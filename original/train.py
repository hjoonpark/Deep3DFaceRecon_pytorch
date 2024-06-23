"""This script is the training script for Deep3DFaceRecon_pytorch
"""

import os
import time
import numpy as np
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.util import genvalconf
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
import shutil

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def parse_dt(start, end):
    # hours, rem = divmod(end-start, 3600)
    # minutes, seconds = divmod(rem, 60)
    hours = 0
    minutes = 0
    seconds = end-start
    return hours, minutes, seconds

def main(rank, world_size, train_opt):
    val_opt = genvalconf(train_opt, isTrain=False)
    
    device = torch.device(rank)
    torch.cuda.set_device(device)
    use_ddp = train_opt.use_ddp
    
    if use_ddp:
        setup(rank, world_size, train_opt.ddp_port)

    train_dataset = create_dataset(train_opt, rank=rank)
    if rank == 0:
        from data.val_dataset import ValDataset
        val_dataloader = DataLoader(ValDataset(train_opt), batch_size=val_opt.batch_size_val, shuffle=False)

    train_dataset_batches = len(train_dataset) // train_opt.batch_size
    
    model = create_model(train_opt)   # create a model given train_opt.model and other options
    model.setup(train_opt)
    model.device = device
    model.parallelize()

    if rank == 0:
        print('The batch number of training images = %d\n, \
            the batch number of validation images = %d'% (train_dataset_batches, -1))
        model.print_networks(train_opt.verbose)
        visualizer = MyVisualizer(train_opt)   # create a visualizer that display/save images and plots

    total_iters = train_dataset_batches * (train_opt.epoch_count - 1)   # the total number of training iterations
    t_data = 0
    t_val = 0
    optimize_time = 0.1
    batch_size = 1 if train_opt.display_per_batch else train_opt.batch_size

    if use_ddp:
        dist.barrier()

    # ---------------------------------- #
    # logistics
    n_train_imgs = len(train_dataset)
    batch_size = train_opt.batch_size
    n_batches = len(train_dataset) // train_opt.batch_size + 1
    total_iters = n_batches * train_opt.n_epochs
    print('n_train_imgs: {:,} | batch_size: {:,} | n_batches: {:,} | total_iters: {:,}'.format(n_train_imgs, batch_size, n_batches, total_iters))
    # ---------------------------------- #
    n_iters = 0
    log_dts = {'print': None, 'train': None, 'train_chkp': None, 'val': None}
    for k in log_dts.keys():
        log_dts[k] = time.time()

    for epoch in range(train_opt.epoch_count, train_opt.n_epochs + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        train_dataset.set_epoch(epoch)
        for batch_idx, train_data in enumerate(train_dataset):  # inner loop within one epoch

            torch.cuda.synchronize()
            model.set_input(train_data)  # unpack train_data from dataset and apply preprocessing
            model.optimize_parameters(isTrain=True)
            torch.cuda.synchronize()

            if use_ddp:
                dist.barrier()

            # ----------------------- #
            # Only for rank == 0
            if rank != 0:
                continue

            n_iters += 1
            end_time = time.time()

            # print losses
            _, _, seconds = parse_dt(start=log_dts['print'], end=end_time)
            if n_iters == 1 or seconds >= train_opt.print_freq:
                log_dts['print'] = end_time

                losses = model.get_current_losses()
                visualizer.print_current_losses(losses, n_iters, epoch, batch_idx, n_batches)
                visualizer.plot_tensorboard_current_losses(total_iters, losses)
                
            # save train progress
            _, _, seconds = parse_dt(start=log_dts['train'], end=end_time)
            if n_iters == 1 or seconds >= train_opt.display_freq:
                log_dts['train'] = end_time

                model.compute_visuals(isTrain=True)
                # visualizer.display_current_results(model.get_current_visuals(), n_iters, epoch, save_results=True, add_image=False, n_imgs=5)
                img_infos = model.img_infos
                folder_name = os.path.join('train', 'e{:}_{:010d}'.format(epoch, n_iters))
                visualizer.display_current_results_JP(model.get_current_visuals(), folder_name=folder_name, name_prefix='', name_suffix='', img_infos=img_infos, n_max_img=5)

                #-------------------------------- #
                save_dir = os.path.join(visualizer.img_dir, folder_name)
                txt_path = os.path.join(save_dir, '%s.txt' % ('values'))
                out_str = model.get_values_ranges()
                with open(txt_path, "w+") as f:
                    f.write(out_str)
                
                print("Train saved:", save_dir)

            # save more train progress
            _, _, seconds = parse_dt(start=log_dts['train_chkp'], end=end_time)
            if n_iters == 1 or seconds >= train_opt.display_overwrite_freq:
                log_dts['train_chkp'] = end_time

                model.compute_visuals(isTrain=True)
                folder_name = 'train_chkp'
                save_dir = os.path.join(visualizer.img_dir, folder_name)
                if os.path.isdir(save_dir):
                    shutil.rmtree(save_dir)

                name_prefix = ''
                img_infos = model.img_infos
                visualizer.display_current_results_JP(model.get_current_visuals(), folder_name, name_prefix=name_prefix, name_suffix='', img_infos=img_infos, n_max_img=15)
                
                txt_path = os.path.join(save_dir, 'values.txt')
                out_str = model.get_values_ranges()
                with open(txt_path, "w+") as f:
                    f.write(out_str)

                print("Train chkp saved:", save_dir)

                save_suffix = 'latest'
                model.save_networks(save_suffix)
            
            # validation
            _, _, seconds = parse_dt(start=log_dts['val'], end=end_time)
            if seconds >= train_opt.evaluation_freq:
                log_dts['val'] = end_time
                folder_name = 'val/epoch_{}'.format(epoch)
                with torch.no_grad():
                    torch.cuda.synchronize()

                    model.eval()
                    for i, val_data in enumerate(val_dataloader):
                        model.set_input(val_data)
                        model.optimize_parameters(isTrain=False)
                        model.compute_visuals(isTrain=False)
                        img_infos = model.img_infos
                        visualizer.display_current_results_JP(model.get_current_visuals(), folder_name, name_prefix=name_prefix, name_suffix='', img_infos=img_infos, n_max_img=15)

                        txt_path = os.path.join(visualizer.img_dir, folder_name, 'values.txt')
                        out_str = model.get_values_ranges()
                        with open(txt_path, "w+") as f:
                            f.write(out_str)

                    torch.cuda.synchronize()
                model.train()      
                print("Validation saved:", folder_name)
                val_dataloader.dataset.use_cache = True

            if use_ddp:
                dist.barrier()

            # if rank == 0 and (total_iters == batch_size or total_iters % train_opt.save_latest_freq == 0):   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     print(train_opt.name)  # it's useful to occasionally show the experiment name on console
            #     save_suffix = 'iter_%d' % total_iters if train_opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)
            
            if use_ddp:
                dist.barrier()
            
            iter_data_time = time.time()

        # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.n_epochs, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        
        # if rank == 0 and epoch % train_opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        if use_ddp:
            dist.barrier()

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    train_opt = TrainOptions().parse()   # get training options
    world_size = train_opt.world_size               

    if train_opt.use_ddp:
        mp.spawn(main, args=(world_size, train_opt), nprocs=world_size, join=True)
    else:
        main(0, world_size, train_opt)

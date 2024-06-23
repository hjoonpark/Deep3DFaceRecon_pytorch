"""This script defines the visualizer for Deep3DFaceRecon_pytorch
"""

import numpy as np
import os, json
import sys
import ntpath
import time
from . import util, html
from .timer import Timer

import matplotlib.pyplot as plt

from subprocess import Popen, PIPE
from torch.utils.tensorboard import SummaryWriter

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

class MyVisualizer:
    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a tensorboard writer
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the optio
        self.name = opt.name
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'results')
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        
        self.timer = Timer()
        self.timer.tic()

        if opt.phase != 'test':
            self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'logs'))
            # create a logging file to store training losses
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)


    def display_current_results(self, visuals, n_iters, epoch, n_imgs, dataset='train', save_results=False, count=0, name=None, add_image=False):
        """Display current results on tensorboad; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            n_iters (int) -- total iterations
            epoch (int) - - the current epoch
            dataset (str) - - 'train' or 'val' or 'test'
        """
        save_paths = []
        for label, image in visuals.items():
            if n_imgs > 0:
                image = image[:n_imgs]
            for i in range(image.shape[0]):
                image_numpy = util.tensor2im(image[i])
                if add_image:
                    self.writer.add_image(label + '%s_%02d'%(dataset, i + count), image_numpy, n_iters, dataformats='HWC')

                if save_results:
                    save_dir = os.path.join(self.img_dir, dataset, 'epoch_%s_%06d'%(epoch, n_iters))
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    if name is not None:
                        img_path = os.path.join(save_dir, '%s.png' % name)
                    else:
                        img_path = os.path.join(save_dir, '%s_%03d.png' % (label, i + count))
                    util.save_image(image_numpy, img_path)
                    save_paths.append(img_path)
        return save_paths

    def display_current_results_JP(self, visuals, folder_name, name_prefix, name_suffix, img_infos, n_max_img):
        """Display current results on tensorboad; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            n_iters (int) -- total iterations
            epoch (int) - - the current epoch
            dataset (str) - - 'train' or 'val' or 'test'
        """
        save_paths = []
        for label, image in visuals.items():
            for i in range(min(n_max_img, image.shape[0])):
                image_numpy = util.tensor2im(image[i])

                # output directory
                save_dir = os.path.join(self.img_dir, folder_name)
                img_dir = save_dir
                os.makedirs(img_dir, exist_ok=True)

                img_name = img_infos[i]['img_name']
                img_path = os.path.join(img_dir, '{}{}{}.png'.format(name_prefix, img_name, name_suffix))
                util.save_image(image_numpy, img_path)

                # save meta data as json
                json_dir = os.path.join(save_dir, 'jsons')
                os.makedirs(json_dir, exist_ok=True)
                md = img_infos[i]
                md_path = os.path.join(json_dir, '{}{}{}.json'.format(name_prefix, img_name, name_suffix))
                with open(md_path, 'w+') as f:
                    json.dump(md, f, indent=4)

                save_paths.append(img_path)
        return save_paths

    def plot_tensorboard_current_losses(self, n_iters, losses, dataset='train'):
        for name, value in losses.items():
            self.writer.add_scalar(name + '/%s'%dataset, value, n_iters)

    # losses: same format as |losses| of plot_tensorboard_current_losses
    def print_current_losses(self, losses, n_iters, epoch, n_epoch_iters, total_n_epoch_iters):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """

        now_str, dt_str = self.timer.toc()
        message = '[%s] %s n_iters: %d, epoch: %d, n_epoch_iters: %d/%d | ' % (now_str, dt_str, n_iters, epoch, n_epoch_iters, total_n_epoch_iters)
        for k, v in losses.items():
            message += '%s: %.4f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


# class Visualizer():
#     """This class includes several functions that can display/save images and print/save logging information.

#     It uses a Python library tensprboardX for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
#     """

#     def __init__(self, opt):
#         """Initialize the Visualizer class

#         Parameters:
#             opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
#         Step 1: Cache the training/test options
#         Step 2: create a tensorboard writer
#         Step 3: create an HTML object for saveing HTML filters
#         Step 4: create a logging file to store training losses
#         """
#         self.opt = opt  # cache the option
#         self.use_html = opt.isTrain and not opt.no_html
#         self.writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'logs', opt.name))
#         self.win_size = opt.display_winsize
#         self.name = opt.name
#         self.saved = False
#         if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
#             self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
#             self.img_dir = os.path.join(self.web_dir, 'images')
#             print('create web directory %s...' % self.web_dir)
#             util.mkdirs([self.web_dir, self.img_dir])
#         # create a logging file to store training losses
#         self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
#         with open(self.log_name, "a") as log_file:
#             now = time.strftime("%c")
#             log_file.write('================ Training Loss (%s) ================\n' % now)

#     def reset(self):
#         """Reset the self.saved status"""
#         self.saved = False


#     def display_current_results(self, visuals, n_iters, epoch, save_result):
#         """Display current results on tensorboad; save current results to an HTML file.

#         Parameters:
#             visuals (OrderedDict) - - dictionary of images to display or save
#             n_iters (int) -- total iterations
#             epoch (int) - - the current epoch
#             save_result (bool) - - if save the current results to an HTML file
#         """
#         for label, image in visuals.items():
#             self.writer.add_image(label, util.tensor2im(image), n_iters, dataformats='HWC')

#         if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
#             self.saved = True
#             # save images to the disk
#             for label, image in visuals.items():
#                 image_numpy = util.tensor2im(image)
#                 print("....................", image_numpy.shape)
#                 img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
#                 util.save_image(image_numpy, img_path)

#             # update website
#             webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
#             for n in range(epoch, 0, -1):
#                 webpage.add_header('epoch [%d]' % n)
#                 ims, txts, links = [], [], []

#                 for label, image_numpy in visuals.items():
#                     image_numpy = util.tensor2im(image)
#                     img_path = 'epoch%.3d_%s.png' % (n, label)
#                     ims.append(img_path)
#                     txts.append(label)
#                     links.append(img_path)
#                 webpage.add_images(ims, txts, links, width=self.win_size)
#             webpage.save()

#     def plot_tensorboard_current_losses(self, n_iters, losses):
#         # G_loss_collection = {}
#         # D_loss_collection = {}
#         # for name, value in losses.items():
#         #     if 'G' in name or 'NCE' in name or 'idt' in name:
#         #         G_loss_collection[name] = value
#         #     else:
#         #         D_loss_collection[name] = value
#         # self.writer.add_scalars('G_collec', G_loss_collection, n_iters)
#         # self.writer.add_scalars('D_collec', D_loss_collection, n_iters)
#         for name, value in losses.items():
#             self.writer.add_scalar(name, value, n_iters)

#     # losses: same format as |losses| of plot_tensorboard_current_losses
#     def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
#         """print current losses on console; also save the losses to the disk

#         Parameters:
#             epoch (int) -- current epoch
#             iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
#             losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
#             t_comp (float) -- computational time per data point (normalized by batch_size)
#             t_data (float) -- data loading time per data point (normalized by batch_size)
#         """
#         message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
#         for k, v in losses.items():
#             message += '%s: %.3f ' % (k, v)

#         print(message)  # print the message
#         with open(self.log_name, "a") as log_file:
#             log_file.write('%s\n' % message)  # save the message


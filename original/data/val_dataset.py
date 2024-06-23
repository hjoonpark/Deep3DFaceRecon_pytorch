import os, glob
import numpy as np
import torch
from PIL import Image
import multiprocessing as mp
from ctypes import c_float
from models.retinaface import RetinaFace, load_model
from models.retinaface.utils.box_utils import align_for_dense_lm_detection
import torchvision.transforms as transforms

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()

        self.img_paths = []
        with open(opt.flist_val, 'r') as f:
            paths = f.readlines()
            for path in paths:
                self.img_paths.append(path.strip())

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.device = torch.device("cuda")
        self.retinaface_net = RetinaFace()
        load_path = 'models/retinaface/retinaface_resnet50_2020-07-20_old_torch.pth'
        self.retinaface_net = load_model(self.retinaface_net, load_path, load_to_cpu=False).eval()

        # pre-align test images {index (int) : image (torch.tensor)}
        nb_samples = len(self.img_paths)
        c, h, w = 3, 224, 224
        shared_array_base = mp.Array(c_float, nb_samples*c*h*w)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(nb_samples, c, h, w)
        self.shared_array = torch.from_numpy(shared_array)
        self.use_cache = False

    def __getitem__(self, index):
        if not self.use_cache:
            # only need input image
            img_path = self.img_paths[index]
            print("  Caching test image[{}]={}".format(index, os.path.basename(img_path)))
            img_raw_PIL = Image.open(img_path).convert('RGB')
            img_raw_np = np.array(img_raw_PIL)
            raw_height, raw_width, _ = img_raw_np.shape

            # Preprocess
            paded = self.retinaface_net.resize_and_pad(img_raw_np)
            pads = paded['pads']
            image_torch = torch.from_numpy((paded['image'])).permute(2,0,1).unsqueeze(0)

            # Face detection
            scores, boxes, lm_5p= self.retinaface_net(image_torch, postprocess=True)
            # conf = F.softmax(conf, dim=-1)
            
            scores = scores.detach().cpu().numpy().astype(np.float64)
            boxes = boxes.cpu().numpy()
            lm_5p = lm_5p.cpu().numpy().reshape([-1, 2])

            # Unpad
            boxes, lm_5p = self.retinaface_net.unpad_and_resize(pads, bboxes=boxes, lm_5p=lm_5p, raw_height=raw_height, raw_width=raw_width)

            # Align
            img_small, _, _, _ = align_for_dense_lm_detection(img_raw_np, lm_5p, None)  # align for 68 landmark detection

            # Cache
            img_small = self.transform(img_small)
            self.shared_array[index] = img_small

        imgs = self.shared_array[index]
        return {"imgs": imgs, "im_paths": self.img_paths[index]}
        
    def __len__(self):
        return len(self.img_paths)


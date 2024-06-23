import os, glob
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from models.retinaface import RetinaFace, load_model
import torchvision.transforms as transforms
from models.retinaface.utils.box_utils import decode, decode_landm, priorbox, pad_to_size, unpad_from_size, align_for_dense_lm_detection
import albumentations as A
from torchvision.ops import nms
from datasets.util.preprocessor import Preprocessor
import face_alignment

import cv2
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy.io import loadmat

import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda")
    max_size: int = 1024

    # Init model
    net = RetinaFace()

    # Load pretrained weights
    model_path = "/home/joonp/1_Projects/HRN/assets/pretrained_models/retinaface_resnet50_2020-07-20_old_torch.pth"
    assert os.path.isfile(model_path)
    net = load_model(net, model_path, load_to_cpu=False)
    net.eval()
    net = net.to(device)

    # Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
    # Apply Contrast Limited Adaptive Histogram Equalization to the input image, p: probability of applying the transform
    augment = A.Compose([A.LongestMaxSize(max_size=max_size, p=1), A.Normalize(p=1)])

    # Test images
    # img_dir = "/home/joonp/1_Projects/image-to-avatar/data/deep3d/test/images"
    img_dir = "/home/joonp/1_Projects/HRN/assets/examples/single_view_image"
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    prior_box = priorbox(image_size=(max_size, max_size)).to(device)

    variance = [0.1, 0.2]
    confidence_threshold: float = 0.7
    nms_threshold: float = 0.4

    save_dir = "debug"
    os.makedirs(save_dir, exist_ok=True)

    bfm_dir = "/home/joonp/1_Projects/image-to-avatar/data/BFM"
    lm3d_std = Preprocessor.load_lm3d(bfm_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    
    bbox_params = None
    DETECT_DENSE_LANDMARKS = False
    if DETECT_DENSE_LANDMARKS:
        lm_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    for i, p in enumerate(img_paths):
        img_raw_PIL = Image.open(p).convert("RGB")
        img_raw_np = np.array(img_raw_PIL)
        
        raw_height, raw_width, _ = img_raw_np.shape
        scale_landmarks = torch.from_numpy(np.tile([max_size, max_size], 5)).to(device).float()

        # Resize image then add padding if needed
        image_resized = augment(image=img_raw_np)['image']
        paded = pad_to_size(target_size=(max_size, max_size), image=image_resized)
        pads = paded['pads']
        image_torch = torch.from_numpy((paded['image'])).permute(2,0,1).unsqueeze(0).to(device)

        # Face detection
        loc, conf, land = net(image_torch)
        conf = F.softmax(conf, dim=-1)

        # Extract bboxes
        boxes = decode(loc.data[0], prior_box, variance)
        scale_bboxes = torch.from_numpy(np.tile([max_size, max_size], 2)).to(device).float()
        boxes *= scale_bboxes

        # Extract landmarks
        lm_5p_candidates = decode_landm(land.data[0], prior_box, variance)
        lm_5p_candidates *= scale_landmarks

        # Ignore low scores
        scores = conf[0][:, 1]
        valid_index = scores > confidence_threshold
        scores = scores[valid_index]
        boxes = boxes[valid_index]
        lm_5p_candidates = lm_5p_candidates[valid_index]

        # Sort from high to low
        order = scores.argsort(descending=True)
        boxes = boxes[order]
        lm_5p_candidates = lm_5p_candidates[order]
        scores = scores[order]

        # NMS (Non-maximum Suppression)
        annotations: List[Dict[str, Union[List, float]]] = []
        keep = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep, :].int()
        lm_5p = lm_5p_candidates[keep]

        scores = scores[keep].detach().cpu().numpy().astype(np.float64)
        boxes = boxes.cpu().numpy()
        lm_5p = lm_5p.cpu().numpy().reshape([-1, 2])

        # Unpad
        unpadded = unpad_from_size(pads, bboxes=boxes, keypoints=lm_5p)
        resize_coeff = max(raw_height, raw_width) / max_size
        boxes = (unpadded['bboxes'] * resize_coeff).astype(int)
        lm_5p = (unpadded['keypoints'].reshape(-1, 10) * resize_coeff).astype(int)

        assert len(boxes) <= 1
        if len(boxes) == 0:
            assert False, "No landmarks detected! Is there a face?"

        for box_id, bbox in enumerate(boxes):
            x_min, y_min, x_max, y_max = bbox

            x_min = np.clip(x_min, 0, raw_width - 1)
            x_max = np.clip(x_max, x_min + 1, raw_width - 1)

            if x_min >= x_max:
                continue

            y_min = np.clip(y_min, 0, raw_height - 1)
            y_max = np.clip(y_max, y_min + 1, raw_height - 1)

            if y_min >= y_max:
                continue

            annotations += [{
                'bbox':
                bbox.tolist(),
                'score':
                scores[box_id],
                'landmarks':
                lm_5p[box_id].reshape(-1, 2).tolist(),
            }]

        # ==================================================
        # Align and resize image to (3, 224, 224)
        # Using (68, 2) image landmarks, we use 5 (preset indices) to align image
        # five_points = np.array(annotations[0]).reshape()

        R = 1
        C = 4 if DETECT_DENSE_LANDMARKS else 2
        fig = plt.figure(figsize=(5*C, 5*R+2))
        ax = fig.add_subplot(R, C, 1)
        ax.imshow(img_raw_np)
        L = lm_5p.squeeze().reshape((5, 2))
        ax.scatter(L[:,0], L[:,1])
        ax.set_title("[A] {} ({:.1f}, {:.1f}) {}".format(img_raw_np.shape, img_raw_np.min(), img_raw_np.max(), img_raw_np.dtype))

        # align_for_dense_lm_detection(img=[1024, 1024, 3] (0, 255) uint8, five_points=[5, 2])
        img_small, scale, bbox, bbox_params = align_for_dense_lm_detection(img_raw_np, lm_5p, bbox_params)  # align for 68 landmark detection

        if not DETECT_DENSE_LANDMARKS:
            lmk_raw = lm_5p.squeeze().reshape((5, 2))
            lmk_raw[:, 0] = lmk_raw[:, 0] - bbox[0]
            lmk_raw[:, 1] = lmk_raw[:, 1] - bbox[1]
            lmk_raw = lmk_raw[:, :2] * scale
            # lmk_raw[:, -1] = raw_height - 1 - lmk_raw[:, -1]
        
        ax = fig.add_subplot(R, C, 2)
        ax.imshow(img_small)
        if not DETECT_DENSE_LANDMARKS:
            ax.scatter(lmk_raw[:,0], lmk_raw[:,1])
        ax.set_title("[B] {} ({:.1f}, {:.1f}) {}".format(img_small.shape, img_small.min(), img_small.max(), img_small.dtype))
        
        if DETECT_DENSE_LANDMARKS:
            # get_landmarks_from_image(img=[224, 224, 3] (0., 255.) float32)
            img_small_float32 = img_small.astype(np.float32)
            lmk_small = lm_detector.get_landmarks_from_image(img_small_float32)[0] # input: (224, 224, 3)

            ax = fig.add_subplot(R, C, 3)
            ax.imshow(img_small)
            ax.scatter(lmk_small[:,0], lmk_small[:,1])
            ax.set_title("[C] {} ({:.1f}, {:.1f}) {}\nwith landmarks".format(img_small.shape, img_small.min(), img_small.max(), img_small.dtype))
            
            lmk_raw = lmk_small[:, :2] / scale
            lmk_raw[:, 0] = lmk_raw[:, 0] + bbox[0]
            lmk_raw[:, 1] = lmk_raw[:, 1] + bbox[1]
            lmk_raw[:, -1] = raw_height - 1 - lmk_raw[:, -1]

            lmk_raw_5p = Preprocessor.extract_5p(lmk_raw)
            _, img_small_aligned_PIL, lmk_small_aligned, _ = Preprocessor.align_img(img_raw_PIL, lmk_raw_5p, lm3d_std)
            lmk_small_aligned[:, -1] = img_small_aligned_PIL.size[1] - 1 - lmk_small_aligned[:, -1]

            ax = fig.add_subplot(R, C, 4)
            I = np.array(img_small_aligned_PIL)
            ax.imshow(I)
            lmk = lmk_small_aligned
            ax.scatter(lmk[:,0], lmk[:,1])
            ax.set_title("[D] Final output\n{} ({:.1f}, {:.1f}) {}\nwith landmarks".format(I.shape, I.min(), I.max(), I.dtype))

            img_small_aligned = torch.tensor(np.array(img_small_aligned_PIL) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            lmk_small_aligned = torch.tensor(lmk_small_aligned).unsqueeze(0)

            print("img_small_aligned: {} ({:.1f}, {:.1f}) {}".format(img_small_aligned.shape, img_small_aligned.min(), img_small_aligned.max(), img_small_aligned.dtype), ", lmk_small_aligned:", lmk_small_aligned.shape)
        plt.suptitle(p)
        plt.tight_layout()
        plt.savefig("debug/final_{}.jpg".format(i), dpi=300)
        print(i)

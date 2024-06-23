import os
import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, Tuple
import itertools
import numpy as np

from models.retinaface.net import MobileNetV1 as MobileNetV1
from models.retinaface.net import FPN as FPN
from models.retinaface.net import SSH as SSH
from models.retinaface.utils.box_utils import decode, decode_landm, priorbox, pad_to_size, unpad_from_size, align_for_dense_lm_detection
from typing import Any, Dict, List, Optional, Union, Tuple
from torchvision.ops import nms
import albumentations as A

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('  Missing pretrained-model keys:{}'.format(len(missing_keys)))
    print('  Unused pretrained-model checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('  Used pretrained-model keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    assert os.path.isfile(pretrained_path)
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, name:str="Resnet50", pretrain:bool=False, \
                in_channel:int=256, out_channel:int=256, \
                max_size:int=1024, 
                return_layers:Dict[str, int]= {'layer2': 1, 'layer3': 2, 'layer4': 3}):
        super().__init__()

        backbone = None
        self.max_size = max_size
        if name == 'mobilenet0.25':
            raise NotImplementedError(f'Only Resnet50 backbone is supported but got {name}')
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif name == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=pretrain)


        self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
        in_channel_list = [
            in_channel * 2,
            in_channel * 4,
            in_channel * 8,
        ]
        self.fpn = FPN(in_channel_list, out_channel)
        self.ssh1 = SSH(out_channel, out_channel)
        self.ssh2 = SSH(out_channel, out_channel)
        self.ssh3 = SSH(out_channel, out_channel)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channel)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channel)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channel)

        params = itertools.chain(backbone.parameters(), self.fpn.parameters(), self.ssh1.parameters(), self.ssh2.parameters(), self.ssh3.parameters(), self.ClassHead.parameters(), self.BboxHead.parameters(), self.LandmarkHead.parameters())
        for param in params:
            param.requires_grad = False

        self.resizer = A.Compose([A.LongestMaxSize(max_size=max_size, p=1), A.Normalize(p=1)])

        # Hyperparameters
        self.prior_box = priorbox(image_size=(max_size, max_size))
        self.variance = [0.1, 0.2]
        self.confidence_threshold: float = 0.7
        self.nms_threshold: float = 0.4
        self.scale_bboxes = torch.from_numpy(np.tile([self.max_size, self.max_size], reps=2)).float()
        self.scale_landmarks = torch.from_numpy(np.tile([max_size, max_size], reps=5)).float()
        
    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def resize_and_pad(self, img_raw):
        """
        Input
            img_raw: np.array (H, W, C)
        Output
            img_torch: np.array (H, W, C) - resized and padded
        """
        # Resize image then add padding if needed
        image_resized = self.resizer(image=img_raw)['image']
        paded = pad_to_size(target_size=(self.max_size, self.max_size), image=image_resized)

        return paded
    
    def unpad_and_resize(self, pads, bboxes, lm_5p, raw_height, raw_width):
        unpadded = unpad_from_size(pads, bboxes=bboxes, keypoints=lm_5p)
        resize_coeff = max(raw_height, raw_width) / self.max_size
        
        bboxes_out = (unpadded['bboxes'] * resize_coeff).astype(int)
        lm_5p_out = (unpadded['keypoints'].reshape(-1, 10) * resize_coeff).astype(int)

        return bboxes_out, lm_5p_out
    
    def forward(self, inputs, postprocess=True):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        phase = 'test'
        if phase == 'train':
            raise NotImplementedError(f'We do not train here.')
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        if not postprocess:
            return output
    
        loc, conf, land = output

        # Extract bboxes
        boxes = decode(loc.data[0], self.prior_box, self.variance)
        boxes *= self.scale_bboxes

        # Extract landmarks
        lm_5p_candidates = decode_landm(land.data[0], self.prior_box, self.variance)
        lm_5p_candidates *= self.scale_landmarks
    
        # Ignore low scores
        scores = conf[0][:, 1]
        valid_index = scores > self.confidence_threshold
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
        keep = nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep, :].int()
        lm_5p = lm_5p_candidates[keep]
        scores = scores[keep]

        return scores, boxes, lm_5p
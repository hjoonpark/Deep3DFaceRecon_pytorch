"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch, os
from .base_model import BaseModel
from . import networks
from .bfm import ParametricFaceModel
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss
from util import util 
from util.nvdiffrast import MeshRenderer
from util.preprocess import estimate_norm_torch

import trimesh
from scipy.io import savemat

class FaceReconModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='./BFM')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)
        parser.add_argument('--use_opengl', type=util.str2bool, nargs='?', const=True, default=True, help='use opengl context or not')

        if is_train:
            # training parameters
            parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'], help='face recog network structure')
            parser.add_argument('--net_recog_path', type=str, default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
            parser.add_argument('--use_crop_face', type=util.str2bool, nargs='?', const=True, default=False, help='use crop mask for photo loss')
            parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False, help='use predefined M for predicted face')

            
            # augmentation parameters
            parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')
            parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')
            parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')

            # loss weights
            parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
            parser.add_argument('--w_color', type=float, default=1.92, help='weight for loss loss')
            parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
            parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
            parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
            parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
            parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
            parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
            parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')



        opt, _ = parser.parse_known_args()
        parser.set_defaults(
                focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.
            )
        if is_train:
            parser.set_defaults(
                use_crop_face=True, use_predef_M=False
            )
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        
        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer']

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        )

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )
        
        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center), use_opengl=opt.use_opengl
        )

        if self.isTrain:
            self.loss_names = ['all', 'feat', 'color', 'lm', 'reg', 'gamma', 'reflc']

            self.net_recog = networks.define_net_recog(
                net_recog=opt.net_recog, pretrained_path=opt.net_recog_path
                )
            # loss func name: (compute_%s_loss) % loss_name
            self.compute_feat_loss = perceptual_loss
            self.comupte_color_loss = photo_loss
            self.compute_lm_loss = landmark_loss
            self.compute_reg_loss = reg_loss
            self.compute_reflc_loss = reflectance_loss

            self.optimizer = torch.optim.Adam(self.net_recon.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
            self.parallel_names += ['net_recog']
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

        self.angles = (torch.tensor([[0, 45, 0], [0,0,0], [0, -45, 0]])[:, None, :] * np.pi/180.0)
        self.trans = torch.zeros((3, 1, 3))
        self.photo_loss_mask = None

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device) 
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device)  if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def forward(self):
        output_coeff = self.net_recon(self.input_img)
        self.facemodel.to(self.device)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(output_coeff)
        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
        
        self.pred_coeffs_dict = self.facemodel.split_coeff(output_coeff)


    def compute_losses(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        assert self.net_recog.training == False
        trans_m = self.trans_m
        if not self.opt.use_predef_M:
            trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])

        pred_feat = self.net_recog(self.pred_face, trans_m)
        gt_feat = self.net_recog(self.input_img, self.trans_m)
        self.loss_feat = self.opt.w_feat * self.compute_feat_loss(pred_feat, gt_feat)

        face_mask = self.pred_mask
        if self.opt.use_crop_face:
            face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)
        
        self.photo_loss_mask = self.atten_mask * face_mask
        self.loss_color = self.opt.w_color * self.comupte_color_loss(self.pred_face, self.input_img, self.photo_loss_mask)
        
        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        self.loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(self.pred_tex, self.facemodel.skin_mask)

        self.loss_all = self.loss_feat + self.loss_color + self.loss_reg + self.loss_gamma \
                        + self.loss_lm + self.loss_reflc
            

    def optimize_parameters(self, isTrain=True):
        self.forward()               
        """Update network weights; it will be called in every training iteration."""
        if isTrain:
            self.compute_losses()
            self.optimizer.zero_grad()  
            self.loss_all.backward()         
            self.optimizer.step()        

    def compute_visuals(self, isTrain: bool):
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            out_imgs_row1 = [input_img_numpy]
            
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
            out_imgs_row1.append(output_vis_numpy_raw)
            
            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')
                out_imgs_row1.append(output_vis_numpy)
            

            # mask
            if isTrain:
                if self.photo_loss_mask is not None:
                    out_imgs_row1.append(255. * self.photo_loss_mask.detach().permute(0, 2, 3, 1).repeat(1,1,1,3).cpu().numpy())

            # Rotate mesh
            for angle, trans in zip(self.angles, self.trans):
                face_shape = self.facemodel.face_shape.detach()
                rotation = self.facemodel.compute_rotation(angle.to(face_shape.device))
                face_shape_transformed = self.facemodel.transform(face_shape, rotation, trans.to(face_shape.device))
                pred_vertex = self.facemodel.to_camera(face_shape_transformed)
                vertex_normals = self.facemodel.compute_norm(pred_vertex)
                pred_color = torch.ones_like(self.pred_color) * 0.5
                _, _, pred_face = self.renderer(pred_vertex, self.facemodel.face_buf, pred_color)
                pred_face = (pred_face-pred_face.min()) / (pred_face.max()-pred_face.min())
                # pred_face = torch.clip(pred_face, 0, 1)
                out_imgs_row1.append(255. * pred_face.detach().permute(0, 2, 3, 1).cpu().numpy())

            self.output_vis = np.concatenate(out_imgs_row1, axis=-2)
            self.output_vis = torch.tensor(self.output_vis / 255., dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

        # [JP] keep meta data of images
        self.img_infos = []
        for i in range(len(self.image_paths)):
            self.img_infos.append({
            'img_name': os.path.basename(self.image_paths[i]).replace('.', '_'),
        })
    def save_mesh(self, name):

        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8), process=False)
        mesh.export(name)

    def save_coeff(self,name):

        pred_coeffs = {key:self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        pred_lm = np.stack([pred_lm[:,:,0],self.input_img.shape[2]-1-pred_lm[:,:,1]],axis=2) # transfer to image coordinate
        pred_coeffs['lm68'] = pred_lm
        savemat(name,pred_coeffs)



    def get_values_ranges(self):
        # [JP]
        with torch.no_grad():
            N = 40
            tensors = {
                'self.input_img': self.input_img.detach().cpu().numpy(),
                'self.pred_face': self.pred_face.detach().cpu().numpy(),
                'self.pred_mask': self.pred_mask.detach().cpu().numpy(),
                'self.pred_tex': self.pred_tex.detach().cpu().numpy(),
            }

            out_str = ''
            
            i = 1
            for name, c in self.pred_coeffs_dict.items():
                out_str += '{}. {} {}{}min/max=({:.3f}, {:.3f}) mean/std=({:.3f}, {:.3f})\n'.format(i, name, list(c.shape), " "*(N-len(str(list(c.shape)))-len(name)-len(str(i))), c.min(), c.max(), c.mean(), c.max())
                i += 1
            out_str += '\n'

            for name, t in tensors.items():
                # (B, H, W, 3) for pred_tex. Others are (B, c, H, W)
                out_str += '{}. {} {}{}min/max=({:.3f}, {:.3f}) {}\n'.format(i, name, t.shape, " "*(N-len(str(list(t.shape)))-len(name)-len(str(i))), t.min(), t.max(), t.dtype)
                i += 1

            out_str += '\nself.pred_coeffs_dict["trans"] z-axis:\n['
            for z in self.pred_coeffs_dict['trans'][:, 2].detach().cpu().numpy().flatten().tolist():
                out_str += '{:.2f}, '.format(z)
            out_str += ']\n'
            return out_str

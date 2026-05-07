import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import os
import hydra
import torch
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

# Global variable to store the desired device for monkeypatching
PHALP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Monkeypatch torch.load to set weights_only=False by default to avoid issues with newer PyTorch versions
orig_load = torch.load
def new_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    if not torch.cuda.is_available() and 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu'
    return orig_load(*args, **kwargs)
torch.load = new_load

# Monkeypatch torch.Tensor.cuda
orig_cuda = torch.Tensor.cuda
def new_cuda(self, *args, **kwargs):
    return self.to(device=PHALP_DEVICE)
torch.Tensor.cuda = new_cuda

# Monkeypatch torch.nn.Module.cuda
orig_module_cuda = torch.nn.Module.cuda
def new_module_cuda(self, *args, **kwargs):
    return self.to(device=PHALP_DEVICE)
torch.nn.Module.cuda = new_module_cuda

# Monkeypatch torch.Tensor.to
orig_tensor_to = torch.Tensor.to
def new_tensor_to(self, *args, **kwargs):
    if len(args) > 0:
        if isinstance(args[0], torch.device) and args[0].type == 'cuda' and PHALP_DEVICE == 'cpu':
            return orig_tensor_to(self, torch.device('cpu'), *args[1:], **kwargs)
        if isinstance(args[0], str) and 'cuda' in args[0] and PHALP_DEVICE == 'cpu':
            return orig_tensor_to(self, 'cpu', *args[1:], **kwargs)
    if 'device' in kwargs and isinstance(kwargs['device'], str) and 'cuda' in kwargs['device'] and PHALP_DEVICE == 'cpu':
        kwargs['device'] = 'cpu'
    return orig_tensor_to(self, *args, **kwargs)
torch.Tensor.to = new_tensor_to

# Monkeypatch torch.nn.Module.to
orig_module_to = torch.nn.Module.to
def new_module_to(self, *args, **kwargs):
    if len(args) > 0:
        if isinstance(args[0], torch.device) and args[0].type == 'cuda' and PHALP_DEVICE == 'cpu':
            return orig_module_to(self, torch.device('cpu'), *args[1:], **kwargs)
        if isinstance(args[0], str) and 'cuda' in args[0] and PHALP_DEVICE == 'cpu':
            return orig_module_to(self, 'cpu', *args[1:], **kwargs)
    if 'device' in kwargs and isinstance(kwargs['device'], str) and 'cuda' in kwargs['device'] and PHALP_DEVICE == 'cpu':
        kwargs['device'] = 'cpu'
    return orig_module_to(self, *args, **kwargs)
torch.nn.Module.to = new_module_to

# Monkeypatch phalp.visualize.visualizer.Visualizer to avoid hardcoded cuda
import phalp.visualize.visualizer as phalp_visualizer
orig_visualizer_init = phalp_visualizer.Visualizer.__init__
def new_visualizer_init(self, cfg, hmar):
    orig_visualizer_init(self, cfg, hmar)
    self.device = PHALP_DEVICE
    if hasattr(self, 'face_detector') and self.face_detector is not None:
        self.face_detector.to(self.device)
phalp_visualizer.Visualizer.__init__ = new_visualizer_init

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.configs.base import CACHE_DIR
from phalp.utils.utils import smpl_to_pose_camera_vector
from phalp.utils.utils_detectron2 import DefaultPredictor_Lazy
import phalp.utils.utils_detectron2 as utils_detectron2
from detectron2 import model_zoo
from detectron2.modeling import build_model
utils_detectron2.build_model = build_model

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out
    
class HMR2023TextureSampler(HMR2Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/bmap_256.npy'))
        fmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256         #self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.   #self.cfg.EXTRA.FOCAL_LENGTH

        import neural_renderer as nr
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
                                          image_size=self.img_size,
                                          light_intensity_ambient=1,
                                          light_intensity_directional=0,
                                          anti_aliasing=False)

    def forward(self, x):
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # from hmr2.models.prohmr_texture import unproject_uvmap_to_mesh

        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = (fmap >= 0)

            fmap_flat = fmap[valid_mask]      # N
            bmap_flat = bmap[valid_mask,:]    # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :] # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat) # B,N,3

            return map_verts, valid_mask

        pred_verts = model_out['pred_vertices'] + model_out['pred_cam_t'].unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor) # B,N,3

        # Project map_verts to image using K,R,t
        # map_verts_view = einsum('bij,bnj->bni', R, map_verts) + t # R=I t=0
        focal = self.focal_length / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3] # B,N,2
        map_verts_depth = map_verts[:, :, 2] # B,N

        # Render Depth. Annoying but we need to create this
        K = torch.eye(3, device=device)
        K[0, 0] = K[1, 1] = self.focal_length
        K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
        K = K.unsqueeze(0)
        R = torch.eye(3, device=device).unsqueeze(0)
        t = torch.zeros(3, device=device).unsqueeze(0)
        rend_depth = self.neural_renderer(pred_verts,
                                        face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(),
                                        # textures=texture_atlas_rgb,
                                        mode='depth',
                                        K=K, R=R, t=t)

        rend_depth_at_proj = torch.nn.functional.grid_sample(rend_depth[:,None,:,:], map_verts_proj[:,None,:,:]) # B,1,1,N
        rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1) # B,N

        img_rgba = torch.cat([batch['img'], batch['mask'][:,None,:,:]], dim=1) # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:,None,:,:]) # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2) # B,4,N

        visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4) # B,N
        img_rgba_at_proj[:,3,:][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((batch['img'].shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        out = {
            'uv_image':  uv_image,
            'uv_vector' : self.hmar_old.process_uv_image(uv_image),
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam':  model_out['pred_cam'],
        }
        return out

from phalp.external.deep_sort_.detection import Detection

class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2Predictor(self.cfg)
        if self.device.type == 'cuda':
            log.info("Moving HMR2 to CPU to save GPU memory...")
            self.HMAR.to("cpu")

    def setup_detectron2(self):
        log.info("Loading Detection model...")
        if self.cfg.phalp.detector == 'maskrcnn':
            # Use a smaller model to save memory: ResNet-50 instead of RegNet-400ep-LSJ
            self.detectron2_cfg = model_zoo.get_config('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', trained=True)
            self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            self.detectron2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
            self.detectron2_cfg.MODEL.DEVICE = str(self.device)
            # Reduce input size
            self.detectron2_cfg.INPUT.MIN_SIZE_TEST = 400
            self.detectron2_cfg.INPUT.MAX_SIZE_TEST = 600
            
            self.detector       = DefaultPredictor_Lazy(self.detectron2_cfg)
            self.class_names    = self.detector.metadata.get('thing_classes')
        else:
            # Fallback to original behavior for other detectors
            super().setup_detectron2()
            if hasattr(self, 'detectron2_cfg'):
                self.detectron2_cfg.MODEL.DEVICE = str(self.device)

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        log.info(f"Frame {t_}: Detecting humans...")
        (
            pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes, 
            ground_truth_track_id, ground_truth_annotations
        ) =  super().get_detections(image, frame_name, t_, additional_data, measurments)

        # Pad bounding boxes 
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)

        return (
            pred_bbox, pred_bbox_padded, pred_masks, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        )
    
    def get_human_features(self, image, seg_mask, bbox, bbox_pad, score, frame_name, cls_id, t_, measurments, gt=1, ann=None, extra_data=None):
        if self.device.type == 'cuda':
            log.info(f"Frame {t_}: Extracting features... GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        else:
            log.info(f"Frame {t_}: Extracting features...")
        NPEOPLE = len(score)

        if(NPEOPLE==0): return []

        img_height, img_width, new_image_size, left, top = measurments                
        ratio = 1.0/int(new_image_size)*self.cfg.render.res
        masked_image_list = []
        center_list = []
        scale_list = []
        rles_list = []
        selected_ids = []
        for p_ in range(NPEOPLE):
            if bbox[p_][2]-bbox[p_][0]<self.cfg.phalp.small_w or bbox[p_][3]-bbox[p_][1]<self.cfg.phalp.small_h:
                continue
            masked_image, center_, scale_, rles, center_pad, scale_pad = self.get_croped_image(image, bbox[p_], bbox_pad[p_], seg_mask[p_])
            masked_image_list.append(masked_image)
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            rles_list.append(rles)
            selected_ids.append(p_)
        
        if(len(masked_image_list)==0): return []

        masked_image_list = torch.stack(masked_image_list, dim=0)
        BS = masked_image_list.size(0)
        
        # HMR2 device
        hmr_device = next(self.HMAR.parameters()).device

        with torch.no_grad():
            extra_args      = {}
            hmar_out        = self.HMAR(masked_image_list.to(hmr_device), **extra_args) 
            uv_vector       = hmar_out['uv_vector']
            appe_embedding  = self.HMAR.autoencoder_hmar(uv_vector, en=True)
            appe_embedding  = appe_embedding.view(appe_embedding.shape[0], -1)
            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam  = self.HMAR.get_3d_parameters(hmar_out['pose_smpl'], hmar_out['pred_cam'],
                                                                                               center=(np.array(center_list) + np.array([left, top]))*ratio,
                                                                                               img_size=self.cfg.render.res,
                                                                                               scale=np.max(np.array(scale_list), axis=1, keepdims=True)*ratio)
            pred_smpl_params = [{k:v[i].cpu().numpy() for k,v in pred_smpl_params.items()} for i in range(BS)]
            
            if(self.cfg.phalp.pose_distance=="joints"):
                pose_embedding  = pred_joints.cpu().view(BS, -1)
            elif(self.cfg.phalp.pose_distance=="smpl"):
                pose_embedding = []
                for i in range(BS):
                    pose_embedding_  = smpl_to_pose_camera_vector(pred_smpl_params[i], pred_cam[i])
                    pose_embedding.append(torch.from_numpy(pose_embedding_[0]))
                pose_embedding = torch.stack(pose_embedding, dim=0)
            else:
                raise ValueError("Unknown pose distance")
            pred_joints_2d_ = pred_joints_2d.reshape(BS,-1)/self.cfg.render.res
            pred_cam_ = pred_cam.view(BS, -1)
            pred_joints_2d_.contiguous()
            pred_cam_.contiguous()
            loca_embedding  = torch.cat((pred_joints_2d_, pred_cam_, pred_cam_, pred_cam_), 1)
        
        # keeping it here for legacy reasons (T3DP), but it is not used.
        full_embedding    = torch.cat((appe_embedding.cpu(), pose_embedding, loca_embedding.cpu()), 1)
        
        detection_data_list = []
        for i, p_ in enumerate(selected_ids):
            detection_data = {
                                "bbox"            : np.array([bbox[p_][0], bbox[p_][1], (bbox[p_][2] - bbox[p_][0]), (bbox[p_][3] - bbox[p_][1])]),
                                "mask"            : rles_list[i],
                                "conf"            : score[p_], 
                                
                                "appe"            : appe_embedding[i].cpu().numpy(), 
                                "pose"            : pose_embedding[i].numpy(), 
                                "loca"            : loca_embedding[i].cpu().numpy(), 
                                "uv"              : uv_vector[i].cpu().numpy(), 
                                
                                "embedding"       : full_embedding[i], 
                                "center"          : center_list[i],
                                "scale"           : scale_list[i],
                                "smpl"            : pred_smpl_params[i],
                                "camera"          : pred_cam_[i].cpu().numpy(),
                                "camera_bbox"     : hmar_out['pred_cam'][i].cpu().numpy(),
                                "3d_joints"       : pred_joints[i].cpu().numpy(),
                                "2d_joints"       : pred_joints_2d_[i].cpu().numpy(),
                                "size"            : [img_height, img_width],
                                "img_path"        : frame_name,
                                "img_name"        : frame_name.split('/')[-1] if isinstance(frame_name, str) else None,
                                "class_name"      : cls_id[p_],
                                "time"            : t_,

                                "ground_truth"    : gt[p_],
                                "annotations"     : ann[p_],
                                "extra_data"      : extra_data[p_] if extra_data is not None else None
                            }
            detection_data_list.append(Detection(detection_data))

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return detection_data_list
    
    def get_uv_distance(self, t_uv, d_uv):
        hmr_device = next(self.HMAR.parameters()).device
        t_uv         = torch.from_numpy(t_uv).to(hmr_device).float()
        d_uv         = torch.from_numpy(d_uv).to(hmr_device).float()
        d_mask       = d_uv[3:, :, :]>0.5
        t_mask       = t_uv[3:, :, :]>0.5
        
        mask_dt      = torch.logical_and(d_mask, t_mask)
        mask_dt      = mask_dt.repeat(4, 1, 1)
        mask_        = torch.logical_not(mask_dt)
        
        t_uv[mask_]  = 0.0
        d_uv[mask_]  = 0.0

        with torch.no_grad():
            t_emb    = self.HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
            d_emb    = self.HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
        t_emb        = t_emb.view(-1)/10**3
        d_emb        = d_emb.view(-1)/10**3
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask_dt).cpu().numpy()/4/256/256/2
    

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192,256)
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function for running the PHALP tracker."""
    global PHALP_DEVICE
    PHALP_DEVICE = cfg.device

    phalp_tracker = HMR2_4dhuman(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()

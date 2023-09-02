from typing import Dict

import pytorch_lightning as pl
import torch
from yacs.config import CfgNode

from ..utils.pylogger import get_pylogger
from .heads import build_texture_head
from .hmr2 import HMR2

log = get_pylogger(__name__)


def unnormalize(img):
    img = img * torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1,3,1,1)
    img = img + torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1,3,1,1)
    return img

def normalize(img):
    img = img - torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1,3,1,1)
    img = img / torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1,3,1,1)
    return img

def sample_textures(texture_flow, images):
    """
    texture_flow: B x ... x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N
    output: B x ... x 3
    """
    b = texture_flow.shape[0]
    assert(texture_flow.shape[-1]==2)
    # Reshape into B x 1 x . x 2
    flow_grid_bx1xdx2 = texture_flow.view(b, 1, -1, 2)
    # B x 3 x 1 x .
    samples_bx3x1xd = torch.nn.functional.grid_sample(images, flow_grid_bx1xdx2)
    # B x 3 x F x T x T
    samples_bx1xdx3 = samples_bx3x1xd.permute(0,2,3,1)
    samples_bxdddx3 = samples_bx1xdx3.view((b,)+texture_flow.shape[1:-1]+(3,))
    return samples_bxdddx3


class HMR2_Texture(pl.LightningModule):
    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup texture model.
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        self.cfg = cfg

        # Setup base model
        cfg.defrost()
        cfg.BASE_MODEL_CONFIG.SMPL = cfg.SMPL
        cfg.freeze()
        self.base_model = HMR2(cfg.BASE_MODEL_CONFIG, init_renderer=init_renderer)
        self.smpl = self.base_model.smpl

        # Save hyperparameters
        self.save_hyperparameters(dict(cfg=self.cfg),
                                logger=False, ignore=['init_renderer'])

        # Create texture decoder backbone
        self.texture_head = build_texture_head(cfg.TEXTURE.MODEL)

        # These buffers will load from checkpoint
        self.register_buffer('uv_sampler', torch.zeros(1, 13776, 36, 2, dtype=torch.float32))
        self.register_buffer('tex_bmap', torch.zeros(256, 256, 3, dtype=torch.float32))
        self.register_buffer('tex_fmap', torch.zeros(256, 256, dtype=torch.int64))
        self.F = self.uv_sampler.size(1)
        self.T = self.uv_sampler.size(2)   #6

        # We will do backward manualy
        self.automatic_optimization = False

    def forward(self, batch: Dict, render: bool = False) -> Dict:
        """ Run a forward step of the network in val mode"""
        return self.forward_step(batch, train=False, render=render)

    def forward_step(self, batch: Dict, train: bool = False, render: bool = False) -> Dict:

        with torch.no_grad():
            base_out = self.base_model.forward_step(batch, train=False, return_feat=True)

        pred_tex_uv_flow = self.texture_head(base_out['conditioning_feats'])

        # Get image and mask
        batch_size = batch['img'].shape[0]
        device=batch['img'].device
        image = batch['img']
        mask = batch.get('mask', torch.ones_like(image[:,0]))
        mask_image_ = image * mask.unsqueeze(1)

        new_outputs = {
            'pred_tex_uv_flow': pred_tex_uv_flow,
            'mask_image': mask_image_,
        }

        return base_out | new_outputs

    def get_uv_texture(self, batch, output):
        pred_tex_uv_flow = output['pred_tex_uv_flow']

        uv_images = torch.nn.functional.grid_sample(output['mask_image'], pred_tex_uv_flow.permute(0,2,3,1))
        uv_images = unnormalize(uv_images)

        masks = batch.get('mask', torch.ones_like(batch['img'][:,0]))
        uv_mask = torch.nn.functional.grid_sample(masks.unsqueeze(1), pred_tex_uv_flow.permute(0,2,3,1))
        return uv_images, uv_mask

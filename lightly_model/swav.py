
import os
import torch
import torch.nn as nn
import torchvision
import math

from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes

from .model_base import ModelBase

class SwaV(ModelBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs['args']
        self.model_dir = kwargs['model_dir']
        self.accum_iters = kwargs['accum_iters']

        self.param_dir = os.path.join(self.model_dir, 'params')
        if not os.path.exists(self.param_dir):
            os.makedirs(self.param_dir)
        backbone_name = self.args.arch

        if backbone_name == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=True)
            input_dim = 2048
        elif backbone_name == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=True)
            input_dim = 512
        else:
            raise RuntimeError 
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SwaVProjectionHead(input_dim, input_dim, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)
        self.criterion = SwaVLoss()
        

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        q = self.projection_head(x)
        x = nn.functional.normalize(q, dim=1, p=2)
        p = self.prototypes(x)
        return p, q

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        crops, _, _ = batch
        if batch_idx == 0 and self.current_epoch == 0:
            self.save_input_to_check(batch_idx, crops)

        multi_crop_features = [ ]
        for x in crops:
            features, q = self.forward(x.to(self.device))
            multi_crop_features.append(features)
        
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        
        
        output_std = torch.std(q.detach(), 0).mean()
        self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std
        collapse_level = max(0., 1 - math.sqrt(q.shape[1]) * self.avg_output_std)
        
        # if batch_idx > 0 and (batch_idx+1) % (self.accum_iters/2) == 0 :
        #     self.save_gradient_to_check(batch_idx)

        if (batch_idx + 1) % self.accum_iters == 0:
            self.save_params_to_check(batch_idx)
        
        
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("collapse_level", collapse_level, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "collapse_level": collapse_level}

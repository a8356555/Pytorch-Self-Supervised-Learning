import os
import math
import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamProjectionHead
from lightly.models.modules import SimSiamPredictionHead

from .model_base import ModelBase

class SimSiam(ModelBase):
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
        self.projection_head = SimSiamProjectionHead(input_dim, input_dim, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        if batch_idx == 0 and self.current_epoch == 0:
            self.save_input_to_check(batch_idx, x0)

        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        
        output_std = torch.std(z0.detach(), 0).mean()
        self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std
        collapse_level = max(0., 1 - math.sqrt(z0.shape[1]) * self.avg_output_std)
        
        # if batch_idx > 0 and (batch_idx+1) % (self.accum_iters/2) == 0 :
        #     self.save_gradient_to_check(batch_idx)

        if (batch_idx + 1) % self.accum_iters == 0:
            self.save_params_to_check(batch_idx)
        
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("collapse_level", collapse_level, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "collapse_level": collapse_level}

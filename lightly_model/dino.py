
import copy
import os
import torch
import torch.nn as nn
import torchvision
import math

from lightly.data import DINOCollateFunction
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum

from .model_base import ModelBase

class DINO(ModelBase):
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
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            input_dim = 2048
        elif backbone_name == 'resnet18':
            resnet = torchvision.models.resnet18()
            backbone = nn.Sequential(*list(resnet.children())[:-1])
            input_dim = 512
        elif backbone_name == 'vit':
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
            input_dim = backbone.embed_dim
        else:
            raise RuntimeError
        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    
    def on_train_batch_start(self, batch, batch_idx):
        self.avg_output_std_teacher = self.avg_output_std_student = 0

    def training_step(self, batch, batch_idx):
        update_momentum(self.student_backbone, self.teacher_backbone, m=0.99)
        update_momentum(self.student_head, self.teacher_head, m=0.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        if batch_idx == 0 and self.current_epoch == 0:
            self.save_input_to_check(batch_idx, views)

        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        print(len(teacher_out), teacher_out[0].shape)
        print(len(student_out), student_out[0].shape)
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        teacher_out = teacher_out[0].detach()
        student_out = student_out[0].detach()
        output_std_teacher = torch.std(teacher_out, 0).mean()
        output_std_student = torch.std(student_out, 0).mean()
        self.avg_output_std_teacher = 0.9 * self.avg_output_std_teacher + 0.1 * output_std_teacher
        self.avg_output_std_student = 0.9 * self.avg_output_std_student + 0.1 * output_std_student
        collapse_level_teacher = max(0., 1 - math.sqrt(teacher_out.shape[1]) * self.avg_output_std_teacher)
        collapse_level_student = max(0., 1 - math.sqrt(student_out.shape[1]) * self.avg_output_std_student)
        print(teacher_out[:3], student_out[:3])
        
        # if batch_idx > 0 and (batch_idx+1) % (self.accum_iters/2) == 0 :
        #     self.save_gradient_to_check(batch_idx)

        if (batch_idx + 1) % self.accum_iters == 0:
            self.save_params_to_check(batch_idx)
        
        
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("collapse_level_teacher", collapse_level_teacher, on_step=True, prog_bar=True, logger=True)
        self.log("collapse_level_student", collapse_level_student, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "collapse_level_teacher": collapse_level_teacher, "collapse_level_student": collapse_level_student}
    
    def training_epoch_end(self, training_step_outputs):
        total_loss = 0
        total_collapse_level_teacher = 0
        total_collapse_level_student = 0
        length = len(training_step_outputs)
        for output in training_step_outputs:
            total_loss += output['loss'].item()
            total_collapse_level_teacher += output['collapse_level_teacher']
            total_collapse_level_student += output['collapse_level_student']
        
        self.log('train_loss', total_loss/length, on_epoch=True)
        self.log('avg_collapse_level_teacher', total_collapse_level_teacher/length, on_epoch=True)
        self.log('avg_collapse_level_student', total_collapse_level_student/length, on_epoch=True)

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)


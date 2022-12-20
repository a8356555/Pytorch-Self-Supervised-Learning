
import os
import torch
import pytorch_lightning as pl
import numpy as np

class ModelBase(pl.LightningModule):
    def on_train_batch_start(self, *args):
        self.avg_output_std = 0
    
    def training_epoch_end(self, training_step_outputs):
        total_loss = 0
        total_collapse_level = 0

        length = len(training_step_outputs)
        for output in training_step_outputs:
            total_loss += output['loss'].item()
            total_collapse_level += output['collapse_level']

        self.log('train_loss', total_loss/length, on_epoch=True)
        self.log('avg_collapse_level', total_collapse_level/length, on_epoch=True)
        
    def save_input_to_check(self, i, views):
        for i, view in enumerate(views):
            np.save(os.path.join(self.model_dir, f'input_check_{i}.npy'), view.detach().cpu().numpy()*255)

    def save_gradient_to_check(self, i):
        all_grads = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                all_grads.append(params.grad.mean().detach().cpu().numpy())
        np.save(os.path.join(self.param_dir, f'epoch{self.current_epoch}_iter{i}_grad.npy'), np.stack(all_grads))

    def save_params_to_check(self, i):    
        all_params = []
        for i, (name, params) in enumerate(self.named_parameters()):
            if params.requires_grad:
                all_params.append(params.mean().detach().cpu().numpy())
        np.save(os.path.join(self.param_dir, f'epoch{self.current_epoch}_iter{i}_param.npy'), np.stack(all_params))
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), self.args.lr,
                                     weight_decay=self.args.weight_decay)
        return optim

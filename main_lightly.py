import argparse
import torch
import pytorch_lightning as pl


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightly.data import LightlyDataset
from datetime import datetime
import os
import numpy as np

from lightly_model.framework_factory import framework_factory

model_names = ['resnet18', 'resnet50', 'vit']
frameworks = ['dino', 'swav', 'byol', 'simsiam', 'mae']
parser = argparse.ArgumentParser(description='SSL Pre-Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-f', '--framework', default='dino',
                    choices=frameworks,
                    help='paper framework: ' +
                        ' | '.join(frameworks) +
                        ' (default: dino)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--image-size', default=224, type=int,
                    help='')
parser.add_argument('--global-batch-size', default=4096, type=int,
                    help='')
parser.add_argument('--clipnorm', default=1.0, type=float,
                    help='')


def main():
    args = parser.parse_args()
    model_dir = f'model_weights/lightly/{args.framework}_{args.arch}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    accum_iters = args.global_batch_size//args.batch_size

    model_cls, collate_fn = framework_factory(args.framework)
    if args.resume:
        model = model_cls.load_from_checkpoint(checkpoint_path=args.resume, **{'args': args, 'model_dir': model_dir, 'accum_iters': accum_iters})
    else:
        model = model_cls(args=args, model_dir=model_dir, accum_iters=accum_iters)
        
    print(model)
    dataset = LightlyDataset(args.data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    gpus = 1 if torch.cuda.is_available() else 0

    logger = TensorBoardLogger(save_dir=model_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        dirpath=model_dir,
        filename='{epoch}{loss:.2f}',
        save_top_k = 3,
        every_n_epochs=3,
        mode='min',
        save_weights_only=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs, 
        gpus=gpus, 
        accumulate_grad_batches=accum_iters,
        gradient_clip_val=args.clipnorm,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback, lr_monitor],
        )
        
    trainer.fit(model=model, train_dataloaders=dataloader)

if __name__ == '__main__':
    main()

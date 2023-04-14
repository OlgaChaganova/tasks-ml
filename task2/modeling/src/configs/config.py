import os
import uuid

import pytorch_lightning as pl
from torch import nn

from configs.base import (
    Callbacks,
    Common,
    Config,
    Criterion,
    Dataset,
    LRScheduler,
    Model,
    Optimizer,
    Project,
    Train,
)


TASK_NAME = 'unet_autoencoder_' + uuid.uuid4().hex[:6]  # unique run id

CONFIG = Config(
    project=Project(
        project_name='audio_denoising',
        task_name=TASK_NAME,
    ),

    common=Common(seed=8),

    dataset=Dataset(
        train_root_dir='data/train/',
        val_root_dir='data/val/',
        max_len=1024,
        n_mels=80,
        file_ext='npy',
        batch_size=64,
        num_workers=4,
    ),

    model=Model(
        in_channels=1,
    ),

    train=Train(
        trainer_params={
            'devices': 1,
            'accelerator': 'auto',
            'accumulate_grad_batches': 4,
            'gradient_clip_val': 0,
            'benchmark': True,
            'precision': 32,
            'profiler': 'simple',
            'max_epochs': 100,
        },

        callbacks=Callbacks(
            model_checkpoint=pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join('checkpoints', TASK_NAME),
                save_top_k=2,
                monitor='val_loss_epoch',
                mode='min',
            ),

            lr_monitor=pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ),

        optimizer=Optimizer(
            name='Adam',
            params={
                'lr': 0.001,
                'weight_decay': 0.0001,
            },
        ),

        lr_scheduler=LRScheduler(
            name='CosineAnnealingWarmRestarts',
            params={
                'T_0': 100,
                'T_mult': 1,
                'eta_min': 1e-7,
            },
        ),

        criterion=Criterion(
            loss=nn.HuberLoss(),
        ),
        ckpt_path=None,
    ),
)

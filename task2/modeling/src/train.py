"""Train model."""

import argparse
import logging
import os
import typing as tp
from runpy import run_path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from configs.base import Config
from data.datamodule import MelSpectrogramDataModule
from model.model import UNetPredictNoise, UNetRecoverClean
from configs.utils import get_config_dict


def load_model(
    config: Config,
    full_restore: bool,
    model_module: UNetRecoverClean | UNetPredictNoise,
) -> tp.Tuple[pl.LightningModule, str]:
    if not full_restore:
        pretrained_model_ckpt = torch.load(config.train.ckpt_path)
        pretrained_state_dict = pretrained_model_ckpt['state_dict']
        model_dict = model_module.state_dict()

        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_state_dict)
        model_module.load_state_dict(pretrained_state_dict)
        logging.info(f'Load only weights of pretrained model from the checkpoint {config.train.ckpt_path}')
        ckpt_path = None

    elif full_restore:
        model = model_module.load_from_checkpoint(config.train.ckpt_path, criterion=config.train.criterion)
        logging.info(f'Fully restoring training from the checkpoint {config.train.ckpt_path}')
        ckpt_path = config.train.ckpt_path
    return model, ckpt_path


def get_wandb_logger(model: pl.LightningModule, datamodule: pl.LightningDataModule, config: Config) -> WandbLogger:
    config_dict = get_config_dict(model=model, datamodule=datamodule, config=config)
    wandb_logger = WandbLogger(
        project=config.project.project_name,
        name=config.project.task_name,
        config=config_dict,
        log_model='all',
    )
    wandb_logger.watch(
        model=model,
        log='all',
        log_freq=300,  # log gradients and parameters every log_freq batches
    )
    return wandb_logger


def save_files(args: tp.Any) -> None:
    """
    Save config file in wandb for experiment reproducibility
    """
    base_path = os.path.split(args.config)[0]
    wandb.save(args.config, base_path=base_path, policy='now')


def get_trainer(logger: WandbLogger, config: Config):
    trainer_params = config.train.trainer_params
    callbacks = list(config.train.callbacks.__dict__.values())
    callbacks = filter(lambda callback: callback is not None, callbacks)
    trainer = Trainer(
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=10), *callbacks],
        **trainer_params,
    )
    return trainer


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join('src', 'configs', 'config.py'),
        type=str,
        help='Path to experiment config file (*.py).',
    )
    parser.add_argument(
        '--full_restore',
        action='store_false',
        help='Continue training with the same hyperparameters; '
             'Otherwise, load only weights and use hyperparameters from config',
    )
    parser.add_argument(
        '--model_type',
        choices=['recover', 'predict_noise'],
        default='recover',
        help="""Model type to be trained.
             `recover`: recovering clean sound from noised one;
             `predict_noise`: predict noise on the sound (clean = noisy - predicted noise)"""
    )
    return parser.parse_args()


def main(args, config: Config):
    if args.model_type == 'recover':
        model = UNetRecoverClean(
            in_channels=config.model.in_channels,
            optimizer=config.train.optimizer,
            lr_scheduler=config.train.lr_scheduler,
            criterion=config.train.criterion,
        )

    elif args.model_type == 'predict_noise':
        model = UNetPredictNoise(
            in_channels=config.model.in_channels,
            optimizer=config.train.optimizer,
            lr_scheduler=config.train.lr_scheduler,
            criterion=config.train.criterion,
        )
    else:
        raise ValueError(f'Available model types are `recover`, `predict_noise`, but got {args.model_type}')

    if config.train.ckpt_path is not None:
        model, ckpt_path = load_model(config=config, full_restore=args.full_restore, model=model)
    else:
        ckpt_path = None

    ModelSummary(model)

    cd = config.dataset
    datamodule = MelSpectrogramDataModule(
        train_root_dir=cd.train_root_dir,
        val_root_dir=cd.val_root_dir,
        max_len=cd.max_len,
        n_mels=cd.n_mels,
        batch_size=cd.batch_size,
        num_workers=cd.num_workers,
        file_ext=cd.file_ext,
    )

    wandb_logger = get_wandb_logger(model, datamodule, config)
    save_files(args)
    trainer = get_trainer(wandb_logger, config)

    logging.info(
        f'\nProject {config.project.project_name} was initialized. The name of the run is {config.project.task_name}.',
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )


if __name__ == '__main__':
    args = parse()
    logging.basicConfig(level=logging.INFO)
    config_module = run_path(args.config)
    exp_config = config_module['CONFIG']

    seed_everything(exp_config.common.seed, workers=True)
    main(args, exp_config)

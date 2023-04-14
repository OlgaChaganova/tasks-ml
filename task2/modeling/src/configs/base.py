import typing as tp
from dataclasses import dataclass

import pytorch_lightning as pl
from torch import nn


@dataclass
class Project:
    project_name: str
    task_name: str


@dataclass
class Common:
    seed: int = 8


@dataclass
class Dataset:
    train_root_dir: str
    val_root_dir: str
    max_len: int
    n_mels: int
    file_ext: str
    batch_size: int
    num_workers: int


@dataclass
class Model:
    in_channels: int


@dataclass
class Callbacks:
    model_checkpoint: pl.callbacks.ModelCheckpoint
    early_stopping: tp.Optional[pl.callbacks.EarlyStopping] = None
    lr_monitor: tp.Optional[pl.callbacks.LearningRateMonitor] = None
    model_summary: tp.Optional[tp.Union[pl.callbacks.ModelSummary, pl.callbacks.RichModelSummary]] = None
    timer: tp.Optional[pl.callbacks.Timer] = None


@dataclass
class Optimizer:
    name: str
    params: dict


@dataclass
class LRScheduler:
    name: str
    params: dict


@dataclass
class Criterion:
    loss: nn.Module


@dataclass
class Train:
    trainer_params: dict
    callbacks: Callbacks
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    criterion: Criterion
    ckpt_path: tp.Optional[str] = None


@dataclass
class Config:
    project: Project
    common: Common
    dataset: Dataset
    model: Model
    train: Train

import logging
import os
import typing as tp
from pathlib import Path

import albumentations as alb
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations import pytorch as alb_pt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tvf


class MelSpectrogramDataset(Dataset):
    """Mel spectrogram dataset."""
    def __init__(
        self,
        root_dir: str,
        max_len: int,
        n_mels: int,
        file_ext: str,
    ):
        """
        Init Mel spectrogram dataset.

        Parameters
        ----------
        root_dir: str
            Path to root directory with data.
        max_len: int
            Maximum len of spectrogram.
        n_mels: int
            Number of mel-filters.
        file_ext: str
            Extension of files.
        """
        self.root_dir = root_dir
        self.max_len = max_len
        self.n_mels = n_mels

        clean_data_dir = os.path.join(root_dir, 'clean')
        noisy_data_dir = os.path.join(root_dir, 'noisy')
        self.clean_data = sorted(list(Path(clean_data_dir).rglob(f'*.{file_ext}')))
        self.noisy_data = sorted(list(Path(noisy_data_dir).rglob(f'*.{file_ext}')))

        self.transform = alb.Compose(
            [
                alb.PadIfNeeded(
                    min_height=n_mels,
                    min_width=max_len,
                    border_mode=0,
                    value=0,  # silence (no sound)
                    always_apply=True,
                    position='bottom_left',
                ),
                alb_pt.ToTensorV2(),
            ],
        )

    def __len__(self):
        return len(self.clean_data)

    def _get(self, ind: int, mode: tp.Literal['clean', 'noisy']) -> torch.tensor:
        spectrum = np.load(self.clean_data[ind]) if mode == 'clean' else np.load(self.noisy_data[ind])
        spectrum = spectrum.T
        spectrum = spectrum.astype(np.float32)
        spectrum = self.transform(image=spectrum)['image']
        if spectrum.shape != (self.n_mels, self.max_len):
            spectrum = tvf.resize(spectrum, size=[self.n_mels, self.max_len], antialias=True)
        return spectrum

    def __getitem__(self, ind) -> tp.Tuple[torch.tensor, torch.tensor]:
        clean_spectrum = self._get(ind, mode='clean')
        noisy_spectrum = self._get(ind, mode='noisy')
        return clean_spectrum, noisy_spectrum


class MelSpectrogramDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root_dir: str,
        val_root_dir: str,
        max_len: int,
        n_mels: int,
        file_ext: str,
        batch_size: int,
        num_workers: int,
    ):
        """Create Data Module for Mel-spectrogram dataset.

        Parameters
        ----------
        train_root_dir: str
            Path to root directory with train data.
        val_root_dir: str
            Path to root directory with val data.
        max_len: int
            Maximum len of spectrogram.
        n_mels: int
            Number of mel-filters.
        file_ext: str
            Extension of files.
        batch_size : int
            Batch size for dataloaders.
        num_workers : int
            Number of workers in dataloaders.
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir
        self.max_len = max_len
        self.n_mels = n_mels
        self.file_ext = file_ext

    def setup(self, stage: tp.Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MelSpectrogramDataset(
                root_dir=self.train_root_dir,
                max_len=self.max_len,
                n_mels=self.n_mels,
                file_ext=self.file_ext,
            )
            num_train_files = len(self.train_dataset)
            logging.info(f'Mode: train, number of files: {num_train_files}')

            self.val_dataset = MelSpectrogramDataset(
                root_dir=self.val_root_dir,
                max_len=self.max_len,
                n_mels=self.n_mels,
                file_ext=self.file_ext,
            )
            num_val_files = len(self.val_dataset)
            logging.info(f'Mode: val, number of files: {num_val_files}')

        elif stage == 'test':
            self.test_dataset = MelSpectrogramDataset(
                root_dir=self.val_root_dir,
                max_len=self.max_len,
                n_mels=self.n_mels,
                file_ext=self.file_ext,
            )
            num_test_files = len(self.test_dataset)
            logging.info(f'Mode: test, number of files: {num_test_files}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False,
        )

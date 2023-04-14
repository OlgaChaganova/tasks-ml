import pytorch_lightning as pl
import torch

from configs.base import LRScheduler, Optimizer, Criterion
from model.autoencoder import UNet


class UNetRecoverClean(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
        criterion: Criterion,
    ):
        """Ligtning module for UNet model recovering clean sound from noised one.
        Input: noisy sound
        Output: clean sound

        Parameters
        ----------
        in_channels: int
            Count of input channels.
        optimizer: Optimizer
            Optimizer.
        lr_scheduler: LRScheduler
            Learning rate schediler.
        criterion: Criterion
            Loss function for classification.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['criterion'])  # criterion is already saved during checkpointing
        self.learning_rate = optimizer.params['lr']

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion.loss
        self.model = UNet(in_channels=in_channels)
        self.metric = torch.nn.MSELoss()  # loss using for validation as metric

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_clean, x_noisy = batch
        x_clean_reconstr = self(x_clean)
        x_noisy_reconstr = self(x_noisy)

        clean_clean_loss = self.criterion(x_clean, x_clean_reconstr)
        noisy_clean_loss = self.criterion(x_clean, x_noisy_reconstr)
        loss = clean_clean_loss + noisy_clean_loss

        self.log('train_clean_clean_loss_batch', clean_clean_loss, on_epoch=False, on_step=True)
        self.log('train_noisy_clean_loss_batch', noisy_clean_loss, on_epoch=False, on_step=True)
        self.log('train_loss_batch', loss, on_epoch=False, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_clean, x_noisy = batch
        x_clean_reconstr = self(x_clean)
        x_noisy_reconstr = self(x_noisy)

        clean_clean_loss = self.metric(x_clean, x_clean_reconstr)
        clean_noisy_loss = self.metric(x_clean, x_noisy_reconstr)
        loss = clean_clean_loss + clean_noisy_loss

        self.log('val_clean_clean_loss_epoch', clean_clean_loss, on_epoch=True, on_step=False)
        self.log('val_clean_noisy_loss_epoch', clean_noisy_loss, on_epoch=True, on_step=False)
        self.log('val_loss_epoch', loss, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer.name)(
            self.parameters(),
            **self.optimizer.params,
        )
        optim_dict = {
            'optimizer': optimizer,
            'monitor': 'val_loss',
        }

        if self.lr_scheduler is not None:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler.name)(
                optimizer,
                **self.lr_scheduler.params,
            )
            optim_dict.update({'lr_scheduler': lr_scheduler})

        return optim_dict


class UNetPredictNoise(UNetRecoverClean):
    def __init__(
        self,
        in_channels: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
        criterion: Criterion,
    ):
        """Ligtning module for UNet model predicting noise on the sound.
        Input: noisy sound
        Output: predicted noise (=> clean = noisy - noise)

        Parameters
        ----------
        in_channels: int
            Count of input channels.
        optimizer: Optimizer
            Optimizer.
        lr_scheduler: LRScheduler
            Learning rate schediler.
        criterion: Criterion
            Loss function for classification.
        """
        super().__init__(in_channels, optimizer, lr_scheduler, criterion)
        self.save_hyperparameters(ignore=['criterion'])  # criterion is already saved during checkpointing
        self.learning_rate = optimizer.params['lr']

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_clean, x_noisy = batch
        noise = x_noisy - x_clean
        zero_noise = torch.zeros_like(x_clean)

        noise_prediction = self(noise)
        zero_noise_prediction = self(zero_noise)

        noise_loss = self.criterion(x_noisy, noise_prediction)
        zero_noise_loss = self.criterion(x_clean, zero_noise_prediction)
        loss = noise_loss + zero_noise_loss

        self.log('train_noise_loss_batch', noise_loss, on_epoch=False, on_step=True)
        self.log('train_zero_noise_loss_batch', zero_noise_loss, on_epoch=False, on_step=True)
        self.log('train_loss_batch', loss, on_epoch=False, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_clean, x_noisy = batch
        noise = x_noisy - x_clean
        zero_noise = torch.zeros_like(x_clean)

        noise_prediction = self(noise)
        zero_noise_prediction = self(zero_noise)

        noise_loss = self.metric(x_noisy, noise_prediction)
        zero_noise_loss = self.metric(x_clean, zero_noise_prediction)
        loss = noise_loss + zero_noise_loss

        self.log('val_noise_loss_epoch', noise_loss, on_epoch=True, on_step=False)
        self.log('val_zero_noise_loss_epoch', zero_noise_loss, on_epoch=True, on_step=False)
        self.log('val_loss_epoch', loss, on_epoch=True, on_step=False)

        return loss

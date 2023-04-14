import typing as tp

import torch
import torch.nn as nn


class UNet(nn.Module):
    """UNet-based autoencoder"""
    def __init__(self, in_channels: int = 1):
        super().__init__()
        down_kernel_size = (2, 2)
        down_stride = (2, 2)
        self.down_conv_layer_1 = DownConvBlock(in_channels, 64, kernel_size=down_kernel_size, stride=down_stride, norm=False)
        self.down_conv_layer_2 = DownConvBlock(64, 128, kernel_size=down_kernel_size, stride=down_stride)
        self.down_conv_layer_3 = DownConvBlock(128, 256, kernel_size=down_kernel_size, stride=down_stride)
        self.down_conv_layer_4 = DownConvBlock(256, 256, dropout=0.5, kernel_size=down_kernel_size, stride=down_stride)
        self.down_conv_layer_5 = DownConvBlock(256, 256, dropout=0.5, kernel_size=down_kernel_size, stride=down_stride)

        self.up_conv_layer_1 = UpConvBlock(256, 256, kernel_size=(3, 2), stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_2 = UpConvBlock(512, 256, kernel_size=down_kernel_size, stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_3 = UpConvBlock(512, 256, kernel_size=down_kernel_size, stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_4 = UpConvBlock(384, 128, dropout=0.5)

        self.final_conv_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(192, in_channels, 3, padding=1),
            # nn.ReLU()
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        enc1 = self.down_conv_layer_1(x)
        enc2 = self.down_conv_layer_2(enc1)
        enc3 = self.down_conv_layer_3(enc2)
        enc4 = self.down_conv_layer_4(enc3)
        enc5 = self.down_conv_layer_5(enc4)

        dec1 = self.up_conv_layer_1(enc5, enc4)
        dec2 = self.up_conv_layer_2(dec1, enc3)
        dec3 = self.up_conv_layer_3(dec2, enc2)
        dec4 = self.up_conv_layer_4(dec3, enc1)
        return self.final_conv_block(dec4)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        ip_sz: int,
        op_sz: int,
        kernel_size: int | tp.Tuple[int, int] = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = [
            nn.ConvTranspose2d(ip_sz, op_sz, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(op_sz),
            nn.ReLU(),
        ]
        if dropout:
            self.layers += [nn.Dropout(dropout)]
        self.up_convblock = nn.Sequential(*self.layers)

    def forward(self, x: torch.tensor, enc_ip: torch.tensor) -> torch.tensor:
        x = self.up_convblock(x)
        return torch.cat((x, enc_ip), dim=1)


class DownConvBlock(nn.Module):
    def __init__(
        self,
        ip_sz: int,
        op_sz: int,
        kernel_size: int | tp.Tuple[int, int],
        stride: int | tp.Tuple[int, int],
        padding: int | tp.Tuple[int, int] = 0,
        norm: bool = True,
        dropout: tp.Optional[float] = None,
    ):
        super().__init__()
        self.layers = [nn.Conv2d(ip_sz, op_sz, kernel_size, stride=stride, padding=padding)]
        if norm:
            self.layers.append(nn.InstanceNorm2d(op_sz))
        self.layers += [nn.LeakyReLU(0.2)]
        if dropout:
            self.layers += [nn.Dropout(dropout)]
        self.down_convblock = nn.Sequential(*(self.layers))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.down_convblock(x)

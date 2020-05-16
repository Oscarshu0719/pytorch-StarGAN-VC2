import argparse
import numpy as np
import torch
import torch.nn as nn

class DownsampleBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super(DownsampleBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_out,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.InstanceNorm2d(num_features=dim_out, affine=True),
            nn.GLU(dim=1)
        )
        self.conv_gated = nn.Sequential(
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_out,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.InstanceNorm2d(num_features=dim_out, affine=True),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        # GLU
        return self.conv_layer(x) * torch.sigmoid(self.conv_gated(x))

class UpSampleBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super(UpSampleBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim_in,
                               out_channels=dim_out,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias),
            nn.PixelShuffle(2),
            nn.GLU(dim=1)
        )
        self.conv_gated = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim_in,
                               out_channels=dim_out,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias),
            nn.PixelShuffle(2),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        # GLU
        return self.conv_layer(x) * torch.sigmoid(self.conv_gated(x))

class ConditionalInstanceNormalisation(nn.Module):
    """
        CIN block.
    """

    def __init__(self, dim_in, style_num):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        self.style_num = style_num
        self.gamma = nn.Linear(style_num, dim_in)
        self.beta = nn.Linear(style_num, dim_in)

    def forward(self, x, c):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        gamma = self.gamma(c.to(self.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta(c.to(self.device))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, style_num):
        super(ResidualBlock, self).__init__()

        self.conv_layer = nn.Conv1d(in_channels=dim_in,
                                    out_channels=dim_out,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)
        self.cin = ConditionalInstanceNormalisation(dim_in=dim_out, style_num=style_num)
        self.glu = nn.GLU(dim=1)

    def forward(self, x, c_):
        x_ = self.conv_layer(x)
        x_ = self.cin(x_, c_)
        x_ = self.glu(x_)

        return x + x_

class Generator(nn.Module):
    def __init__(self, num_speakers=4):
        super(Generator, self).__init__()

        self.num_speakers = num_speakers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7)),
            nn.GLU(dim=1)
        )

        # Down-sampling layers.
        self.down_sample_1 = DownsampleBlock(dim_in=64,
                                             dim_out=256,
                                             kernel_size=(5, 5),
                                             stride=(2, 2),
                                             padding=(2, 2),
                                             bias=False)
        self.down_sample_2 = DownsampleBlock(dim_in=128,
                                             dim_out=512,
                                             kernel_size=(5, 5),
                                             stride=(2, 2),
                                             padding=(2, 2),
                                             bias=False)

        # Reshape data (This operation is done in forward function).

        # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

        # Bottleneck layers.
        self.residual_1 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)
        self.residual_2 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)
        self.residual_3 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)
        self.residual_4 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)
        self.residual_5 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)
        self.residual_6 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)
        self.residual_7 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)
        self.residual_8 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)
        self.residual_9 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers)

        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # Reshape data (This operation is done in forward function).

        # Up-sampling layers.
        self.up_sample_1 = UpSampleBlock(dim_in=256,
                                         dim_out=1024,
                                         kernel_size=(5, 5),
                                         stride=(1, 1),
                                         padding=2,
                                         bias=False)
        self.up_sample_2 = UpSampleBlock(dim_in=128,
                                         dim_out=512,
                                         kernel_size=(5, 5),
                                         stride=(1, 1),
                                         padding=2,
                                         bias=False)

        # TODO: The last layer differs from the paper.
        self.out = nn.Conv2d(in_channels=64,
                             out_channels=1, # 35 in paper
                             kernel_size=(5, 15),
                             stride=(1, 1),
                             padding=(2, 7),
                             bias=False)

    def forward(self, x, c_):
        width_size = x.size(3)

        x = self.conv_layer_1(x)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)

        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

        x = self.residual_1(x, c_)
        x = self.residual_2(x, c_)
        x = self.residual_1(x, c_)
        x = self.residual_1(x, c_)
        x = self.residual_1(x, c_)
        x = self.residual_1(x, c_)
        x = self.residual_1(x, c_)
        x = self.residual_1(x, c_)
        x = self.residual_1(x, c_)

        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)

        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        
        out = self.out(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, num_speakers=4):
        super(Discriminator, self).__init__()

        self.num_speakers = num_speakers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GLU(dim=1)
        )
        self.conv_gated_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GLU(dim=1)
        )

        # Down-sampling layers.
        self.down_sample_1 = DownsampleBlock(dim_in=64,
                                             dim_out=256,
                                             kernel_size=(3, 3),
                                             stride=(2, 2),
                                             padding=1,
                                             bias=False)
        self.down_sample_2 = DownsampleBlock(dim_in=128,
                                             dim_out=512,
                                             kernel_size=(3, 3),
                                             stride=(2, 2),
                                             padding=1,
                                             bias=False)
        self.down_sample_3 = DownsampleBlock(dim_in=256,
                                             dim_out=1024,
                                             kernel_size=(3, 3),
                                             stride=(2, 2),
                                             padding=1,
                                             bias=False)
        self.down_sample_4 = DownsampleBlock(dim_in=512,
                                             dim_out=1024,
                                             kernel_size=(1, 5),
                                             stride=(1, 1),
                                             padding=(0, 2),
                                             bias=False)

        # Fully connected layer.
        self.fully_connected = nn.Linear(in_features=512, out_features=1)

        # Projection.
        self.projection = nn.Linear(self.num_speakers * 2, 512)

    def forward(self, x, c, c_):
        c_onehot = torch.cat((c, c_), dim=1).to(self.device)
        x = self.conv_layer_1(x) * torch.sigmoid(self.conv_gated_1(x))

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x_ = self.down_sample_4(x)

        h = torch.sum(x_, dim=(2, 3))

        x = self.fully_connected(h)

        p = self.projection(c_onehot)

        x += torch.sum(p * h, dim=1, keepdim=True)

        return x

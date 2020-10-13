import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm import tqdm

nonlinearity = partial(F.relu, inplace=True)

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return {"lr": param_group["lr"], "momentum": param_group["momentum"]}

class CustomSoftmax(nn.Module):
    """modified Softmax
    input: N*C*H*W*(Z)
    output: exp(xi-max)/sum(exp(xi-max))
    """

    def __init__(self, dim, logit=False):
        super(CustomSoftmax, self).__init__()
        self.dim = dim
        self.softmax = nn.LogSoftmax(dim=dim) if logit else nn.Softmax(dim=dim)

    def forward(self, x):
        # DONE: check again later
        max, _ = torch.max(x, dim=self.dim, keepdim=True)
        x = x - max
        return self.softmax(x)

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                        2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(
            channel, channel, kernel_size=1, dilation=1, padding=0)
        # self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        # self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(
            self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x))))
        )
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        self.layer1 = F.upsample(
            self.conv(self.pool1(x)), size=(h, w), mode="bilinear", align_corners=True
        )
        self.layer2 = F.upsample(
            self.conv(self.pool2(x)), size=(h, w), mode="bilinear", align_corners=True
        )
        self.layer3 = F.upsample(
            self.conv(self.pool3(x)), size=(h, w), mode="bilinear", align_corners=True
        )
        self.layer4 = F.upsample(
            self.conv(self.pool4(x)), size=(h, w), mode="bilinear", align_corners=True
        )

        out = torch.cat([self.layer1, self.layer2,
                         self.layer3, self.layer4, x], 1)
        # out = torch.cat([self.layer1, self.layer2, x], 1)

        return out

class UNetSSLContrastive(nn.Module):
    '''
        Features are extracted at the last layer of decoder. 
    '''
    def __init__(self, n_channels, n_classes, embedding_dim=64):
        super(UNetSSLContrastive, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1028, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.dac = DACblock(512)
        self.spp = SPPblock(512)
        self.squash = CustomSoftmax(dim=1)
        self.projection_head = nn.Sequential(nn.Conv2d(
            64, embedding_dim, 1), nn.ReLU(), nn.Conv2d(embedding_dim, embedding_dim, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.dac(x5)
        x5 = self.spp(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        y = self.outc(x)
        return self.squash(y), self.projection_head(x)

class UNetSSLContrastive_v1(nn.Module):
    '''
        Features are extracted few layers before the last layer of the decoder
    '''
    def __init__(self, n_channels, n_classes, embedding_dim=64):
        super(UNetSSLContrastive_v1, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1028, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.dac = DACblock(512)
        self.spp = SPPblock(512)
        self.squash = CustomSoftmax(dim=1)
        self.projection_head = nn.Sequential(nn.Conv2d(
            64, embedding_dim, 1), nn.ReLU(), nn.Conv2d(embedding_dim, embedding_dim, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.dac(x5)
        x5 = self.spp(x5)

        x = self.up1(x5, x4)
        x_out = self.up2(x, x3)

        x = self.up3(x_out, x2)
        x = self.up4(x, x1)
        y = self.outc(x)
        return self.squash(y), self.projection_head(x), x_out

if __name__ == "__main__":
    device = "cuda"
    x = torch.randn(2, 1, 200, 400).to(device)
    ssl_net = UNetSSLContrastive_v1(1, 8).to(device)
    y = ssl_net(x)
    print(y[0].shape, y[1].shape, y[2].shape)
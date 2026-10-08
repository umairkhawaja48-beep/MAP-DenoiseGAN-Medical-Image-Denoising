"""
MAP-DenoiseGAN Path A (map_project): model architectures. Fresh
self-contained copies of the generator (UNet2D: 4 encoder/4 decoder
blocks, widths 32/64/128/256, sigmoid output) and discriminator
(conditional PatchGAN) already established and validated in the earlier
feasibility experiment - kept architecturally IDENTICAL for continuity
(this is deliberate: Phase A1's bias-variance study must use the same
architecture the feasibility numbers came from, or the whole comparison
is invalid), but copied here as fresh, independent code per "Path A is a
fresh classical-only project."
"""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, widths=(32, 64, 128, 256)):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.enc1 = conv_block(in_channels, w1)
        self.enc2 = conv_block(w1, w2)
        self.enc3 = conv_block(w2, w3)
        self.enc4 = conv_block(w3, w4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(w4, w4)

        self.up4 = nn.ConvTranspose2d(w4, w4, 2, stride=2)
        self.dec4 = conv_block(w4 + w4, w4)
        self.up3 = nn.ConvTranspose2d(w4, w3, 2, stride=2)
        self.dec3 = conv_block(w3 + w3, w3)
        self.up2 = nn.ConvTranspose2d(w3, w2, 2, stride=2)
        self.dec2 = conv_block(w2 + w2, w2)
        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1 = conv_block(w1 + w1, w1)

        self.out_conv = nn.Conv2d(w1, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        def block(cin, cout, stride=2, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, 32, norm=False),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256, stride=1),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )

    def forward(self, noisy, candidate):
        x = torch.cat([noisy, candidate], dim=1)
        return self.net(x)

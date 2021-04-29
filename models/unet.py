import torch
import torch.nn as nn
from torch.nn import init

class Conv_Block(nn.Module):
    def __init__(self, chn_in, chn_out, ksize, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(chn_in, chn_out, ksize, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(chn_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        ## Encoder
        self.enc_conv0 = Conv_Block(in_channels, 48, 3, stride=1, padding=1)
        self.enc_conv1 = Conv_Block(48, 48, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_conv2 = Conv_Block(48, 48, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc_conv3 = Conv_Block(48, 48, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc_conv4 = Conv_Block(48, 48, 3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        self.enc_conv5 = Conv_Block(48, 48, 3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2)

        self.enc_conv6 = Conv_Block(48, 48, 3, stride=1, padding=1)

        ## decoder
        self.upsample5 = nn.UpsamplingNearest2d(scale_factor=2)
        #  concat pool4
        self.dec_conv5a = Conv_Block(96, 96, 3, stride=1, padding=1)
        self.dec_conv5b = Conv_Block(96, 96, 3, stride=1, padding=1)
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=2)
        #  concat pool3
        self.dec_conv4a = Conv_Block(144, 96, 3, stride=1, padding=1)
        self.dec_conv4b = Conv_Block(96, 96, 3, stride=1, padding=1)
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        #  concat pool2
        self.dec_conv3a = Conv_Block(144, 96, 3, stride=1, padding=1)
        self.dec_conv3b = Conv_Block(96, 96, 3, stride=1, padding=1)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        #  concat pool1
        self.dec_conv2a = Conv_Block(144, 96, 3, stride=1, padding=1)
        self.dec_conv2b = Conv_Block(96, 96, 3, stride=1, padding=1)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        #  concat input
        self.dec_conv1a = Conv_Block(96+in_channels, 64, 3, stride=1, padding=1)
        self.dec_conv1b = Conv_Block(64, 32, 3, stride=1, padding=1)
        self.dec_conv1c = nn.Conv2d(32, out_channels, 3, stride=1, padding=1, bias=True)


    def forward(self, x):
        N, C, H, W = x.shape
        
        # Encoder part------------------------------------------
        skip_connecs = [x]
        
        out = self.enc_conv0(x)
        out = self.enc_conv1(out)
        out = self.pool1(out)
        skip_connecs.append(out)

        out = self.enc_conv2(out)
        out = self.pool2(out)
        skip_connecs.append(out)

        out = self.enc_conv3(out)
        out = self.pool3(out)
        skip_connecs.append(out)

        out = self.enc_conv4(out)
        out = self.pool4(out)
        skip_connecs.append(out)

        out = self.enc_conv5(out)
        out = self.pool5(out)
        out = self.enc_conv6(out)

        # Decoder part------------------------------------------
        out = self.upsample5(out)
        out = torch.cat([out, skip_connecs.pop()], dim=1)
        out = self.dec_conv5a(out)
        out = self.dec_conv5b(out)

        out = self.upsample4(out)
        out = torch.cat([out, skip_connecs.pop()], dim=1)
        out = self.dec_conv4a(out)
        out = self.dec_conv4b(out)

        out = self.upsample3(out)
        out = torch.cat([out, skip_connecs.pop()], dim=1)
        out = self.dec_conv3a(out)
        out = self.dec_conv3b(out)

        out = self.upsample2(out)
        out = torch.cat([out, skip_connecs.pop()], dim=1)
        out = self.dec_conv2a(out)
        out = self.dec_conv2b(out)

        out = self.upsample1(out)
        out = torch.cat([out, skip_connecs.pop()], dim=1)
        out = self.dec_conv1a(out)
        out = self.dec_conv1b(out)

        out = self.dec_conv1c(out)

        return out


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        # self.selu = nn.SeLU()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
class InitBN(nn.Module):
    """(BN)"""
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.init_bn = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        return self.init_bn(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        initd = x
        resBlock = self.double_conv(x)
        resBlock = torch.add(resBlock,initd)
        return resBlock


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class finalconcat(nn.Module):
    def __init__(self,outchannels):
        super(finalconcat,self).__init__()
        self.reluu = nn.PReLU()
        self.conv2dn = nn.Sequential(nn.Conv2d(outchannels*2, outchannels, kernel_size=1),
                                     nn.BatchNorm2d(outchannels)
                                     )

    def forward(self, x1, x2):

        x2=self.reluu(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv2dn(x)
        return x



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv_2dn = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(in_channels//2),
                                      nn.PReLU()
                                      )
        self.conv = DoubleConv(in_channels//2, out_channels)
        # self.conv2 = DoubleConv(out_channels,out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 = self.conv_2dn(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x3 = self.conv_2dn(x)
        # x = self.conv1(x)
        return self.conv(x3)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBatchNorm, self).__init__()
        self.convbn = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.PReLU())
    def forward(self, x):
        return self.convbn(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(AttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size=1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size=1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.psi(nn.ReLU()(x1 + g1))
        return out * x


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels//2, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(out_channels, in_channels//2, int(in_channels // 2))
        self.conv_bn1 = ConvBatchNorm(in_channels , out_channels)
        self.conv_bn2 = ConvBatchNorm(out_channels, out_channels)
        self.conv_bn3 = ConvBatchNorm(out_channels,out_channels)

    def forward(self, x, x_skip):
        # note : x_skip is the skip connection and x is the input from the previous block
        # apply the attention block to the skip connection, using x as context
        x_attention = self.attention(x_skip, x)
        # upsample x to have th same size as the attention map
        # x = nn.functional.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.upsample(x)
        # stack their channels to feed to both convolution blocks
        x = torch.cat((x_attention, x), dim=1)
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        return self.conv_bn3(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels//2, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionBlock(out_channels, in_channels//2, int(in_channels // 2))
        self.conv_bn1 = ConvBatchNorm(in_channels , out_channels)
        self.conv_bn2 = ConvBatchNorm(out_channels, out_channels)
        self.conv_bn3 = ConvBatchNorm(out_channels,out_channels)

    def forward(self, x, x_skip):
        # note : x_skip is the skip connection and x is the input from the previous block
        # apply the attention block to the skip connection, using x as context
        # x_attention = self.attention(x_skip, x)
        # upsample x to have th same size as the attention map
        # x = nn.functional.interpolate(x, x_skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.upsample(x)
        # stack their channels to feed to both convolution blocks
        x = torch.cat((x_skip, x), dim=1)
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        return self.conv_bn3(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.RReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.prelu(residual)

        return x + residual




class ResUnetpp(nn.Module):
    def __init__(self, n_channels, n_classes, scale_factor, bilinear=False):
        super(ResUnetpp, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scale_factor = scale_factor

        # Unmixing Network
        self.block1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = ResidualBlock(64)
        self.block8 = ResidualBlock(64)
        self.block9 = ResidualBlock(64)
        self.block10 = ResidualBlock(64)
        self.block11 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # Unmixing Network
        self.upf = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # Supixel Location Network

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(1, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        # self.spatten = SpatialAttentionModule()
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

        # Subpixel Location Network

    def forward(self, x):

        # Unmixing Network
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        block9 = self.block9(block8)
        block10 = self.block10(block9)
        block11 = self.block11(block10)
        water_Frac = (torch.tanh(block11) + 1) / 2
        # Unmixing Network
        water_FracU = self.upf(water_Frac)
        # Subpixel Location Network

        x0_0 = self.conv0_0(water_FracU)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # x0_4 = self.spatten(x0_4)


        water_map = self.final(x0_4)
        # Subpixel Location Network
        return water_Frac,water_map



class ResUnetppModified(nn.Module):
    def __init__(self, n_channels, n_classes, scale_factor, bilinear=False):
        super(ResUnetppModified, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scale_factor = scale_factor

        # Unmixing Network
        self.block1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = ResidualBlock(64)
        self.block8 = ResidualBlock(64)
        self.block9 = ResidualBlock(64)
        self.block10 = ResidualBlock(64)
        self.block11 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # Unmixing Network
        self.upf = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # Supixel Location Network

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(1, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        # self.spatten = SpatialAttentionModule()
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        
        self.output_linear = nn.Linear(64,64)
        
        # Subpixel Location Network

    def forward(self, x, topo_1, topo_2):

        # Unmixing Network
        # block1 = self.block1(x)
        # block2 = self.block2(block1)
        # block3 = self.block3(block2)
        # block4 = self.block4(block3)
        # block5 = self.block5(block4)
        # block6 = self.block6(block5)
        # block7 = self.block7(block6)
        # block8 = self.block8(block7)
        # block9 = self.block9(block8)
        # block10 = self.block10(block9)
        # block11 = self.block11(block10)
        # water_Frac = (torch.tanh(block11) + 1) / 2
        # Unmixing Network
        water_FracU = self.upf(x)
        # Subpixel Location Network

        x0_0 = self.conv0_0(water_FracU)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # x0_4 = self.spatten(x0_4)


        water_map = self.final(x0_4)
        
        # Subpixel Location Network
        x = torch.sigmoid(self.output_linear(water_map))
        
        return x

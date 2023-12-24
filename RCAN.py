from torch import nn
import torch

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class RCAN(nn.Module):
    def __init__(self, scale, num_features, num_rg, num_rcab, reduction, topo_inclusion, dropout_prob_input, dropout_prob_topo_1, 
                 dropout_prob_topo_2):
        super(RCAN, self).__init__()
        num_channels = 1
        self.topo_inclusion = topo_inclusion
        self.sf = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale))

        self.dropout_topo_1 = nn.Dropout(dropout_prob_topo_1) 
        self.dropout_topo_2 = nn.Dropout(dropout_prob_topo_2)
        self.dropout_input = nn.Dropout(dropout_prob_input)

        if topo_inclusion == "beggining":
            self.sfe1_topo_1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
            self.sfe2_topo_1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
            self.sfe1_topo_2 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
            self.sfe2_topo_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
            
            self.rgs_topo_1 = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
            self.conv1_topo_1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            
            self.rgs_topo_2 = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
            self.conv1_topo_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            
            
        elif topo_inclusion == "vertical":
            self.sfe1_topo_1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
            self.sfe2_topo_1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
            self.rgs_topo_1 = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
            self.conv1_topo_1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            
        elif topo_inclusion == "horizontal":
            self.sfe1_topo_2 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
            self.sfe2_topo_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)    
            self.rgs_topo_2 = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
            self.conv1_topo_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
    
        # self.conv2 = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)

        if topo_inclusion == "beggining":
            self.conv2 = nn.Conv2d(num_features*3, num_channels, kernel_size=3, padding=3 // 2)
        elif topo_inclusion in ["vertical", "horizontal"] :
            self.conv2 = nn.Conv2d(num_features*2, num_channels, kernel_size=3, padding=3 // 2)
        elif topo_inclusion == "none":
            self.conv2 = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=3 // 2)
            
    

    def forward(self, x, topo_1=None, topo_2=None):

        if self.topo_inclusion in ["beggining", "vertical"]:
            sfe1_topo_1 = self.sfe1_topo_1(topo_1)
            sfe1_topo_1 = self.dropout_topo_1(sfe1_topo_1)
            sfe2_topo_1 = self.sfe2_topo_1(sfe1_topo_1)
            residual = sfe2_topo_1
            sfe2_topo_1 = self.rgs_topo_1(sfe2_topo_1)
            sfe2_topo_1 = self.conv1_topo_1(sfe2_topo_1)
            sfe2_topo_1 += residual

        if self.topo_inclusion in ["beggining", "horizontal"]:
            sfe1_topo_2 = self.sfe1_topo_2(topo_2)
            sfe1_topo_2 = self.dropout_topo_2(sfe1_topo_2)
            
            sfe2_topo_2 = self.sfe2_topo_2(sfe1_topo_2)
            residual = sfe2_topo_2
            sfe2_topo_2 = self.rgs_topo_2(sfe2_topo_2)
            sfe2_topo_2 = self.conv1_topo_2(sfe2_topo_2)
            sfe2_topo_2 += residual

        x = self.sf(x)
        x = self.dropout_input(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.upscale(x)

        if self.topo_inclusion == "beggining":
            x = torch.cat((x, sfe2_topo_1), dim=1)
            x = torch.cat((x, sfe2_topo_2), dim=1)
        elif self.topo_inclusion == "vertical":
            x = torch.cat((x, sfe2_topo_1), dim=1)
            # x = torch.cat((x, sfe2_topo_2), dim=1)
        elif self.topo_inclusion == "horizontal":
            # x = torch.cat((x, sfe2_topo_1), dim=1)
            x = torch.cat((x, sfe2_topo_2), dim=1)

        x = self.conv2(x)
        
        x = torch.sigmoid(x)
        
        return x
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN_comb(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers,topo_inclusion):
        super(RDN_comb, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.topo_inclusion = topo_inclusion

        self.sfe1_pm = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2_pm = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
            
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # residual dense blocks
        self.rdbs_pm = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs_pm.append(RDB(self.G, self.G, self.C))



        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # # up-sampling
        # if scale_factor == 2 or scale_factor == 4:
        #     self.upscale = []
        #     for _ in range(scale_factor // 2):
        #         self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
        #                              nn.nn.PixelShuffle(2)])
                            
        #     self.upscale = nn.Sequential(*self.upscale)
        # else:
        #     self.upscale = nn.Sequential(
        #         nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
        #         nn.PixelShuffle(scale_factor)
        #     )

        self.output_2 = nn.Conv2d(num_features*2, num_channels, kernel_size=3, padding=3 // 2)
        

    def forward(self, x_high_res, x_poor_mans):

        sfe1 = self.sfe1(x_high_res)
        sfe2 = self.sfe2(sfe1)
        x_high_res = sfe2
        local_features = []

        sfe1_pm = self.sfe1_pm(x_poor_mans)
        sfe2_pm = self.sfe2_pm(sfe1_pm)
        x_poor_mans = sfe2_pm
        local_features_pm = []


        for i in range(self.D):
            x_high_res = self.rdbs[i](x_high_res)
            local_features.append(x_high_res)
        x_high_res = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning

        #x_high_res = self.upscale(x_high_res)

        for i in range(self.D):
            x_poor_mans = self.rdbs_pm[i](x_poor_mans)
            local_features_pm.append(x_poor_mans)
        x_poor_mans = self.gff(torch.cat(local_features_pm, 1)) + sfe1  # global residual learning
        #x_poor_mans = self.upscale(x_poor_mans)
        

        x_comb = torch.cat((x_high_res, x_poor_mans), dim=1)
        x = self.output_2(x_comb)
        x = torch.sigmoid(x)
        
        return x

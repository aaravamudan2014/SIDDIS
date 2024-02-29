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


class RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, 
                 num_layers,topo_inclusion, dropout_prob_input, dropout_prob_topo_1, 
                 dropout_prob_topo_2):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.topo_inclusion = topo_inclusion

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        self.dropout_topo_1 = nn.Dropout(dropout_prob_topo_1) 
        self.dropout_topo_2 = nn.Dropout(dropout_prob_topo_2)
        self.dropout_input = nn.Dropout(dropout_prob_input)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        
        # residual dense blocks topo 1
        self.rdbs_topo_1 = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs_topo_1.append(RDB(self.G, self.G, self.C))

        # residual dense blocks topo 2
        self.rdbs_topo_2 = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs_topo_2.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # global feature fusion: topo 1
        self.gff_topo_1 = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        
        # global feature fusion: topo 2
        self.gff_topo_2 = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        
        # up-sampling
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output_1 = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
        
        if topo_inclusion == "beggining":
            self.output_2 = nn.Conv2d(num_features*3, num_channels, kernel_size=3, padding=3 // 2)
        elif topo_inclusion in ["vertical", "horizontal"] :
            self.output_2 = nn.Conv2d(num_features*2, num_channels, kernel_size=3, padding=3 // 2)
        elif topo_inclusion == "none":
            self.output_2 = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=3 // 2)
            
    
        self.output_3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=3 // 2)
        self.output_4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=3 // 2)
        self.output_5 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=3 // 2)
        

    def forward(self, x, topo_1=None, topo_2=None):

        if self.topo_inclusion == True and (topo_1 == None or topo_2 == None):
            raise Exception('Topo inputs must be included if topo_inclusion flag is set to True')

        sfe1 = self.sfe1(x)
        sfe1 = self.dropout_input(sfe1)
        
        sfe2 = self.sfe2(sfe1)

        if self.topo_inclusion in ["beggining", "horizontal"]:
            sfe1_topo_1 = self.sfe1_topo_1(topo_1)
            sfe1_topo_1 = self.dropout_topo_1(sfe1_topo_1)
            sfe2_topo_1 = torch.relu(self.sfe2_topo_1(sfe1_topo_1))

            # x = sfe2_topo_1
            # local_features_topo_1 = []
            # for i in range(self.D):
            #     x = self.rdbs_topo_1[i](x)
            #     local_features_topo_1.append(x)
            # x_topo_1 = self.gff(torch.cat(local_features_topo_1, 1)) + sfe1_topo_1 
            x_topo_1 = sfe2_topo_1
        
        if self.topo_inclusion in ["beggining", "vertical"]:
            sfe1_topo_2 = self.sfe1_topo_2(topo_2)
            sfe1_topo_2 = self.dropout_topo_2(sfe1_topo_2)
            sfe2_topo_2 = torch.relu(self.sfe2_topo_2(sfe1_topo_2))

            # x = sfe2_topo_2
            # local_features_topo_2 = []
            # for i in range(self.D):
            #     x = self.rdbs_topo_2[i](x)
            #     local_features_topo_2.append(x)
            # x_topo_2 = self.gff(torch.cat(local_features_topo_2, 1)) + sfe1_topo_2 
            x_topo_2 = sfe2_topo_2
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        
        if self.topo_inclusion == "beggining":
            x = torch.cat((x, x_topo_1), dim=1)
            x = torch.cat((x, x_topo_2), dim=1)
        elif self.topo_inclusion == "horizontal":
            x = torch.cat((x, x_topo_1), dim=1)
        elif self.topo_inclusion == "vertical":
            x = torch.cat((x, x_topo_2), dim=1)
        
        x = self.output_2(x)
        x = torch.relu(x)
        x = self.output_3(x)
        x = torch.relu(x)
        x = self.output_4(x)
        x = torch.sigmoid(x)
        
        return x

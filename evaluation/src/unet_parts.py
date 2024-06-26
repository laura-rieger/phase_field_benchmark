""" Full assembly of the parts to form the complete network """
# from https://github.com/milesial/Pytorch-UNet 

# /*
#  * Milesial
#  * 2022
#  *
#  * This file is part of https://github.com/milesial/Pytorch-UNet
#  *
#  * Phase Field Standard is a derivative work based on https://github.com/milesial/Pytorch-UNet.
#  *
#  *  Phase Field Standard is free software: you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation, either version 3 of the License, or
#  * (at your option) any later version.
#  *
#  * Phase Field Standard is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with Phase Field Standard  If not, see <http://www.gnu.org/licenses/>.
#  */

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
def add_frame(input_tensor, num_pad = 1):
    output_tensor = F.pad(input_tensor, (num_pad, num_pad, num_pad, num_pad)) 
    output_tensor[:,:, num_pad:-num_pad, :num_pad] = input_tensor[:,:,:,-num_pad:]
    output_tensor[:,:, num_pad:-num_pad, -num_pad:] = input_tensor[:,:,:,:num_pad]
    output_tensor[:,:, :num_pad, num_pad:-num_pad] = input_tensor[:,:,-num_pad:]
    output_tensor[:,:, -num_pad:, num_pad:-num_pad] = input_tensor[:,:,:num_pad]   
    
    output_tensor[:,:, :num_pad, :num_pad] = input_tensor[:,:,-num_pad:,-num_pad:]
    output_tensor[:,:, :num_pad, -num_pad:] = input_tensor[:,:,-num_pad:,:num_pad]
    output_tensor[:,:, -num_pad:, :num_pad] = input_tensor[:,:,:num_pad,-num_pad:]
    output_tensor[:,:, -num_pad:, -num_pad:] = input_tensor[:,:,:num_pad,:num_pad]
    

    return output_tensor

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        return self.double_conv(add_frame(x, num_pad=2))


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

if __name__ == "__main__":
#h dim is given by image size and model
    test_model = DoubleConv(1,1)
    

    test_input = torch.zeros((8, 1, 5, 5,))
    
    test_input[:,:, -1:] =.25
    test_input[:,:, :1] =.75
    test_input[:,:, :,-1:] =1
    test_input[:,:, :,:1] =.5
    my_output = test_model(test_input)
    print(my_output.shape)
    # print(test_model(test_input).shape)
    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(test_input[0,0].detach().cpu().numpy())
    axes[1].imshow(my_output[0,0].detach().cpu().numpy())
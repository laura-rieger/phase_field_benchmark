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

import torch.nn.functional as F
import numpy as np
from unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, in_factor =32, use_small = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.use_small = use_small

        self.inc = DoubleConv(n_channels, in_factor)
        self.down1 = Down(in_factor, in_factor*2)
        self.down2 = Down(in_factor*2, in_factor*4)
        
        if not self.use_small:
            self.down3 = Down(in_factor*4, in_factor*8)
            self.down4 = Down(in_factor*8, in_factor*16 // factor)
            self.up1 = Up(in_factor*16, in_factor*8 // factor, bilinear)
        else:
            self.down3 = Down(in_factor*4, in_factor*8//factor)
        self.up2 = Up(in_factor*8, in_factor*4 // factor, bilinear)
        self.up3 = Up(in_factor*4, in_factor*2 // factor, bilinear)
        self.up4 = Up(in_factor*2, in_factor, bilinear)
        self.outc = OutConv(in_factor, n_classes)
        self.my_sigmoid = torch.nn.Sigmoid()

    

    def forward(self, x):
        x_sum = x.sum(axis = [1,2,3])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if not self.use_small:

            x5 = self.down4(x4)


            x = self.up1(x5, x4)
            x = self.up2(x, x3)
        else:
            x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.my_sigmoid(logits)
        return torch.clip(logits * x_sum[:,None,None,None]/ logits.sum(axis = [1,2,3])[:,None,None,None], min =0, max =1)
        # my_out = self.output(logits)
        # return my_out[:,0]
        

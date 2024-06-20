from segformer_pytorch import Segformer
import torch
import torch.nn as nn
class MySegformer(nn.Module):

    def __init__(
        self, num_input, num_classes, device
    ):
        # todo check if device
        super().__init__()
       
        self.my_sigmoid = torch.nn.Sigmoid()
        pad_layer = torch.nn.ZeroPad2d(14)
        pad_layer_rev = torch.nn.ZeroPad2d(-14)
        feature_extractor = Segformer(channels=num_input, num_classes=1).to(
            device=device
        )
        upsample_layer = torch.nn.Upsample(scale_factor=(4, 4), mode="bilinear")
        self.model = torch.nn.Sequential(
            pad_layer, feature_extractor, upsample_layer, pad_layer_rev
        ).to(device=device)

    def forward(self, x):
 
        x_sum = x.sum(axis = [1,2,3])
        logits = self.my_sigmoid(self.model(x))
        return torch.clip(logits * x_sum[:,None,None,None]/ logits.sum(axis = [1,2,3])[:,None,None,None], min =0, max =1)

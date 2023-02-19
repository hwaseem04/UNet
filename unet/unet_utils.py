import torch
import torch.nn as nn
import torch.nn.functional as F

#  Remarks
## Unlike original paper, same padding is done
## Center Cropping the contractive path feature map is not done
## Dropout layer not used but used in Original paper 

class InsideBlockConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(InsideBlockConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1), # For same padding: pad=1 for filter=3
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True), # inplace=True doesn't create additonal memory. Not always correct operation. But here there is no issue
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1):
        return self.double_conv(x1)
    
class ContractionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ContractionBlock, self).__init__()
        self.contraction = nn.Sequential(
            nn.MaxPool2d(2),
            InsideBlockConv(in_features, out_features),
        )
        
    def forward(self, x1):
        return self.contraction(x1)

class ExpansionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ExpansionBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_features, in_features//2, kernel_size=2, stride=2) # During upscaling channel depth is halved
        self.conv = InsideBlockConv(in_features, out_features) # After concatenation
    def forward(self, x1, x2):
        # x1 -> Feature map to be upscaled
        # x2 -> Feature map values from symmetrical contractive block, to concatenate with x1
        x1 = self.up(x1)

        # Padding 
        # to match the feature map in contractive path. Hence avoiding center cropping the contractive path
        diffX = x2.shape[2] - x1.shape[2]
        diffY = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX, diffX, diffY, diffY])

        # Concatenation. Input in PyTorch manipulation is usually of (Batch, Channel, Height, Width). 
        # We need to concatenate along channels. hence axis=1
        print(x1.shape, x2.shape)
        try:
            x = torch.cat([x2,x1], axis=1) # x2,x1 and not as x1,x2 because following it the simlar way of original paper.
        except:
            raise ValueError(f"{x1.shape}, {x2.shape}")
            
        return self.conv(x)

class finalMap(nn.Module):
    def __init__(self, in_features, out_features):
        super(finalMap, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1)
    def forward(self, x1):
        return self.conv(x1)



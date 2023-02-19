# Actual Architecture
# referred from https://github.com/milesial/Pytorch-UNet - Changed as needed.

from .unet_utils import *
from torch.utils import checkpoint

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.initial = InsideBlockConv(in_channels, 64)
        self.down1 = ContractionBlock(64, 128)
        self.down2 = ContractionBlock(128, 256)
        self.down3 = ContractionBlock(256,512)
        self.down4 = ContractionBlock(512, 1024)
        # self.down_list = nn.ModuleList([self.initial, self.down1, self.down2, self.down3, self.down4])
        self.up1 = ExpansionBlock(1024, 512)
        self.up2 = ExpansionBlock(512, 256)
        self.up3 = ExpansionBlock(256, 128)
        self.up4 = ExpansionBlock(128, 64)
        # self.up_list = nn.ModuleList([self.up1, self.up2, self.up3, self.up4])
        self.output = finalMap(64, n_classes)

    def forward(self, x):
        # Downward
        x1 = self.initial(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Upward
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final mask prediction
        logits = self.output(x) # Feature map logitss
        return logits


    # def store_block_outputs(self):
    #     self.initial = checkpoint(self.initial) # topBlock
    #     self.down1 = checkpoint(self.downBlock1) # maxpool, 2 convs
    #     self.down2 = checkpoint(self.downBlock2)
    #     self.down3 = checkpoint(self.downBlock3)
    #     self.down4 = checkpoint(self.downBlock4)
    #     self.up1 = checkpoint(self.upBlock1) # transpose conv, concat, 2 convs
    #     self.up2 = checkpoint(self.upBlock2)
    #     self.up3 = checkpoint(self.upBlock3)
    #     self.up4 = checkpoint(self.upBlock4)
    #     self.output = checkpoint(self.output) # output mask


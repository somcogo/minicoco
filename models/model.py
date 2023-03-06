import torch.nn as  nn

from .UNetBlock  import UNetBlock

class AttUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1)
        self.block1 = UNetBlock(in_channels=4, out_channels=64, down=True, attention=True)
        self.block2 = UNetBlock(in_channels=64, out_channels=128, down=True, attention=True)
        self.block3 = UNetBlock(in_channels=128, out_channels=256, down=True, attention=True)
        self.block4 = UNetBlock(in_channels=256, out_channels=128, up=True, attention=True)
        self.block5 = UNetBlock(in_channels=128, out_channels=64, up=True, attention=True)
        self.block6 = UNetBlock(in_channels=64, out_channels=32, up=True, attention=True)
        self.conv = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        out = self.conv(x)
        return out
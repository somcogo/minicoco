import torch.nn as  nn
import math

class SampleModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Linear(in_channels, out_channels)
        self.layer2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d
        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.layer2(x)
        x = self.relu(x)
        out = self.batch_norm(x)

        return out
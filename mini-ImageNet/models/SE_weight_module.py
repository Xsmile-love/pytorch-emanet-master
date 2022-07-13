import math

import torch
import torch.nn as nn

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, channel):
#         super(eca_layer, self).__init__()
#         self.channel = channel
#         self.k_size = self.Coverage_size(channel)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def Coverage_size(self, dim, gamma=2, b=1):
#         with torch.no_grad():
#             t = int(abs((math.log(dim, 2) + b) / gamma))
#             k = t if t % 2 else t + 1
#             return k
#
#     def forward(self, x):
#
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#         # out = y.expand_as(x)
#         return y


# if __name__ =='__main__':
#     input = torch.randn(64,5,32,32)
#     out = eca_layer(channel=5)
#     print(out(input))

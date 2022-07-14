import torch
import torch.nn as nn
import math
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class _BatchAttNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_BatchAttNorm, self).__init__(num_features, eps, momentum, affine)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias_readjust = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust.data.fill_(0)
        self.bias_readjust.data.fill_(-1)
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        self._check_input_dim(input)

        # Soft Batch norm
        attention = self.sigmoid(self.avg(input) * self.weight_readjust + self.bias_readjust)
        bn_w = self.weight * self.softmax(attention)

        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training, self.momentum, self.eps)
        out_bn = out_bn * bn_w + self.bias

        return out_bn


class BAN2d(_BatchAttNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class SEWModule(nn.Module):
    """
    MLP Module
    channel statistical information modeling
    """
    def __init__(self, channels, reduction=16):
        super(SEWModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        out = self.fc(avg_out)
        weight = self.sigmoid(out)

        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Residual(nn.Module):
    """Residual module"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer1(in_planes, out_planes, kernel_size=3, mixer_size=9, stride=1, padding=1, dilation=1, groups=1):
    """
    CNN-style transformer architecture
    in_planes: input channel
    out_planes: out channel
    mixer_size: spatial mixing kernel size
    """
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, groups=groups, bias=False),
        nn.GELU(),
        nn.BatchNorm2d(out_planes),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(out_planes, out_planes, mixer_size, groups=out_planes, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(out_planes)
                )),
                Residual(nn.Sequential(
                    BAN2d(out_planes),
                    nn.Conv2d(out_planes, out_planes, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(out_planes)
                ))
        )]
    )


class MAModule(nn.Module):
    """
    MA module
    inplans: input channel
    outplans: output channel
    conv_kernals: different branches convolutional kernel size
    conv_groups: different branches convolutional groups size
    """
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(MAModule, self).__init__()
        self.conv_1 = ConvMixer1(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                                 stride=stride, groups=conv_groups[0])
        self.conv_2 = ConvMixer1(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                                 stride=stride, groups=conv_groups[1])
        self.conv_3 = ConvMixer1(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                                 stride=stride, groups=conv_groups[2])
        self.conv_4 = ConvMixer1(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                                 stride=stride, groups=conv_groups[3])
        self.se = SEWModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        # splicing of different branches
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        # calculate different branch weights
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        # re-correct different branch weights and multiply them with the corresponding channel feature maps
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors

        # (B,4,C/4,H,W)-->(B,C,H,W)
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class EMABlock(nn.Module):
    """EMA Bottleneck bloch,the MA module is used to replace the 3x3 convolution in the ResNet residual block"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(EMABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = MAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class EMANet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(EMANet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layers(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layers(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def emanet50(pretrained=False):
    model = EMANet(EMABlock, [3, 4, 6, 3], num_classes=10)
    if pretrained:
        state_dict = torch.load("./model_data")
        model.load_state_dict(state_dict)
    # ----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier


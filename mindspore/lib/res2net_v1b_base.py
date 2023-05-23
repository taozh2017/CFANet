import pdb

from mindspore import Tensor, ops, Parameter, nn, common
import mindspore as ms
import math

__all__ = ['Res2Net', 'res2net50_v1b']

# model_urls = {
#     'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
#     'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
# }


class Bottle2neck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, pad_mode='valid', has_bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, pad_mode="same")
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, pad_mode='same', has_bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.CellList(convs)
        self.bns = nn.CellList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, pad_mode='valid', has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp = None
        spx = ops.Split(1, self.scale)(out)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = ops.Concat(1)((out, sp))
        if self.scale != 1 and self.stype == 'normal':
            out = ops.Concat(1)((out, spx[self.nums]))  # torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = ops.Concat(1)((out, self.pool(spx[self.nums])))  # torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Cell):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode='same', has_bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Dense(512 * block.expansion, num_classes)

        # for m in self.cells():
        #     if isinstance(m, nn.Conv2d):
        #         common.initializer.HeNormal(mode='fan_out', nonlinearity='relu')(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         print(m)
        #         common.initializer.Constant(1)(m.weight)
        #         common.initializer.Constant(0)(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, pad_mode='valid', has_bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.SequentialCell(*layers)

    def construct(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x6 = x5.view(x5.size(0), -1)
        x7 = self.fc(x6)

        return x7


class Res2Net_Ours(nn.Cell):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net_Ours, self).__init__()

        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, pad_mode='same', has_bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # for m in self.cells():
        #     if isinstance(m, nn.Conv2d):
        #         common.initializer.HeNormal(mode='fan_out', nonlinearity='relu')(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         print(m)
        #         common.initializer.Constant(1)(m.weight)
        #         common.initializer.Constant(0)(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, pad_mode='valid', has_bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.SequentialCell(*layers)

    def construct(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4


def res2net50_v1b(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'],map_location='cpu'))
        param_dict = ms.load_checkpoint(
            './lib/res2net50_v1b_26w_4s-3cf99910.ckpt')
        ms.load_param_into_net(model, param_dict)

    return model


def res2net50_v1b_Ours(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net_Ours(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        param_dict = ms.load_checkpoint(
            './lib/res2net50_v1b_26w_4s-3cf99910.ckpt')
        ms.load_param_into_net(model, param_dict)
    return model


def Res2Net_model(ind=50):
    if ind == 50:
        model_base = res2net50_v1b(pretrained=False)
        model = res2net50_v1b_Ours()

    pretrained_dict = model_base.parameters_dict()
    model_dict = model.parameters_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    ms.load_param_into_net(model, pretrained_dict)

    return model

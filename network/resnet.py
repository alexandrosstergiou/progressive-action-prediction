'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import torch
import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch.nn.functional as F

'''
===  S T A R T  O F  C L A S S  C O N V 3 D S I M P L E ===

    [About]

        nn.Sequential class for creating a 3D Convolution operation used as a building block for the 3D CNN.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: None. Only used to ensure continouity with class `Conv2Plus1D`.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class Conv3DSimple(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3),
                      stride=(stride, stride, stride), padding=(padding, padding, padding),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
'''
===  E N D  O F  C L A S S  C O N V 3 D S I M P L E ===
'''


'''
===  S T A R T  O F  C L A S S  C O N V 2 P L U S 1 D ===

    [About]

        nn.Sequential class for creating a (2+1)D Convolution operation used as a building block for the 3D CNN.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: Integer for the number of intermediate channels as calculated by the (2+1)D operation.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
'''
===  E N D  O F  C L A S S  C O N V 2 P L U S 1 D ===
'''


'''
===  S T A R T  O F  C L A S S  C O N V 3 D N O T E M P O R A L ===

    [About]

        nn.Sequential class for creating a 2D Convolution operations based on a 3D Conv wrapper used as a building block for the 3D CNN.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: None. Only used to ensure continouity with class `Conv2Plus1D`.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(1, padding, padding),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)
'''
===  E N D  O F  C L A S S  C O N V 3 D N O T E M P O R A L ===
'''


'''
===  S T A R T  O F  C L A S S  C O N V 3 D D E P T H W I S E ===

    [About]

        nn.Sequential class for creating a 3D Depthwise Convolution operations used as a building block for the 3D CNN.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: None. Only used to ensure continouity with class `Conv2Plus1D`.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class Conv3DDepthwise(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DDepthwise, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3),
                      stride=(stride, stride, stride), padding=(padding, padding, padding),
                      groups=in_planes, bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
'''
===  S T A R T  O F  C L A S S  C O N V 3 D D E P T H W I S E ===
'''


'''
===  S T A R T  O F  C L A S S  I P C O N V 3 D D E P T H W I S E ===

    [About]

        nn.Sequential class for creating interaction-preserving 3D Depthwise Convolution operations used as a building block for the 3D CNN.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: None. Only used to ensure continouity with class `Conv2Plus1D`.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class IPConv3DDepthwise(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        assert in_planes == out_planes
        super(IPConv3DDepthwise, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
            Conv3DDepthwise(in_planes=out_planes, out_planes=out_planes, padding=1, stride=stride)
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
'''
===  E N D  O F  C L A S S  I P C O N V 3 D D E P T H W I S E ===
'''



'''
===  S T A R T  O F  C L A S S  C O N V 3 D N O T E M P O R A L ===

    [About]

        nn.Conv3d Class for creating a 3D Convolution without any temporal extend (i.e. kernel size
        is limited to a shape of 1 x k x k, where k is the kernel size).

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: None. Only used to ensure continouity with class `Conv2Plus1D`.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: Integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)
'''
===  E N D  O F  C L A S S  C O N V 3 D N O T E M P O R A L ===
'''


'''
===  S T A R T  O F  C L A S S  B A S I C B L O C K ===

    [About]

        nn.Module Class for creating a spatio-temporal `BasicBlock` for ResNets.

    [Init Args]

        - in_planes: Integer for the number input channels to the block.
        - planes: Integer for the number of output channels to the block.
        - conv_builder: nn.Module for the Convolution type to be used.
        - stride: Integer for the kernel stride. Defaults to 1.
        - downsample: nn.Module in the case that downsampling is to be used for the residual connection.
        Defaults to None.
        - groups: Integer for the number of groups to be used.
        - base_width: Only used for contiouity with the `Bottleneck` block. Defaults to 64.

    [Methods]

        - __init__ : Class initialiser
        - forward : Function for operation calling.

'''
class BasicBlock(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, groups=1, base_width=64):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)


        return out
'''
===  E N D  O F  C L A S S  B A S I C B L O C K ===
'''


'''
===  S T A R T  O F  C L A S S  B O T T L E N E C K ===

    [About]

        nn.Module Class for creating a spatio-temporal `BottleNeck` block for ResNets.

    [Init Args]

        - in_planes: Integer for the number input channels to the block.
        - planes: Integer for the number of output channels to the block.
        - conv_builder: nn.Module for the Convolution type to be used.
        - stride: Integer for the kernel stride. Defaults to 1.
        - downsample: nn.Module in the case that downsampling is to be used for the residual connection.
        Defaults to None.
        - groups: Integer for the number of groups to be used.
        - base_width: Used to define the width of the bottleneck. Defaults to 64.

    [Methods]

        - __init__ : Class initialiser
        - forward : Function for operation calling.

'''
class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, groups=1, base_width=64):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        width = int(planes * (base_width / 64.)) * groups
        mid_width = int(midplanes * (base_width / 64.)) * groups

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, width, kernel_size=1, bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True)
        )

        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(width, width, mid_width, stride),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
'''
===  E N D  O F  C L A S S  B O T T L E N E C K ===
'''


'''
===  S T A R T  O F  C L A S S  B A S I C S T E M ===

    [About]

        nn.Sequential Class for the initial 3D convolution.

    [Init Args]

        - None

    [Methods]

        - __init__ : Class initialiser
'''
class BasicStem(nn.Sequential):
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
'''
===  E N D  O F  C L A S S  B A S I C S T E M ===
'''


'''
===  S T A R T  O F  C L A S S  R 2 P L U S 1 D S T E M ===

    [About]

        nn.Sequential Class for the initial (2+1)D convolution.

    [Init Args]

        - None

    [Methods]

        - __init__ : Class initialiser
'''
class R2Plus1dStem(nn.Sequential):
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
'''
===  E N D  O F  C L A S S  R 2 P L U S 1 D S T E M ===
'''


'''
===  S T A R T  O F  C L A S S  V I D E O R E S N E T ===

    [About]

        nn.Module for creating the 3D ResNet.

    [Init Args]

        - block: nn.Module used as resnet building block.
        - conv_makers: List of Functions that create each layer.
        - layers: List of Integers specifying the number of blocks per layer.
        - stem: nn.Module for the Resnet stem to be used 3D/(2+1)D. Defaults to None.
        - num_classes: Integer for the dimension of the final FC layer. Defaults to 400.
        - zero_init_residual: Boolean for zero init bottleneck residual BN. Defaults to False.

    [Methods]

        - __init__ : Class initialiser
        - forward: Function for performing the main sequence of operations.
        - _make_layer: Function for creating a sequence (nn.Sequential) of layers.
        - _initialise_weights: Function for weight initialisation.
'''
class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400, groups=1, width_per_group=64,
                 zero_init_residual=False):

        super(VideoResNet, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.stem = stem()

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)

        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)

        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)

        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialise_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample, groups=self.groups, base_width=self.base_width))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
'''
===  E N D  O F  C L A S S  V I D E O R E S N E T ===
'''


'''
---  S T A R T  O F  N E T W O R K  C R E A T I O N  F U N C T I O N S ---
    [About]
        All below functions deal with the creation of networks. Networks are specified based on their
        function names.
'''


def r3d_18(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[2, 2, 2, 2], stem=BasicStem, **kwargs)


def r3d_34(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def r3d_50(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def r3d_101(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, **kwargs)


def r3d_152(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 8, 36, 3], stem=BasicStem,**kwargs)

def r3d_200(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 24, 36, 3], stem=BasicStem,**kwargs)

def r3dxt34_32d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)

def r3dxt50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def r3dxt101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, **kwargs)


def wide_r3d50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def wide_r3d101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, **kwargs)


def ir_csn_50(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DDepthwise]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def ir_csn_101(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DDepthwise]*4, layers=[3, 4, 23, 3], stem=BasicStem, **kwargs)


def ir_csn_152(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DDepthwise]*4, layers=[3, 8, 36, 3], stem=BasicStem,**kwargs)


def ip_csn_50(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[IPConv3DDepthwise]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def ip_csn_101(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[IPConv3DDepthwise]*4, layers=[3, 4, 23, 3], stem=BasicStem, **kwargs)


def ip_csn_152(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[IPConv3DDepthwise]*4, layers=[3, 8, 36, 3], stem=BasicStem,**kwargs)

'''
---  E N D  O F  N E T W O R K  C R E A T I O N  F U N C T I O N S ---
'''


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    tmp = (3,16,224,224)
    #--- TEST 1 ---
    net = ir_csn_101(num_classes=400)#.cuda()

    macs, params = get_model_complexity_info(net, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('network 1 test passed \n')

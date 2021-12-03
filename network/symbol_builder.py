'''
---  I M P O R T  S T A T E M E N T S  ---
'''

import coloredlogs, logging
coloredlogs.install()
from .mtnet import MTNet_xs, MTNet_s, MTNet_m, MTNet_l, MTNet_xl, MTNet_xxl, MTNet_xs_g8, MTNet_s_g8, MTNet_m_g8, MTNet_l_g8
from .srtg_resnet import r3d_18, r3d_34, r3d_50, r3d_101, r3d_152, r3d_200, r3dxt50_32x4d, r3dxt101_32x8d, wide_r3d50_2,wide_r3d101_2, r2plus1d_18, r2plus1d_34, r2plus1d_50, r2plus1d_101, r2plus1d_152, r2plus1d_200, r2plus1dxt50_32x4d, r2plus1dxt101_32x8d, wide_r2plus1d50_2,wide_r2plus1d101_2, srtg_r3d_18, srtg_r3d_34, srtg_r3d_50, srtg_r3d_101, srtg_r3d_152, srtg_r3d_200, srtg_r3dxt50_32x4d, srtg_r3dxt101_32x8d, srtg_wide_r3d50_2, srtg_wide_r3d101_2, srtg_r2plus1d_18, srtg_r2plus1d_34, srtg_r2plus1d_50, srtg_r2plus1d_101, srtg_r2plus1d_152, srtg_r2plus1d_200, srtg_r2plus1dxt50_32x4d, srtg_r2plus1dxt101_32x8d, srtg_wide_r2plus1d50_2, srtg_wide_r2plus1d101_2

from .temper_h import TemPer_h

from .config import get_config

import torch
import torch.nn.parallel
import torch.nn.functional as F

from einops import reduce, rearrange
from einops.layers.torch import Reduce, Rearrange

from ptflops import get_model_complexity_info
from torchinfo import summary


#import adapool_cuda
#from adaPool import IDWPool1d, EMPool1d, EDSCWPool1d, AdaPool1d


def beautify_net(net):
    converted = []
    net_string = str(net)
    for name, layer in net.named_modules():
        names = name.split('.')
        layer_n = str(type(layer).__name__)

        if layer_n not in converted:
            converted.append(layer_n)
            new_ln = '\033[34m' + layer_n + '\033[0m'
            net_string = net_string.replace(layer_n, new_ln)

        for n in names:
            if n not in converted:
                converted.append(n)
                new_n = '\033[35m' + n + '\033[0m'
                net_string = net_string.replace('('+n+')', '('+new_n+')')
    # beautify for readability
    print(net_string)



'''
---  S T A R T  O F  F U N C T I O N  G E T _ S Y M B O L ---
    [About]
        Function for loading PyTorch models.
    [Args]
        - name: String for the backbone/head network name.
        - print_net: Boolean for printing the architecture. Defaults to False.
        - headless: Boolean for not using the classifer part of the network. Defaults to False.
    [Returns]
        - net: Module for the loaded Pytorch network.
        - config: Dictionary that includes a `mean` and `std` terms spcifying the mean and standard deviation.
                  See `network/config.py` for more info.
'''
def get_symbol(name, samplers=4, pool=None, headless=False, **kwargs):

    # TemPer_h
    if ('TEMPER' in name.upper()):
        net = TemPer_h(depth=samplers, return_acts=True, pool=pool, **kwargs)

    # Multi-Temporal net
    elif "MTNET" in name.upper():
        if "MTNET_XS" in name.upper():
            if "G8" in name.upper():
                net = MTNet_xs_g8(**kwargs, return_acts=True)
            else:
                net = MTNet_s(**kwargs, return_acts=True)
        elif "MTNET_S" in name.upper():
            if "G8" in name.upper():
                net = MTNet_s_g8(**kwargs, return_acts=True)
            else:
                net = MTNet_s(**kwargs, return_acts=True)
        elif "MTNET_M" in name.upper():
            if "G8" in name.upper():
                net = MTNet_m_g8(**kwargs, return_acts=True)
            else:
                net = MTNet_m(**kwargs, return_acts=True)
        elif "MTNET_L" in name.upper():
            if "G8" in name.upper():
                net = MTNet_l_g8(**kwargs, return_acts=True)
            else:
                net = MTNet_l(**kwargs, return_acts=True)
        elif "MTNET_XL" in name.upper():
            net = MTNet_l(**kwargs, return_acts=True)
        elif "MTNET_XXL" in name.upper():
            net = MTNet_xxl(**kwargs, return_acts=True)
        else:
            net = MTNet_m(**kwargs, return_acts=True)
    # Res_net 3D
    elif "R3D" in name.upper():
        if "R3D_18" in name.upper():
            net = r3d_18(**kwargs, return_acts=True)
        elif "R3D_34" in name.upper():
            net = r3d_34(**kwargs, return_acts=True)
        elif "R3D_50" in name.upper():
            net = r3d_50(**kwargs, return_acts=True)
        elif "R3D_101" in name.upper():
            net = r3d_101(**kwargs, return_acts=True)
        elif "R3D_152" in name.upper():
            net = r3d_152(**kwargs, return_acts=True)
        elif "R3D_200" in name.upper():
            net = r3d_200(**kwargs, return_acts=True)
        elif "R3DXT50" in name.upper():
            net = r3dxt50_32x4d(**kwargs, return_acts=True)
        elif "R3DXT101" in name.upper():
            net = r3dxt101_32x8d(**kwargs, return_acts=True)
        elif "WIDE_R3D50" in name.upper():
            net = wide_r3d50_2(**kwargs, return_acts=True)
        else:
            net = wide_r3d101_2(**kwargs, return_acts=True)
    # Res_net (2+1)D
    elif "R2PLUS1D" in name.upper():
        if "R2PLUS1D_18" in name.upper():
            net = r2plus1d_18(**kwargs, return_acts=True)
        elif "R2PLUS1D_34" in name.upper():
            net = r2plus1d_34(**kwargs, return_acts=True)
        elif "R2PLUS1D_50" in name.upper():
            net = r2plus1d_50(**kwargs, return_acts=True)
        elif "R2PLUS1D_101" in name.upper():
            net = r2plus1d_101(**kwargs, return_acts=True)
        elif "R2PLUS1D_152" in name.upper():
            net = r2plus1d_152(**kwargs, return_acts=True)
        elif "R2PLUS1D_200" in name.upper():
            net = r2plus1d_200(**kwargs, return_acts=True)
        elif "R2PLUS1DXT50" in name.upper():
            net = r2plus1dxt50_32x4d(**kwargs, return_acts=True)
        elif "R2PLUS1DXT101" in name.upper():
            net = r2plus1dxt101_32x8d(**kwargs, return_acts=True)
        elif "WIDE_R2PLUS1D50" in name.upper():
            net = wide_r2plus1d50_2(**kwargs, return_acts=True)
        else:
            net = wide_r2plus1d101_2(**kwargs, return_acts=True)
    # Res_net 3D + SRTG
    elif "SRTG_R3D" in name.upper():
        if "SRTG_R3D_18" in name.upper():
            net = srtg_r3d_18(**kwargs, return_acts=True)
        elif "SRTG_R3D_34" in name.upper():
            net = srtg_r3d_34(**kwargs, return_acts=True)
        elif "SRTG_R3D_50" in name.upper():
            net = srtg_r3d_50(**kwargs, return_acts=True)
        elif "SRTG_R3D_101" in name.upper():
            net = srtg_r3d_101(**kwargs, return_acts=True)
        elif "SRTG_R3D_152" in name.upper():
            net = srtg_r3d_152(**kwargs, return_acts=True)
        elif "SRTG_R3D_200" in name.upper():
            net = srtg_r3d_200(**kwargs, return_acts=True)
        elif "SRTG_R3DXT50" in name.upper():
            net = srtg_r3dxt50_32x4d(**kwargs, return_acts=True)
        elif "SRTG_R3DXT101" in name.upper():
            net = srtg_r3dxt101_32x8d(**kwargs, return_acts=True)
        elif "SRTG_WIDE_R3D50" in name.upper():
            net = srtg_wide_r3d50_2(**kwargs, return_acts=True)
        else:
            net = srtg_wide_r3d101_2(**kwargs, return_acts=True)
    elif "SRTG_R2PLUS1D" in name.upper():
        if "SRTG_R2PLUS1D_18" in name.upper():
            net = srtg_r2plus1d_18(**kwargs, return_acts=True)
        elif "SRTG_R2PLUS1D_34" in name.upper():
            net = srtg_r2plus1d_34(**kwargs, return_acts=True)
        elif "SRTG_R2PLUS1D_50" in name.upper():
            net = srtg_r2plus1d_50(**kwargs, return_acts=True)
        elif "SRTG_R2PLUS1D_101" in name.upper():
            net = srtg_r2plus1d_101(**kwargs, return_acts=True)
        elif "SRTG_R2PLUS1D_152" in name.upper():
            net = srtg_r2plus1d_152(**kwargs, return_acts=True)
        elif "SRTG_R2PLUS1D_200" in name.upper():
            net = srtg_r2plus1d_200(**kwargs, return_acts=True)
        elif "SRTG_R2PLUS1DXT50" in name.upper():
            net = srtg_r2plus1dxt50_32x4d(**kwargs, return_acts=True)
        elif "SRTG_R2PLUS1DXT101" in name.upper():
            net = srtg_r2plus1dxt101_32x8d(**kwargs, return_acts=True)
        elif "SRTG_WIDE_R2PLUS1D50" in name.upper():
            net = srtg_wide_r2plus1d50_2(**kwargs, return_acts=True)
        else:
            net = srtg_wide_r2plus1d101_2(**kwargs, return_acts=True)
    else:
        logging.error("network '{}'' not implemented".format(name))
        raise NotImplementedError()

    return net
'''
---  E N D  O F  F U N C T I O N  G E T _ S Y M B O L ---
'''

def get_pooling(name, samplers):
    if name.upper() == 'AVG':
        pool = torch.nn.AdaptiveAvgPool1d((1))
    elif name.upper() == 'MAX':
        pool = torch.nn.AdaptiveMaxPool1d((1))
        '''
    elif name.upper() == 'EM':
        pool = EMPool1d(kernel_size=(num_samplers))
    elif name.upper() == 'EDSCW':
        pool = EDSCWPoll1d(kernel_size=(num_samplers))
    elif name.upper() == 'IDW':
        pool = IDWPool1d(kernel_size=(num_samplers))
    elif name.upper() == 'ADA':
        pool = AdaPool1d(kernel_size=(num_samplers), beta=(1))
        '''
    else:
        logging.error("Pooling method '{}'' not implemented".format(name))
        raise NotImplementedError()
    return pool


class Combined(torch.nn.Module):
    def __init__(self,
                 backbone,
                 head=None,
                 pool=None,
                 print_net=False,
                 num_samplers=4,
                 precision='fp32',
                 **kwargs):
        super(Combined, self).__init__()

        assert num_samplers >= 1, 'Cannot create model with `num_samplers` being less than 1!'
        self.samplers = num_samplers
        self.precision = precision

        # Get backbone model, input configuration, head model and pooling method
        self.backbone = get_symbol(backbone, samplers=self.samplers, headless =True, **kwargs)
        self.backbone.requires_grad_(False)

        if pool is not None:
            pool = get_pooling(pool, samplers=self.samplers)
            self.pred_fusion = torch.nn.Sequential(
                                 Rearrange('b s c -> b c s'),
                                 pool,
                                 Rearrange('b c 1 -> b c')) if pool is not None else None
        else:
            self.pred_fusion = None

        if (head is not None):
            self.head = get_symbol(head, samplers=self.samplers, **kwargs)

        # model printing
        converted = []
        ks = []

        if print_net:
            beautify_net(self.backbone)
            if self.head is not None:
                beautify_net(self.head)


    def forward(self,x):

        # Assume x of shape [B S C T H W] and reshape to [BxS C T H W]
        B, _, _, _, _, _ = x.shape
        x = rearrange(x, 'b s c t h w -> (b s) c t h w')
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred, x = self.backbone(x)
            if self.precision=='fp32':
                pred = pred.float()
                x = x.float()
        # feature pooling
        _, _, t, _, _ = x.shape
        x = F.adaptive_avg_pool3d(x, (t,4,4))
        # Rearrange features to [B S C' t 4 4] and predictions [B S C]
        x = rearrange(x, '(b s) c t h w -> b c s t h w',b=B)
        pred = rearrange(pred, '(b s) c -> b s c',b=B)

        if self.head is not None:
            pred = self.head(x)
            if self.pred_fusion is not None:
                pred = self.pred_fusion(pred)
            else:
                pred_list = rearrange(pred_list, 's b c -> b c s')
                pred = reduce(pred_list, 'b c s -> b c', 'mean')
        return pred





if __name__ == "__main__":

    net = Combined(backbone='mtnet_s',
                   head='TemPer_h',
                   pool='avg',
                   print_net=False,
                   num_samplers=4,
                   t_dim=16).cuda()
    tmp = torch.rand([1, 4, 3, 16, 186, 186]).cuda()
    out = net(tmp)
    print('Exited sucessfully',out.shape)

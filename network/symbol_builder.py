'''
---  I M P O R T  S T A T E M E N T S  ---
'''

import coloredlogs, logging
coloredlogs.install()
from .resnet import r3d_18, r3d_34, r3d_50, r3d_101, r3d_152, r3d_200, r3dxt50_32x4d, r3dxt101_32x8d

from .swin import get_swin_ssv2

from .movinets import MoViNet
from .movinets.config import _C

from .tempr_h import TemPr_h

from .config import get_config

import torch
import torch.nn.parallel
import torch.nn.functional as F

from einops import reduce, rearrange
from einops.layers.torch import Reduce, Rearrange

from ptflops import get_model_complexity_info
from torchinfo import summary

import adapool_cuda
from adaPool import IDWPool1d, EMPool1d, EDSCWPool1d, AdaPool1d

import sys



'''
---  S T A R T  O F  F U N C T I O N  B E A U TI F Y _ N E T ---
    [About]
        Function for creating a string to visualise the network modules.
    [Args]
        - net: torch.nn.module for the network to be visualised.
    [Returns]
        - None
'''
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
---  E N D  O F  F U N C T I O N  B E A U TI F Y _ N E T ---
'''


'''
---  S T A R T  O F  F U N C T I O N  G E T _ S Y M B O L ---
    [About]
        Function for loading PyTorch models.
    [Args]
        - name: String for the backbone/head network name.
        - samplers: Integer for the number of scales.
        - pool: String for the ensemble function to be used. Defaults to None.
        - headless: Boolean for runing solely the feature extractor part of the model. Defaults to False.
    [Returns]
        - net: Module for the loaded Pytorch network.
'''
def get_symbol(name, samplers, pool=None, headless=False, **kwargs):
    
    # MOVINET
    if 'MOVINET' in name.upper():
        net =  MoViNet(_C.MODEL.MoViNetA4, causal = False, pretrained = True, return_embed=True)
        net.classifier = torch.nn.Identity()
        in_channels = 856
    # X3D
    elif 'X3D' in name.upper():
        net = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
        net.blocks[5] = torch.nn.Identity()
        in_channels = 192
    # TemPr_h
    elif ('TEMPR' in name.upper()):
        net = TemPr_h(depth=samplers, return_acts=True, pool=pool, **kwargs)
        in_channels = None
    # Swin-B (ssv2)
    elif "SWIN" in name.upper():
        net = get_swin_ssv2(**kwargs)
        in_channels = 192
    # Res_net 3D
    elif "R3D" in name.upper():
        in_channels = 512
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
        else:
            net = r3dxt101_32x8d(**kwargs, return_acts=True)
    else:
        logging.error("network '{}'' not implemented".format(name))
        raise NotImplementedError()

    return (net, in_channels)
'''
---  E N D  O F  F U N C T I O N  G E T _ S Y M B O L ---
'''


'''
---  S T A R T  O F  F U N C T I O N  G E T _ P O O L I N G ---
    [About]
        Function for loading PyTorch models.
    [Args]
        - name: String for the ensemble method to be used.
        - samplers: Integer for the number of scales (to be used as kernel size).
    [Returns]
        - pool: Module corresponding to the ensemble method chosen.
'''
def get_pooling(name, samplers):
    if name.upper() == 'AVG':
        pool = torch.nn.AdaptiveAvgPool1d((1))
    elif name.upper() == 'MAX':
        pool = torch.nn.AdaptiveMaxPool1d((1))
    elif name.upper() == 'EM':
        pool = EMPool1d(kernel_size=(samplers))
    elif name.upper() == 'EDSCW':
        pool = EDSCWPool1d(kernel_size=(samplers))
    elif name.upper() == 'IDW':
        pool = IDWPool1d(kernel_size=(samplers))
    elif name.upper() == 'ADA':
        pool = AdaPool1d(kernel_size=(samplers), beta=(1))
    else:
        logging.error("Pooling method '{}'' not implemented".format(name))
        raise NotImplementedError()
    return pool
'''
---  E N D  O F  F U N C T I O N  G E T _ P O O L I N G ---
'''


'''
===  S T A R T  O F  C L A S S  C O N T I G U O U S ===

    [About]

        nn.Module helper class for converting tensors to contiguous memory.

    [Init Args]

        - None

    [Methods]

        - __init__ : Class initialiser
        - forward: Function for operation calling.
'''
class Contiguous(torch.nn.Module):
    def __init__(self):
        super(Contiguous, self).__init__()

    def forward(self,x):
        return x.contiguous()
'''
===  E N D  O F  C L A S S  C O N T I G U O U S ===
'''


'''
===  S T A R T  O F  C L A S S  C O M B I N E D ===

    [About]

        nn.Module class that combined the feature extractor (backbone), attention towers (head), and predictor aggregation function (pool).

    [Init Args]

        - backbone: String for the backbone feature extractor to be used.
        - head: String for the model to be used as head. If `None`/None, no head network is used and the
        predictions are made directly from the backbone. Defaults to None.
        - pool: String for the ensemble function to aggregate together individual predictors. If None, then
        the ouput will include prediction(s) over each scale in range of `num_samplers`. Defaults to None.
        - print_net: Boolean for printing the network in a per-string format for each module/sub-module. Defaults
        to False.
        - num_samplers: Integer for the number of scales to be used. Defaults to 4.
        - precision: [Depricated] String for using either `mixed` or `fp32` precision. This only applies
        for the head network. You use this configuration by looking at lines #318-322. Defaults to `fp32`.

    [Methods]

        - __init__ : Class initialiser
        - forward: Function for operation calling.
'''
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
        self.has_head = head is not None

        # Get backbone model, input configuration, head model and pooling method
        self.backbone, in_channels = get_symbol(backbone, samplers=self.samplers, headless =True, **kwargs)
        self.backbone.requires_grad_(False)

        self.rarrange = torch.nn.Sequential(
                             Rearrange('b s c -> s b c'))

        if pool is not 'none' and pool is not None:
            pool = get_pooling(pool, samplers=self.samplers)
            make_contiguous = Contiguous()
            self.pred_fusion = torch.nn.Sequential(
                                 Rearrange('b s c -> b c s'),
                                 make_contiguous,
                                 pool,
                                 Rearrange('b c 1 -> b c')) if pool is not None else None
        else:
            self.pred_fusion = None

        if head!='none' and head is not None:
            self.backbone.requires_grad_(False)
            self.head,_ = get_symbol(head, samplers=self.samplers, input_channels=in_channels, **kwargs)
        else:
            self.fc = torch.nn.Linear(in_channels,kwargs['num_classes'])

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
        with torch.set_grad_enabled(self.has_head):
            with torch.cuda.amp.autocast():
                x = self.backbone(x)
            if self.precision=='fp32':
                #pred = pred.float()
                x = x.float()
        # feature pooling
        _, _, t, _, _ = x.shape
        x = F.adaptive_avg_pool3d(x, (t,4,4))
        

        # Rearrange features to [B S C' t 4 4] and predictions [B S C]
        x = rearrange(x, '(b s) c t h w -> b c s t h w',b=B)
        #pred = rearrange(pred, '(b s) c -> b s c',b=B)

        if hasattr(self, 'head'):
            # if self.precision=='mixed':
            #with torch.cuda.amp.autocast():
            pred = self.head(x)
            #else:
            #pred = self.head(x)
            if self.pred_fusion is not None:
                preds = pred
                pred = self.pred_fusion(pred)
                return (pred, preds)
            else:
                return self.rarrange(pred)
        else:
            x = reduce(x, 'b c s t h w -> b 1 c','mean')
            pred = self.fc(x)

        return (pred.squeeze(1), pred)

'''
===  E N D  O F  C L A S S  C O M B I N E D ===
'''




if __name__ == "__main__":

    net = Combined(backbone='x3d',
                   head='TemPr_h',
                   pool='ada',
                   print_net=False,
                   num_samplers=4,
                   t_dim=16).cuda()
    
    tmp = torch.rand([32, 4, 3, 16, 224, 224]).cuda()
    out = net(tmp)
    print('Exited sucessfully',out[0].shape)

    runs = []
    for i in range(0, 1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = net(tmp)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        runs.append(time)
        print(i, 'elapsed time: ', time, end='\r')
    print('Avg elapsed time:',sum(runs)/len(runs))


from math import pi, log
from functools import wraps
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce, Rearrange


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi # [T, H, W, T, num_bands]
    x = torch.cat([x.sin(), x.cos()], dim = -1) # [T, H, W, T, 2 x num_bands]
    x = torch.cat((x, orig_x), dim = -1) # [T, H, W, T, (2 x num_bands)+1]
    return x

'''
if self.use_gates or self.use_soft_gating:
    # Should aways exit at least at the final frame
    if (i==self.depth-1):
        gates_t += torch.ones((b), device=data.device)
    else:
        # Main gating functionality
        gate = self.gates[i]
        # First gate (no previous frame to concatenate)
        if (i<1):
            gates_t = gate(x_t)
            # Main gating functionality
        else:
            x_cat = rearrange([x_t,x_prev], 'l b n d -> b n (d l)')
            gates_t += gate(x_cat)
    # Update the gate values
    gs[i] = gates_t
    # Fool profing
    gates_t += gates_t


if self.use_gates:
# Gate values > 1 denote duplicates
gs[gs>1.] *= 0.

# zero-out temporal outputs
gs = rearrange(gs, 't b -> t b 1 1')
x_list = rearrange(x_list, 't b n d -> t b n d')
x_list = x_list * gs
# create tensor (can also use `sum` as reduction method)
x = reduce(x_list, 't b n d -> b n d', 'max')

if self.use_soft_gating:

# Scale > 1. values back to 1
gs = torch.pow(gs,0)

# zero-out non informative frames
gs = rearrange(gs, 't b -> t b 1 1')
x_list = rearrange(x_list, 't b n d -> t b n d')
x_list = x_list * gs


if self.use_temporal_fusion or self.use_soft_gating:
x_list = rearrange(x_list, 't b n d -> t b n d')
if self.mode == 'max':
    x = reduce(x_list, 't b n d -> b n d', 'max')
elif self.mode == 'avg':
    x = reduce(x_list, 't b n d -> b n d', 'mean')
else:
    n_dim = x_list[0].shape[-2]
    x = rearrange(x_list, 't b n d -> b (n d) t')
    if self.mode=='em':
        x = empool1d(x,kernel_size=self.depth)
    elif self.mode=='idw':
        x = idwpool1d(x,kernel_size=self.depth)
    elif self.mode=='edscw':
        x = edscwpool1d(x,kernel_size=self.depth)
    elif self.mode=='ada':
        x = self.pool(x)
    x = rearrange(x, 'b (n d) 1 -> b n d', n=n_dim)
else:
x = x_t
'''

'''
class ExitingGate(nn.Module):
    def __init__(self, latent_dim, thres=0.5):
        super(ExitingGate, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.attn = PreNorm(latent_dim, Attention(latent_dim, heads = 4))

        self.linear = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1, bias=True))
        self.thres = thres

    def forward(self, x):

        #if isinstance(x,list):
        #    x = rearrange(x, 'l b n d -> b (l n) d') # concatenate over

        #if (self.type == 'cross'):
        #    x, x_prev = x
        #    x = self.attn(x, context = x_prev, mask = None) + x
        #else:
        x = self.attn(x) + x

        x = self.linear(x) # [B, 2N, D] => [B, 1]
        x = rearrange(x, 'b 1 -> b')
        x = self.sigmoid(x)
        # Exit flag [0/1] for each element in batch nased on set `thres`
        x[x >= self.thres] = 1
        x[x < self.thres] = 0
        # Case of first gate
        return x
'''

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )


    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.temp = 2.0

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def stable_softmax(self,x):
        x = torch.nan_to_num(x)
        x -= reduce(x, '... d -> ... 1', 'max')
        return x.softmax(dim = -1)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        #sim /= self.temp
        #sim = torch.clamp(sim, min=1e-8, max=1e+8)
        attn = self.stable_softmax(sim)
        #attn = torch.nan_to_num(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class
class TemPr_h(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands=10,
        depth,
        max_freq=10.,
        input_channels = 512,
        num_latents = 256,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True,
        **kwargs
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Number of frames (for the network depth).
          max_freq: Maximum frequency, hyperparameter depending on how fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Perceiver paper uses 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = 3
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.depth = depth

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (self.input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        torch.nn.init.kaiming_uniform_(self.latents)
        self.num_classes = num_classes

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(self.depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns,
                Reduce('b n d -> b d', 'mean')if final_classifier_head else nn.Identity(),
                nn.Linear(latent_dim, self.num_classes) if final_classifier_head else nn.Identity()
            ]))


        #self.fc = nn.Linear(latent_dim, self.num_classes)


    def forward(self, data, mask = None, return_embeddings = False):
        data = rearrange(data, 'b c s t h w -> s b t h w c')
        s, b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data has %d axes, expected %d'%(len(axis),self.input_axis)

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis)) # [T, H , W]
            pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1) # [T, H, W, 3]
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)# [T, H, W, 3, (2 x num_bands)+1]
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')# [T, H, W, 3 x ((2 x num_bands)+1)]
            enc_pos = repeat(enc_pos, '... -> s b ...', s = s, b = b)# [S, B, T, H, W, 3 x ((2 x num_bands)+1)]

            # Create a concat version of the data and the PE
            data = torch.cat((data, enc_pos), dim = -1)# [S, B, T, H, W, D]

        # flatten spatio-temporal dim
        # [S, B, T, H, W, D] => [S, B, T x H x W, D]
        data = rearrange(data, 's b ... d -> s b (...) d')

        # Repeat latents over batch dim
        x_l = repeat(self.latents, 'n d -> b n d', b = b)

        # layers
        x_list = []

        # Main calls
        for i,(cross_attn, cross_ff, self_attns, reduce, fc) in enumerate(self.layers):
            x_t = x_l
            x_prev = x_t
            # Cross attention
            x_t = cross_attn(x_t, context = data[i], mask = mask) + x_t
            x_t = cross_ff(x_t) + x_t

            # Latent Transformer
            for self_attn, self_ff in self_attns:
                x_t = self_attn(x_t) + x_t
                x_t = self_ff(x_t) + x_t

            x_t = reduce(x_t)
            x_t = fc(x_t)
            x_list.append(x_t)

        # to tensor
        pred = torch.stack(x_list,dim=0)
        pred = rearrange(pred, 's b d -> b s d')

        # used for fetching embeddings
        if return_embeddings:
            return pred, x
        return pred




if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    from torchinfo import summary

    ####################################
    ##### N E T W O R K  T E S T S  ####
    ####################################

    depth = 4

    #--- TEST 1 --- (train -- fp32)
    tmp = torch.rand(64,512,depth,16,4,4).cuda()
    net = torch.nn.DataParallel(TemPr_h(num_freq_bands=10, depth=depth, max_freq=10., input_channels=512, latent_dim = 256, latent_heads = 8).cuda())
    out = net(tmp)
    print('--- TEST 1 (train -- fp32) passed ---','input:',tmp.shape,'exited the network with new shape:',out.shape,'\n')
    del out, net, tmp

    #--- TEST 2 --- (inference -- fp32)
    tmp = torch.rand(64,512,depth,16,4,4).cuda()
    net = torch.nn.DataParallel(TemPr_h(num_freq_bands=10, depth=depth, max_freq=10., input_channels=512, latent_dim = 256, latent_heads = 8).cuda())
    with torch.no_grad():
        out = net(tmp)
    print('--- TEST 2 (inference -- fp32) passed ---','input:',tmp.shape,'exited the network with new shape:',out.shape,'\n')
    del out, net, tmp

    #--- TEST 3 --- (train -- mixed)
    tmp = torch.rand(128,512,depth,12,4,4).cuda().half()
    net = torch.nn.DataParallel(TemPr_h(num_freq_bands=10, depth=depth, max_freq=10., input_channels=512, latent_dim = 256, latent_heads = 8).cuda())
    out = net(tmp)
    print('--- TEST 3 (mixed) passed ---','input:',tmp.shape,'exited the network with new shape:',out.shape,'\n')
    del out, net, tmp

    #--- TEST 4 --- (inference -- mixed)
    tmp = torch.rand(128,512,depth,12,4,4).cuda().half()
    net = torch.nn.DataParallel(TemPr_h(num_freq_bands=10, depth=depth, max_freq=10., input_channels=512, latent_dim = 256, latent_heads = 8).cuda())
    with torch.no_grad():
        out = net(tmp)
    print('--- TEST 4 (inference -- mixed) passed ---','input:',tmp.shape,'exited the network with new shape:',out.shape,'\n')
    del out, net, tmp

    tmp = (512,depth,16,4,4)
    net = TemPr_h(num_freq_bands=10, depth=depth, max_freq=10., input_channels=512, num_latents=256, latent_dim = 256)

    macs, params = get_model_complexity_info(net, tmp, as_strings=True,print_per_layer_stat=False, verbose=False)
    print('-- TEST 5 passed --- ')
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('\n')

    #--- TEST 6 --- summary
    net = TemPr_h(num_freq_bands=10, depth=depth, max_freq=10., input_channels=512, num_latents=256, latent_dim = 256).cuda()
    summary(net, (8,512,depth,16,4,4))
    print('--- TEST 6 passed --- \n')

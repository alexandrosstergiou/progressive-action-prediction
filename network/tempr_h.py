from math import pi, log
from functools import wraps
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce, Rearrange

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

        #! Use of stability (in case on Nans)
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
                self_attns
            ]))

        self.reduce = nn.Sequential(
            Reduce('s b n d -> s b d', 'mean'),
            Rearrange('s b d -> b s d'),
            nn.LayerNorm(latent_dim)
        ) if final_classifier_head else nn.Identity()

        self.fc = nn.Linear(latent_dim, self.num_classes) if final_classifier_head else nn.Identity()


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
        for i,(cross_attn, cross_ff, self_attns) in enumerate(self.layers):
            x_t = x_l
            x_prev = x_t
            # Cross attention
            x_t = cross_attn(x_t, context = data[i], mask = mask) + x_t
            x_t = cross_ff(x_t) + x_t

            # Latent Transformer
            for self_attn, self_ff in self_attns:
                x_t = self_attn(x_t) + x_t
                x_t = self_ff(x_t) + x_t


            x_list.append(x_t)

        # to logits
        x = self.reduce(torch.stack(x_list,dim=0))

        # class predictions
        pred = self.fc(x)
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

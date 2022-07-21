import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
import math
import warnings
from mmedit.models.common import (ResidualBlockNoBN,
                                  flow_warp, make_layer)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x.permute(0,1,3,4,2)).permute(0,1,4,2,3)
        return self.fn(x, *args, **kwargs)

# Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, dim, num_resblocks):
        super().__init__()
        main = []
        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_resblocks, mid_channels=dim))
        self.net = nn.Sequential(*main)

    def forward(self, x):
        out = self.net(x)
        return out

# Flow-Guided Sparse Window-based Multi-head Self-Attention
class FGSW_MSA(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(5,4,4),
        dim_head=64,
        heads=8,
        shift=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift
        inner_dim = dim_head * heads

        # position embedding
        q_l = self.window_size[1]*self.window_size[2]
        kv_l = self.window_size[0]*self.window_size[1]*self.window_size[2]
        self.static_a = nn.Parameter(torch.Tensor(1, heads, q_l , kv_l))
        trunc_normal_(self.static_a)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 3, 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 3, 1, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)

    def forward(self, q_inp, k_inp, flow):
        """
        :param q_inp: [n,1,c,h,w]
        :param k_inp: [n,2r+1,c,h,w]  (r: temporal radius of neighboring frames)
        :param flow: list: [[n,2,h,w],[n,2,h,w]]
        :return: out: [n,1,c,h,w]
        """
        b,f_q,c,h,w = q_inp.shape
        fb,hb,wb = self.window_size

        [flow_f, flow_b] = flow
        # sliding window
        if self.shift:
            q_inp, k_inp = map(lambda x: torch.roll(x, shifts=(-hb//2, -wb//2), dims=(-2, -1)), (q_inp, k_inp))
            if flow_f is not None:
                flow_f = torch.roll(flow_f, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
            if flow_b is not None:
                flow_b = torch.roll(flow_b, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
        k_f, k_r, k_b = k_inp[:, 0], k_inp[:, 1], k_inp[:, 2]

        # retrive key elements
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False
        grid = grid.type_as(k_f)
        if flow_f is not None:
            vgrid = grid + flow_f.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            # index the nearest token
            # k_f = F.grid_sample(k_f.float(), vgrid_scaled, mode='bilinear')
            k_f = F.grid_sample(k_f.float(), vgrid_scaled, mode='nearest')
        if flow_b is not None:
            vgrid = grid + flow_b.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            # index the nearest token
            # k_b = F.grid_sample(k_b.float(), vgrid_scaled, mode='bilinear')
            k_b = F.grid_sample(k_b.float(), vgrid_scaled, mode='nearest')

        k_inp = torch.stack([k_f, k_r, k_b], dim=1)
        # norm
        q = self.norm_q(q_inp.permute(0,1,3,4,2)).permute(0,1,4,2,3)
        kv = self.norm_kv(k_inp.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        q = self.to_q(q.flatten(0, 1))
        k, v = self.to_kv(kv.flatten(0, 1)).chunk(2, dim=1)

        # split into (B,N,C)
        q, k, v = map(lambda t: rearrange(t, '(b f) c (h p1) (w p2)-> (b h w) (f p1 p2) c', p1=hb, p2=wb, b=b), (q, k, v))

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # scale
        q *= self.scale

        # attention
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.static_a
        attn = sim.softmax(dim=-1)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        # merge windows back to original feature map
        out = rearrange(out, '(b h w) (f p1 p2) c -> (b f) c (h p1) (w p2)', b=b, h=(h // hb), w=(w // wb),
                        p1=hb, p2=wb)

        # combine heads
        out = self.to_out(out).view(b, f_q, c, h, w)

        # inverse shift
        if self.shift:
            out = torch.roll(out, shifts=(hb//2, wb//2), dims=(-2, -1))

        return out

class FGAB(nn.Module):
    def __init__(
        self,
        q_dim,
        emb_dim,
        window_size=(3,4,4),
        dim_head=64,
        heads=8,
        num_resblocks=5,
        shift=False
    ):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.embed_dim = emb_dim
        self.q_dim = q_dim
        self.attn = FGSW_MSA(q_dim,window_size,dim_head,heads,shift=shift)
        self.feed_forward = FeedForward(q_dim,num_resblocks)
        self.conv = nn.Conv2d(q_dim+emb_dim, q_dim,3,1,1,bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.shift = shift

    def forward(self, x, flows_forward, flows_backward, cpu_cache):
        """
        :param x: [n,t,c,h,w]
        :param flows_forward: [n,t,2,h,w]
        :param flows_backward: [n,t,2,h,w]
        :return: outs: [n,t,c,h,w]
        """
        t = len(x)
        n,c,h,w = x[0].shape
        outs = []
        embedding = flows_forward[0].new_zeros(n, self.embed_dim, h, w)
        for i in range(0, t):
            flow_f, flow_b = None, None
            if i>0:
                flow_f = flows_forward[i-1]
                if cpu_cache:
                    flow_f = flow_f.cuda()
                    embedding = embedding.cuda()
                embedding = flow_warp(embedding, flow_f.permute(0, 2, 3, 1))
                k_f = x[i-1]
            else:
                k_f = x[i]
            if i<t-1:
                flow_b = flows_backward[i]
                if cpu_cache:
                    flow_b = flow_b.cuda()
                k_b = x[i+1]
            else:
                k_b = x[i]
            x_current = x[i]
            if cpu_cache:
                embedding = embedding.cuda()
                x_current = x_current.cuda()
                k_f = k_f.cuda()
                k_b = k_b.cuda()
            q_inp = self.lrelu(self.conv(torch.cat((embedding,x_current),dim=1))).unsqueeze(1)
            k_inp = torch.stack([k_f, x_current, k_b], dim=1)
            out = self.attn(q_inp=q_inp, k_inp=k_inp, flow=[flow_f,flow_b]) + q_inp
            out = out.squeeze(1)
            out = self.feed_forward(out) + out
            embedding = out
            if cpu_cache:
                out = out.cpu()
                torch.cuda.empty_cache()
            outs.append(out)
        return outs

class FGABs(nn.Module):
    def __init__(
            self,
            q_dim,
            emb_dim,
            window_size=(3,3,3),
            heads=4,
            dim_head=32,
            num_resblocks=20,
            num_FGAB = 1,
            reverse=(True,False),
            shift=(True,False)

    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FGAB(
                q_dim=q_dim,emb_dim=emb_dim, window_size=window_size, heads=heads, dim_head=dim_head, num_resblocks=num_resblocks, shift=shift
                )
            for _ in range(num_FGAB)])
        self.reverse = reverse

    def forward(self, video, flows_forward, flows_backward, cpu_cache):
        """
        :param video: [n,t,c,h,w]
        :param flows: list [[n,t-1,2,h,w],[n,t-1,2,h,w]]
        :return: x: [n,t,c,h,w]
        """
        for i in range(len(self.layers)):
            layer = self.layers[i]
            reverse = self.reverse[i]
            if not reverse:
                video = layer(video, flows_forward=flows_forward, flows_backward=flows_backward, cpu_cache=cpu_cache)
            else:
                video = layer(video[::-1], flows_forward=flows_backward[::-1],
                          flows_backward=flows_forward[::-1], cpu_cache=cpu_cache)
                video = video[::-1]
        return video



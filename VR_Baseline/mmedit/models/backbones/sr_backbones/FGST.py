import torch.nn.functional as F
import torch.nn as nn
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.backbones.sr_backbones.FGST_util import FGABs
import torch
from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmedit.models.common import (ResidualBlockNoBN, make_layer)
from mmcv.runner import load_checkpoint

def Forward(x, model, cpu_cache):
    feat = []
    t = len(x)
    for i in range(0, t):
        feat_i = x[i]
        if cpu_cache:
            feat_i = feat_i.cuda()
        feat_i = model(feat_i)
        if cpu_cache:
            feat_i = feat_i.cpu()
            torch.cuda.empty_cache()
        feat.append(feat_i)
    return feat

@BACKBONES.register_module()
class FGST(nn.Module):
    def __init__(self, dim=32, spynet_pretrained=None, cpu_cache_length=30):
        super(FGST, self).__init__()
        self.dim = dim
        self.cpu_cache_length = cpu_cache_length

        #### optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        #### embedding
        self.embedding = ResidualBlocksWithInputConv(3, dim, 5)

        #### transformer blocks
        self.window_size = [3,3,3]
        self.emb_dim = 32
        self.encoder_1 = FGABs(
            q_dim=self.dim,
            emb_dim=self.emb_dim,
            window_size=self.window_size,
            num_FGAB=1,
            heads=2,
            dim_head=32,
            num_resblocks=5,
            reverse=[False],
            shift=[False]
        )
        self.parchmerge_1 = nn.Conv2d(self.dim, self.dim*2, 4, 2, 1, bias=False)
        self.encoder_2 = FGABs(
            q_dim=self.dim*2,
            emb_dim=self.emb_dim*2,
            window_size=self.window_size,
            num_FGAB=1,
            heads=4,
            dim_head=32,
            num_resblocks=5,
            reverse=[False],
            shift=[False]
        )
        self.parchmerge_2 = nn.Conv2d(self.dim*2, self.dim*4, 4, 2, 1, bias=False)
        self.bottle_neck = FGABs(
            q_dim=self.dim*4,
            emb_dim=self.emb_dim*4,
            window_size=self.window_size,
            num_FGAB=2,
            heads=8,
            dim_head=32,
            num_resblocks=5,
            reverse=[False,True],
            shift=[False, True]
        )
        self.patchexpand_1 = nn.ConvTranspose2d(self.dim*4, self.dim*2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.fution_1 = nn.Conv2d(self.dim*4, self.dim*2, 3, 1, 1, bias=False)
        self.decoder_1 = FGABs(
            q_dim=self.dim*2,
            emb_dim=self.emb_dim*2,
            window_size=self.window_size,
            num_FGAB=1,
            heads=4,
            dim_head=32,
            num_resblocks=5,
            reverse=[True],
            shift=[True]
        )
        self.patchexpand_2 = nn.ConvTranspose2d(self.dim * 2, self.dim * 1, stride=2, kernel_size=2, padding=0,
                                          output_padding=0)
        self.fution_2 = nn.Conv2d(self.dim*2, self.dim, 3, 1, 1, bias=False)
        self.decoder_2 = FGABs(
            q_dim=self.dim,
            emb_dim=self.emb_dim,
            window_size=self.window_size,
            num_FGAB=1,
            heads=2,
            dim_head=32,
            num_resblocks=5,
            reverse=[True],
            shift=[True]
        )
        # residual blocks after transformer
        main = []
        main.append(make_layer(ResidualBlockNoBN, 5, mid_channels=dim))
        main.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(nn.Conv2d(self.dim, 3, 3, 1, 1, bias=True))
        self.tail = nn.Sequential(*main)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def spatial_padding(self, lqs):

        n, t, c, h, w = lqs.shape
        tb, hb, wb = self.window_size
        hb *= 4
        wb *= 4
        pad_h = (hb - h % hb) % hb
        pad_w = (wb - w % wb) % wb

        # padding
        lqs = lqs.view(-1, c, h, w)
        lqs = F.pad(lqs, [0, pad_w, 0, pad_h], mode='reflect')

        return lqs.view(n, t, c, h + pad_h, w + pad_w)

    def compute_flow(self, lqs, flows):
        n, t, c, h, w = lqs.size()
        flows['forward'], flows['backward'], flows['forward_ds2'], flows['backward_ds2'], \
        flows['forward_ds4'], flows['backward_ds4'] = [], [], [], [], [], []
        lqs_1 = lqs[:, :-1, :, :, :]
        lqs_2 = lqs[:, 1:, :, :, :]
        for i in range(t-1):
            lq_1,lq_2 = lqs_1[:,i,:,:,:], lqs_2[:,i,:,:,:]
            flow_backward_ = self.spynet(lq_1, lq_2)
            flow_forward_ = self.spynet(lq_2, lq_1)
            flow_backward_ds2_ = F.avg_pool2d(flow_backward_, kernel_size=2,stride=2)/2.0
            flow_forward_ds2_ = F.avg_pool2d(flow_forward_, kernel_size=2, stride=2) / 2.0
            flow_backward_ds4_ = F.avg_pool2d(flow_backward_ds2_, kernel_size=2, stride=2) / 2.0
            flow_forward_ds4_ = F.avg_pool2d(flow_forward_ds2_, kernel_size=2, stride=2) / 2.0
            if self.cpu_cache:
                flow_backward_, flow_forward_ = flow_backward_.cpu(), flow_forward_.cpu()
                flow_backward_ds2_, flow_forward_ds2_ = flow_backward_ds2_.cpu(), flow_forward_ds2_.cpu()
                flow_backward_ds4_, flow_forward_ds4_ = flow_backward_ds4_.cpu(), flow_forward_ds4_.cpu()
                torch.cuda.empty_cache()
            flows['forward'].append(flow_forward_)
            flows['backward'].append(flow_backward_)
            flows['forward_ds2'].append(flow_forward_ds2_)
            flows['backward_ds2'].append(flow_backward_ds2_)
            flows['forward_ds4'].append(flow_forward_ds4_)
            flows['backward_ds4'].append(flow_backward_ds4_)
        return flows

    def forward(self, x):
        """
        :param x: [n,t,c,h,w]
        :return: out: [n,t,c,h,w]
        """
        n, t, c, h_input, w_input = x.size()

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and x.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        # pad the input and make sure that it can be reshape into several windows
        lqs = self.spatial_padding(x)
        h, w = lqs.size(3), lqs.size(4)

        # compute optical flow
        flows = {}
        flows = self.compute_flow(lqs, flows)

        feats = {}
        # embedding
        if self.cpu_cache:
            feats['encoder1'] = []
            for i in range(0, t):
                feat_ = self.embedding(lqs[:, i, :, :, :]).cpu()
                feats['encoder1'].append(feat_)
                torch.cuda.empty_cache()
        else:
            feats_ = self.embedding(lqs.flatten(0, 1)).view(n, t, self.dim, h, w)
            feats['encoder1'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # FGABs
        feats['encoder1'] = self.encoder_1(feats['encoder1'], flows['forward'], flows['backward'], self.cpu_cache)
        feats['encoder2'] = Forward(feats['encoder1'], self.parchmerge_1, self.cpu_cache)
        feats['encoder2'] = self.encoder_2(feats['encoder2'], flows['forward_ds2'], flows['backward_ds2'], self.cpu_cache)

        feats['bottle'] = Forward(feats['encoder2'], self.parchmerge_2, self.cpu_cache)
        feats['bottle'] = self.bottle_neck(feats['bottle'], flows['forward_ds4'], flows['backward_ds4'], self.cpu_cache)
        feats['decoder1'] = Forward(feats['bottle'], self.patchexpand_1, self.cpu_cache)
        if self.cpu_cache:
            del feats['bottle']

        for i in range(0,t):
            feat_encoder_2 = feats['encoder2'][i]
            feat_decoder_1 = feats['decoder1'][i]
            if self.cpu_cache:
                feat_encoder_2 = feat_encoder_2.cuda()
                feat_decoder_1 = feat_decoder_1.cuda()
            feat_ = self.lrelu(self.fution_1(torch.cat((feat_encoder_2,feat_decoder_1),dim=1)))
            if self.cpu_cache:
                feat_ = feat_.cpu()
                torch.cuda.empty_cache()
            feats['decoder1'][i] = feat_
        if self.cpu_cache:
            del feats['encoder2']

        feats['decoder1'] = self.decoder_1(feats['decoder1'], flows['forward_ds2'], flows['backward_ds2'], self.cpu_cache)
        feats['decoder2'] = Forward(feats['decoder1'], self.patchexpand_2, self.cpu_cache)
        if self.cpu_cache:
            del feats['decoder1'], flows['forward_ds2'], flows['backward_ds2']

        for i in range(0,t):
            feat_encoder_1 = feats['encoder1'][i]
            feat_decoder_2 = feats['decoder2'][i]
            if self.cpu_cache:
                feat_encoder_1 = feat_encoder_1.cuda()
                feat_decoder_2 = feat_decoder_2.cuda()
            feat_ = self.lrelu(self.fution_2(torch.cat((feat_encoder_1,feat_decoder_2),dim=1)))
            if self.cpu_cache:
                feat_ = feat_.cpu()
                torch.cuda.empty_cache()
            feats['decoder2'][i] = feat_
        if self.cpu_cache:
            del feats['encoder1']
        feats['decoder2'] = self.decoder_2(feats['decoder2'],flows['forward'], flows['backward'], self.cpu_cache)
        if self.cpu_cache:
            del flows

        # tail
        feats['decoder2'] = Forward(feats['decoder2'], self.tail, self.cpu_cache)

        outputs = []
        for i in range(0, t):
            feature_i = feats['decoder2'].pop(0)
            if self.cpu_cache:
                feature_i = feature_i.cuda()
            out = feature_i + lqs[:, i, :, :, :]
            if self.cpu_cache:
                out = out.cpu()
                torch.cuda.empty_cache()
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        return outputs[:, :, :, :h_input, :w_input]

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')




















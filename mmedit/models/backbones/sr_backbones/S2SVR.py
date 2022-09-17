import torch
import torch.nn as nn
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.backbones.sr_backbones.S2SVR_util import Encoder, Decoder
from mmedit.models.backbones.sr_backbones.pwclite import PWCLite, resize_flow, restore_model
from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmcv.runner import load_checkpoint
from easydict import EasyDict

@BACKBONES.register_module()
class S2SVR(nn.Module):
    def __init__(self, dim=48, num_blocks=13, num_layers=3, is_low_res_input=False, pwclite_pretrained=None, cpu_cache_length=25):
        super(S2SVR, self).__init__()
        self.cpu_cache_length = cpu_cache_length
        self.encoder = Encoder(input_dim=3, hidden_dim=dim, num_blocks=num_blocks, num_layers=num_layers)
        self.decoder = Decoder(hidden_dim=dim, num_blocks=num_blocks, num_layers=num_layers, is_low_res_input=is_low_res_input)

        # optical flow
        cfg = {
            'model': {
                'upsample': True,
                'n_frames': 2,
                'reduce_dense': True,
            },
            'pretrained_model': pwclite_pretrained,
            'test_shape': (256, 256),
        }
        self.flow_cfg = EasyDict(cfg)
        self.pwclite = PWCLite(self.flow_cfg.model)
        if self.flow_cfg.pretrained_model is not None:
            self.pwclite = restore_model(self.pwclite, self.flow_cfg.pretrained_model)

    def compute_flow(self, lqs, flows):

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :]
        lqs_2 = lqs[:, 1:, :, :, :]
        flows['forward'], flows['backward'] = [], []
        for i in range(t-1):
            lq_1, lq_2 = lqs_1[:, i, :, :, :], lqs_2[:, i, :, :, :]
            if self.flow_cfg.test_shape != (h, w):
                lq_1 = torch.nn.functional.interpolate(lq_1, self.flow_cfg.test_shape, mode='bilinear',align_corners=True)
                lq_2 = torch.nn.functional.interpolate(lq_2, self.flow_cfg.test_shape, mode='bilinear',align_corners=True)
            img_pair = torch.cat([lq_2, lq_1], 1)
            res = self.pwclite(img_pair)
            if self.flow_cfg.test_shape != (h, w):
                flow_forward = resize_flow(res['flows_fw'][0], (h, w))
                flow_backward = resize_flow(res['flows_bw'][0], (h, w))
            else:
                flow_forward = res['flows_fw'][0]
                flow_backward = res['flows_bw'][0]
            if self.cpu_cache:
                flow_forward = flow_forward.cpu()
                flow_backward = flow_backward.cpu()
                torch.cuda.empty_cache()
            flows['forward'].append(flow_forward)
            flows['backward'].append(flow_backward)
        return flows

    def forward(self, x):
        '''
        :param x: [n,t,c,h,w]
        :return: out: [n,t,c,h*scale,w*scale]
        '''
        n,t,c,h,w = x.shape

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and x.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        # compute optical flow
        flows = {}
        flows = self.compute_flow(x, flows)

        # encoder
        encoder_out, encoder_hidden_state = self.encoder(x, flows, self.cpu_cache)

        # decoder
        out = self.decoder(x, encoder_out, encoder_hidden_state, flows, self.cpu_cache)
        return out

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

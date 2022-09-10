import torch
import torch.nn as nn
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.backbones.sr_backbones.S2SVR_util import Encoder, Decoder
from mmedit.models.backbones.sr_backbones.pwclite import PWCLite, resize_flow, restore_model
from mmcv.runner import load_checkpoint
from easydict import EasyDict

@BACKBONES.register_module()
class S2SVR(nn.Module):
    def __init__(self, dim=48, num_blocks=13, num_layers=3, is_low_res_input=False, pwclite_pretrained=None, max_infer_length=25):
        super(S2SVR, self).__init__()
        self.encoder = Encoder(input_dim=3, hidden_dim=dim, num_blocks=num_blocks, num_layers=num_layers)
        self.decoder = Decoder(hidden_dim=dim, num_blocks=num_blocks, num_layers=num_layers, is_low_res_input=is_low_res_input)
        self.max_infer_length = max_infer_length
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

    def compute_flow(self, lqs):

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :]
        lqs_2 = lqs[:, 1:, :, :, :]
        flows_forward = []
        flows_backward = []
        for i in range(t-1):
            lq_1, lq_2 = lqs_1[:, i, :, :, :], lqs_2[:, i, :, :, :]
            if self.flow_cfg.test_shape != (h, w):
                lq_1 = torch.nn.functional.interpolate(lq_1, self.flow_cfg.test_shape, mode='bilinear',align_corners=True)
                lq_2 = torch.nn.functional.interpolate(lq_2, self.flow_cfg.test_shape, mode='bilinear',align_corners=True)
            img_pair = torch.cat([lq_1, lq_2], 1)
            res = self.pwclite(img_pair)
            if self.flow_cfg.test_shape != (h, w):
                flows_forward.append(resize_flow(res['flows_fw'][0], (h, w)))
                flows_backward.append(resize_flow(res['flows_bw'][0], (h, w)))
            else:
                flows_forward.append(res['flows_fw'][0], (h, w))
                flows_backward.append(res['flows_bw'][0], (h, w))
        return torch.stack(flows_forward, dim=1), torch.stack(flows_backward, dim=1)

    def forward(self, x):
        b,n,c,h,w = x.shape
        if n > self.max_infer_length:
            out = []
            for i_start in range(0,n,self.max_infer_length):
                i_end = min(n, i_start+self.max_infer_length)
                x_clip = x[:,i_start:i_end]
                flows_clip = self.compute_flow(x_clip)
                encoder_out_clip, encoder_hidden_state_clip = self.encoder(x_clip, flows_clip)
                out_clip = self.decoder(x_clip, encoder_out_clip, encoder_hidden_state_clip, flows_clip)
                out.append(out_clip)
                del flows_clip, encoder_out_clip, encoder_hidden_state_clip, out_clip
            out = torch.cat(out, dim=1)
        else:
            flows = self.compute_flow(x)
            encoder_out, encoder_hidden_state = self.encoder(x, flows)
            out = self.decoder(x, encoder_out, encoder_hidden_state, flows)
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

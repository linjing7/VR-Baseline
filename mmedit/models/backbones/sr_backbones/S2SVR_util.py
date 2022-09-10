import torch
import torch.nn as nn
import torch.nn.functional as F
from mmedit.models.common import (ResidualBlockNoBN, flow_warp, make_layer)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h

class ResConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128, num_blocks=15, is_backward=False):
        super(ResConvGRU, self).__init__()
        self.is_backward = is_backward
        self.conv_gru = ConvGRU(hidden_dim=hidden_dim,input_dim=input_dim)
        self.resblocks = make_layer(ResidualBlockNoBN, num_blocks, mid_channels=hidden_dim)
        if self.is_backward:
            self.fusion = nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, hidden_state, input, hidden_state_f=None):
        hidden_state = self.conv_gru(hidden_state, input)
        if self.is_backward and hidden_state_f is not None:
            hidden_state = self.lrelu(self.fusion(torch.cat([hidden_state, hidden_state_f], dim=1)))
        hidden_state = self.resblocks(hidden_state)
        return hidden_state

class Encoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_blocks=15, num_layers=3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.convgru_layers = nn.ModuleList()

        # 1st GRU layer, bi-directional
        self.encoder_layer1_f = ResConvGRU(hidden_dim=hidden_dim, input_dim=input_dim, num_blocks=num_blocks)
        self.encoder_layer1_b = ResConvGRU(hidden_dim=hidden_dim, input_dim=input_dim, num_blocks=num_blocks, is_backward=True)
        self.fution = nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)

        # ResConvGru layers, uni-direction
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers-1):
            self.encoder_layers.append(ResConvGRU(hidden_dim=hidden_dim,input_dim=hidden_dim, num_blocks=num_blocks))

        # To out
        self.conv_1 = nn.Conv2d(hidden_dim, hidden_dim * 4, 3, 1, 1, bias=True)
        self.conv_2 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, 1, 1, bias=True)
        self.conv_out = nn.Conv2d(hidden_dim * 2, 48, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self,x, flows):
        B, N, C, H, W = x.shape
        flows_forward, flows_backward = flows
        encoder_hidden_state = []

        # 1st ResConvGRU layer, bi-directional
        # forward
        hidden_states_f = []
        hidden_state = x.new_zeros(B, self.hidden_dim, H, W)
        for i in range(0, N):
            if i > 0:  # no warping required for the first timestep
                flow = flows_forward[:, i - 1, :, :, :]
                hidden_state = flow_warp(hidden_state, flow.permute(0, 2, 3, 1))
            hidden_state = self.encoder_layer1_f(hidden_state, x[:, i])
            hidden_states_f.append(hidden_state)
        encoder_hidden_state.append(hidden_state)
        # backward
        hidden_states_b = []
        hidden_state = x.new_zeros(B, self.hidden_dim, H, W)
        for i in range(N-1, -1, -1):
            if i < N-1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                hidden_state = flow_warp(hidden_state, flow.permute(0, 2, 3, 1))
            hidden_state = self.encoder_layer1_b(hidden_state, x[:, i], hidden_states_f[i])
            hidden_states_b.append(hidden_state)
        hidden_states_b = hidden_states_b[::-1]

        # 2nd-4th ResConvGRU layers, uni-direction
        inputs = torch.stack(hidden_states_b, 1)  # B,N,C,H,W
        for layer in range(len(self.encoder_layers)):
            hidden_states = []
            hidden_state = x.new_zeros(B, self.hidden_dim, H, W)
            for i in range(0, N):
                if i > 0:  # no warping required for the first timestep
                    flow = flows_forward[:, i - 1, :, :, :]
                    hidden_state = flow_warp(hidden_state, flow.permute(0, 2, 3, 1))
                hidden_state = self.encoder_layers[layer](hidden_state, inputs[:,i])
                hidden_states.append(hidden_state)
            hidden_states = torch.stack(hidden_states, 1)  # B,N,C,H,W
            inputs = inputs + hidden_states
            encoder_hidden_state.append(inputs[:,-1])
        encoder_out = inputs
        return encoder_out, encoder_hidden_state

class Decoder(nn.Module):
    def __init__(self, hidden_dim=64, num_blocks=15, num_layers=3, is_low_res_input=False):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.is_low_res_input = is_low_res_input

        # ResConvGRU layers, uni-direction
        self.convgru_layers = nn.ModuleList()
        self.recon_layers = nn.ModuleList()
        self.fution_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.convgru_layers.append(ConvGRU(hidden_dim=hidden_dim,input_dim=hidden_dim))
            self.recon_layers.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=hidden_dim))
        for _ in range(num_layers-1):
            self.fution_layers.append(nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True))
        self.attention = Attention(hidden_dim, hidden_dim)

        # Convert hidden state to output
        if self.is_low_res_input:
            self.conv_out_1 = nn.Conv2d(hidden_dim, hidden_dim * 4, 3, 1, 1, bias=True)
            self.conv_out_2 = nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, 1, 1, bias=True)
            self.conv_out_3 = nn.Conv2d(hidden_dim * 2, 48, 3, 1, 1, bias=True)
            self.PixelShuffle = nn.PixelShuffle(2)
        else:
            self.conv_out_1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
            self.conv_out_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
            self.conv_out_3 = nn.Conv2d(hidden_dim, 3, 3, 1, 1, bias=True)

        # Convert t-th output to (t+1)-th input
        if self.is_low_res_input:
            self.conv_inp_1 = nn.Conv2d(3, self.hidden_dim//2, 3, 1, 1, bias=True)
            self.conv_inp_2 = nn.Conv2d(self.hidden_dim//2, self.hidden_dim, 5, 2, 2, bias=True)
            self.conv_inp_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim*2, 5, 2, 2, bias=True)
            self.conv_inp_4 = nn.Conv2d(self.hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)
        else:
            self.conv_inp_1 = nn.Conv2d(3, self.hidden_dim // 2, 3, 1, 1, bias=True)
            self.conv_inp_2 = nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, 3, 1, 1, bias=True)
            self.conv_inp_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, 3, 1, 1, bias=True)
            self.conv_inp_4 = nn.Conv2d(self.hidden_dim * 2, hidden_dim, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def hidden2out(self, x_fea, base):
        out = self.lrelu(self.conv_out_1(x_fea))
        out = self.lrelu(self.conv_out_2(out))
        out = self.conv_out_3(out)
        if self.is_low_res_input:
            out = self.PixelShuffle(self.PixelShuffle(out))
        return out + base

    def out2inp(self, x):
        out = self.lrelu(self.conv_inp_1(x))
        out = self.lrelu(self.conv_inp_2(out))
        out = self.lrelu(self.conv_inp_3(out))
        out = self.lrelu(self.conv_inp_4(out))
        return out

    def forward(self, x, encoder_out, encoder_hidden_state, flows):
        '''
        encoder_out: the output of the top layer [B,N,C,H,W]
        encoder_hidden_state: the hidden state of every layers Lx[B,C,H,W]
        flows: flows_forward, flows_backward
        '''
        B, N, C, H, W = x.shape
        flows_forward, flows_backward = flows

        if self.is_low_res_input:
            base = F.interpolate(x.view(-1, C, H, W), scale_factor=4, mode='bilinear', align_corners=False).view(B,N,C,4*H,4*W)
        else:
            base = x

        # GRU layers, uni-directional
        # initial hidden state
        hidden_state = encoder_hidden_state
        # backward
        outputs = []
        input = encoder_out[:,-1]
        for i in range(N - 1, -1, -1):
            for layer in range(len(self.convgru_layers)):
                if i < N-1:  # no warping required for the last timestep
                    flow = flows_backward[:, i, :, :, :]
                    hidden_state[layer] = flow_warp(hidden_state[layer], flow.permute(0, 2, 3, 1))
                    if layer==0:
                        input = flow_warp(input, flow.permute(0, 2, 3, 1))
                if layer == 0:
                    # The context vectors is computed once and shared on all layers
                    hidden_state[layer] = self.convgru_layers[layer](hidden_state[layer], input)
                    input = self.recon_layers[layer](hidden_state[layer])
                    if i == N - 1:
                        flow_f = flows_forward[:, i - 1]
                        encoder_outputs_l = flow_warp(encoder_out[:, i - 1], flow_f.permute(0, 2, 3, 1))
                        encoder_outputs_m = encoder_outputs_r = encoder_out[:, i]
                    elif i == 0:
                        flow_b = flows_backward[:, i]
                        encoder_outputs_r = flow_warp(encoder_out[:, i + 1], flow_b.permute(0, 2, 3, 1))
                        encoder_outputs_m = encoder_outputs_l = encoder_out[:, i]
                    else:
                        flow_f = flows_forward[:, i - 1]
                        flow_b = flows_backward[:, i]
                        encoder_outputs_r = flow_warp(encoder_out[:, i + 1], flow_b.permute(0, 2, 3, 1))
                        encoder_outputs_l = flow_warp(encoder_out[:, i - 1], flow_f.permute(0, 2, 3, 1))
                        encoder_outputs_m = encoder_out[:, i]
                    context = self.attention(input,torch.stack([encoder_outputs_l, encoder_outputs_m, encoder_outputs_r],dim=1))
                else:
                    residual = input
                    input = self.lrelu(self.fution_layers[layer-1](torch.cat((input, context), dim=1)))
                    hidden_state[layer] = self.convgru_layers[layer](hidden_state[layer], input)
                    input = self.recon_layers[layer](hidden_state[layer]) + residual
            output = self.hidden2out(input, base[:,i])
            input = self.out2inp(output)
            outputs.append(output)
        outputs = outputs[::-1]
        outputs = torch.stack(outputs, dim=1)
        return outputs

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):

        super().__init__()
        self.attn = nn.Conv2d(enc_hid_dim + dec_hid_dim, dec_hid_dim, 3, 1, 1, bias=False)
        self.v = nn.Conv2d(dec_hid_dim, dec_hid_dim, 3, 1, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size,N,enc_hid_dim,H,W = encoder_outputs.shape
        dec_hid_dim = hidden.shape[1]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1, 1, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2).flatten(0,1)))
        attention = self.v(energy)
        attention = F.softmax(attention.view(batch_size,N,dec_hid_dim,H,W), dim=1)
        context = torch.sum(attention*encoder_outputs, dim=1)
        return context


import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import SpatialGradient
from torch import Tensor
import torch.fft
import math

##########################################################################
class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        GELU()
    )

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

##########################################################################
def downsample_conv(in_channels, out_channels) :
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        GELU()
    )

##########################################################################
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, is_last=False):
        super(ConvLayer, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)

    def forward(self, x):
        out = self.conv2d(x)
        return out

##########################################################################
class EdgeDetect(nn.Module):
    def __init__(self):
        super(EdgeDetect, self).__init__()
        self.spatial = SpatialGradient('diff')
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        y = self.max_pool(u)
        return y

##########################################################################
class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.ed = EdgeDetect()

    def forward(self, ir, rgb):

        ir_attention = ((ir * 127.5) + 127.5) / 255
        rgb_attention = ((rgb * 127.5) + 127.5) / 255
        ir_edgemap = self.ed(ir_attention)
        rgb_edgemap = self.ed(rgb_attention)
        edgemap_ir = ir_edgemap / (ir_edgemap + rgb_edgemap + 0.00001)
        edgemap_ir = (edgemap_ir - 0.5) * 2

        edgemap_rgb = rgb_edgemap / (ir_edgemap + rgb_edgemap + 0.00001)
        edgemap_rgb = (edgemap_rgb - 0.5) * 2

        return edgemap_ir, edgemap_rgb

##########################################################################
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer_vis = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=3, stride=1, padding=1, groups=self.groups, bias=False)
        self.conv_layer_ir = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=3, stride=1, padding=1, groups=self.groups, bias=False)

        self.conv = ConvLayer(in_channels * 4, out_channels * 2, kernel_size=1, stride=1, padding=0)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

        self.Prelu1 = nn.PReLU()
        self.Prelu2 = nn.PReLU()

    def forward(self, vis, ir):
        batch = vis.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = vis.shape[-2:]
            vis = F.interpolate(vis, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = vis.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted_vis = torch.fft.rfftn(vis, dim=fft_dim, norm=self.fft_norm)
        ffted_vis = torch.stack((ffted_vis.real, ffted_vis.imag), dim=-1)
        ffted_vis = ffted_vis.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted_vis = ffted_vis.view((batch, -1,) + ffted_vis.size()[3:])

        ffted_ir = torch.fft.rfftn(ir, dim=fft_dim, norm=self.fft_norm)
        ffted_ir = torch.stack((ffted_ir.real, ffted_ir.imag), dim=-1)
        ffted_ir = ffted_ir.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted_ir = ffted_ir.view((batch, -1,) + ffted_ir.size()[3:])

        ffted_vis = self.conv_layer_vis(ffted_vis)  # (batch, c*2, h, w/2+1)
        ffted_ir = self.conv_layer_ir(ffted_ir)  # (batch, c*2, h, w/2+1)
        ffted_ir = self.Prelu1(ffted_vis)
        ffted_ir = self.Prelu2(ffted_ir)

        ffted = torch.cat([ffted_vis, ffted_ir], dim= 1)
        ffted = self.conv(ffted)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = vis.shape[-3:] if self.ffc3d else vis.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

##########################################################################
class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.fu = FourierUnit(
            in_channels, out_channels, groups, **fu_kwargs)

    def forward(self, vis, ir):

        output = self.fu(vis, ir)

        return output


############################################################################
class Reconstruction(nn.Module):
    def __init__(self, in_ch, out_ch, expansion):
        super(Reconstruction, self).__init__()
        exp_ch = int(in_ch * expansion)
        self.se_conv = nn.Conv2d(in_ch, exp_ch, 3, stride=1, padding=1, groups=in_ch)
        self.se_bn = nn.BatchNorm2d(exp_ch)
        self.hd_conv = nn.Conv2d(exp_ch, exp_ch, 3, stride=1, padding=1, groups=in_ch)
        self.hd_bn = nn.BatchNorm2d(exp_ch)
        self.gelu = GELU()
        self.cp_conv = nn.Conv2d(exp_ch, in_ch, 1, stride=1, padding=0, groups=in_ch)
        self.cp_bn = nn.BatchNorm2d(in_ch)
        self.pw_conv = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_sig = nn.Sigmoid()

        self.conv8 = ConvLayer(64, 1, kernel_size=1, stride=1, padding=0)
        self.fused = ConvLayer(64, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, enh_vis, enh_ir, vis, ir):

        enh_result = torch.cat([enh_vis, enh_ir], 1)
        enh_result = self.fused(enh_result)
        x = self.se_conv(enh_result)
        x = self.gelu(x)
        x = self.hd_conv(x)
        x = self.gelu(x)
        x = self.cp_conv(x)
        x = self.pw_conv(x)
        final_result = torch.tanh(x)

        return final_result

##########################################################################
class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Residual Feature Distillation Block
class RFDB(nn.Module):
    def __init__(self, distillation_rate=0.25):
        super(RFDB, self).__init__()
        in_channels = 32
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = nn.Conv2d(in_channels, self.dc, kernel_size=1, padding=0)
        self.c1_r = nn.Conv2d(in_channels, self.rc, kernel_size=3, padding=1)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=1, padding=0)
        self.c2_r = nn.Conv2d(self.remaining_channels, self.rc, kernel_size=3, padding=1)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=1, padding=0)
        self.c3_r = nn.Conv2d(self.remaining_channels, self.rc, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=3, padding=1)
        self.act = GELU()
        self.c5 = nn.Conv2d(self.dc*4, in_channels, kernel_size=1, padding=0)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


##########################################################################
class CMT(nn.Module):
    def __init__(self):
        super(CMT, self).__init__()

        self.channel_conv_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.channel_conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.spatial_conv_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.spatial_conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv11 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, query, key):

        ######################################## channel
        chn_key = self.channel_conv_1(key)  # H * W * 1
        chn_query = self.channel_conv_2(query)  # H * W * 32

        B, C, H, W = chn_query.size()

        chn_query_unfold = chn_query.view(B, C, H * W)  # HW * 32
        chn_key_unfold = chn_key.view(B, 1, H * W)  # HW * 1

        chn_key_unfold = chn_key_unfold.permute(0, 2, 1)

        chn_query_relevance = torch.bmm(chn_query_unfold, chn_key_unfold)
        chn_query_relevance_ = torch.sigmoid(chn_query_relevance) #softmax?
        chn_query_relevance_ = 1 - chn_query_relevance_ #irrelevance map(channel)
        inv_chn_query_relevance_ = chn_query_relevance_.unsqueeze(2)
        chn_value_final = inv_chn_query_relevance_ * query

        ######################################## spatial
        spa_key = self.spatial_conv_1(key)  # H * W * 32
        spa_query = self.spatial_conv_2(query)  # H * W *32

        B, C, H, W = spa_query.size()

        spa_query_unfold = spa_query.view(B, H * W, C)  # HW * 32
        spa_key_unfold = spa_key.view(B, H * W, C)  # HW * 32

        spa_key_unfold = torch.mean(spa_key_unfold, dim=1)
        spa_key_unfold = spa_key_unfold.unsqueeze(2)

        spa_query_relevance = torch.bmm(spa_query_unfold, spa_key_unfold)
        spa_query_relevance = torch.sigmoid(spa_query_relevance) #softmax?

        inv_spa_query_relevance = 1 - spa_query_relevance #irrelevance map(spatial)
        inv_spa_query_relevance_ = inv_spa_query_relevance.permute(0, 2, 1)
        inv_spa_query_relevance_ = inv_spa_query_relevance_.view(B, 1, H, W)
        spa_value_final = inv_spa_query_relevance_ * query

        key_relevance = torch.cat([chn_value_final, spa_value_final], dim =1)
        key_relevance = self.conv11(key_relevance)

        return key_relevance

##########################################################################
class CMT_transformers(nn.Module):
    def __init__(self):
        super(CMT_transformers, self).__init__()

        self.bot_conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bot_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fusionTransformer_vis = CMT()
        self.fusionTransformer_ir = CMT()

    def forward(self, rgb, ir):

        #gated_bottleneck
        bot_feature = rgb + ir
        bot_rgb = self.bot_conv1(bot_feature)
        bot_rgb_ = torch.sigmoid(bot_rgb)

        bot_ir = self.bot_conv2(bot_feature)
        bot_ir_ = torch.sigmoid(bot_ir)

        #normalization
        bot_rgb_ = bot_rgb_ / (bot_rgb_ + bot_ir_)
        bot_ir_ = bot_ir_ / (bot_ir_ + bot_rgb_)

        #transformer
        rgb_hat = self.fusionTransformer_vis(rgb, ir*bot_ir_)
        ir_hat = self.fusionTransformer_ir(ir, rgb*bot_rgb_)

        return rgb_hat, ir_hat


##########################################################################
class CMTFusion(nn.Module):
    def __init__(self):
        super(CMTFusion, self).__init__()
        nb_filter = [32, 32, 48, 64]
        kernel_size = 3
        stride = 1
        self.M = 16
        self.attention = attention()
        self.conv_ir1 = ConvLayer(2, nb_filter[0], kernel_size, stride)
        self.conv_rgb1 = ConvLayer(2, nb_filter[0], kernel_size, stride)
        self.conv_g_ir = ConvLayer(2, nb_filter[0], kernel_size, stride)
        self.conv_g_rgb = ConvLayer(2, nb_filter[0], kernel_size, stride)
        self.conv_pre1 = ConvLayer(1, nb_filter[0], kernel_size, stride)

        self.down1 = downsample_conv(32, 32)
        self.down2 = downsample_conv(32, 32)
        self.down3 = downsample_conv(32, 32)
        self.down4 = downsample_conv(32, 32)

        self.ir_encoder_level1 = RFDB()
        self.ir_encoder_level2 = RFDB()
        self.ir_encoder_level3 = RFDB()
        self.rgb_encoder_level1 = RFDB()
        self.rgb_encoder_level2 = RFDB()
        self.rgb_encoder_level3 = RFDB()

        self.up_eval1 = UpsampleReshape_eval(32, 32)
        self.up_eval2 = UpsampleReshape_eval(32, 32)
        self.up_eval3 = UpsampleReshape_eval(32, 32)
        self.up_eval4 = UpsampleReshape_eval(32, 32)
        self.up_eval5 = UpsampleReshape_eval(32, 32)
        self.up_eval6 = UpsampleReshape_eval(32, 32)
        self.up_eval7 = UpsampleReshape_eval(32, 32)
        self.up_eval8 = UpsampleReshape_eval(32, 32)

        self.conv1_1 = ConvLayer(2 * nb_filter[0], nb_filter[0], kernel_size=1, stride=1, padding=0)
        self.conv1_2 = ConvLayer(2 * nb_filter[0], nb_filter[0], kernel_size=1, stride=1, padding=0)
        self.conv1_3 = ConvLayer(2 * nb_filter[0], nb_filter[0], kernel_size=1, stride=1, padding=0)
        self.conv1_4 = ConvLayer(2 * nb_filter[0], nb_filter[0], kernel_size=1, stride=1, padding=0)
        self.conv2_1 = ConvLayer(2 * nb_filter[0], nb_filter[0], kernel_size=1, stride=1, padding=0)
        self.conv2_2 = ConvLayer(2 * nb_filter[0], nb_filter[0], kernel_size=1, stride=1, padding=0)
        self.conv2_3 = ConvLayer(2 * nb_filter[0], nb_filter[0], kernel_size=1, stride=1, padding=0)
        self.conv2_4 = ConvLayer(2 * nb_filter[0], nb_filter[0], kernel_size=1, stride=1, padding=0)

        self.Stage1_1 = CMT_transformers()
        self.Stage2_1 = CMT_transformers()
        self.Stage3_1 = CMT_transformers()

        self.conv11_1 = ConvLayer(32, 32, 1, stride, 0)
        self.conv11_2 = ConvLayer(32, 32, 1, stride, 0)
        self.conv11_3 = ConvLayer(32, 32, 1, stride, 0)
        self.conv11_4 = ConvLayer(32, 32, 1, stride, 0)
        self.conv11_5 = ConvLayer(32, 32, 1, stride, 0)
        self.conv11_6 = ConvLayer(32, 32, 1, stride, 0)

        self.conv6 = Reconstruction(32, 1, 4)
        self.conv7 = Reconstruction(32, 1, 4)
        self.conv8 = Reconstruction(32, 1, 4)

        self.conv9 = ConvLayer(nb_filter[0], 1, 1, stride, 0)
        self.fft = SpectralTransform(32, 32)


    def forward(self, rgb, ir):

        edgemap_ir, edgemap_rgb = self.attention(ir, rgb)

        ir_input = torch.cat([ir, edgemap_ir], 1)
        rgb_input = torch.cat([rgb, edgemap_rgb], 1)

        ir_level1 = self.conv_ir1(ir_input)
        ir_level2 = self.down1(ir_level1)
        ir_level3 = self.down2(ir_level2)

        rgb_level1 = self.conv_rgb1(rgb_input)
        rgb_level2 = self.down3(rgb_level1)
        rgb_level3 = self.down4(rgb_level2)

        ir_level_3 = self.conv11_1(ir_level3)
        rgb_level_3 = self.conv11_2(rgb_level3)

        ir_level_2 = self.conv11_3(ir_level2)
        rgb_level_2 = self.conv11_4(rgb_level2)

        ir_level_1 = self.conv11_5(ir_level1)
        rgb_level_1 = self.conv11_6(rgb_level1)

        ir_level_3 = self.ir_encoder_level3(ir_level_3)
        rgb_level_3 = self.rgb_encoder_level3(rgb_level_3)

        ir_level_2 = self.ir_encoder_level2(ir_level_2)
        rgb_level_2 = self.rgb_encoder_level2(rgb_level_2)

        ir_level_1 = self.ir_encoder_level1(ir_level_1)
        rgb_level_1 = self.rgb_encoder_level1(rgb_level_1)
        ##################################################Level3

        rgb_level_3_1, ir_level_3_1 = self.Stage3_1(rgb_level_3, ir_level_3)
        output3 = self.conv6(rgb_level_3_1, ir_level_3_1, rgb_level_3, ir_level_3)

        ##################################################Level2

        rgb_up_3_1 = self.up_eval1(rgb_level_2, rgb_level_3_1)
        ir_up_3_1 = self.up_eval2(ir_level_2, ir_level_3_1)
        rgb_input_2 = torch.cat([rgb_level_2, rgb_up_3_1], 1)
        rgb_input_2 = self.conv1_1(rgb_input_2)
        ir_input_2 = torch.cat([ir_level_2, ir_up_3_1], 1)
        ir_input_2 = self.conv1_2(ir_input_2)

        rgb_level_2_1, ir_level_2_1 = self.Stage2_1(rgb_input_2, ir_input_2)
        output2 = self.conv7(rgb_level_2_1, ir_level_2_1, rgb_level_2, ir_level_2)

        ##################################################Level2

        rgb_up_2_1 = self.up_eval5(rgb_level_1, rgb_level_2_1)
        ir_up_2_1 = self.up_eval6(ir_level_1, ir_level_2_1)
        rgb_input_1 = torch.cat([rgb_level_1, rgb_up_2_1], 1)
        rgb_input_1 = self.conv2_1(rgb_input_1)
        ir_input_1 = torch.cat([ir_level_1, ir_up_2_1], 1)
        ir_input_1 = self.conv2_2(ir_input_1)

        rgb_level_1_1, ir_level_1_1 = self.Stage1_1(rgb_input_1, ir_input_1)
        fused_output = self.conv8(rgb_level_1_1, ir_level_1_1, rgb_level_1, ir_level_1)

        return fused_output, output2, output3

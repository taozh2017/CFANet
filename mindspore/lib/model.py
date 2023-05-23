import pdb

from mindspore import Tensor, ops, Parameter, nn, common
import mindspore as ms
from lib.res2net_v1b_base import Res2Net_model


class global_module(nn.Cell):
    def __init__(self, channels=64, r=4):
        super(global_module, self).__init__()
        out_channels = int(channels // r)
        # local_att

        self.global_att = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(out_channels, use_batch_statistics=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(channels, use_batch_statistics=True)
        )

        self.sig = nn.Sigmoid()

    def construct(self, x):
        xg = self.global_att(x)
        out = self.sig(xg)

        return out


class BasicConv2d(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, pad_mode='valid', dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              pad_mode=pad_mode, dilation=dilation, has_bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ChannelAttention(nn.Cell):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        # self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, pad_mode='valid', has_bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, pad_mode='valid', has_bias=False)

        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        _, x = ops.max(x, axis=2, keep_dims=True)
        _, x = ops.max(x, axis=3, keep_dims=True)
        # x = self.max_pool(x)
        max_out = self.fc2(self.relu1(self.fc1(x)))
        out = max_out
        return self.sigmoid(out)


class GateFusion(nn.Cell):
    def __init__(self, in_planes):
        self.init__ = super(GateFusion, self).__init__()

        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, pad_mode='valid', has_bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, pad_mode='valid', has_bias=True)

        self.softmax = nn.Softmax(axis=1)

    def construct(self, x1, x2):
        ###
        cat_fea = ops.concat((x1, x2), axis=1)

        ###
        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)

        att_vec_cat = ops.concat((att_vec_1, att_vec_2), axis=1)
        att_vec_soft = self.softmax(att_vec_cat)

        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2

        return x_fusion


class BAM(nn.Cell):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(BAM, self).__init__()

        self.relu = nn.ReLU()

        self.global_att = global_module(channel)

        self.conv_layer = BasicConv2d(channel * 2, channel, 3, pad_mode='same')

    def construct(self, x, x_boun_atten):
        out1 = self.conv_layer(ops.concat((x, x_boun_atten), axis=1))
        out2 = self.global_att(out1)
        out3 = out1 * out2

        out = x + out3

        return out


class CFF(nn.Cell):
    def __init__(self, in_channel1, in_channel2, out_channel):
        self.init__ = super(CFF, self).__init__()

        act_fn = nn.ReLU()

        ## ---------------------------------------- ##
        self.layer0 = BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv2d(in_channel2, out_channel // 2, 1)

        self.layer3_1 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer3_2 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer5_1 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer5_2 = nn.SequentialCell(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer_out = nn.SequentialCell(
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(out_channel), act_fn)

    def construct(self, x0, x1):
        ## ------------------------------------------------------------------ ##
        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)

        x_3_1 = self.layer3_1(ops.concat((x0_1, x1_1), axis=1))
        x_5_1 = self.layer5_1(ops.concat((x1_1, x0_1), axis=1))

        x_3_2 = self.layer3_2(ops.concat((x_3_1, x_5_1), axis=1))
        x_5_2 = self.layer5_2(ops.concat((x_5_1, x_3_1), axis=1))

        out = self.layer_out(x0_1 + x1_1 + x_3_2 * x_5_2)

        return out


###############################################################################
## 2022/01/03
###############################################################################
class CFANet(nn.Cell):
    # resnet based encoder decoder
    def __init__(self, channel=64, opt=None):
        super(CFANet, self).__init__()

        act_fn = nn.ReLU()

        self.resnet = Res2Net_model(50)
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)

        ## ---------------------------------------- ##

        self.layer0 = nn.SequentialCell(
            nn.Conv2d(64, channel, kernel_size=3, stride=2, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer1 = nn.SequentialCell(
            nn.Conv2d(256, channel, kernel_size=3, stride=2, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)

        self.low_fusion = GateFusion(channel)

        self.high_fusion1 = CFF(256, 512, channel)
        self.high_fusion2 = CFF(1024, 2048, channel)

        ## ---------------------------------------- ##
        self.layer_edge0 = nn.SequentialCell(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer_edge1 = nn.SequentialCell(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer_edge2 = nn.SequentialCell(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(64 ),
            act_fn)
        self.layer_edge3 = nn.Conv2d(64, 1, kernel_size=1, pad_mode='valid', has_bias=True)

        ## ---------------------------------------- ##
        self.layer_cat_ori1 = nn.SequentialCell(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer_hig01 = nn.SequentialCell(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)

        self.layer_cat11 = nn.SequentialCell(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer_hig11 = nn.SequentialCell(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)

        self.layer_cat21 = nn.SequentialCell(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer_hig21 = nn.SequentialCell(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(64 ),
            act_fn)

        self.layer_cat31 = nn.SequentialCell(
            nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(64 ),
            act_fn)
        self.layer_hig31 = nn.Conv2d(64, 1, kernel_size=1, pad_mode='valid', has_bias=True)

        self.layer_cat_ori2 = nn.SequentialCell(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer_hig02 = nn.SequentialCell(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)

        self.layer_cat12 = nn.SequentialCell(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer_hig12 = nn.SequentialCell(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)

        self.layer_cat22 = nn.SequentialCell(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(channel ),
            act_fn)
        self.layer_hig22 = nn.SequentialCell(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(64 ),
            act_fn)

        self.layer_cat32 = nn.SequentialCell(
            nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(64),
            act_fn)
        self.layer_hig32 = nn.Conv2d(64, 1, kernel_size=1, pad_mode='valid', has_bias=True)

        self.layer_fil = nn.Conv2d(64, 1, kernel_size=1, pad_mode='valid', has_bias=True)

        ## ---------------------------------------- ##

        self.atten_edge_0 = ChannelAttention(channel)
        self.atten_edge_1 = ChannelAttention(channel)
        self.atten_edge_2 = ChannelAttention(channel)
        self.atten_edge_ori = ChannelAttention(channel)

        self.cat_01 = BAM(channel)
        self.cat_11 = BAM(channel)
        self.cat_21 = BAM(channel)
        self.cat_31 = BAM(channel)

        self.cat_02 = BAM(channel)
        self.cat_12 = BAM(channel)
        self.cat_22 = BAM(channel)
        self.cat_32 = BAM(channel)

        ## ---------------------------------------- ##
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ResizeBilinear()

    def construct(self, xx):
        # ---- feature abstraction -----

        x0, x1, x2, x3, x4 = self.resnet(xx)

        ## -------------------------------------- ## 

        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)

        low_x = self.low_fusion(x0_1, x1_1)  # 64*44

        edge_out0 = self.layer_edge0(self.up(low_x, scale_factor=2, align_corners=True))  # 64*88
        edge_out1 = self.layer_edge1(self.up(edge_out0, scale_factor=2, align_corners=True))  # 64*176
        edge_out2 = self.layer_edge2(self.up(edge_out1, scale_factor=2, align_corners=True))  # 64*352
        edge_out3 = self.layer_edge3(edge_out2)

        etten_edge_ori = self.atten_edge_ori(low_x)
        etten_edge_0 = self.atten_edge_0(edge_out0)
        etten_edge_1 = self.atten_edge_1(edge_out1)
        etten_edge_2 = self.atten_edge_2(edge_out2)

        ## -------------------------------------- ##
        high_x01 = self.high_fusion1(self.downSample(x1), x2)
        high_x02 = self.high_fusion2(self.up(x3, scale_factor=2, align_corners=True),
                                     self.up(x4, scale_factor=4, align_corners=True))

        ## --------------- high 1 ----------------------- # 
        cat_out_01 = self.cat_01(high_x01, low_x * etten_edge_ori)
        hig_out01 = self.layer_hig01(self.up(cat_out_01, scale_factor=2, align_corners=True))

        cat_out11 = self.cat_11(hig_out01, edge_out0 * (etten_edge_0))
        hig_out11 = self.layer_hig11(self.up(cat_out11, scale_factor=2, align_corners=True))

        cat_out21 = self.cat_21(hig_out11, edge_out1 * (etten_edge_1))
        hig_out21 = self.layer_hig21(self.up(cat_out21, scale_factor=2, align_corners=True))

        cat_out31 = self.cat_31(hig_out21, edge_out2 * (etten_edge_2))
        sal_out1 = self.layer_hig31(cat_out31)

        ## ---------------- high 2 ---------------------- ##
        cat_out_02 = self.cat_02(high_x02, low_x * (etten_edge_ori))
        hig_out02 = self.layer_hig02(self.up(cat_out_02, scale_factor=2, align_corners=True))

        cat_out12 = self.cat_12(hig_out02, edge_out0 * (etten_edge_0))
        hig_out12 = self.layer_hig12(self.up(cat_out12, scale_factor=2, align_corners=True))

        cat_out22 = self.cat_22(hig_out12, edge_out1 * (etten_edge_1))
        hig_out22 = self.layer_hig22(self.up(cat_out22, scale_factor=2, align_corners=True))

        cat_out32 = self.cat_32(hig_out22, edge_out2 * (etten_edge_2))
        sal_out2 = self.layer_hig32(cat_out32)

        ## --------------------------------------------- ##
        sal_out3 = self.layer_fil(cat_out31 + cat_out32)

        # ---- output ----
        return edge_out3, sal_out1, sal_out2, sal_out3

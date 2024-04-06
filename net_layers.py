import torch.nn as nn
import torch
from torch.nn import Mish, Dropout
from mmcv.cnn import build_norm_layer
from torch.nn import functional as F

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, out_channels):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            Dropout(0.2),
        )

        self.atten = Attention(out_channels, key_dim=8, num_heads=8, activation=nn.ReLU)

    def forward(self, x):
        y1_1 = self.avg_pool1(x)
        y1_2 = self.max_pool1(x)
        y2 = torch.cat([y1_1, y1_2], dim=1)
        y2_1 = self.fc1(y2)
        y3 = self.atten(y2_1)

        return y3

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels + i * out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                Dropout(0.2),
            ))

    def forward(self, x):
        for layer in self.layers:
            y = layer(x)
            x = torch.cat((x,y), dim=1)
        return x

class Guodu(nn.Module):
    def __init__(self, out_channels):
        super(Guodu, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            return x

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Mish(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.Mish(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, bias=False, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0], bias=bias
            ),
            Mish(),
            nn.BatchNorm2d(out_dims),
            Dropout(0.2),
        )

        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1], bias=bias
            ),
            Mish(),
            nn.BatchNorm2d(out_dims),
            Dropout(0.2),
        )

        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2], bias=bias
            ),
            Mish(),
            nn.BatchNorm2d(out_dims),
            Dropout(0.2),
        )

        self.output = nn.Conv2d(len(rate)* out_dims, out_dims, 1, bias=bias)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                with torch.no_grad():
                    m.weight[torch.rand_like(m.weight) < 0.05] = 0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SPPFCSPC(nn.Module):
    def __init__(self, c1, c2, bias=False, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, bias=bias)
        self.cv2 = nn.Conv2d(c1, c_, 1, bias=bias)
        self.cv3 = nn.Conv2d(c_, c_, 3, padding=1, bias=bias)
        self.cv4 = nn.Conv2d(c_, c_, 1, bias=bias)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = nn.Conv2d(4 * c_, c_, 1, bias=bias)
        self.cv6 = nn.Conv2d(c_, c_, 3, padding=1, bias=bias)
        self.cv7 = nn.Conv2d(2 * c_, c2, 1, bias=bias)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride, padding):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.upsample(x)

class DualScaleAttentionBlock(nn.Module):
    def __init__(self, input_channels1, input_channels2, output_dim1, output_dim2, output_dim3):
        super(DualScaleAttentionBlock, self).__init__()

        self.conv_encoder1 = nn.Sequential(
            nn.BatchNorm2d(input_channels1),
            nn.ReLU(),
            nn.Conv2d(input_channels1, output_dim1, 3, padding=1),
            nn.AvgPool2d(kernel_size=4, stride=4),
        )

        self.conv_encoder2 = nn.Sequential(
            nn.BatchNorm2d(input_channels1),
            nn.ReLU(),
            nn.Conv2d(input_channels1, output_dim1, 5, padding=2),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        self.conv_decoder1 = nn.Sequential(
            nn.BatchNorm2d(input_channels2),
            nn.ReLU(),
            nn.Conv2d(input_channels2, output_dim1, 3, padding=1),
        )

        self.conv_decoder2 = nn.Sequential(
            nn.BatchNorm2d(input_channels2),
            nn.ReLU(),
            nn.Conv2d(input_channels2, output_dim1, 5, padding=2),
        )

        self.conv_attn2 = nn.Sequential(
            nn.BatchNorm2d(output_dim2),
            nn.ReLU(),
            nn.Conv2d(output_dim2, output_dim3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        print(x1.shape, x2.shape)
        print(self.conv_encoder1(x1).shape, self.conv_decoder1(x2).shape)
        out1 = self.conv_encoder1(x1) + self.conv_decoder1(x2)
        out2 = self.conv_encoder2(x1) + self.conv_decoder2(x2)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv_attn2(out)
        return out * x2

class DualScaleAttentionNetwork(nn.Module):
    def __init__(self, input_channels1, input_channels2, output_dim1, output_dim2, output_dim3):
        super(DualScaleAttentionNetwork, self).__init__()
        self.attention_block = DualScaleAttentionBlock(input_channels1, input_channels2, output_dim1, output_dim2, output_dim3)

    def forward(self, x1, x2):
        attention_output = self.attention_block(x1, x2)
        return attention_output

class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bicubic", scale_factor=scale, align_corners=True)
        self.dropout = Dropout(0.2)

    def forward(self, x):
        x = self.upsample(x)
        return self.dropout(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)

        return x

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=2,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))

        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))

        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)

        xx = self.sigmoid(xx) * qkv
        return xx
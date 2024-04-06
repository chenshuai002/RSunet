import torch.nn as nn
import torch

from net_layers import (
    Guodu,
    DenseBlock,
    ResidualConv,
    Squeeze_Excite_Block,
    Attention,
    Upsample_,
    SPPFCSPC,
)

class DenseResUnetPlus(nn.Module):
    def __init__(self, channel, num_class):
        super(DenseResUnetPlus, self).__init__()

        self.input_layer1 = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )

        self.input_layer2 = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
        )

        self.input_layer3 = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=7, padding=3),
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.input_layer4 = nn.Sequential(
            nn.Conv2d(67, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(64)
        self.dense_conv1 = DenseBlock(64, 32, num_layers=2)
        self.guodu1 = Guodu(128)

        self.squeeze_excite2 = Squeeze_Excite_Block(128)
        self.dense_conv2 = DenseBlock(128, 64, num_layers=2)
        self.guodu2 = Guodu(256)

        self.SPPFCSPC_bridge = SPPFCSPC(128, 64)

        self.attn1 = Attention(dim=64, key_dim=4, num_heads=4, activation=nn.ReLU)
        self.upsample1 = Upsample_(4)
        self.up_residual_conv1 = ResidualConv(64, 32, 1, 1)

        self.attn2 = Attention(dim=32, key_dim=4, num_heads=4, activation=nn.ReLU)
        self.upsample2 = Upsample_(4)
        self.up_residual_conv2 = ResidualConv(32, 16, 1, 1)

        self.SPPFCSPC_out = SPPFCSPC(16, num_class, bias=True)

    def forward(self, x):
        x1_1 = self.input_layer1(x)
        x1_2 = self.input_layer2(x)
        x1_3 = self.input_layer3(x)
        x1_4 = self.input_skip(x)
        x1_5 = torch.cat([x1_1, x1_2, x1_3, x1_4, x], dim=1)
        x1 = self.input_layer4(x1_5)

        x2_1 = self.squeeze_excite1(x1)
        x2_2 = self.dense_conv1(x2_1)
        x2 = self.guodu1(x2_2)

        x3_1 = self.squeeze_excite2(x2)
        x3_2 = self.dense_conv2(x3_1)
        x3 = self.guodu2(x3_2)

        x5 = self.SPPFCSPC_bridge(x3)

        x6_1 = self.attn1(x5)
        x6_2 = self.upsample1(x6_1)
        x6_3 = nn.functional.interpolate(x6_2, size=(x2.shape[2], x2.shape[3]), mode='nearest')
        x6 = self.up_residual_conv1(x6_3)

        x7_1 = self.attn2(x6)
        x7_2 = self.upsample2(x7_1)
        x7_3 = nn.functional.interpolate(x7_2, size=(x.shape[2], x.shape[3]), mode='nearest')
        x7 = self.up_residual_conv2(x7_3)

        out = self.SPPFCSPC_out(x7)
        return out

# models = DenseResUnetPlus(channel=3, num_class=3)
# x = torch.randn(size=(1,3,54,30))
# for layer in models.modules():
#     x = layer(x)
#     print(layer.__class__.__name__,x.shape)
#     break
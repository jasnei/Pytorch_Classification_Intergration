import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        # out = hard_sigmoid(out)

        return out * x

class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        padding = (kernal_size - 1) // 2

        self.use_connect = stride == 1 and in_channels == out_channels

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        # MobileNetV2
        out = self.conv(x)
        out = self.depth_conv(out)

        # Squeeze and Excite
        if self.SE:
            out = self.squeeze_block(out)

        # point-wise conv
        out = self.point_conv(out)

        # connection
        if self.use_connect:
            return x + out
        else:
            return out

# 自己加的，以便多输出
class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernal_size,stride,padding):
        super(ConvolutionalLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernal_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):
        return self.conv(x)

class ConvolutionalSetLayer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ConvolutionalSetLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channel,out_channel,kernal_size=1,stride=1,padding=0),
            ConvolutionalLayer(out_channel,in_channel,kernal_size=3,stride=1,padding=1),
            ConvolutionalLayer(in_channel,out_channel,kernal_size=1,stride=1,padding=0),
            ConvolutionalLayer(out_channel,in_channel,kernal_size=3,stride=1,padding=1),
            ConvolutionalLayer(in_channel,out_channel,kernal_size=1,stride=1,padding=0)
        )
    def forward(self,x):
        return self.conv(x)

class DownSampleLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownSampleLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channels,out_channels,kernal_size=3,stride=2,padding=1)
        )
    def forward(self,x):
        return self.conv(x)

class NewNet(nn.Module):

    def __init__(self, arch: object, num_classes=1000) -> object:
        super(NewNet, self).__init__()
        self.in_channels = 3
        self.conv3_32     = self.__make_layer(32,  arch[0], kernel_size=3, padding=1, se=False, exp_size=None)
        self.conv3_64     = self.__make_layer(64,  arch[1], kernel_size=3, padding=1, se=True, exp_size=64)
        self.conv3_128    = self.__make_layer(128, arch[2], kernel_size=3, padding=1, se=True, exp_size=128)
        self.conv3_256a   = self.__make_layer(256, arch[3], kernel_size=3, padding=1, se=True, exp_size=256)
        self.conv3_256b   = self.__make_layer(256, arch[4], kernel_size=3, padding=1, se=True, exp_size=256)
        self.conv1x1_512  = self.__make_layer(512, 1, kernel_size=1, padding=0,      se=False, exp_size=512)
        self.conv1x1_1024 = self.__make_layer(1024, 1, kernel_size=1, padding=0,      se=False, exp_size=None)
        self.conv1x1_out  = self.__make_layer(num_classes, 1, kernel_size=1, padding=0, se=False, exp_size=None)
        
    def __make_layer(self, channels, num, kernel_size, padding, se, exp_size):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, kernel_size, stride=1, padding=padding, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            if se:
                layers.append(SqueezeBlock(exp_size))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_32(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_64(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256b(out)
        out = F.max_pool2d(out, 2)
        out = self.conv1x1_512(out)
        batch, channel, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[height, width])
        out = self.conv1x1_1024(out)
        batch, channels, height, width = out.size()
        out = self.conv1x1_out(out).view(batch, -1)
        return out


def NewNet_11(num_classes=1000):
    return NewNet([2, 2, 3, 3, 2], num_classes=num_classes)

class NewNetSE(nn.Module):
    def __init__(self, model_mode="LARGE", num_classes=1000, multiplier=1.0, dropout_rate=0.0):
        super(NewNetSE, self).__init__()
        self.num_classes = num_classes

        if model_mode == "LARGE":
            layers_1 = [
                [16, 16, 3, 1, "RE", False, 16],
                [16, 24, 3, 2, "RE", False, 64],
                [24, 24, 3, 1, "RE", True, 72],                           
            ]
            
            layers_2 = [
                [24, 40, 5, 2, "RE", True, 72],
                [40, 40, 5, 1, "RE", True, 120],
                [40, 40, 5, 1, "RE", True, 120],
            ]

            layers_3 = [
                [40, 80, 3, 2, "HS", False, 240],
                [80, 80, 3, 1, "HS", False, 200],
                [80, 80, 3, 1, "HS", False, 184],
                [80, 80, 3, 1, "HS", False, 184],
                [80, 112, 3, 1, "HS", True, 480],
                [112, 112, 3, 1, "HS", True, 672],
            ]

            layers_4 = [
                [112, 160, 5, 2, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 960],
            ]

            layers_5 = [
                [160, 220, 3, 2, "HS", False, 960],
                [220, 220, 3, 1, "HS", False, 960],
                [220, 220, 3, 1, "HS", False, 1280],
            ]
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(init_conv_out),
                h_swish(inplace=True),
            )
            
            self.block_1 = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers_1:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block_1.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block_1 = nn.Sequential(*self.block_1)

            # [-1, 24, 56, 56]
            # out 1
            # Convolution set
            self.conv2d_set_1 = nn.Sequential(
                ConvolutionalSetLayer(24, 24)
                )
                
            out1_conv2_in = _make_divisible(24 * multiplier)
            out1_conv2_out = _make_divisible(24 * multiplier)
            self.out1_conv2 = nn.Sequential(
                nn.Conv2d(out1_conv2_in, out1_conv2_out, kernel_size=1, stride=1),
                h_swish(inplace=True),
                nn.Conv2d(out1_conv2_out, self.num_classes, kernel_size=1, stride=1)
            )

            self.block_2 = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers_2:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block_2.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block_2 = nn.Sequential(*self.block_2)

            # Convolution set 2
            self.conv2d_set_2 = nn.Sequential(
                ConvolutionalSetLayer(40, 40)
                )

            # out 2
            out2_conv2_in = _make_divisible(40 * multiplier)
            out2_conv2_out = _make_divisible(80 * multiplier)
            self.out2_conv2 = nn.Sequential(
                nn.Conv2d(out2_conv2_in, out2_conv2_out, kernel_size=1, stride=1),
                h_swish(inplace=True),
                nn.Conv2d(out2_conv2_out, self.num_classes, kernel_size=1, stride=1)
            )

            self.block_3 = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers_3:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block_3.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block_3 = nn.Sequential(*self.block_3)

            # Convolution set 3
            self.conv2d_set_3 = nn.Sequential(
                ConvolutionalSetLayer(112, 112)
                )

            # out 3
            out3_conv2_in = _make_divisible(112 * multiplier)
            out3_conv2_out = _make_divisible(224 * multiplier)
            self.out3_conv2 = nn.Sequential(
                nn.Conv2d(out3_conv2_in, out3_conv2_out, kernel_size=1, stride=1),
                h_swish(inplace=True),
                nn.Conv2d(out3_conv2_out, self.num_classes, kernel_size=1, stride=1)
            )

            self.block_4 = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers_4:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block_4.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block_4 = nn.Sequential(*self.block_4)

            self.block_5 = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers_5:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block_5.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block_5 = nn.Sequential(*self.block_5)

            # out 4
            out4_conv1_in = _make_divisible(160 * multiplier)
            out4_conv1_out = _make_divisible(960 * multiplier)
            self.out4_conv1 = nn.Sequential(
                nn.Conv2d(out4_conv1_in, out4_conv1_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(out4_conv1_out),
                h_swish(inplace=True),
            )

            # out
            out4_conv2_in = _make_divisible(960 * multiplier)
            out4_conv2_out = _make_divisible(1280 * multiplier)
            self.out4_conv2 = nn.Sequential(
                nn.Conv2d(out4_conv2_in, out4_conv2_out, kernel_size=1, stride=1),
                h_swish(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out4_conv2_out, self.num_classes, kernel_size=1, stride=1),
            )

            self.conv2d_BN1 = nn.Sequential(
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(24)
                )

            # self.identity1 = Identity(24, 24, stride=1)

            self.conv2d_BN2 = nn.Sequential(
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(48)
                )
            # self.identity2 = Identity(24, 48, stride=2, se=True)
            # out 4
            # self.conv2d
            self.newnet = NewNet_11(self.num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.init_conv(x) # [-1, 16, 112, 112]

        out = self.block_1(out) # [-1, 24, 56, 56]
        # print('1', out.size())

        # out1 for big image
        out1 = self.conv2d_set_1(out)
        # print('2', out1.size())
        batch1, channels1, height1, width1 = out1.size()
        out1 = F.avg_pool2d(out1, kernel_size=[height1, width1]) # [2, 96, 1, 1]
        out1 = self.out1_conv2(out1).view(batch1, -1)            # [2, 4] 

        # print('3', out.size())                
        out = self.block_2(out)                                  # [2, 40, 28, 28]

        # print('4', out.size())    
        out2 = self.conv2d_set_2(out)                            # [-1, 80, 14, 14]
        # print('5', out2.size())    
        batch2, channels2, height2, width2 = out2.size()
        out2 = F.avg_pool2d(out2, kernel_size=[height2, width2])
        # print('6', out2.size()) 
        out2 = self.out2_conv2(out2).view(batch2, -1)

        out = self.block_3(out) # [-1, 112, 14, 14]
        # print('7', out.size())
        out3 = self.conv2d_set_3(out)
        batch3, channels3, height3, width3 = out3.size()
        out3 = F.avg_pool2d(out3, kernel_size=[height3, width3])
        # print('8', out3.size())
        out3 = self.out3_conv2(out3).view(batch3, -1)

        out = self.block_4(out) # [-1, 160, 7, 7]
        # print('9', out.size())
        out4 = self.out4_conv1(out)
        # print('10', out4.size())
        batch4, channels4, height4, width4 = out4.size()
        out4 = F.avg_pool2d(out4, kernel_size=[height4, width4])
        out4 = self.out4_conv2(out4).view(batch4, -1)

        out5 = self.newnet(x)

        return out1, out2, out3, out4, out5


if __name__ == '__main__':
    temp = torch.zeros((1, 3, 299, 299))
    model = NewNetSE(model_mode="LARGE", num_classes=4, multiplier=1.0)
    # print(model)
    
    # summary(model.cuda(), (3, 224, 224))
    summary(model, (3, 224, 224), device='cpu')
    # print(model(temp).shape)
    # print(get_model_parameters(model))
    model.eval()
    with torch.no_grad():
        output1, output2, output3, output4, output5 = model(temp)
        print(output1, output2, output3, output4, output5)
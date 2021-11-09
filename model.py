import torch
import torch.nn as nn
from torchstat import stat
import torch.nn.functional as F


class B2_VGG_A(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self):
        super(B2_VGG_A, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))

        self.conv1 = conv1
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4_1 = nn.Sequential()
        conv4_1.add_module('pool3_1', nn.AvgPool2d(2, stride=2))
        conv4_1.add_module('conv4_1_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4_1.add_module('relu4_1_1', nn.ReLU())
        conv4_1.add_module('conv4_2_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_1.add_module('relu4_2_1', nn.ReLU())
        conv4_1.add_module('conv4_3_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_1.add_module('relu4_3_1', nn.ReLU())
        self.conv4_1 = conv4_1

        conv5_1 = nn.Sequential()
        conv5_1.add_module('pool4_1', nn.AvgPool2d(2, stride=2))
        conv5_1.add_module('conv5_1_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_1_1', nn.ReLU())
        conv5_1.add_module('conv5_2_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_2_1', nn.ReLU())
        conv5_1.add_module('conv5_3_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_3_1', nn.ReLU())
        self.conv5_1 = conv5_1

        pre_train = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth', progress=True)
        self._initialize_weights(pre_train)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4_1(x)
        x1 = self.conv5_1(x1)
        return x1

    def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())
        self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])
        self.conv4_1.conv4_1_1.weight.data.copy_(pre_train[keys[14]])
        self.conv4_1.conv4_2_1.weight.data.copy_(pre_train[keys[16]])
        self.conv4_1.conv4_3_1.weight.data.copy_(pre_train[keys[18]])
        self.conv5_1.conv5_1_1.weight.data.copy_(pre_train[keys[20]])
        self.conv5_1.conv5_2_1.weight.data.copy_(pre_train[keys[22]])
        self.conv5_1.conv5_3_1.weight.data.copy_(pre_train[keys[24]])

        self.conv1.conv1_1.bias.data.copy_(pre_train[keys[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])
        self.conv4_1.conv4_1_1.bias.data.copy_(pre_train[keys[15]])
        self.conv4_1.conv4_2_1.bias.data.copy_(pre_train[keys[17]])
        self.conv4_1.conv4_3_1.bias.data.copy_(pre_train[keys[19]])
        self.conv5_1.conv5_1_1.bias.data.copy_(pre_train[keys[21]])
        self.conv5_1.conv5_2_1.bias.data.copy_(pre_train[keys[23]])
        self.conv5_1.conv5_3_1.bias.data.copy_(pre_train[keys[25]])
        
        
class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x1, x2, x3):
        # x1: 1/16 x2: 1/8 x3: 1/2
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.upsample(self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2))) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.upsample(self.conv_upsample5(self.upsample(x2_2)))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size-1)//2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels, self.in_channels//2, (1, self.kernel_size), padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels//2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels, self.in_channels//2, (self.kernel_size, 1), padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels//2, 1, (1, self.kernel_size), padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels//4)
        self.linear_2 = nn.Linear(self.in_channels//4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))
        
        # Activity regularizer
        # ca_act_reg = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats


class LGFA(nn.Module):
    def __init__(self, channel=32):
        super(LGFA, self).__init__()
        self.name = 'LGFA'
        self.vgg = B2_VGG_A()
        self.sa_low = SpatialAttention(in_channels=128)
        self.rfb2_1 = RFB(128,channel)
        
        # self.rfb3_1 = RFB(256, channel)
        
        
        self.rfb4_1 = RFB(512, channel)
        self.ca_high = nn.Sequential()
        self.ca_high.add_module('ca_1',ChannelwiseAttention(in_channels=512))
        
        
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        modules = [self.vgg, self.rfb4_1, self.rfb5_1, self.agg1, self.upsample,self.sa_low,self.ca_high]
        modules_names = ['vgg', 'rfb4_1', 'rfb5_1', 'agg1', 'upsample','sa_low','ca_high']
        print('Parameters')
        for module, name in zip(modules, modules_names):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print('{}\t{}'.format(name, params))

    def forward(self, x):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        # 浅层空间注意力机制
        x2_sa = self.sa_low(x2)
        x2_sa = torch.mul(x2_sa,x2)
        # rfb:金字塔融合(空洞卷积)
        x2_1 = self.rfb2_1(x2_sa)

        x3 = self.vgg.conv3(x2)
        # x3_1 = self.rfb3_1(x3)
        x4_1 = self.vgg.conv4_1(x3)
        x4_1_ca = self.ca_high(x4_1)
        x4_1_ca = torch.mul(x4_1_ca,x4_1)

        x5_1 = self.vgg.conv5_1(x4_1)
        x4_1 = self.rfb4_1(x4_1_ca)

        x5_1 = self.rfb5_1(x5_1)
        attention = self.agg1(x5_1, x4_1, x2_1)

        return self.upsample(attention)
    

def test():
    dummy_input = torch.randn(2, 3, 256, 256)

    model = LGFA()
    att, detection = model(dummy_input)

    print(model)
    print('\nModel input shape :', dummy_input.size())
    print('Model att shape :', att.size())
    print('Model detection shape :', detection.size())
    print(stat(model, (3, 256, 256)))


if __name__ == '__main__':
    test()
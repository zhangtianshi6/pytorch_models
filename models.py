import torch
import torch.nn as nn
import torch.nn.functional as F


# alexnet model
class AlexNet(nn.Module):
    def __init__(self, channels, class_num):
        super(AlexNet, self).__init__()
        self.in_channel = channels
        self.conv1 = nn.Conv2d(self.in_channel, 96, kernel_size=11, stride=4, padding=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.fc1 = nn.Linear(7*7*256, out_features=4096)
        self.fc2 = nn.Linear(256, 4096)
        self.fc3 = nn.Linear(4096, class_num)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.maxpool1(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool2(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = self.maxpool3(out)
        out = F.relu(F.avg_pool2d(out, out.size()[3]))
        out = out.view(-1, out.size()[1]*out.size()[2]*out.size()[3])
        #out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)


# vggnet model
class VGGNet16(nn.Module):
    def __init__(self, channels, class_num):
        super(VGGNet16, self).__init__()
        self.in_channel = channels
        ratio = 0.5
        self.conv1_1 = nn.Conv2d(self.in_channel, int(64*ratio), kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(int(64*ratio), int(64*ratio), kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(int(64*ratio), int(128*ratio), kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(int(128*ratio), int(128*ratio), kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(int(128*ratio), int(256*ratio), kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(int(256*ratio), int(256*ratio), kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(int(256*ratio), int(256*ratio), kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(int(256*ratio), int(512*ratio), kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(int(512*ratio), int(512*ratio), kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(int(512*ratio), int(512*ratio), kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(int(512*ratio), int(512*ratio), kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(int(512*ratio), int(512*ratio), kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(int(512*ratio), int(512*ratio), kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        #self.fc1 = nn.Linear(int(6*6*512*ratio), out_features=int(4096*ratio))
        self.fc2 = nn.Linear(int(512*ratio), int(4096*ratio))
        self.fc3 = nn.Linear(int(4096*ratio), class_num)
    
    def forward(self, x):
        out = F.relu(self.conv1_1(x))
        out = F.relu(self.conv1_2(out))
        out = self.maxpool1(out)
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.maxpool2(out)
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.maxpool3(out)
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        out = self.maxpool4(out)
        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.maxpool5(out)
        out = F.relu(F.avg_pool2d(out, out.size()[3]))
        out = out.view(-1, out.size()[1]*out.size()[2]*out.size()[3])
        #out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)


class Conv_bn_relu(nn.Module):
    def __init__(self, in_channel, num_filter, ksize, stride=1, padding=1):
        super(Conv_bn_relu, self).__init__()
        self.conv2d = nn.Conv2d(in_channel, num_filter, ksize, stride, padding)
        self.bn = nn.BatchNorm2d(num_filter)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Basic_block(nn.Module):
    
    def __init__(self, in_channel, num_filter, stride):
        super(Basic_block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channel, num_filter, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter)
        if self.stride== 2:
            self.conv3 = nn.Conv2d(in_channel, num_filter, kernel_size=3, stride=stride, padding=1)
            self.bn3 = nn.BatchNorm2d(num_filter)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.stride == 2:
            x = self.bn3(self.conv3(x))
        out += x
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):

    def __init__(self, in_channel, num_filter, stride):
        super(Bottleneck,self).__init__()
        self.stride = stride
        self.slide = 0
        self.conv1 = nn.Conv2d(in_channel, num_filter, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filter)
        self.conv3 = nn.Conv2d(num_filter, num_filter*4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(num_filter*4)
        if self.stride == 2 or in_channel==num_filter:
            self.slide = 1
            self.conv4 = nn.Conv2d(in_channel, num_filter*4, kernel_size=1, stride=stride)
            self.bn4 = nn.BatchNorm2d(num_filter*4)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.slide == 1:
            x = self.bn4(self.conv4(x))
        out += x
        out = F.relu(x)
        return out

# resnet18
class Resnet18(nn.Module):
    def __init__(self, channels, num_class):
        super(Resnet18, self).__init__()
        self.channels = channels
        self.conv1 = Conv_bn_relu(self.channels, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1_1 = Basic_block(64, 64, 1)
        self.block1_2 = Basic_block(64, 64, 1)
        self.block2_1 = Basic_block(64, 128, 2)
        self.block2_2 = Basic_block(128, 128, 1)
        self.block3_1 = Basic_block(128, 256, 2)
        self.block3_2 = Basic_block(256, 256, 1)
        self.block4_1 = Basic_block(256, 512, 2)
        self.block4_2 = Basic_block(512, 512, 1)
        self.linear = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.maxpool1(self.conv1(x))
        out = self.block1_1(out)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(-1, out.size()[1]*out.size()[2]*out.size()[3])
        out = self.linear(out)
        return F.log_softmax(out, dim=1)

# resnet34
class Resnet34(nn.Module):
    def __init__(self, channels, num_class):
        super(Resnet34, self).__init__()
        self.channels = channels
        self.conv1 = Conv_bn_relu(self.channels, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1_1 = Basic_block(64, 64, 1)
        self.block1_2 = Basic_block(64, 64, 1)
        self.block1_3 = Basic_block(64, 64, 1)
        self.block2_1 = Basic_block(64, 128, 2)
        self.block2_2 = Basic_block(128, 128, 1)
        self.block2_3 = Basic_block(128, 128, 1)
        self.block2_4 = Basic_block(128, 128, 1)
        self.block3_1 = Basic_block(128, 256, 2)
        self.block3_2 = Basic_block(256, 256, 1)
        self.block3_3 = Basic_block(256, 256, 1)
        self.block3_4 = Basic_block(256, 256, 1)
        self.block3_5 = Basic_block(256, 256, 1)
        self.block3_6 = Basic_block(256, 256, 1)
        self.block4_1 = Basic_block(256, 512, 2)
        self.block4_2 = Basic_block(512, 512, 1)
        self.block4_3 = Basic_block(512, 512, 1)
        self.linear = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.block1_1(out)
        out = self.block1_2(out)
        out = self.block1_3(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block2_3(out)
        out = self.block2_4(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)
        out = self.block3_5(out)
        out = self.block3_6(out)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(-1, out.size()[1]*out.size()[2]*out.size()[3])
        out = self.linear(out)
        return F.log_softmax(out, dim=1)

# resnet50
class Resnet50(nn.Module):
    def __init__(self, channels, num_class):
        super(Resnet50, self).__init__()
        self.channels = channels
        self.conv1 = Conv_bn_relu(self.channels, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1_1 = Bottleneck(64, 64, 1)
        self.block1_2 = Bottleneck(256, 64, 1)
        self.block1_3 = Bottleneck(256, 64, 1)
        self.block2_1 = Bottleneck(256, 128, 2)
        self.block2_2 = Bottleneck(512, 128, 1)
        self.block2_3 = Bottleneck(512, 128, 1)
        self.block2_4 = Bottleneck(512, 128, 1)
        self.block3_1 = Bottleneck(512, 256, 2)
        self.block3_2 = Bottleneck(1024, 256, 1)
        self.block3_3 = Bottleneck(1024, 256, 1)
        self.block3_4 = Bottleneck(1024, 256, 1)
        self.block3_5 = Bottleneck(1024, 256, 1)
        self.block3_6 = Bottleneck(1024, 256, 1)
        self.block4_1 = Bottleneck(1024, 512, 2)
        self.block4_2 = Bottleneck(2048, 512, 1)
        self.block4_3 = Bottleneck(2048, 512, 1)
        self.linear = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.maxpool1(self.conv1(x))
        out = self.block1_1(out)
        out = self.block1_2(out)
        out = self.block1_3(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block2_3(out)
        out = self.block2_4(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block3_3(out)
        out = self.block3_4(out)
        out = self.block3_5(out)
        out = self.block3_6(out)
        out = self.block4_1(out)
        out = self.block4_2(out)
        out = self.block4_3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(-1, out.size()[1]*out.size()[2]*out.size()[3])
        out = self.linear(out)
        return F.log_softmax(out, dim=1)


def resnet():
    return Resnet50(1000)


from torch.autograd import Variable
def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

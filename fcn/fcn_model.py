import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models


class FCN8s(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(FCN8s, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        net = models.vgg16(pretrained=True)  # 从预训练模型加载VGG16网络参数
        self.premodel = net.features  # 只使用Vgg16的五层卷积层（特征提取层）（3，224，224）----->（512，7，7）

        # self.conv6 = nn.Conv2d(512,512,kernel_size=1,stride=1,padding=0,dilation=1)
        # self.conv7 = nn.Conv2d(512,512,kernel_size=1,stride=1,padding=0,dilation=1)
        # (512,7,7)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)  # x2
        self.bn1 = nn.BatchNorm2d(512)
        # (512, 14, 14)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)  # x2
        self.bn2 = nn.BatchNorm2d(256)
        # (256, 28, 28)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)  # x2
        self.bn3 = nn.BatchNorm2d(128)
        # (128, 56, 56)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)  # x2
        self.bn4 = nn.BatchNorm2d(64)
        # (64, 112, 112)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)  # x2
        self.bn5 = nn.BatchNorm2d(32)
        # (32, 224, 224)
        self.classifier = nn.Conv2d(32, self.n_classes, kernel_size=1)
        # (num_classes, 224, 224)

    def forward(self, input):
        x = input
        for i in range(len(self.premodel)):
            x = self.premodel[i](x)
            if i == 16:
                x3 = x  # maxpooling3的feature map (1/8)
            if i == 23:
                x4 = x  # maxpooling4的feature map (1/16)
            if i == 30:
                x5 = x  # maxpooling5的feature map (1/32)

        # 五层转置卷积，每层size放大2倍，与VGG16刚好相反。两个skip-connect
        score = self.relu(self.deconv1(x5))  # out_size = 2*in_size (1/16)
        score = self.bn1(score + x4)

        score = self.relu(self.deconv2(score))  # out_size = 2*in_size (1/8)
        score = self.bn2(score + x3)

        score = self.bn3(self.relu(self.deconv3(score)))  # out_size = 2*in_size (1/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # out_size = 2*in_size (1/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # out_size = 2*in_size (1)

        score = self.classifier(score)  # size不变，使输出的channel等于类别数

        return score


if __name__ == "__main__":
    model = FCN8s(n_channels=3, n_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(3, 3, 64, 224)
    # model = model.to(device)
    print(model)
    output = model(input)
    print(output.shape)

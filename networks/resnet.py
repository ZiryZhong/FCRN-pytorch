import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models


"""
    resnet 部分
    本实验中主要用到了resnet50
"""
class base_resnet(nn.Module):
    def __init__(self, model_type ,model_pth_path=None):
        super(base_resnet, self).__init__()
        if model_type == "resnet50":
            self.model = models.resnet50(pretrained=False)
        elif model_type == "resnet34":
            self.model = models.resnet34(pretrained=False)
        elif model_type == "resnet101":
            self.model = models.resnet101(pretrained=False)
        # 使用自行下载后读取，pretrained=False
        # self.model = models.resnet50(pretrained=False)
        # self.model.load_state_dict(torch.load(model_pth_path))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        print('Loading Finished!')

    def forward(self, x):
        x = self.model.conv1(x)
        #print("1 {}".format(x.shape))
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        #print("2 {}".format(x.shape))
        x = self.model.layer1(x)
        #print("3 {}".format(x.shape))
        x = self.model.layer2(x)
        #print("4 {}".format(x.shape))
        x = self.model.layer3(x)
        #print("5 {}".format(x.shape))
        x = self.model.layer4(x)
        #print("6 {}".format(x.shape))
        #x = self.model.avgpool(x)
        #print("7 {}".format(x.shape))

        return x
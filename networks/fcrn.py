import torch
import torch.nn as nn
from networks.resnet import base_resnet
import numpy as np
from torchvision import models


"""
    改进二的网络结构
"""
class multiscale(nn.Module):

    def __init__(self):
        super(multiscale, self).__init__()
        backbone = "resnet50"
        backbone_pth = "weights/test/resnet-pretrain/resnet50-19c8e357.pth"
        self.resnet = base_resnet(backbone, backbone_pth)
        self.conv0 = nn.Conv2d(2048, 1024, (1,1))
        self.block1_1 = ms_blcok1(1024, 512)
        self.conv1 = nn.Conv2d(1024,256,(3, 3))
        self.block1_2 = ms_blcok1(256, 128)
        self.conv2 = nn.Conv2d(256, 128, (3, 3))
        self.block1_3 = ms_blcok1(128, 64)
        self.block1_4 = ms_blcok1(128, 64)
        self.conv22 = nn.Conv2d(128, 64, (1,1))

        self.conv3 = nn.Conv2d(128,32,(3,3))
        self.conv4 = nn.Conv2d(32,16,(3,3))

        self.upsample = nn.Upsample((57, 102), mode="bilinear")# 102
        self.upsample_1 = nn.Upsample((26, 46), mode="bilinear")

        self.upsample_s = nn.Upsample((228,405), mode="bilinear")
        self.conv = nn.Conv2d(16,1,(3,3))
        self.relu = nn.ReLU()

    def forward(self, x):

        # out = self.resnet(x)
        x = self.resnet.model.conv1(x)
        x = self.resnet.model.bn1(x)
        x = self.resnet.model.relu(x)
        x = self.resnet.model.maxpool(x)
        x1 = x

        #print(x.shape)
        x = self.resnet.model.layer1(x)
        #print(x.shape)
        x = self.resnet.model.layer2(x)

        x = self.resnet.model.layer3(x)
        #print(x.shape)
        x = self.resnet.model.layer4(x)
        #print(x.shape)

        out = self.conv0(x)
        #print(out.shape)

        out = self.block1_1(out)
        out = self.conv1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.block1_2(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = self.block1_3(out)
        out = self.conv22(out)
        out = self.relu(out)

        out = self.upsample(out)
        out = torch.cat([x1, out], dim=1)

        out = self.block1_4(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv(out)
        out = self.relu(out)
        out = self.upsample_s(out)
        #print(out.shape)
        return out


"""
    多尺度上采样模块一 
"""
class ms_blcok1(nn.Module):

    def __init__(self, input_ch, output_ch):
        super(ms_blcok1, self).__init__()

        self.conv1 = nn.Conv2d(input_ch, output_ch, (1,1))
        self.conv2 = nn.Conv2d(input_ch, output_ch, (3,3), stride=(1,1), padding=(1,1))
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")


    def forward(self, x):

        out1 = self.conv1(x)
        out2 = self.conv2(x)

        out = torch.cat([out1, out2],dim=1)
        out = self.upsample(out)

        return out


"""
    多尺度上采样模块二
"""
class ms_block2(nn.Module):

    def __init__(self, input_ch, output_ch):
        super(ms_block2, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, output_ch, (3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(input_ch, output_ch, (3,3), padding=(2,2), dilation=(2,2))
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):

        out1 = self.conv1(x)
        out2 = self.conv2(x)

        out = torch.cat([out1, out2], dim=1)
        out = self.upsample(out)

        return out


"""
    FCRN的改进一
"""

class extend_fcrn(nn.Module):

    def __init__(self):
        super(extend_fcrn, self).__init__()
        backbone_type = "resnet50"  # 以resnet50作为特征提取的骨架
        backbone_pth_path = "weights/exp/resnet50-19c8e357.pth"  # 待会下载一下权重文件
        print('Loading weights into state dict...')
        state = torch.load("./weights/exp5/CKPT-Epoch8-Total_Loss0.1208-Val_Loss0.0265.pth")

        self.fcrn = base_fcrn(backbone_type, backbone_pth_path) # fcrn结构
        self.fcrn.load_state_dict(state["model"])
        #self.fcrn_freeze()

        self.bn = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,3),stride=(1,1),padding=(1,1))

        self.fine1 = nn.Conv2d(3,1, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.fine2 = nn.Conv2d(2,8, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv = nn.Conv2d(8,16, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv1 = nn.Conv2d(16,32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(32,16, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.fine3 = nn.Conv2d(16,1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.relu = nn.ReLU(inplace=False)
        # self.upsample = nn.Upsample((228,405))

    def forward(self, x):

        out1 = self.fcrn(x)
        out2 = self.fine1(x)
        out2 = self.bn(out2)
        out2 = self.relu(out2)
        out2 = self.maxpool(out2)
        #print("out1 {} out2 {}".format(out1.shape, out2.shape))
        #out2 = self.relu(out2)

        out3 = torch.cat([out2, out1],dim=1)

        out4 = self.fine2(out3)

        out4 = self.relu(out4)

        out_ = self.conv(out4)
        out_ = self.relu(out_)

        out_ = self.conv1(out_)
        out_ = self.relu(out_)
        out_ = self.conv2(out_)
        out_ = self.relu(out_)

        out = self.fine3(out_)
        out = self.relu(out)

        return out

    def fcrn_freeze(self):
        for param in self.fcrn.parameters():
            param.requires_grad = False



"""
    多尺度融合的FCRN
"""
class mutiscale_fcrn(nn.Module):

    def __init__(self):
        super(mutiscale_fcrn, self).__init__()
        backbone_type = "resnet50"  # 以resnet50作为特征提取的骨架
        backbone_pth_path = "weights/exp/resnet50-19c8e357.pth"  # 待会下载一下权重文件

        self.resnet = models.resnet50(pretrained=False)
        self.fcrn = base_fcrn(backbone_type, backbone_pth_path)
        self.upsample_2 = nn.Upsample((128,208), mode="bilinear")
        #self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear")
        #self.upsample_8 = nn.Upsample(scale_factor=8, mode="bilinear")
        #self.upsample_16 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.conv = nn.Conv2d(64,1,(1,1))
        # self.conv_1 = nn.Conv2d(448,224,(1,1))
        # self.conv_2 = nn.Conv2d(224, 112, (1,1))
        # self.conv_3 = nn.Conv2d(112, 56, (1,1))
        # self.conv_4 = nn.Conv2d(56, 1, (1,1))
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample((228,405), mode="bilinear")
        self.downsample = nn.Conv2d(256, 128, (1,1))
        self.downsample_1 = nn.Conv2d(256, 128, (1,1))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        #print(x.shape)
        up1 = self.upsample_2(x)
        x = self.resnet.layer1(x)
        #print(x.shape)
        up2 = self.upsample_2(x)
        up2 = self.downsample(up2)
        #print(up2.shape)
        x = self.resnet.layer2(x)
        # #print(x.shape)

        # up3 = self.upsample_2(x)
        x = self.resnet.layer3(x)
        # #print(x.shape)

        #up4 = self.upsample_16(x)
        x = self.resnet.layer4(x)

        x = self.fcrn.conv1(x)
        x = self.fcrn.fast_up_1(x)
        x = self.fcrn.relu(x)
        x = self.fcrn.fast_up_2(x)
        x = self.fcrn.relu(x)
        x = self.fcrn.fast_up_3(x)
        x = self.fcrn.relu(x)
        x = self.upsample_2(x)

        x = torch.cat([up2,x],dim=1)
        x = self.downsample_1(x)
        x = self.fcrn.fast_up_4(x)
        x = self.fcrn.relu(x)

        out = self.conv(x)
        # out = self.conv_1(out)
        # out = self.conv_2(out)
        # out = self.conv_3(out)
        # out = self.conv_4(out)

        out = self.relu(out)

        out = self.upsample(out)

        return out



"""
    原始的FCRN
"""
class base_fcrn(nn.Module):

    def __init__(self, backbone, backbone_pth):
        super(base_fcrn,self).__init__()

        self.resnet = base_resnet(backbone, backbone_pth)
        self.conv1 = nn.Conv2d(2048,1024,(1,1))
        self.fast_up_1 = base_upprojection(1024, 512).cuda(torch.device(0))
        self.fast_up_2 = base_upprojection(512, 256).cuda(torch.device(0))
        self.fast_up_3 = base_upprojection(256, 128).cuda(torch.device(0))
        self.fast_up_4 = base_upprojection(128, 64).cuda(torch.device(0))
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.4)
        self.conv2 = nn.Conv2d(64,1,(3,3),stride=(1,1),padding=(1,1)) # TODO:这边的大小要修正一下
        self.upsample = nn.Upsample((228, 405), mode="bilinear")

    def forward(self, x):

        # 特征提取模块
        #print("8 {}".format(x.shape))
        out = self.resnet(x)
        out1 = self.conv1(out)
        #print("9 {}".format(out1.shape))
        # 上采样模块
        out2 = self.fast_up_1(out1)
        out2 = self.relu(out2)
        #print("10 {}".format(out2.shape))
        out3 = self.fast_up_2(out2)
        out3 = self.relu(out3)
        #print("11 {}".format(out3.shape))
        out4 = self.fast_up_3(out3)
        out4 = self.relu(out4)
        #print("12 {}".format(out4.shape))
        out5 = self.fast_up_4(out4)
        out5 = self.relu(out5)
        #print("13 {}".format(out5.shape))
        out5 = self.dropout(out5) # TODO：测试dropout有没有用

        # 输出预测的深度图 大小接近原图的一半
        out6 = self.conv2(out5)
        out6 = self.relu(out6)
        out6 = self.upsample(out6)
        #print("14 {}".format(out6.shape))

        return out6

    def freezebone(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreezebone(self):
        for param in self.resnet.parameters():
            param.requires_grad = True



class base_upprojection(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(base_upprojection, self).__init__()

        # 输入输出通道数
        self.input_channel = input_channel
        self.output_channel = output_channel
        # 四层小卷积核
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (2,3), stride=(1,1), padding=(0,1))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (3,2), stride=(1,1), padding=(1,0))
        self.conv4 = nn.Conv2d(input_channel, output_channel, (2,2), stride=(1,1), padding=(0,0))
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(output_channel, output_channel, (3,3), stride=(1,1), padding=(1,1)) # TODO:这边上投影模块再注意一下

        # 保存interleaving模板
        self.il_rows = []
        self.il_cols = []
        self.inter_leaving_flag = False

    def forward(self, x):

        outs = []

        out1 = self.conv1(x)
        [b,c,h,w] = out1.shape

        concat_h = torch.zeros(b, c, 1, w).cuda(torch.device(0))
        concat_w = torch.zeros(b, c, h, 1).cuda(torch.device(0))

        out2 = self.conv2(x)

        out2 = torch.cat([out2, concat_h],dim=2)

        out3 = self.conv3(x)
        out3 = torch.cat([out3, concat_w],dim=3)

        concat_h_special = torch.zeros(b, c, 1, w-1).cuda(torch.device(0))

        out4 = self.conv4(x)
        out4 = torch.cat([out4, concat_h_special],dim=2)
        out4 = torch.cat([out4, concat_w],dim=3)

        outs.append(out1)
        outs.append(out2)
        outs.append(out3)
        outs.append(out4)

        tmp_res_1 = self.interleaving(outs, b, c, h, w, 2)
        tmp_res_2 = self.relu(tmp_res_1)
        tmp_res_2 = self.conv5(tmp_res_2)

        tmp_res = tmp_res_1 + tmp_res_2
        res = self.relu(tmp_res)

        return res


    def interleaving(self, x, b, c, h, w, scale):

        out = torch.zeros(b, c, h*scale, w*scale).cuda(torch.device(0))

        list_col = torch.linspace(0, int(w * scale ) - 2, int(w * scale / 2)).view(1,-1)
        list_row = torch.linspace(0, int(h * scale ) - 2, int(h * scale / 2)).view(1,-1)
        tmp_mat_col = torch.zeros(int(h * scale / 2) ,int(w * scale / 2))
        tmp_mat_col = tmp_mat_col.copy_(list_col)

        tmp_mat_row = torch.zeros(int(w * scale / 2), int(h * scale / 2))
        tmp_mat_row = tmp_mat_row.copy_(list_row)

        col = tmp_mat_col.view(-1)
        tmp_m_r = tmp_mat_row.T
        tmp_m = tmp_m_r.contiguous()
        row = tmp_m.view(-1)[0:]

        row = row.detach().numpy()
        col = col.detach().numpy()

        out[:,:,row,col] = x[0].view(b,c,-1)
        out[:,:,row,col+1] = x[1].view(b,c,-1)
        out[:,:,row+1,col] = x[2].view(b,c,-1)
        out[:,:,row+1,col+1] = x[3].view(b,c,-1)
        self.inter_leaving_flag = True

        return out
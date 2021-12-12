import torch
import torch.nn as nn
from networks.resnet import base_resnet
import numpy as np

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
        self.conv2 = nn.Conv2d(64,1,(3,3),stride=(1,1),padding=(1,1)) # TODO:这边的大小要修正一下
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

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

        # if self.inter_leaving_flag:
        #
        #     out[:, :, torch.tensor(self.il_rows[0], dtype=torch.long).cuda(torch.device(0)),
        #               torch.tensor(self.il_cols[0], dtype=torch.long).cuda(torch.device(0))] = x[0].view(b, c, -1)
        #
        #     out[:, :, torch.tensor(self.il_rows[0], dtype=torch.long).cuda(torch.device(0)),
        #               torch.tensor(self.il_cols[1], dtype=torch.long).cuda(torch.device(0))] = x[1].view(b, c, -1)
        #
        #     out[:, :, torch.tensor(self.il_rows[1], dtype=torch.long).cuda(torch.device(0)),
        #               torch.tensor(self.il_cols[0], dtype=torch.long).cuda(torch.device(0))] = x[2].view(b, c, -1)
        #
        #     out[:, :, torch.tensor(self.il_rows[1], dtype=torch.long).cuda(torch.device(0)),
        #               torch.tensor(self.il_cols[1], dtype=torch.long).cuda(torch.device(0))] = x[3].view(b, c, -1)

        # else:
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

        # self.il_rows.append(row)
        # self.il_rows.append(row+1)
        # self.il_cols.append(col)
        # self.il_cols.append(col+1)
        row = row.detach().numpy()\
            #.cuda(torch.device(0))
        col = col.detach().numpy()\
            #.cuda(torch.device(0))

        #TODO: 这边在做内插的时候有个bug 是分片操作时候的失误 导致输出的图片成块状 待解决
        #TODO: 考虑提前生成这些索引矩阵 提高速度
        out[:,:,row,col] = x[0].view(b,c,-1)

        out[:,:,row,col+1] = x[1].view(b,c,-1)

        out[:,:,row+1,col] = x[2].view(b,c,-1)

        out[:,:,row+1,col+1] = x[3].view(b,c,-1)

        self.inter_leaving_flag = True

        return out
import torch
import torch.nn.functional as F
import torch.nn as nn


'''原始实现的Berhu'''
def Berhu_Loss(pred, ground_truth):

    pred = pred.permute(0, 2, 3, 1).float()

    ground_truth = ground_truth.permute(0, 2, 3, 1).float()
    [b,h,w,c] = ground_truth.shape
    ground_truth = ground_truth[:,:,:,0].view(b,h,w,1)
    #print(pred.shape)
    #print(ground_truth.shape)
    error = abs(pred - ground_truth).view(b,h,w)
    #print(ground_truth)
    #print(pred)
    #print(error)
    num_of_batch = pred.shape[0] # batch 大小
    #c, idx = torch.max(error,dim=0)
    #print(c.shape)

    error = pow(error, 2)
    # for i in range(0, b-1):
    #     error[i, error[i,:,:] > 0.2 * c] = (pow(error[i,:,:], 2) + pow(0.2* c, 2)) / (2 * 0.2 * c)
    # sum_error = error.sum(axis=[1, 2]) # 求和得到每张图的误差
    # print(sum_error.shape)
    # #print(sum_error)
    # sum_error[sum_error > c] = (pow(sum_error, 2) + pow(c, 2)) / (2 * c)

    loss = error.sum() / (b*h*w)

    return loss


'''开源Berhu实现一'''
class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss


'''开源Berhu实现二'''
class loss_huber(nn.Module):
    def __init__(self):
        super(loss_huber,self).__init__()

    def forward(self, pred, truth):
        # [b c h w] 为四个通道
        c = pred.shape[1] #通道
        h = pred.shape[2] #高
        w = pred.shape[3] #宽

        pred = pred.view(-1, c * h * w).cuda()
        truth = truth.view(-1, c * h * w).cuda()
        # 根据当前batch所有像素计算阈值

        t = 0.2 * torch.max(torch.abs(pred - truth))

        # 计算L1范数
        l1 = torch.mean(torch.mean(torch.abs(pred - truth), 1), 0)
        # 计算论文中的L2
        l2 = torch.mean(torch.mean(((pred - truth)**2 + t**2) / t / 2, 1), 0)

        if l1 > t:
            return l2
        else:
            return l1


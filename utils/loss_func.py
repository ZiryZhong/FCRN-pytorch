import torch
import torch.nn.functional as F


def Berhu_Loss(pred, ground_truth):

    pred = pred.permute(0, 2, 3, 1).float()

    ground_truth = ground_truth.permute(0, 2, 3, 1).float()
    [b,h,w,c] = ground_truth.shape
    ground_truth = ground_truth[:,:,:,0].view(b,h,w,1)
    #print(pred.shape)
    #print(ground_truth.shape)
    error = abs(pred - ground_truth).view(b,h,w)

    num_of_batch = pred.shape[0] # batch 大小
    c = 0.2 * torch.max(error).sum() / num_of_batch
    #print(c.shape)
    sum_error = error.sum(axis=[1, 2]) # 求和得到每张图的误差
    #print(sum_error.shape)
    #print(sum_error)
    #sum_error[sum_error > c] = (pow(sum_error, 2) + pow(c, 2)) / (2 * c)

    loss = sum_error.sum() / num_of_batch

    return loss
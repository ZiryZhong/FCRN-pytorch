import torch
import torch.nn as nn
from utils.loss_func import Berhu_Loss, loss_huber
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import  matplotlib.pyplot as plot

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

loss_fn = loss_huber()

def fit_one_epoch(model, net, epoch, epoch_size, total_epoch, epoch_size_val, optimizer, gen, gen_val, Freeze_Epoch, cuda):

    total_loss = 0
    val_loss = 0
    torch.autograd.set_detect_anomaly(True)
    iteration = 0
    model.train()
    with tqdm(total=total_epoch, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for batch in gen:  # enumerate 可以返回 索引 和 内容
            iteration += 1

            batch_img, batch_dpm = batch
            input_img = Variable(batch_img.type(torch.cuda.FloatTensor))
            input_dpm = Variable(batch_dpm.type(torch.cuda.FloatTensor))

            optimizer.zero_grad()

            pred_dpm = model(input_img)
            loss = loss_fn(pred_dpm, input_dpm)

            total_loss = total_loss + loss.item()  # item是将张量类型转换为浮点类型

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数值

            # 进度条
            pbar.set_postfix(**{'total_r_loss': total_loss / (iteration),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if True:
        iteration = 0
        model.eval()
        print('Start Validation')
        with tqdm(total=20, desc=f'Epoch {epoch + 1}/{20}', postfix=dict, mininterval=0.3) as pbar:
            for batch in gen_val:
                iteration += 1
                with torch.no_grad():

                    batch_img, batch_dpm = batch
                    input_img = Variable(batch_img.type(torch.cuda.FloatTensor))
                    input_dpm = Variable(batch_dpm.type(torch.cuda.FloatTensor))

                    pred_dpm = model(input_img)
                    loss = loss_fn(pred_dpm,input_dpm)

                    val_loss += loss.item()

                pbar.set_postfix(**{'total_loss': val_loss / (iteration )})
                pbar.update(1)

        print('Finish Validation')
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (iteration), val_loss / (iteration)))

        print('Saving state, iter:', str(epoch + 1))
        if epoch % 5 == 0:
            state = {"model": model.state_dict(),"optimizer":optimizer.state_dict()}
            torch.save(state, './weights/train/CKPT-Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
                (epoch), total_loss / (iteration), val_loss / (iteration)))

        input_gt_depth_image = input_dpm[0][0].data.cpu().numpy().astype(np.float32)
        pred_depth_image = pred_dpm[0].data.squeeze().cpu().numpy().astype(np.float32)

        input_gt_depth_image /= np.max(input_gt_depth_image)
        pred_depth_image /= np.max(pred_depth_image)

        plot.imsave('./results/train/exp/gt_depth_epoch_{}.png'.format(epoch + 1), input_gt_depth_image,
                    cmap="viridis")
        plot.imsave('./results/train/exp/pred_depth_epoch_{}.png'.format(epoch + 1), pred_depth_image,
                    cmap="viridis")

    return val_loss / (iteration)


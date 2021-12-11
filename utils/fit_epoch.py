import torch
import torch.nn as nn
from utils.loss_func import Berhu_Loss
from torch.autograd import Variable
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(model, net, epoch, epoch_size, total_epoch, epoch_size_val, optimizer, gen, gen_val, Freeze_Epoch, cuda):

    total_loss = 0
    val_loss = 0
    torch.autograd.set_detect_anomaly(True)

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):  # enumerate 可以返回 索引 和 内容
            if iteration >= epoch_size:
                break
            with torch.no_grad():  # 把数据从numpy类型转换为tensor类型
                if cuda:
                    batch = [Variable(torch.from_numpy(ann.astype(float)).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann.astype(float)).type(torch.FloatTensor)) for ann in batch]

            batch_img, batch_dpm = batch


            optimizer.zero_grad()

            pred_dpm = net(batch_img)
            loss = Berhu_Loss(pred_dpm, batch_dpm)

            total_loss = total_loss + loss.item()  # item是将张量类型转换为浮点类型

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数值

            # 进度条
            pbar.set_postfix(**{'total_r_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_size_val:
                break
            with torch.no_grad():
                if cuda:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in batch]
                else:
                    batch = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in batch]

                batch_img, batch_dpm = batch

                pred_dpm = net(batch_img)
                loss = Berhu_Loss(pred_dpm, batch_dpm)

                val_loss += loss.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(total_epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), './weights/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    return val_loss / (epoch_size_val + 1)


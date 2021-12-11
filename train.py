import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from networks.fcrn import base_fcrn
from data_process.dataset import FcrnDataSet, Fcrn_Dataset_Collate
from utils.loss_func import Berhu_Loss
from torch.utils.data import DataLoader
from utils.fit_epoch import fit_one_epoch


# 超参数
batch_size = 8
num_of_epoch = 20
learning_rate = 0.0001
freeze_epoch = 10
cuda_available = 1

# 构建数据集
data_path = "./script/train.txt"
val_rate = 0.1



if __name__ == '__main__':

    # 初始化模型
    backbone_type = "resnet50" # 以resnet50作为特征提取的骨架
    backbone_pth_path = "weights/resnet50-19c8e357.pth" # 待会下载一下权重文件
    print('Loading weights into state dict...')
    model = base_fcrn(backbone_type, backbone_pth_path)

    #model_dict = model.state_dict()
    #pretrained_dict = torch.load(backbone_pth_path)
    # 加载模型 剔除加载模型中和现有模型不一样的结构
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    #model_dict.update(pretrained_dict)  # 更新model_dict
    #model.load_state_dict(model_dict)
    print('Finished!')

    # 采用GPU方式训练
    net = model.train()
    net = torch.nn.DataParallel(model)  # 多GPU的时候并行处理
    cudnn.benchmark = True  # 提高运行效率 找到最高效的运行方式 （但有条件）
    net = net.cuda()

    # 划分数据集
    with open(data_path) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * val_rate)

    num_train = len(lines) - num_val


    '''
        参数解冻前的训练
    '''
    # 优化器模块
    optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    # 数据加载模块
    img_set = []
    dpm_set = []

    for l in lines:
        l = l.rstrip("\n")
        img_set.append(l.split(" ")[0])
        dpm_set.append(l.split(" ")[1])

    print(num_train)
    train_dataset = FcrnDataSet(img_set[:400], dpm_set[:400])
    val_dataset = FcrnDataSet(img_set[400:480], dpm_set[400:480])
    gen = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, pin_memory=True,
                     drop_last=True, collate_fn=Fcrn_Dataset_Collate)
    gen_val = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, pin_memory=True,
                         drop_last=True, collate_fn=Fcrn_Dataset_Collate)

    epoch_size = 400 // batch_size
    epoch_size_val = 80 // batch_size

    model.freezebone()

    for epoch in range(0, epoch_size-1):
        val_loss = fit_one_epoch(model, net, epoch, epoch_size, num_of_epoch, epoch_size_val, optimizer,gen, gen_val, freeze_epoch, cuda_available)
        lr_scheduler.step(val_loss)


    '''
        参数解冻后的训练
    '''
    # optimizer = torch.optim.Adam(net.parameters(), learning_rate, weight_decay=5e-4)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    #
    # # 数据加载模块
    # img_set = []
    # dpm_set = []
    #
    # for l in lines:
    #     img_set.append(l.split(" ")[0])
    #     dpm_set.append(l.split(" ")[1])
    #
    # train_dataset = FcrnDataSet(img_set[:num_train], dpm_set[:num_train])
    # val_dataset = FcrnDataSet(img_set[num_train:], dpm_set[num_train:])
    # gen = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True,
    #                  drop_last=True, collate_fn=Fcrn_Dataset_Collate)
    # gen_val = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True,
    #                      drop_last=True, collate_fn=Fcrn_Dataset_Collate)
    #
    # epoch_size = num_train // batch_size
    # epoch_size_val = num_val // batch_size

    model.unfreezebone()

    for epoch in range(epoch_size, epoch_size_val):
        val_loss = fit_one_epoch(net, epoch, epoch_size, num_of_epoch, epoch_size_val, optimizer,gen, gen_val, freeze_epoch, cuda_available)
        lr_scheduler.step(val_loss)


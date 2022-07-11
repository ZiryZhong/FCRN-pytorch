import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from networks.fcrn import base_fcrn, extend_fcrn, mutiscale_fcrn, multiscale
from data_process.dataset import FcrnDataSet, Fcrn_Dataset_Collate, NyuDepthLoader, load_split
from utils.loss_func import Berhu_Loss
from torch.utils.data import DataLoader
from utils.fit_epoch import fit_one_epoch
import os

'''超参数'''
batch_size = 8
num_of_epoch = 80
learning_rate = 0.0001
freeze_epoch = 10
cuda_available = 1

'''数据集相关'''
'''
    data_path: 加载自制的球场深度数据集(train_v3.txt是最新数据集，包含两种不同视角)
    data_path_new: nyu_v2 数据集 (暂时不用官方数据集)
    val_rate: 指定数据集中验证集的比例 (暂时不用)
'''
data_path = "./script/train_v3.txt"
#data_path_new = "./data_process/nyu_depth_v2_labeled.mat"
val_rate = 0.1

if __name__ == '__main__':

    print(torch.cuda.is_available())  # 是否有可用的gpu
    print(torch.cuda.device_count())  # 有几个可用的gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 声明gpu
    dev = torch.device('cuda:0')  # 调用哪个gpu
    torch.cuda.empty_cache()

    '''初始化模型'''
    backbone_type = "resnet50" # 以resnet50作为特征提取的骨架
    backbone_pth_path = "weights/test/resnet-pretrain/resnet50-19c8e357.pth" # 待会下载一下权重文件
    print('Loading weights into state dict...')

    # 改进版FCRN模型
    model = multiscale()
    state = torch.load("./weights/test/fcrn-modify/fcrn-multiscale.pth")
    model.load_state_dict(state["model"])

    # 原版FCRN模型
    #model = base_fcrn(backbone_type, backbone_pth_path)
    #state = torch.load("./weights/test/fcrn-origin/fcrn.pth")
    #model.load_state_dict(state["model"])

    print('Finished!')

    '''采用GPU方式训练'''
    net = model.train()
    net = torch.nn.DataParallel(model)  # 多GPU的时候并行处理
    cudnn.benchmark = True  # 提高运行效率 找到最高效的运行方式 （但有条件）
    net = net.cuda(dev)
    model = model.cuda(dev)

    '''划分数据集'''
    with open(data_path) as f:
        lines = f.readlines()

    np.random.seed(101011)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * val_rate)
    num_train = len(lines) - num_val


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

    '''nyu数据集加载'''
    #train_lists, val_lists, test_lists = load_split()
    #train_dataset = NyuDepthLoader(data_path_new, train_lists)
    #val_dataset = NyuDepthLoader(data_path_new, val_lists)

    '''普通数据集加载'''
    train_dataset = FcrnDataSet(img_set[0:400], dpm_set[0:400])
    val_dataset = FcrnDataSet(img_set[400:464], dpm_set[400:464])
    gen = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,
                     drop_last=True)
    gen_val = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,
                         drop_last=True, shuffle=True)

    epoch_size = 400 // batch_size
    epoch_size_val = 64 // batch_size

    '''模型训练循环'''
    for epoch in range(0, 100):
        val_loss = fit_one_epoch(model, net, epoch, epoch_size, num_of_epoch, epoch_size_val, optimizer,gen, gen_val, freeze_epoch, cuda_available)
        # if epoch%2 == 0:
        lr_scheduler.step(val_loss)

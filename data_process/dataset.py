from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 as cv
import torch
# 返回图片路径
# 将图片路径预处理好存储到txt文件进行读取
# 所有图像路径在外围读取存在一个变量中
# 这边给dataset传入一堆路径
# 然后每次索引的时候按照index 提取对应的行
# 再对对应行的数据进行拆分


class FcrnDataSet(Dataset):

    def __init__(self, img_paths, depth_paths):
        self.img_paths = img_paths
        self.depth_paths = depth_paths
        self.length = len(img_paths)

    def __len__(self):
        return self.length

    #TODO: 要做一个resize的操作
    def getDataList(self, index):
        img_path = self.img_paths[index]
        dpm_path = self.depth_paths[index]

        img = cv.imread(img_path)
        dpm = cv.imread(dpm_path)

        img = img.transpose((2,0,1))
        dpm = dpm.transpose((2,0,1))

        return img, dpm

    def __getitem__(self, item):
        img, dpm = self.getDataList(item)
        return img, dpm




def Fcrn_Dataset_Collate(batch):

    batch_img, batch_dpm = [], []

    for img, dpm in batch:
        batch_img.append(img)
        batch_dpm.append(dpm)

    batch_img = np.array(batch_img)
    batch_dpm = np.array(batch_dpm)

    return batch_img, batch_dpm
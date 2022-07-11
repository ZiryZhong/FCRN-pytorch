from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 as cv
import h5py
from PIL import Image
import torchvision.transforms as transforms
import os


'''构建训练数据集'''
class FcrnDataSet(Dataset):

    def __init__(self, img_paths, depth_paths):
        self.img_paths = img_paths
        self.depth_paths = depth_paths
        self.length = len(img_paths)

    def __len__(self):
        return self.length


    def getDataList(self, index):
        img_path = self.img_paths[index]
        dpm_path = self.depth_paths[index]

        img = cv.imread(img_path)
        dpm = cv.imread(dpm_path)
        dpm = dpm[:,:,0]

        img = img.transpose((0,1,2))
        dpm = dpm.transpose((0,1))
        #print(dpm.shape)
        img = Image.fromarray(img)
        dpm = Image.fromarray(dpm)

        input_transform = transforms.Compose([transforms.Resize(228),
                                              transforms.ToTensor()])

        target_depth_transform = transforms.Compose([transforms.Resize(228),
                                                     transforms.ToTensor()])

        img = input_transform(img)
        dpm = target_depth_transform(dpm)
        #print(img.shape)
        return img, dpm

    def __getitem__(self, item):
        img, dpm = self.getDataList(item)
        return img, dpm


'''Nyu数据集加载'''
class NyuDepthLoader(Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0) # HWC
        dpt = self.dpts[img_idx].transpose(1, 0)
        #print(img)
        img = Image.fromarray(img)
        #print(img)
        dpt = Image.fromarray(dpt)
        # compose 将图像预处理
        input_transform = transforms.Compose([transforms.Resize(228),
                                              transforms.ToTensor()])

        target_depth_transform = transforms.Compose([transforms.Resize(228),
                                                     transforms.ToTensor()])

        img = input_transform(img)
        #print(img.shape)
        dpt = target_depth_transform(dpt)
        return img, dpt

    def __len__(self):
        return len(self.lists)


def Fcrn_Dataset_Collate(batch):

    batch_img, batch_dpm = [], []

    for img, dpm in batch:
        batch_img.append(img)
        batch_dpm.append(dpm)

    batch_img = np.array(batch_img)
    batch_dpm = np.array(batch_dpm)

    return batch_img, batch_dpm


'''根据trainIdxs.txt和testIdxs.txt分割数据集'''
def load_split():
    current_directoty = os.getcwd()
    train_lists_path = current_directoty + '/data_process/trainIdxs.txt'
    test_lists_path = current_directoty + '/data_process/testIdxs.txt'

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []

    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    val_start_idx = int(len(train_lists) * 0.8)

    val_lists = train_lists[val_start_idx:-1]
    train_lists = train_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists
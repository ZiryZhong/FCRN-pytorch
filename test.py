'''
    进行模型测试（多张图）
'''

import PIL.Image as Image
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import torch
from networks.fcrn import extend_fcrn, multiscale, base_fcrn
from torch.autograd import Variable
import matplotlib.pyplot as plot
from torch.utils.data import DataLoader
from data_process.dataset import FcrnDataSet
import time

dtype = torch.cuda.FloatTensor

''' 评价指标 '''
REL = 0
MAE = 0
RMSE = 0
LOG_MAE = 0
LOG_RMSE = 0
SILOG = 0

''' 测试数据集加载部分 '''
data_path = "./script/test.txt"

with open(data_path) as f:
    lines = f.readlines()

np.random.seed(101011)
np.random.shuffle(lines)
np.random.seed(None)
num_test = len(lines)
print(num_test)
test_set = []
dpm_set = []

batch_size = 1

for l in lines:
    l = l.rstrip("\n")
    test_set.append(l.split(" ")[0])
    dpm_set.append(l.split(" ")[1])

test_dataset = FcrnDataSet(test_set, dpm_set)

gen_test = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                      drop_last=True)


'''模型加载部分'''
#model_state = torch.load("./weights/test/fcrn-origin/fcrn.pth") # 原版FCRN模型
model_state = torch.load("./weights/test/fcrn-modify/fcrn-multiscale.pth") # 改进二FCRN的模型

#model = base_fcrn("resnet50","weights/test/resnet-pretrain/resnet50-19c8e357.pth") # 原版FCRN模型
model = multiscale() # 改进版FCRN模型

model = model.cuda()
model.load_state_dict(model_state['model'])

model.eval()
test_loss = 0
cnt = 0
start = time.time()

for img, dpm in test_dataset:
    print("cnt {} REL {} RMSE {}".format(cnt ,REL, RMSE))
    cnt += 1
    with torch.no_grad():
        input_var = Variable(img.type(dtype)).view(1, 3, 228, 405)
        input_dpm = Variable(dpm.type(dtype)).view(1, 1, 228, 405)

        out = model(input_var)

        pred_depth_image = out[0].data.squeeze().cpu().numpy().astype(np.float32)
        gt_depth_image = input_dpm[0][0].data.cpu().numpy().astype(np.float32)

        pred_depth_image /= np.max(pred_depth_image)
        gt_depth_image /= np.max(gt_depth_image)

        #print('predict complete.')
        #plot.imsave('Test_input_rgb_{:05d}.png'.format(11), gt_depth_image, cmap="viridis")
        #plot.imsave('Test_pred_depth_{:05d}.png'.format(11), pred_depth_image, cmap="viridis")
        gt_depth_image[gt_depth_image == 0] = 0.1

        REL += np.abs( (gt_depth_image - pred_depth_image) / gt_depth_image ).sum() / (cnt * 228*405)

        RMSE += np.sum( (gt_depth_image - pred_depth_image) ** 2 / (cnt*228*405))

duration = start - time.time()

REL = REL
RMSE = RMSE

print("REL: {} RMSE: {} time {}".format(REL, RMSE, duration /cnt))

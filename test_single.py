"""
    进行单张图片的模型测试
"""
import PIL.Image as Image
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import torch
from networks.fcrn import extend_fcrn, multiscale, base_fcrn
from torch.autograd import Variable
import matplotlib.pyplot as plot

dtype = torch.cuda.FloatTensor

''' 数据集路径 '''
#img_path = "G:\\football-game-data\\new\\2\\2-frame65235.rdc_1.jpg"
#dpm_path = "G:\\football-game-data\\new\\2\\2-frame65235.rdc_2.png"
#img_path = "G:\\football-game-video\\00138.jpg"
img_path = "G:\\football-game-data\\t\\test\\12.jpg"
dpm_path = "G:\\football-game-data\\t\\test\\12.png"

''' 数据集准备 '''
img = cv.imread(img_path)
dpm = cv.imread(dpm_path)
dpm = dpm[:, :, 0]

dpm = dpm.transpose((0, 1))

img = Image.fromarray(img)

dpm = Image.fromarray(dpm)
input_transform = transforms.Compose([transforms.Resize(228),
                                              transforms.ToTensor()])
target_depth_transform = transforms.Compose([transforms.Resize(228),
                                                     transforms.ToTensor()])

img = input_transform(img)
dpm = target_depth_transform(dpm)
batch_size = 1

''' 模型加载 '''
#model_state = torch.load("./weights/test/fcrn-origin/fcrn.pth")
model_state = torch.load("./weights/test/fcrn-modify/fcrn-multiscale.pth")

#model = base_fcrn("resnet50","weights/test/resnet-pretrain/resnet50-19c8e357.pth")
model = multiscale()
model = model.cuda()
model.load_state_dict(model_state['model'])


''' 单张图片测试 '''
model.eval()

with torch.no_grad():

    input_var = Variable(img.type(dtype)).view(1,3,228,405)
    input_dpm = Variable(dpm.type(dtype)).view(1,1,228,405)

    out = model(input_var)

    pred_depth_image = out[0].data.squeeze().cpu().numpy().astype(np.float32)
    gt_depth_image = input_dpm[0][0].data.cpu().numpy().astype(np.float32)

    pred_depth_image /= np.max(pred_depth_image)
    gt_depth_image /= np.max(gt_depth_image)

    print('predict complete.')
    plot.imsave('Test_input_rgb_{:05d}.png'.format(11), gt_depth_image, cmap="viridis")
    plot.imsave('Test_pred_depth_{:05d}.png'.format(11), pred_depth_image, cmap="viridis")
    REL = np.abs((gt_depth_image - pred_depth_image) / gt_depth_image).sum() / (228*405)
    print(REL)
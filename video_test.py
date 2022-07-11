'''
    测试模型实时预测视频
'''

import PIL.Image as Image
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import torch
from networks.fcrn import extend_fcrn, multiscale, base_fcrn
from torch.autograd import Variable
import matplotlib.pyplot as plotl
import time

dtype = torch.cuda.FloatTensor

model_state = torch.load("./weights/test/fcrn-modify/fcrn-multiscale.pth")

cap = cv.VideoCapture('G:\\football-game-video\\录制_2021_11_15_21_10_05_671.mp4')
#cap = cv.VideoCapture('G:\\football-game-video\\1(12).mp4')

fps = 25
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
videoWriter = cv.VideoWriter('./video.mp4', cv.VideoWriter_fourcc('M', 'P', 'E', 'G'),fps, (405, 228))
model = multiscale()
model = model.cuda()
print("_____ loading model _____")
model.load_state_dict(model_state['model'])
print("_____    finish!    _____")
input_transform = transforms.Compose([transforms.ToTensor()])
target_depth_transform = transforms.Compose([transforms.Resize(228),
                                                     transforms.ToTensor()])
start = time.time()
cnt = 0


with torch.no_grad():
    while(cap.isOpened()):
        ret, frame = cap.read()
        a = frame[:,:,1]
        cnt += 1
        #frame = frame[300:660,640:1280]
        frame = cv.resize(frame, (405, 228))
        frame = input_transform(frame)
        frame = Variable(frame.type(dtype)).view(1,3,228,405)
        pred = model(frame)
        pred = pred.view(228,405,1)
        #print(type(pred.cpu().numpy()))
        b = pred.cpu().view(228, 405).numpy()
        c = np.array([b for i in range(3)]).transpose(1, 2, 0)
        cv.imshow('frame',c)
        videoWriter.write(c)

        #fps = cnt / (time.time() - start)
        if cnt == 600:
            break
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
videoWriter.release()
import cv2 as cv
import numpy as np
import os


'''将原始红色深度图转为灰色深度图'''

data_path = "G:\\football-game-data\\new\\14"

files = os.listdir(data_path)

for f in files:

    if f.endswith(".png"):
        print(os.path.join(data_path,f))
        dpm = cv.imread(os.path.join(data_path,f))
        dpm = dpm[:,:,2]
        cv.imwrite(os.path.join(data_path,f),dpm)


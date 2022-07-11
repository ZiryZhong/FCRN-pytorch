import cv2 as cv
import numpy as np
import os

'''裁剪数据集（训练中未使用）'''

store_dir = "G:\\football-game-data\\low_resolution\\"
im = cv.imread("G:\\football-game-data\\new\\1\\1-frame17753.rdc_2.png")

data_path = "G:\\football-game-data\\new\\7"

files = os.listdir(data_path)

for f in files:

    if f.endswith(".jpg"):
        print(os.path.join(data_path,f))
        img = cv.imread(os.path.join(data_path, f))
        [h, w, c] = img.shape
        img = img[int(0.25 * h):int(0.5 * h), int(0.25 * w):int(0.5 * w), :]
        cv.imwrite(os.path.join(store_dir, f), img)

    if f.endswith(".png"):
        print(os.path.join(data_path,f))
        dpm = cv.imread(os.path.join(data_path,f))
        [h, w, c] = dpm.shape
        dpm = dpm[int(0.25 * h):int(0.5 * h), int(0.25 * w):int(0.5 * w), :]
        cv.imwrite(os.path.join(store_dir,f),dpm)
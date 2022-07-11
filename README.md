# FCRN-pytorch
DIP课程项目，基于深度学习的单目深度估计


---
###一 这里是本项目运行的说明
###### 1 主要运行环境
> python 3.9 \
> pytorch 1.8.0 gpu版本 \
> opencv-python 4.5 \
> numpy 等科学计算库
###### 2 数据准备部分
    python genDataTxt.py
> 首先进入data_process目录，点击genDataTxt.py,根据文件中给的
> 提示可以在script目录下生成用于训练的.txt文件。

###### 3 模型训练部分
    python train.py
> (1) 点击train.py,根据文件中的提示设置模型类型和数据集加载对应的.txt文件,
> 运行该文件即可开始训练。\
> (2) 训练时可以在results 目录下实时观察每个验证集的ground truth和
> 预测结果。

###### 4 模型评估部分
    python test_single.py
    python test.py
> test_single.py 是对单张图片进行预测 \
> test.py 是对测试集中多张图片进行预测并输出REL和MSE指标 \
> test_single.py 会将运行完的结果保存在项目根目录下，便于观察结果
---
###注意：如果使用者只需要简单的demo，不关注具体的训练，可以运行以下指令
    python video_test.py
> video_test.py输出运行视频，进行实时预测。
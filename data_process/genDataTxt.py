import os

# 本文件生成数据集的加载模块
# root path: 数据集的根路径
# dir path: 数据集的分类文件夹
# img index: 图像的编号

# 生成包含数据路径的所有txt文件

def getDataTxt(ds_root_path = "G:/train_set/", train_set_path = "", valid_set_path = ""):

    f_txt = open("../script/train.txt","w")
    ds_dir_name = os.listdir(ds_root_path)
    ds_file_name = []
    for i in ds_dir_name:
        ds_file_name =  os.listdir(ds_root_path+i)
        for j in ds_file_name:
            path = ds_root_path+str(i)+"/"+str(j)
            if path.endswith(".jpg") :
                f_txt.write(path+" ")
            elif path.endswith(".png") :
                f_txt.write(path+"\n")
            else:
                print("[E] path error occurs !!!")

getDataTxt()







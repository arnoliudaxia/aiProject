# coding: utf-8
import os #导入os库，用于文件处理

rootpath=r"C:\Users\Arnoliu\Desktop\快速临时处理文件夹\Si100bAIproject\train"
for i in range(10):
    image_filename = os.listdir(rootpath+"\\"+str(i))
    for j in image_filename:
        prefix=rootpath+"\\"+str(i)+"\\"
        os.rename(prefix+j,prefix+j.replace(" ",''))

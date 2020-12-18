# coding: utf-8
import os #导入os库，用于文件处理

rootpath=r"C:\Users\Arnoliu\Desktop\快速临时处理文件夹\Si100bAIproject\train"
f=open(rootpath+"\\train_list.txt","a")
for i in range(10):
    image_filename = os.listdir(rootpath+"\\"+str(i))
    for j in image_filename:
        f.write(str(i)+"/"+j+" "+str(i))
        f.write("\n")

f.close()
import os
from classModel import check

serperateSign="\\"
imgpath=[]
rootpath=r"C:\Users\Arnoliu\Desktop\快速临时处理文件夹\Si100bAIproject\test"
for sublabel in range(10):
    for img in os.listdir(rootpath+"/"+str(sublabel)):
        imgpath.append(rootpath+serperateSign+str(sublabel)+serperateSign+img)
imgNumber=len(imgpath)
corrects=0
for img in imgpath:
    if check(img)==int(img[img.rindex("\\")+1]):
        corrects+=1
print("测试数目为：{0}，正确率为：{1}".format(imgNumber,corrects/imgNumber))
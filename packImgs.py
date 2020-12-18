#code utf-8
import json
from main import *
##DONE：20201218：每个循环抽象成了函数并且添加了注释

#预定义变量部分
rootpath=r"C:\Users\Arnoliu\Desktop\快速临时处理文件夹\Si100bAIproject\完整数据集\train"#存放三个文件夹的父文件夹路径
trainpath= rootpath+"\\train"
valpath=rootpath+"\\val"
testpath=rootpath+"\\test"
jsonpath=r"./thousand.json"

#程序开始
data=[[],[],[]]
train_set=[[],[]]
val_set=[[],[]]
test_set=[[],[]]



def imgdataAndLabel(rootURL:str):
    dataset=[[],[]]
    for sublabel in range(10):#遍历十个文件夹
        for img in os.listdir(rootURL + "/" + str(sublabel)):#返回每个文件夹下的图片路径
            im = Image.open(rootURL + "/" + str(sublabel) + "/" + img).convert('L')#打开每个图片
            im = im.resize((75, 75), Image.ANTIALIAS)#resize图片数据，只有符合训练集数据格式的数据才能通过
            im = np.array(im).astype(np.float32).reshape(5625)#坍缩成一位数组
            im = 1.0 - im / 255.#归一化
            im = im.tolist()#将numpy变量化为python list类型，只有python的原有类型才能json.dumps

            dataset[0].append(im)
            dataset[1].append(sublabel)
    return dataset

data[0]= imgdataAndLabel(trainpath)
data[1]=imgdataAndLabel(valpath)
data[2]=imgdataAndLabel(testpath)

with open(jsonpath,"w") as f:
    f.write(json.dumps(data))


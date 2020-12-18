import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
import os
from PIL import Image


import json
import random

place = paddle.fluid.core_avx.CUDAPlace
# 定义数据集读取器
def load_data(mode='train'):

    # 数据文件

    data = json.load(open("./new_mnist.json"))
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]

    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


# 多层卷积神经网络实现
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
        # 激活函数使用relu
        self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 定义一层全连接层，输出维度是1，不使用激活函数
        self.fc = Linear(input_dim=6480, output_dim=10, act="softmax")

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


# 仅修改计算损失的函数，从均方误差（常用于回归问题）到交叉熵误差（常用于分类问题）
def trainModel():

    with fluid.dygraph.guard():
        model = MNIST()
        model.train()
        # 调用加载数据的函数
        train_loader = load_data('train')
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
        EPOCH_NUM = 5
        for epoch_id in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                # 准备数据，变得更加简洁
                image_data, label_data = data
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)

                # 前向计算的过程
                predict = model(image)

                # 计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
                loss = fluid.layers.cross_entropy(predict, label)
                avg_loss = fluid.layers.mean(loss)

                # 每训练了200批次的数据，打印下当前Loss的情况
                if batch_id % 200 == 0:
                    print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

                # 后向传播，更新参数的过程
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()

        # 保存模型参数
        fluid.save_dygraph(model.state_dict(), 'mnist')

# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    #im.show()
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)

    # 图像归一化
    im = 1.0 - im / 255.
    return im


# 定义预测过程
def check():
    with fluid.dygraph.guard():
        model = MNIST()
        params_file_path = 'mnist'
        img_path = './testimg/7004 _ 79.png'
        # 加载模型参数
        model_dict, _ = fluid.load_dygraph("mnist")
        model.load_dict(model_dict)

        model.eval()
        tensor_img = load_image(img_path)
        # 模型反馈10个分类标签的对应概率
        results = model(fluid.dygraph.to_variable(tensor_img))
        # 取概率最大的标签作为预测输出
        lab = np.argsort(results.numpy())
        print("本次预测的数字是: ", lab[0][-1])
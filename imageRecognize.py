# coding: utf-8
import sys, os
from PIL import Image
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from collections import OrderedDict

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 监督数据就是正确标签的内容，正确位置是1，其他位置为0
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        print self.loss
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class Network:

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = np.load('W1.npy')            #weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.load('b1.npy')            #np.zeros(hidden_size1)
        self.params['W2'] = np.load('W2.npy')            #weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.load('b2.npy')            #np.zeros(hidden_size2)
        self.params['W3'] = np.load('W3.npy')            #weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.load('b3.npy')            #np.zeros(output_size)
        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        return grads
    def startTraining(self, iter, learning_rate, x_train, t_train):
        train_size = x_train.shape[0]
        batch_size = 100
        learning_rate = learning_rate
        for i in range(iter):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            # 通过误差反向传播法求梯度
            grad = self.gradient(x_batch, t_batch)
            # 更新
            for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                self.params[key] -= learning_rate * grad[key]



def read_picture(pathIndex, n_C):
        path="C:\Users\\13302\Desktop\\train\\train\\"+str(pathIndex)
        print path
        import matplotlib.pyplot as plt
        # function:读取path路径下的图片，并转为形状为[m,n_H,n_W,n_C]的数组
        # path:str,图片所在路径
        # n_C:int,图像维数，黑白图像输入1，rgb图像输入3
        # datas：返回维度为（m，n_H,n_W,n_C）的array（数组）矩阵
        datas = []
        x_dirs = os.listdir(path)
        for x_file in x_dirs:
            fpath = os.path.join(path, x_file)
            if n_C == 1:
                _x = Image.open(fpath).convert("L")
                plt.imshow(_x,"gray")   #显示图像(只显示最后一张)
            elif n_C == 3:
                _x = Image.open(fpath)
                plt.imshow(_x)         #显示图像(只显示最后一张)
            else:
                print("错误：图像维数错误")
            n_W = _x.size[0]
            n_H = _x.size[1]
            datas.append(np.array(_x))
            _x.close()
        datas = np.array(datas)

        m = datas.shape[0]
        datas = datas.reshape((m, n_H*n_W))
        datas = datas/255
        identity = np.identity(12)
        data_label = identity[pathIndex-1]
        data_labels = np.array([data_label]*m)

        data_trains = datas[0:500]
        data_train_labels = data_labels[0:500]
        data_tests = datas[500:620]
        data_test_labels = data_labels[500:620]
        return data_trains, data_train_labels, data_tests, data_test_labels
if __name__ == '__main__':
    x_train, t_train, x_test, t_test = read_picture(1, 1)
    for i in range(1, 12):
        x_train_temp, t_train_temp, x_test_temp, t_test_temp = read_picture(i+1, 1)
        x_train = np.concatenate((x_train, x_train_temp), axis=0)
        t_train = np.concatenate((t_train, t_train_temp), axis=0)
        x_test = np.concatenate((x_test, x_test_temp), axis=0)
        t_test = np.concatenate((t_test, t_test_temp), axis=0)
    print x_train.shape
    print t_train.shape
    print x_test.shape
    print t_test.shape
    netWork = Network(784, 10000, 100, 12)
    netWork.startTraining(10, 0.01, x_train, t_train)
    # 保存训练结构
    np.save("W1", netWork.params['W1'])
    np.save("b1", netWork.params['b1'])
    np.save('W2', netWork.params['W2'])
    np.save('b2', netWork.params['b2'])
    np.save('W3', netWork.params['W3'])
    np.save('b3', netWork.params['b3'])
    # 测试准确率
    # train_size = x_train.shape[0]
    # batch_size = 100
    # batch_mask = np.random.choice(train_size, batch_size)
    # x_batch = x_train[batch_mask]
    # t_batch = t_train[batch_mask]
    accuracy = netWork.accuracy(x_test, t_test)
    print "accuracy"
    print accuracy
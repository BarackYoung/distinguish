#!/usr/bin/python
# -*- coding:utf8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

class ImageSort:
    def __init__(self, input_size=784, hidden_size=1000, hidden_size2=100, output_size=12):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.params = {'W1': np.random.random((input_size, hidden_size)),
                       'B1': np.zeros(hidden_size),
                       'W2': np.random.random((hidden_size, hidden_size2)),
                       'B2': np.zeros(hidden_size2),
                       'W3': np.random.random((hidden_size2, output_size)),
                       'B3': np.zeros(output_size)
                       }


    @staticmethod
    def softmax(x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)  # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))
    def predict(self, x):
        # 预测函数，输出预测值序列
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params["W3"]
        b1, b2, b3 = self.params['B1'], self.params['B2'], self.params["B3"]
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.softmax(a3)
        return y

    def accuracy(self, x, t):
        y = self.predict(x)
        print y[0]
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

    @staticmethod
    def lost_function(y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['B1'], self.params['B2'], self.params['B3']
        grads = {}
        batch_num = x.shape[0]
        print batch_num
        # forward
        print x
        a1 = np.dot(x, W1) + b1
        print "a1", a1
        z1 = self.sigmoid(a1)
        print "z1", z1
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.softmax(a3)
        print y.shape
        print "y", y
        print t.shape
        # backward
        dy = (y - t) / batch_num
        print "dy", dy
        print "z1.T", z1.T
        grads['W3'] = np.dot(z2.T, dy)
        grads["B3"] = np.sum(dy, axis=0)

        da2 = np.dot(dy, W3.T)
        dz2 = self.sigmoid_grad(a2) * da2

        grads['W2'] = np.dot(z1.T, dz2)
        grads['B2'] = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, W2.T)
        print "da1", da1
        dz1 = self.sigmoid_grad(a1) * da1
        print "a1", a1
        print "dz1", dz1
        grads['W1'] = np.dot(x.T, dz1)
        grads['B1'] = np.sum(dz1, axis=0)
        return grads

    def read_picture(self, pathIndex, n_C):
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
            # 若要对图像进行放大缩小，激活（去掉注释）以下函数
            '''
            rat=0.8          #放大/缩小倍数
            n_W=int(rat*n_W)
            n_H=int(rat*n_H)
            _x=_x.resize((n_W,n_H))  #直接给n_W,n_H赋值可将图像变为任意大小
            '''
            datas.append(np.array(_x))
            _x.close()
        datas = np.array(datas)

        m = datas.shape[0]
        datas = datas.reshape((m, n_H*n_W))
        datas = datas/255
        identity = np.identity(12)
        data_label = identity[pathIndex-1]
        data_labels = np.array([data_label]*m)
        return datas, data_labels
    def start_training(self, lr, epoch):
        for i in range(epoch):
            for m in range(12):
                  data_train, data_label = self.read_picture(m + 1, 1)
                  row_rand_array = np.arange(data_train.shape[0])
                  np.random.shuffle(row_rand_array)
                  row_train = data_train[row_rand_array[0:2]]
                  row_label = data_label[row_rand_array[0:2]]
                  print "row_train"
                  print row_train
                  print "row_label"
                  print row_label
                  grads = self.gradient(row_train, row_label)
                  for key in ('W1', 'B1', 'W2', 'B2', 'W3', 'B3'):
                      self.params[key] -= lr * grads[key]
                  print grads
            print self.params
if __name__ == '__main__':
    imageSort = ImageSort()
    imageSort.start_training(0.1, 1)
    # 训练结束，测试精度
    for i in range(12):
        x, t = imageSort.read_picture(i+1, 1)
        accuracy = imageSort.accuracy(x, t)
        print "accuracy"+str(i+1)+":"+str(accuracy);

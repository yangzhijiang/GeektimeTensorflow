# -*- coding: utf-8 -*-
# @Time    : 2019/2/13 23:33
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : mnistdataset.py

import matplotlib.pyplot as plt
from keras.datasets import mnist


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    fig = plt.figure()
    for i in range(15):
        plt.subplot(3, 5, i + 1)                # 3x5子图形式
        plt.tight_layout()                      # 自动适配子图尺寸
        plt.imshow(X_train[i], cmap='Greys')    # 灰度显示
        plt.title('Label {}'.format(Y_train[i]))
        plt.xticks([])
        plt.yticks([])                          # 删除x，y轴标记
    plt.show()

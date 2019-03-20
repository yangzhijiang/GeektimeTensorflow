# -*- coding: utf-8 -*-
# @Time    : 2019/2/16 17:20
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : mnistsoftmax.py


import os

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from tensorflow import gfile


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # 将数据变换为二维矩阵
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # 数据归一化
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # 标签数量统计
    label, count = np.unique(Y_train, return_counts=True)
    plt.figure()
    plt.bar(label, count, width=0.7, align='center')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(label)
    plt.ylim(0, 7500)
    for l, c in zip(label, count):
        plt.text(l, c, '%d' % c, ha='center', va='bottom', fontsize=10)
    plt.show()

    # 对标签进行One-Hot编码
    n_classes = 10
    print('Shape Before One-Hot Encoding:', Y_train.shape)
    Y_train = np_utils.to_categorical(Y_train, n_classes)
    print('Shape After One-Hot Encoding:', Y_train.shape)
    Y_test = np_utils.to_categorical(Y_test, n_classes)

    # 构建网络
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # 训练模型
    history = model.fit(
        X_train,
        Y_train,
        batch_size=128,
        epochs=5,
        verbose=2,
        validation_data=(X_test, Y_test)
    )

    # 训练可视化
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.tight_layout()
    plt.show()

    # 存储模型
    save_dir = '../model/mnist/'
    if gfile.Exists(save_dir):
        gfile.DeleteRecursively(save_dir)
    gfile.MakeDirs(save_dir)

    model_name = 'keras_mnist.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved Trained Model At %s ' % model_path)

    # 加载模型
    model = load_model(model_path)

    # 统计模型在测试集上的结果
    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
    print('Test Loss: {}'.format(loss_and_metrics[0]))
    print('Test Accuracy: {}%'.format(loss_and_metrics[1] * 100))


# -*- coding: utf-8 -*-
# @Time    : 2019/2/16 19:24
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : mnistcnn.py

import os

import matplotlib.pyplot as plt
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from tensorflow import gfile


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # 数据规范化
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # 数据归一化
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'Train Samples')
    print(X_test.shape[0], 'Test Samples')

    # 对标签进行One-Hot编码
    n_classes = 10
    print('Shape Before One-Hot Encoding:', Y_train.shape)
    Y_train = np_utils.to_categorical(Y_train, n_classes)
    print('Shape After One-Hot Encoding:', Y_train.shape)
    Y_test = np_utils.to_categorical(Y_test, n_classes)

    # 定义网络结构
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    print(model.summary())

    for layer in model.layers:
        print(layer.get_output_at(0).get_shape().as_list())

    # 编译模型
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # 训练模型
    history = model.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=5,
        verbose=2,
        validation_data=(X_test, Y_test),
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

# -*- coding: utf-8 -*-
# @Time    : 2019/3/12 18:29
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : dataprocess.py

import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER    # 验证码字符集
CAPTCHA_LEN = 4             # 验证码长度
CAPTCHA_HEIGHT = 60         # 验证码高度
CAPTCHA_WIDTH = 160         # 验证码宽度

TRAIN_DATA_DIR = '../data/chapter6/train/'  # 训练集数据路径


def plot_images(images, texts, cmap=None):
    """绘制数据图片"""
    plt.figure()
    for i in range(20):
        plt.subplot(5, 4, i + 1)  # 5行4列显示
        plt.tight_layout()  # 自动适配子图
        if cmap:
            plt.imshow(images[i], cmap=cmap)
        else:
            plt.imshow(images[i])
        plt.title('Label: {}'.format(texts[i]))  # 设置标题
        plt.xticks([])  # 删除x轴标记
        plt.yticks([])  # 删除y轴标记
    plt.show()


def rgb2gray(images):
    """将RGB图像转为灰度图"""
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])


def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPTCHA_WIDTH):
    """适配Keras图像数据格式"""
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    return batch, input_shape


def text2vec(text, length=CAPTCHA_LEN, charset=CAPTCHA_CHARSET):
    """对验证码中的每个字符进行One-Hot编码"""
    # 长度校验
    if length != len(text):
        raise ValueError('Error: length should be {}, but got {}'.format(length, len(text)))

    # 生成长度为 CAPTCHA_LEN * CAPTCHA_CHARSET 的一维向量
    vector = np.zeros(length * len(charset))
    for i in range(length):
        vector[charset.index(text[i]) + i * len(charset)] = 1
    return vector


def vec2text(vector):
    """将验证码向量解码为对应标签"""
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text


if __name__ == '__main__':
    # 读取前100张图片并通过文件名解析标签
    images, texts = [], []
    count = 0
    for filename in glob.glob(TRAIN_DATA_DIR + '*.png'):
        images.append(np.array(Image.open(filename)))
        texts.append(filename[-8:-4])
        count += 1
        if count >= 100:
            break
    plot_images(images, texts)
    images = np.array(images, dtype=np.float32)
    print('Origin Images Shape:', images.shape)

    # 将图像转为灰度图
    images = rgb2gray(images)
    plot_images(images, texts, cmap='Greys')
    print('After Gray Images Shape:', images.shape)

    # 数据规范化
    images = images / 255
    print(images[0])

    # 转换为Keras数据格式
    images, input_shape = fit_keras_channels(images)
    print('Image Shape:', images.shape)
    print('Input Shape:', input_shape)

    # 对标签进行编码
    texts = list(texts)
    vectors = [None] * len(texts)
    for i in range(len(texts)):
        vectors[i] = text2vec(texts[i])
    print(texts[1])
    print(vectors[1])

    # 模型对于 3935 的预测向量
    pred_vector = np.array([[2.0792404e-10, 4.3756086e-07, 3.1140310e-10, 9.9823320e-01,
                             5.1135743e-15, 3.7417038e-05, 1.0556480e-08, 9.0933657e-13,
                             2.7573466e-07, 1.7286760e-03, 1.1030550e-07, 1.1852034e-07,
                             7.9457263e-10, 3.4533365e-09, 6.6065012e-14, 2.8996323e-05,
                             7.6345885e-13, 3.1817032e-16, 3.9540555e-05, 9.9993122e-01,
                             5.3814397e-13, 1.2061575e-10, 1.6408040e-03, 9.9833637e-01,
                             6.5149628e-08, 5.2246549e-12, 1.1365444e-08, 9.5700288e-12,
                             2.2725430e-05, 5.2195204e-10, 3.2457771e-13, 2.1413280e-07,
                             7.3547295e-14, 4.4094882e-06, 3.8390007e-07, 9.9230206e-01,
                             6.4467136e-03, 3.9224533e-11, 1.2461344e-03, 1.1253484e-07]],
                           dtype=np.float32)
    pred_text = vec2text(pred_vector)
    print(pred_text)

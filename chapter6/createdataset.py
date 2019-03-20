# -*- coding: utf-8 -*-
# @Time    : 2019/3/12 17:36
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : createdataset.py

import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import gfile
from captcha.image import ImageCaptcha
from PIL import Image


NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWERCASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
UPPERCASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARSET = NUMBER    # 验证码字符集
CAPTCHA_LEN = 4             # 验证码长度
CAPTCHA_HEIGHT = 60         # 验证码高度
CAPTCHA_WIDTH = 160         # 验证码宽度

TRAIN_DATASET_SIZE = 5000                       # 训练数据集大小
TEST_DATASET_SIZE = 1000                        # 测试数据集大小
TRAIN_DATA_DIR = '../data/chapter6/train/'      # 训练集数据路径
TEST_DATA_DIR = '../data/chapter6/test/'        # 测试集数据路径


def gen_random_text(charset=CAPTCHA_CHARSET, length=CAPTCHA_LEN):
    """生成随机字符"""
    text = [random.choice(charset) for _ in range(length)]
    return ''.join(text)


def create_captcha_dataset(size=100, data_dir='./data/', height=60, width=160, image_format='.png'):
    """创建并保存验证码数据集"""
    # 清空存储目录并重新创建
    if gfile.Exists(data_dir):
        gfile.DeleteRecursively(data_dir)
    gfile.MakeDirs(data_dir)

    # 创建ImageCaptcha实例
    captcha = ImageCaptcha(width=width, height=height)

    for _ in range(size):
        # 生成随机验证码
        text = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LEN)
        captcha.write(text, data_dir + text + image_format)


def gen_captcha_dataset(size=100, height=60, width=160):
    """生成并返回验证码数据集"""
    # 创建ImageCaptcha实例
    captcha = ImageCaptcha(width=width, height=height)

    # 创建图像和文本数组
    images, texts = [None] * size, [None] * size
    for i in range(size):
        # 生成随机验证码
        texts[i] = gen_random_text(CAPTCHA_CHARSET, CAPTCHA_LEN)
        # 将生成的验证码使用PIL打开并转换成Numpy数组
        images[i] = np.array(Image.open(captcha.generate(texts[i])))
    return images, texts


def plot_captcha_dataset():
    images, texts = gen_captcha_dataset()
    print('Image Size: ', images[0].shape)

    plt.figure()
    for i in range(20):
        plt.subplot(5, 4, i + 1)                    # 5行4列显示
        plt.tight_layout()                          # 自动适配子图
        plt.imshow(images[i])
        plt.title('Label: {}'.format(texts[i]))     # 设置标题
        plt.xticks([])                              # 删除x轴标记
        plt.yticks([])                              # 删除y轴标记
    plt.show()


if __name__ == '__main__':
    # create_captcha_dataset(TRAIN_DATASET_SIZE, TRAIN_DATA_DIR)
    # create_captcha_dataset(TEST_DATASET_SIZE, TEST_DATA_DIR)
    plot_captcha_dataset()



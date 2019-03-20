# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 14:17
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : trainningmodel.py

import glob
import pickle

import numpy as np
from PIL import Image
from tensorflow import gfile
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense, Concatenate
from keras.utils.vis_utils import plot_model


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
TEST_DATA_DIR = '../data/chapter6/test/'    # 测试集数据路径

BATCH_SIZE = 128                # Batch大小
EPOCHS = 50                     # 训练轮数
OPT = 'adadelta'                    # 优化器
LOSS = 'binary_crossentropy'    # 损失函数

MODEL_DIR = '../model/captcha/'         # 模型文件路径
MODEL_FORMAT = '.h5'                    # 模型文件类型
HISTORY_DIR = '../history/captcha/'     # 训练记录文件路径
HISTORY_FORMAT = '.history'             # 训练记录文件类型

FILENAME = '{}captcha_{}_{}_bs_{}_epochs_{}{}'

# 神经网络结构文件
MODEL_VIS_FILE = 'captcha_classfication.png'
# 模型文件
MODEL_FILE = FILENAME.format(MODEL_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), MODEL_FORMAT)
# 训练记录文件
HISTORY_FILE = FILENAME.format(HISTORY_DIR, OPT, LOSS, str(BATCH_SIZE), str(EPOCHS), HISTORY_FORMAT)


def rgb2gray(images):
    """将RGB图像转为灰度图"""
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])


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


def fit_keras_channels(batch, rows=CAPTCHA_HEIGHT, cols=CAPTCHA_WIDTH):
    """适配Keras图像数据格式"""
    if K.image_data_format() == 'channels_first':
        batch = batch.reshape(batch.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        batch = batch.reshape(batch.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)
    return batch, input_shape


def preprocess(data_dir):
    # 从指定目录获取数据并进行预处理
    X, y = [], []
    for filename in glob.glob(data_dir + '*.png'):
        X.append(np.array(Image.open(filename)))
        y.append(filename[-8:-4])

    # 转换为Numpy数组
    X = np.array(X, dtype=np.float32)
    # 转换为灰度图
    X = rgb2gray(X)
    # 对训练数据标准化
    X = X / 255
    # 根据Keras图像格式进行通道处理
    X, input_shape = fit_keras_channels(X)

    y = list(y)
    # 将标签转化为向量
    for i in range(len(y)):
        y[i] = text2vec(y[i])
    y = np.asarray(y)

    return X, y, input_shape


def build_model(input_shape=(60, 160, 1)):
    # 输入层
    inputs = Input(shape=input_shape, name='inputs')

    # 第1层卷积
    conv1 = Conv2D(32, (3, 3), name='conv1')(inputs)
    relu1 = Activation('relu', name='relu1')(conv1)

    # 第2层卷积
    conv2 = Conv2D(32, (3, 3), name='conv2')(relu1)
    relu2 = Activation('relu', name='relu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool2')(relu2)

    # 第3层卷积
    conv3 = Conv2D(64, (3, 3), name='conv3')(pool2)
    relu3 = Activation('relu', name='relu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same', name='pool3')(relu3)

    # 将上面得到的Tensor摊平
    x = Flatten()(pool3)

    # Dropout
    x = Dropout(0.25)(x)

    # 4个全连接层分别做10分类对应4个字符
    x = [Dense(10, activation='softmax', name='fc%d' % (i + 1))(x) for i in range(4)]

    # 将4个向量进行拼接，作为模型输出
    outs = Concatenate()(x)

    model = Model(inputs=inputs, outputs=outs)
    model.compile(optimizer=OPT, loss=LOSS, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # 获取预处理后的数据
    X_train, y_train, input_shape = preprocess(TRAIN_DATA_DIR)
    X_test, y_test, _ = preprocess(TEST_DATA_DIR)
    print('Train Shape:', X_train.shape)
    print('Test Shape:', X_test.shape)
    print('Input Shape:', input_shape)

    # 构建模型
    model = build_model(input_shape)
    model.summary()
    plot_model(model, to_file=MODEL_VIS_FILE, show_shapes=True)

    # 模型训练
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_data=(X_test, y_test)
    )

    # 测试样例
    print(vec2text(y_test[9]))
    test_predict = model.predict(X_test[9].reshape(1, 60, 160, 1))
    print(vec2text(test_predict))

    # 保存模型
    if not gfile.Exists(MODEL_DIR):
        gfile.MakeDirs(MODEL_DIR)
    model.save(MODEL_FILE)
    print('Save Model at %s' % MODEL_FILE)

    # 保存训练记录
    if not gfile.Exists(HISTORY_DIR):
        gfile.MakeDirs(HISTORY_DIR)
    with open(HISTORY_FILE, 'wb') as f:
        pickle.dump(history.history, f)
    print(HISTORY_FILE)

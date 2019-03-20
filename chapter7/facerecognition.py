# -*- coding: utf-8 -*-
# @Time    : 2019/3/20 13:21
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : facerecognition.py

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
from keras.utils.vis_utils import plot_model

from align import AlignDlib
from model import create_model


class IdentityMetadata(object):
    def __init__(self, base, name, file):
        self.base = base    # 数据集根目录
        self.name = name    # 目录名称
        self.file = file    # 文件名称

    def get_image_path(self):
        return os.path.join(self.base, self.name, self.file)

    def __repr__(self):
        return self.get_image_path()


def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV默认使用BGR通道，转换为RGB通道
    return img[..., ::-1]


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a - p), axis=-1)
        n_dist = K.sum(K.square(a - n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def plot_sample():
    metadata = load_metadata('../data/chapter7/images/')

    # 初始化OpenFace人脸对齐工具，使用Dlib提供的68个关键点
    alignment = AlignDlib('../data/chapter7/landmarks.dat')
    # 加载一张训练图像
    img = load_image(metadata[45].get_image_path())
    # 检测人脸并返回边框
    bounding_box = alignment.getLargestFaceBoundingBox(img)
    # 使用指定的人脸关键点转换图像并截取96*96的人脸图像
    aligned_img = alignment.align(96, img, bounding_box, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    # 绘制原图
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    # 绘制带人脸边框的原图
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.gca().add_patch(patches.Rectangle(
        (bounding_box.left(), bounding_box.top()),
        bounding_box.width(),
        bounding_box.height(),
        fill=False,
        color='red'
    ))
    plt.xticks([])
    plt.yticks([])

    # 绘制对齐后截取的96*96人脸图像
    plt.subplot(1, 3, 3)
    plt.imshow(aligned_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def create_custom_model():
    # 创建模型
    nn4_small2 = create_model()
    # 输入Anchor、Positive、Negative 96*96 RGB图像
    in_a = Input(shape=(96, 96, 3))
    in_p = Input(shape=(96, 96, 3))
    in_n = Input(shape=(96, 96, 3))
    # 输出对应的人脸特征向量
    emb_a = nn4_small2(in_a)
    emb_p = nn4_small2(in_p)
    emb_n = nn4_small2(in_n)
    plot_model(nn4_small2, to_file='../plot/nn4_small2_model.png', show_shapes=True)

    # 添加TripletLoss层
    triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])
    nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)
    plot_model(nn4_small2_train, to_file='../plot/nn4_small2_train.png', show_shapes=True)


def create_pretrain_model():
    nn4_small2 = create_model()
    nn4_small2.load_weights('../model/facenet/nn4.small2.v1.h5')
    alignment = AlignDlib('../data/chapter7/landmarks.dat')

    def align_image(img):
        return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                               landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    metadata = load_metadata('../data/chapter7/images/')
    embedded = np.zeros((metadata.shape[0], 128))

    for i, m in enumerate(metadata):
        img = load_image(m.get_image_path())
        img = align_image(img)
        # 数据规范化
        img = (img / 255.).astype(np.float32)
        # 获取人脸特征向量
        embedded[i] = nn4_small2.predict(np.expand_dims(img, axis=0))[0]
        print('Process', i, m, 'Finish')

    def distance(emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

    def show_pair(idx1, idx2):
        plt.figure(figsize=(8, 3))
        plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
        plt.subplot(121)
        plt.imshow(load_image(metadata[idx1].get_image_path()))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(load_image(metadata[idx2].get_image_path()))
        plt.xticks([])
        plt.yticks([])

    show_pair(1, 45)
    show_pair(28, 47)
    show_pair(46, 47)
    plt.show()
    return embedded


def predict_face():
    # embedded = create_pretrain_model()
    # np.save('emmbedded.npy', embedded)

    embedded = np.load('emmbedded.npy')
    metadata = load_metadata('../data/chapter7/images/')
    targets = np.array([m.name for m in metadata])

    encoder = LabelEncoder()
    encoder.fit(targets)
    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 2 != 0
    test_idx = np.arange(metadata.shape[0]) % 2 == 0

    X_train, X_test = embedded[train_idx], embedded[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    svc = LinearSVC()
    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))
    print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')

    example_idx = 11
    example_image = load_image(metadata[example_idx].get_image_path())
    example_prediction = svc.predict([embedded[example_idx]])
    example_identity = encoder.inverse_transform(example_prediction)[0]

    plt.imshow(example_image)
    plt.title(f'Recognized as {example_identity}');
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # plot_sample()
    # create_custom_model()
    predict_face()
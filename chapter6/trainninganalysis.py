# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 21:49
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : trainninganalysis.py

import glob
import pickle

import matplotlib.pyplot as plt


HISTORY_DIR = '../history/captcha/'     # 训练记录文件路径


def plot_training(history=None, metric='acc', title='Model Accuracy', loc='lower right'):
    model_list = []
    fig = plt.figure(figsize=(10, 8))
    for key, val in history.items():
        model_list.append(key.split('\\')[-1].rstrip('.history'))
        plt.plot(val[metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(model_list, loc=loc)
    plt.show()


if __name__ == '__main__':
    history = {}
    for filename in glob.glob(HISTORY_DIR + '*.history'):
        with open(filename, 'rb') as f:
            print(filename)
            history[filename] = pickle.load(f)

    plot_training(history)
    plot_training(history, metric='loss', title='Model Loss', loc='upper right')
    plot_training(history, metric='val_acc', title='Model Accuracy(val)')
    plot_training(history, metric='val_loss', title='Model Loss(val)', loc='upper right')

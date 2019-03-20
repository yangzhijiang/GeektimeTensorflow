# -*- coding: utf-8 -*-
# @Time    : 2019/2/10 14:35
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : dataprocess.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def normalize_feature(data):
    return data.apply(lambda col: (col - col.mean()) / col.std())


if __name__ == '__main__':
    multi_data = pd.read_csv('../data/chapter4/data1.csv', names=['square', 'bedrooms', 'price'])
    print(multi_data.head())

    single_data = pd.read_csv('../data/chapter4/data0.csv', names=['square', 'price'])
    sns.set(style='whitegrid', palette='dark')
    sns.lmplot('square', 'price', single_data, height=6, fit_reg=True)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('square')
    ax.set_ylabel('bedrooms')
    ax.set_zlabel('price')
    ax.scatter3D(multi_data['square'], multi_data['bedrooms'], multi_data['price'],
                 c=multi_data['price'], cmap='Greens')
    plt.show()

    proc_data = normalize_feature(multi_data)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('square')
    ax.set_ylabel('bedrooms')
    ax.set_zlabel('price')
    ax.scatter3D(proc_data['square'], proc_data['bedrooms'], proc_data['price'],
                 c=proc_data['price'], cmap='Reds')
    plt.show()

    ones_data = pd.DataFrame({'ones': np.ones(len(proc_data))})
    proc_data = pd.concat([ones_data, proc_data], axis=1)
    X_data = proc_data.iloc[:, :3].values
    Y_data = proc_data.iloc[:, -1].values.reshape(-1, 1)
    print(X_data.shape, type(X_data))
    print(Y_data.shape, type(Y_data))


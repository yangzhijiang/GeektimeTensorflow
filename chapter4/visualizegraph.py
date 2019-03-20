# -*- coding: utf-8 -*-
# @Time    : 2019/2/10 19:37
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : visualizegraph.py

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_feature(data):
    return data.apply(lambda col: (col - col.mean()) / col.std())


if __name__ == '__main__':
    multi_data = pd.read_csv('../data/chapter4/data1.csv', names=['square', 'bedrooms', 'price'])
    proc_data = normalize_feature(multi_data)
    ones_data = pd.DataFrame({'ones': np.ones(len(proc_data))})
    proc_data = pd.concat([ones_data, proc_data], axis=1)
    X_data = proc_data.iloc[:, :3].values
    Y_data = proc_data.iloc[:, -1].values.reshape(-1, 1)
    print(X_data.shape, type(X_data))
    print(Y_data.shape, type(Y_data))

    alpha = 1e-2    # 学习率
    epochs = 500    # 训练轮数

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, X_data.shape, name='X')
        Y = tf.placeholder(tf.float32, Y_data.shape, name='Y')

    with tf.name_scope('hypothesis'):
        W = tf.get_variable('weights', (X_data.shape[1], 1), initializer=tf.constant_initializer())
        Y_pred = tf.matmul(X, W, name='Y_pred')

    with tf.name_scope('loss'):
        loss_op = 1 / (2 * len(X_data)) * tf.matmul((Y_pred - Y), (Y_pred - Y), transpose_a=True)

    with tf.name_scope('train'):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('../summary/linear-regression-1', sess.graph)
        loss_data = []

        for epoch in range(1, epochs + 1):
            _, loss, w = sess.run([train_op, loss_op, W], feed_dict={X: X_data, Y: Y_data})
            loss_data.append(float(loss))
            if epoch % 10 == 0:
                log_str = 'Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g'
                print(log_str % (epoch, loss, w[1], w[2], w[0]))
    writer.close()

    sns.set(style='whitegrid', palette='dark')
    ax = sns.lineplot(x='epoch', y='loss', data=pd.DataFrame({
        'epoch': np.arange(epochs),
        'loss': loss_data
    }))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.show()

# -*- coding: utf-8 -*-
# @Time    : 2019/2/10 15:41
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : trainingmodel.py

import numpy as np
import pandas as pd
import tensorflow as tf


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

    X = tf.placeholder(tf.float32, X_data.shape)
    Y = tf.placeholder(tf.float32, Y_data.shape)
    W = tf.get_variable('weights', (X_data.shape[1], 1), initializer=tf.constant_initializer())
    Y_pred = tf.matmul(X, W)

    loss_op = 1 / (2 * len(X_data)) * tf.matmul((Y_pred - Y), (Y_pred - Y), transpose_a=True)
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = opt.minimize(loss_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            sess.run(train_op, feed_dict={X: X_data, Y: Y_data})
            if epoch % 10 == 0:
                loss, w = sess.run([loss_op, W], feed_dict={X: X_data, Y: Y_data})
                log_str = 'Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g'
                print(log_str % (epoch, loss, w[1], w[2], w[0]))


# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 21:18
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : main.py


import tensorflow as tf
import keras


if __name__ == '__main__':
    hello = tf.constant('Hello Tensorflow!')
    with tf.Session() as sess:
        print(sess.run(hello))
    print(tf.__version__)
    print(tf.keras.__version__)
    print(keras.__version__)

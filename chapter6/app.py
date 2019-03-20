# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 22:50
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : app.py

from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.models import load_model
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

MODEL_FILE = '../model/captcha/captcha_adam_binary_crossentropy_bs_128_epochs_10.h5'


def rgb2gray(images):
    """将RGB图像转为灰度图"""
    # Y' = 0.299 R + 0.587 G + 0.114 B
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])


def vec2text(vector):
    """将验证码向量解码为对应标签"""
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text


app = Flask(__name__)


@app.route('/ping', methods=['GET', 'POST'])
def hello_world():
    return 'pong'


@app.route('/predict', methods=['POST'])
def predict():
    response = {'success': False, 'prediction': '', 'debug': 'error'}

    received_image = False
    if request.method == 'POST':
        if request.files.get('image'):
            image = request.files['image'].read()
            received_image = True
            response['debug'] = 'get image'

        if received_image:
            image = np.array(Image.open(BytesIO(image)))
            image = rgb2gray(image).reshape(1, 60, 160, 1).astype('float32') / 255
            with graph.as_default():
                pred = model.predict(image)
            response['prediction'] = response['prediction'] + vec2text(pred)
            response['success'] = True
            response['debug'] = 'predicted'
    else:
        response['debug'] = 'No Post'
    return jsonify(response)


model = load_model(MODEL_FILE)
graph = tf.get_default_graph()


# set FLASK_ENV=development && flask run --host=0.0.0.0
# curl -X POST -F image=@9993.png 'http://localhost:5000/predict'

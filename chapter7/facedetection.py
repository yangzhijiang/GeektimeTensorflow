# -*- coding: utf-8 -*-
# @Time    : 2019/3/15 14:36
# @Author  : LunaFire
# @Email   : gilgemesh2012@gmail.com
# @File    : facedetection.py


import cv2
import matplotlib.pyplot as plt
import face_recognition


IMAGE_PATH = '../data/chapter7/test_face_detection.jpg'
CASC_PATH = '../data/chapter7/haarcascade_frontalface_default.xml'


def face_detect_cv3():
    face_cascade = cv2.CascadeClassifier(CASC_PATH)

    image = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print('Found {} faces!'.format(len(faces)))

    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.title('Faces Found')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def face_detect_fr():
    image = face_recognition.load_image_file(IMAGE_PATH)
    face_locations = face_recognition.face_locations(image)

    print('Found {} faces!'.format(len(face_locations)))

    image = cv2.imread(IMAGE_PATH)

    for top, right, bottom, left in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    plt.title('Faces Found')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    # face_detect_cv3()
    face_detect_fr()

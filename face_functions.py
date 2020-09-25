# -*- coding: utf-8 -*-
"""
face recognition defines using dlib

@author: John Watson and Damen Kelly
"""

import cv2
import dlib
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../dlib-models/shape_predictor_68_face_landmarks.dat")
recogniser = dlib.face_recognition_model_v1("../dlib-models/dlib_face_recognition_resnet_model_v1.dat")


def find(img):
    detections = detector(img)  # detect faces and store rectangles
    shape = predictor(img, detections[0])  # find the shape of the face (use first detected face)
    return detections[0], shape


def recognise(img):
    pos, shape = find(img)
    descriptor = recogniser.compute_face_descriptor(img, shape)
    return descriptor


def swap(img, img2):
    pos, shape = find(img)
    pos2, shape2 = find(img2)

    img3 = img.copy()

    subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))

    for point in shape.parts():
        cv2.circle(img, (point.x, point.y), 2, (0, 255, 0), -1)
        subdiv.insert((point.x, point.y))

    triangles = subdiv.getTriangleList()

    for triangle in triangles:
        point1 = (triangle[0], triangle[1])
        point2 = (triangle[2], triangle[3])
        point3 = (triangle[4], triangle[5])

        cv2.line(img, point1, point2, (255, 255, 255), 1)
        cv2.line(img, point2, point3, (255, 255, 255), 1)
        cv2.line(img, point3, point1, (255, 255, 255), 1)

    subdiv2 = cv2.Subdiv2D((0, 0, img2.shape[1], img2.shape[0]))

    for point in shape2.parts():
        cv2.circle(img2, (point.x, point.y), 2, (0, 255, 0), -1)
        subdiv2.insert((point.x, point.y))

    triangles = subdiv2.getTriangleList()

    for triangle in triangles:
        point1 = (triangle[0], triangle[1])
        point2 = (triangle[2], triangle[3])
        point3 = (triangle[4], triangle[5])

        cv2.line(img2, point1, point2, (255, 255, 255), 1)
        cv2.line(img2, point2, point3, (255, 255, 255), 1)
        cv2.line(img2, point3, point1, (255, 255, 255), 1)

    #cv2.getAffineTransform()
    #cv2.warpAffine()

    return img3
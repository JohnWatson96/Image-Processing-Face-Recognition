# -*- coding: utf-8 -*-
"""
Face recognition project

@author: John Watson and Damen Kelly
"""

import cv2
import numpy as np
import face_functions as face

Video = False

src_img = cv2.imread("John.jpg")
input_img = cv2.imread("Damen.jpg")

#person = face.recognise(img)
#person2 = face.recognise(img2)

cam = cv2.VideoCapture(0)

input_points, input_bbox = face.find(input_img)

if not Video:
    src_points, src_bbox = face.find(src_img)
    out_img = face.swap(src_img, src_points, src_bbox, input_img, input_points, input_bbox)

    src_img = face.draw(src_img)
    input_img = face.draw(input_img)

    cv2.imshow("src_img", src_img)
    cv2.imshow("input_img", input_img)
    cv2.imshow("out_img", out_img)
    cv2.waitKey(0)

else:
    while True:
        ret_val, frame = cam.read()
        frame_points, frame_bbox = face.find(frame)
        frame = face.swap(frame, frame_points, frame_bbox, input_img, input_points, input_bbox)

        #frame = face.draw(frame)

        cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
cv2.destroyAllWindows()


# PROJECT SCOPE
# Open webcam and UI, Button for learn and swap face to
# mode to do emotions
# mode to open mouth
# mode for gaze detection look and blink input into UI

# Face Swapper
# Emotion
# Spiffy UI
# Lip reading (detect if talking)
# Road to deepfakes
#
# Gaze detection
# Heart beat
# Lie Detection (goodluck) :P
# Poker bot 9000
# etc...
# Lip reading (detect if talking)
#
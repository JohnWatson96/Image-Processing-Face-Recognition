# -*- coding: utf-8 -*-
"""
Face recognition project

@author: John Watson and Damen Kelly
"""

import cv2
import numpy as np
import face_functions as face

img = cv2.imread("John.jpg")
img2 = cv2.imread("Damen.jpg")

img = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))

#person = face.recognise(img)
#person2 = face.recognise(img2)

cam = cv2.VideoCapture(0)
color_green = (0, 255, 0)
line_width = 3

img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
img3 = face.swap(img, img2)

out = np.hstack((img, img2, img3))
cv2.imshow("out", out)
cv2.waitKey(0)

while False:
    ret_val, frame = cam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



    dets = face.find(rgb_frame)
    for det in dets:
        #face.swap(img, img2, (det.left(), det.top()), (det.right(), det.bottom()))
        cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
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

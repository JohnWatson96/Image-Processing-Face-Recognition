# -*- coding: utf-8 -*-
"""
Face recognition project

@author: John Watson and Damen Kelly
"""

import dlib
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, \
    QPushButton, QVBoxLayout, QWidget


#GUI

class StreamWindow(QMainWindow):
  def initilise(self):
      self.setWindowTitle("Facial Recognition Project")
      self.initiliseUI()


    
  def Button1_pressed():  
      print('Option A')
      
  def Button2_pressed():  
      print('Option B')
      
  def Button3_pressed():  
      print('Option C')    
      
      
  def initiliseUI(self):
      self.Button1 = QPushButton('Option A', dock_widget)
      self.Button2 = QPushButton('Option B', dock_widget)
      self.Button3 = QPushButton('Option C', dock_widget)
      self.Button1.clicked.connect(self.Button1_pressed)
      self.Button2.clicked.connect(self.Button2_pressed)
      self.Button3.clicked.connect(self.Button2_pressed)



app = QApplication([]) 
stream = StreamWindow()
dock_widget = QWidget()




layout = QVBoxLayout(dock_widget)
layout.addWidget(Button1)
layout.addWidget(Button2)
layout.addWidget(Button3)

stream.setCentralWidget(dock_widget)
stream.show()




detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("../dlib-models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("../dlib-models/dlib_face_recognition_resnet_model_v1.dat")

img = dlib.load_rgb_image("John.jpg")
dets = detector(img, 1)
for d in dets:
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)

cam = cv2.VideoCapture(0)
color_green = (0, 255, 0)
line_width = 3

while True:
    ret_val, frame = cam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_frame)
    for det in dets:
        cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        app.exit(app.exec_())


app.exit(app.exec_())
cam.release()
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

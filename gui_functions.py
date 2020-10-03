# -*- coding: utf-8 -*-
"""
gui defines using PyQt5

@author: John Watson and Damen Kelly
"""

import cv2
import numpy as np
import face_functions as face
from PyQt5.QtWidgets import (QApplication, QMainWindow,
                             QPushButton, QWidget,
                             QGridLayout, QLabel)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot

#globals
recognise = False
swap = False
store = False

#initialisation
input_imgs = [cv2.imread("John.jpg"), cv2.imread("Damen.jpg"), cv2.imread("Laura.jpg"), cv2.imread("Andrew.jpg")]

input_pointses = np.zeros((len(input_imgs), 68, 2), dtype=np.uint32)
input_bboxs = np.zeros((len(input_imgs), 4, 2), dtype=np.uint32)
input_shapes = []

for index in range(0, len(input_imgs)):
    input_points, input_bbox, input_shape = face.find(input_imgs[index])
    input_pointses[index] = input_points
    input_bboxs[index] = input_bbox
    input_shapes.append(input_shape)

#input_descriptor = face.recognise(input_img, input_shape)  # WIP
#input_descriptors = [input_descriptor, input_descriptor]  # WIP


def frame_operation(frame, input_imgs, input_pointses, input_bboxs):

    global recognise
    global swap
    global store

    input_index = 1  # change this for different people
    input_img, input_points, input_bbox = input_imgs[input_index], input_pointses[input_index], input_bboxs[input_index]
    if recognise:
        print("you are")
        recognise = False
    if swap:
        frame_points, frame_bbox, _ = face.find(frame)
        frame = face.swap(frame, frame_points, frame_bbox, input_img, input_points, input_bbox)
    if store:
        print("store")
        store = False
    return frame


class Thread(QThread):
    changePixmapIn = pyqtSignal(QImage, name='In')
    changePixmapOut = pyqtSignal(QImage, name='Out')

    def run(self):
        cam = cv2.VideoCapture(0)
        while True:

            ret, frame = cam.read()

            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmapIn.emit(p)

            frame = frame_operation(frame, input_imgs, input_pointses, input_bboxs)

            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmapOut.emit(p)


# -----------Video Capture--------------------#

# GUI
class AppWindow(QMainWindow):

    # Define Button Presses
    def Button1_pressed(self):
        global recognise
        recognise = True

    def Button2_pressed(self):
        global swap
        swap = ~swap

    def Button3_pressed(self):
        global store
        store = True

        # Initilize main window

    def __init__(self, *args, **kwargs):
        super(AppWindow, self).__init__(*args, **kwargs)

        # Set Title and window dimensions
        self.setWindowTitle("Facial Recognition Project")

        # -----------Buttons-------------------------#
        # Create  buttons
        self.Button1 = QPushButton()
        self.Button2 = QPushButton()
        self.Button3 = QPushButton()
        # Label Buttons
        self.Button1.setText('Who am I?')
        self.Button2.setText('Swap my face!')
        self.Button3.setText('Remember me!')
        # Place buttons
        self.layout = QGridLayout()
        self.layout.addWidget(self.Button1, 3, 3)
        self.layout.addWidget(self.Button2, 4, 3)
        self.layout.addWidget(self.Button3, 5, 3)

        # Connect buttons
        self.Button1.clicked.connect(self.Button1_pressed)
        self.Button2.clicked.connect(self.Button2_pressed)
        self.Button3.clicked.connect(self.Button3_pressed)
        # -----------Images-------------------------#

        self.labelIn = QLabel(self)
        self.pixmapIn = QPixmap()
        self.resize(self.pixmapIn.width(), self.pixmapIn.height())
        self.layout.addWidget(self.labelIn, 0, 0, 2, 2)

        self.labelOut = QLabel(self)
        self.pixmapOut = QPixmap()
        self.resize(self.pixmapOut.width(), self.pixmapOut.height())
        self.layout.addWidget(self.labelOut, 0, 4, 2, 6)

        # Lay widgets location
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        th = Thread(self)
        th.changePixmapIn.connect(self.setImageIn)
        th.changePixmapOut.connect(self.setImageOut)
        th.start()

    # Image Update
    @pyqtSlot(QImage, name='In')
    def setImageIn(self, image):
        # Input image
        self.labelIn.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage, name='Out')
    def setImageOut(self, frame):
        # Output image
        self.labelOut.setPixmap(QPixmap.fromImage(frame))


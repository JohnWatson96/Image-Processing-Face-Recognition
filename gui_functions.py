# -*- coding: utf-8 -*-
"""
gui defines using PyQt5

@author: John Watson and Damen Kelly
"""

import cv2
import numpy as np
import face_functions as face
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, QInputDialog, QLineEdit, QGridLayout,
                             QLabel)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
import glob

#globals
recognise = False
swap = None
save = False
cam = 0

# initialisation
input_imgs = []
input_names = []
for file in glob.glob("*.jpg"):
    input_imgs.append(cv2.imread(file))
    input_names.append(file[:-4])

input_pointses = np.zeros((len(input_imgs), 70, 2), dtype=np.uint32)
input_bboxs = np.zeros((len(input_imgs), 4, 2), dtype=np.uint32)
input_shapes = []
input_descriptors = []

for index in range(0, len(input_imgs)):
    input_points, input_bbox, input_shape = face.find(input_imgs[index])
    input_pointses[index] = input_points
    input_bboxs[index] = input_bbox
    input_shapes.append(input_shape)
    input_descriptors.append(face.recognise(input_imgs[index], input_shape))


def closecam():
    global cam
    cam.release()
    return


def frame_operation(frame):

    global recognise
    global swap
    global save

    if save:
        cv2.imwrite(file, frame)
        save = False

    if recognise:
        _, frame_bbox, frame_shape = face.find(frame)
        if frame_bbox or frame_shape is not None:
            frame_descriptor = face.recognise(frame, frame_shape)
            name_index = face.match(frame_descriptor, input_descriptors)
            if name_index is not None:
                name = input_names[name_index]
                frame = face.draw(frame)
                cv2.putText(frame, name, tuple(frame_bbox[2]), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Not Recognised", (0, 480), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 2)

    if swap is not None:
        frame_points, frame_bbox, _ = face.find(frame)
        frame = face.swap(frame, frame_points, frame_bbox, input_imgs[swap], input_pointses[swap])

    return frame


class Thread(QThread):
    changePixmapIn = pyqtSignal(QImage, name='In')
    changePixmapOut = pyqtSignal(QImage, name='Out')

    def run(self):
        global cam
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

            frame = frame_operation(frame)

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

    def getText(self):
        global file
        text, okPressed = QInputDialog.getText(self, "Get text", "Your name:", QLineEdit.Normal, "")
        if okPressed and text != '':
            global save
            save = True
            file = text + ".jpg"

    # Define Button Presses
    def Button1_pressed(self):
        global recognise
        recognise = ~recognise

    def Button2_pressed(self):
        global swap
        if swap is None:
            swap = 0
        elif swap < len(input_imgs) - 1:
            swap = swap + 1
        else:
            swap = None
        if swap is not None:
            self.Name.setText(input_names[swap])
        else:
            self.Name.setText("")
    def Button3_pressed(self):
        self.getText()

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
        self.Name = QLabel()

        # Label Buttons
        self.Button1.setText('Who am I?')
        self.Button2.setText('Swap my face!')
        self.Button3.setText('Remember me!')
        # Place buttons
        self.layout = QGridLayout()
        self.layout.addWidget(self.Button1, 3, 3)
        self.layout.addWidget(self.Button2, 4, 3)
        self.layout.addWidget(self.Button3, 5, 3)
        self.layout.addWidget(self.Name, 3, 5)

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


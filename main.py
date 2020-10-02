# -*- coding: utf-8 -*-
"""
Face recognition project

@author: John Watson and Damen Kelly
"""

import cv2
import sys
import face_functions as face

from PyQt5.QtWidgets import (QApplication, QMainWindow,
                             QPushButton, QWidget,
                             QGridLayout, QLabel, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot 

Video = True
Exit_flag = False

Option_A = cv2.imread("Damen.jpg")
Option_B = cv2.imread("John.jpg")
Option_C = cv2.imread("Andrew.jpg")

src_img = cv2.imread("John.jpg") 
input_img = [Option_A,Option_B,Option_C]
Cycle = 1


# -----------Video Capture--------------------#
class Thread(QThread):
    changePixmapIn = pyqtSignal(QImage, name = 'In')
    changePixmapOut = pyqtSignal(QImage, name = 'Out')
    def run(self):
        cam = cv2.VideoCapture(0)
        while Exit_flag == False:
            ret, frame = cam.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmapIn.emit(p)
            frame_points, frame_bbox, _ = face.find(frame)
            frame = face.swap(frame, frame_points, frame_bbox, input_img[Cycle], input_points, input_bbox)
            #frame = face.draw(frame)
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmapOut.emit(p)
        sys.quit()   
        

# GUI
class AppWindow(QMainWindow):

    # Define Button Presses
    def Button1_pressed(self):
        print("A")

    def Button2_pressed(self):
        global Cycle
        Cycle = Cycle + 1
        if Cycle == 3:
            Cycle = 0

    def Button3_pressed(self):
        print("C")

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
    @pyqtSlot(QImage, name = 'In')
    def setImageIn(self, image):
        # Input image
        self.labelIn.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot(QImage, name = 'Out')
    def setImageOut(self, frame):
        # Output image
        self.labelOut.setPixmap(QPixmap.fromImage(frame))

    def exitEvent(self, event):
        global Exit_flag
        ExitMsg = QMessageBox.question(self, 'Exit', 'Are you sure?', 
                                       QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if ExitMsg == QMessageBox.Yes:
            event.accept()
            Exit_flag = True
        else:
            event.ignore()

input_points, input_bbox, input_shape = face.find(input_img[Cycle])

input_descriptor = face.recognise(input_img[Cycle], input_shape)  # WIP
input_descriptors = [input_descriptor, input_descriptor]  # WIP



    # Create Application


app = QApplication([])

# Main window
window = AppWindow()

# close the file or device


# Display
window.show()
app.exec_()


        

# if not Video:
#     src_points, src_bbox, src_shape = face.find(src_img)

#     # input_descriptor = face.recognise(src_img, src_shape)  # WIP
#     # face.match(src_descriptor, input_descriptor)  # WIP

#     out_img = face.swap(src_img, src_points, src_bbox, input_img, input_points, input_bbox)

#     src_img = face.draw(src_img)
#     input_img = face.draw(input_img)

#     cv2.imshow("src_img", src_img)
#     cv2.imshow("input_img", input_img)
#     cv2.imshow("out_img", out_img)
#     cv2.waitKey(0)

# else:
# cam = cv2.VideoCapture(0)
# while True:
#         ret_val, frame = cam.read()
#         frame_points, frame_bbox, _ = face.find(frame)
#         frame = face.swap(frame, frame_points, frame_bbox, input_img, input_points, input_bbox)

#         frame = face.draw(frame)

        # cv2.imshow('my webcam', frame)
        # if cv2.waitKey(1) == 27:
        #     break  # esc to quit



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




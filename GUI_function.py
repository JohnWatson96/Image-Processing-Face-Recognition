# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:24:29 2020

@author: Daemo
"""

import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow,
                            QPushButton,  QWidget,
                            QGridLayout, QLabel)
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot




#-----------Video Capture--------------------# 

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

#GUI
class AppWindow(QMainWindow):
       
  #Define Button Presses  
  def Button1_pressed(self):
      print("A")  
  def Button2_pressed(self):
      print("B")     
  def Button3_pressed(self):
      print("C")    
      
      
  #Initilize main window
  def __init__(self, *args, **kwargs):
      super(AppWindow, self).__init__(*args, **kwargs)
      
      #Set Title and window dimensions
      self.setWindowTitle("Facial Recognition Project")


#-----------Buttons-------------------------#      
      #Create  buttons
      self.Button1 = QPushButton()
      self.Button2 = QPushButton()
      self.Button3 = QPushButton()
      #Label Buttons
      self.Button1.setText('Who am I?')     
      self.Button2.setText('Swap my face!')     
      self.Button3.setText('Remember me!')
      #Place buttons
      self.layout = QGridLayout()
      self.layout.addWidget(self.Button1,3,3)
      self.layout.addWidget(self.Button2,4,3)
      self.layout.addWidget(self.Button3,5,3)

      #Connect buttons
      self.Button1.clicked.connect(self.Button1_pressed)
      self.Button2.clicked.connect(self.Button2_pressed)
      self.Button3.clicked.connect(self.Button3_pressed)
 #-----------Images-------------------------#      

      self.labelIn = QLabel(self)
      self.pixmapIn = QPixmap()
      self.resize(self.pixmapIn.width(), self.pixmapIn.height())
      self.layout.addWidget(self.labelIn,0,0,2,2)
      
      self.labelOut = QLabel(self)
      self.pixmapOut = QPixmap()
      self.resize(self.pixmapOut.width(), self.pixmapOut.height())
      self.layout.addWidget(self.labelOut,0,4,2,6)
      
      #Lay widgets location
      self.widget = QWidget()
      self.widget.setLayout(self.layout)
      self.setCentralWidget(self.widget)
      th = Thread(self)
      th.changePixmap.connect(self.setImage)
      th.start()
#Image Update     
  @pyqtSlot(QImage)    
  def setImage(self, image):
            
      #Input image     
      self.labelIn.setPixmap(QPixmap.fromImage(image))   
      
      #Output image     
      self.labelOut.setPixmap(QPixmap.fromImage(image))   
      


#Create Application
app = QApplication([]) 

#Main window
window = AppWindow()

# close the file or device



























# Display
window.show()
app.exec_()



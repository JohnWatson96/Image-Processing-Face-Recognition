# -*- coding: utf-8 -*-
"""
Video processing Lab

@author: John Watson using template by Andrew Busch
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

filename = 'Office Space.mp4'


# this is the mouse handling function
def onMouse(event, x, y, flags, param):
    global doquit
    if event == cv.EVENT_LBUTTONUP:
        # if mouse button is pressed, just set the quit flag to True
        doquit = True


# This command opens the video stream. The argument can be a filename or a number
# If it's a number, it tries to open the corresponding video device on the system. 0 is the first one.
cap = cv.VideoCapture(0)

if cap.isOpened():
    print("Successfully opened video device or file")
else:
    print("Cannot open video device")
    sys.exit()

# create a few windows to display the videos
cv.namedWindow('Video')

# any mouse click in EITHER window will use the callback above
cv.setMouseCallback('Video', onMouse)

# Now try to read a single frame from the source. It returns two results, one a boolean to see if it worked,
# the other the frame (a numpy array with the image)
success, frame = cap.read()

# Define the codec and create VideoWriter object


# now loop until the quit variable is true
while success and not doquit:
    # first, see if a key is pressed. 1 means wait only 1 millisecond before continuing
    key = cv.waitKey(1)

    out, outframe = frame.copy()

    # write the frame
    out.write(np.hstack((frame, outframe)))

    # now display the original and processed images in the windows

    cv.imshow('Video', np.hstack((frame, outframe)))
    # and read the next frame (or try to)
    success, frame = cap.read()

# once the user presses q or clicks mouse and doquit is True, destroy windows and quit
cv.destroyAllWindows()

# close the file or device
cap.release()
out.release()

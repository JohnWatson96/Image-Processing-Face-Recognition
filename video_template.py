# -*- coding: utf-8 -*-
"""
Video processing Lab

@author: John Watson using template by Andrew Busch
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

filename = 'Office Space.mp4'

doquit = False
flip = False
brightness = 0
contrast = 0
vflip = False
gamma = 1
mode = 0
threshold = 137

# this is the mouse handling function
def onMouse(event, x, y, flags, param):
    global doquit
    if event == cv.EVENT_LBUTTONUP:
        # if mouse button is pressed, just set the quit flag to True
        doquit = True


# This command opens the video stream. The argument can be a filename or a number
# If it's a number, it tries to open the corresponding video device on the system. 0 is the first one.
cap = cv.VideoCapture(filename)

height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
framerate = cap.get(cv.CAP_PROP_FPS)
print("Resolution:", height, "x", width)
print("Framerate:", framerate)

if cap.isOpened():
    print("Successfully opened video device or file")
else:
    print("Cannot open video device")
    quit()

# create a few windows to display the videos
cv.namedWindow('Video')

# any mouse click in EITHER window will use the callback above
cv.setMouseCallback('Video', onMouse)

print('Showing video..')

print('Brightness +/-   q/w')
print('Contrast   +/-   a/s')
print('Flip       +/-   f/v')
print('Gamma      +/-   z/x')
print('Threshold  +/-   t/y/u')
print('Gauss Blur       g')
print('Med Blur         m')
print('Canny            e')
print('Reset            r')

# Now try to read a single frame from the source. It returns two results, one a boolean to see if it worked,
# the other the frame (a numpy array with the image)
success, frame = cap.read()

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output2.avi', fourcc, framerate, (2*width,  height))

# now loop until the quit variable is true
while success and not doquit:
    # first, see if a key is pressed. 1 means wait only 1 millisecond before continuing
    key = cv.waitKey(1)

    if key == ord('f'):
        flip = not flip
        print("\rflipped horizontal:", flip, end="")
    elif key == ord('q'):
        brightness = brightness + 10
        print("\rbrightness", brightness, end="")
    elif key == ord('w'):
        brightness = brightness - 10
        print("\rbrightness", brightness, end="")
    elif key == ord('a'):
        contrast = contrast + 10
        print("\rcontrast", contrast, end="")
    elif key == ord('s'):
        contrast = contrast - 10
        print("\rcontrast", contrast, end="")
    elif key == ord('v'):
        vflip = not vflip
        print("\rflipped vertical:", vflip, end="")
    elif key == ord('z'):
        gamma = gamma*1.05
        print("\rgamma", gamma, end="")
    elif key == ord('x'):
        gamma = gamma*0.95
        print("\rgamma", gamma, end="")
    elif key == ord('t'):
        mode = 1
    elif mode == 1 and key == ord('y'):
        threshold = threshold + 10
        print("\rthreshold", threshold, end="")
    elif mode == 1 and key == ord('u'):
        threshold = threshold - 10
        print("\rthreshold", threshold, end="")
    elif key == ord('g'):
        mode = 2
        print("\rgaussian", end="")
    elif key == ord('m'):
        mode = 3
        print("\rmedian", end="")
    elif key == ord('e'):
        mode = 4
        print("\rcanny", end="")
    elif key == ord('r'):
        doquit = False
        flip = False
        brightness = 0
        contrast = 0
        vflip = False
        gamma = 1
        mode = 0
        threshold = 137
        print("\rreset", end="")
    elif key == 27:  # this is the escape key to quit
        doquit = True

    outframe = frame.copy()

    if flip:
        # flip the image left to right so it's mirroring
        cv.flip(outframe, 1, dst=outframe)  # using dst does an "in place" transform which is faster and more efficient
    if vflip:
        # flip the image vertically
        cv.flip(outframe, 0, dst=outframe)  # using dst does an "in place" transform which is faster and more efficient

    # outframe = 255 - frame # simple negative of the image, nice and easy
    # you can modify this to perform any function you want for the output image

    if mode == 0:
        r = np.array(range(0, 256))
        rmax = np.max(frame) + contrast
        rmin = np.min(np.nonzero(frame)) - contrast
        r = (((r + brightness - rmin) * 255 / (rmax - rmin) / 255) ** gamma) * 255
        lut = np.uint8(np.clip(r, 0, 255))
        outframe = cv.LUT(outframe, lut)

    elif mode == 1:  # threshold
        outframe = cv.cvtColor(outframe, cv.COLOR_BGR2GRAY)
        ret, outframe = cv.threshold(outframe, threshold, 255, cv.THRESH_BINARY)
        outframe = cv.cvtColor(outframe, cv.COLOR_GRAY2BGR)

    elif mode == 2: # gaussian
        outframe = cv.GaussianBlur(outframe, (7, 7), 0)

    elif mode == 3:  # median
        outframe = cv.medianBlur(outframe, 7, 0)

    elif mode == 4:  # canny
        outframe = cv.cvtColor(outframe, cv.COLOR_BGR2GRAY)
        outframe = cv.Canny(outframe, 50, 100)
        outframe = cv.cvtColor(outframe, cv.COLOR_GRAY2BGR)

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

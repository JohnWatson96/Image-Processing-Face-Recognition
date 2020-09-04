# -*- coding: utf-8 -*-
"""
Video processing Lab

@author: John Watson using template by Andrew Busch
"""

import cv2 as cv
import numpy as np
import sys
import face_recognition


doquit = False

# this is the mouse handling function
def onMouse(event, x, y, flags, param):
    global doquit
    if event == cv.EVENT_LBUTTONUP:
        # if mouse button is pressed, just set the quit flag to True
        doquit = True


# This command opens the video stream. The argument can be a filename or a number
# If it's a number, it tries to open the corresponding video device on the system. 0 is the first one.
cap = cv.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
john_image = face_recognition.load_image_file("John.jpg")
john_face_encoding = face_recognition.face_encodings(john_image)[0]

# Load a second sample picture and learn how to recognize it.
damen_image = face_recognition.load_image_file("Damen.jpg")
damen_face_encoding = face_recognition.face_encodings(damen_image)[0]

# Load a third sample picture and learn how to recognize it.
laura_image = face_recognition.load_image_file("Laura.jpg")
laura_face_encoding = face_recognition.face_encodings(laura_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    john_face_encoding,
    damen_face_encoding,
    laura_face_encoding
]
known_face_names = [
    "John Watson",
    "Damen Kelly",
    "Laura Currey"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

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

# out = frame.copy()

# now loop until the quit variable is true
while True:

    # outframe = frame.copy()

    # Resize frame of video to 1/4 size for faster face recognition processing

    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # write the frame
    # out = np.hstack((frame, outframe))

    # now display the original and processed images in the windows

    # cv.imshow('Video', np.hstack((frame, outframe)))
    # and read the next frame (or try to)
    success, frame = cap.read()

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# once the user presses q or clicks mouse and doquit is True, destroy windows and quit
cv.destroyAllWindows()

# close the file or device
cap.release()

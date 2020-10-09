# Image-Processing-Face-Recognition

Creates an application that allows recognition, face swapping and saving of faces.

## Installation:

This program requires installation of Dlib. The Nvidia CUDA 11.0 and cuDNN 8.0 packages can be used for GPU acceleration AMD graphics cards are not supported. These packages must be installed before Dlib. The program also assumes model data can be accessed at:
```
dlib.shape_predictor("../dlib-models/shape_predictor_68_face_landmarks.dat")
recogniser = dlib.face_recognition_model_v1("../dlib-models/dlib_face_recognition_resnet_model_v1.dat")
```
In face_functions.py

Models: https://github.com/davisking/dlib-models 

Dlib: pip install dlib

CUDA: https://developer.nvidia.com/cuda-downloads

CuDNN: https://developer.nvidia.com/cudnn

CuDNN Installation Guide: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

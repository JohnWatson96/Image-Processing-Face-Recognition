# -*- coding: utf-8 -*-
"""
face recognition defines using dlib with guidance of wuhuikai/FaceSwap

@author: John Watson and Damen Kelly
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../dlib-models/shape_predictor_68_face_landmarks.dat")
recogniser = dlib.face_recognition_model_v1("../dlib-models/dlib_face_recognition_resnet_model_v1.dat")


def match(src_descriptor, descriptors):
    print(np.linalg.norm(descriptors - src_descriptor, axis=1))
    return


def recognise(img, shape):
    descriptor = recogniser.compute_face_descriptor(img, shape)
    return descriptor


def swap(src_img, src_points, src_bbox, input_img, input_points, input_bbox):
    src_points, src_bbox, _ = find(src_img)
    if src_points is None or src_bbox is None:  # no face detected
        print("No face to swap")
        return src_img

    result_img = warp(src_img, src_points, src_bbox, input_img, input_points, input_bbox)

    ###COPY PASTED###
    ## Mask for blending
    h, w = src_img.shape[:2]
    mask = mask_from_points((h, w), src_points)
    mask_src = np.mean(result_img, axis=2) > 0
    mask = np.asarray(mask * mask_src, dtype=np.uint8)
    # colour correction
    result_img = apply_mask(result_img, mask)
    dst_face_masked = apply_mask(src_img, mask)
    result_img = correct_colours(dst_face_masked, result_img, src_points)
    ## Shrink the mask
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    ##Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    result_img = cv2.seamlessClone(result_img, src_img, mask, center, cv2.NORMAL_CLONE)
    ###COPY PASTED###

    return result_img


###COPY PASTED###
def correct_colours(im1, im2, landmarks1):
    COLOUR_CORRECT_BLUR_FRAC = 0.75
    LEFT_EYE_POINTS = list(range(42, 48))
    RIGHT_EYE_POINTS = list(range(36, 42))

    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.medianBlur(im1, blur_amount, 0)
    im2_blur = cv2.medianBlur(im2, blur_amount, 0)

    # Avoid divide-by-zero errors.
    im2_blur = im2_blur.astype(int)
    im2_blur += 128*(im2_blur <= 1)

    result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
###COPY PASTED###


###COPY PASTED###
def apply_mask(img, mask):

    masked_img = cv2.bitwise_and(img,img,mask=mask)

    return masked_img
###COPY PASTED###


###COPY PASTED###
def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask
###COPY PASTED###


def find(img):
    detections = detector(img)  # detect faces and store rectangles
    if len(detections) == 0:
        print("No face detected")
        return None, None, None
    shape = predictor(img, detections[0])  # find the shape of the face (use first detected face)
    points = np.array(list([point.x, point.y] for point in shape.parts()))

    bbox = [[min(points[:, 0]), min(points[:, 1])], [max(points[:, 0]), min(points[:, 1])],
            [min(points[:, 0]), max(points[:, 1])], [max(points[:, 0]), max(points[:, 1])]]
    '''
    2   3
    0   1
    '''

    return points, bbox, shape


def warp(src_img, src_points, src_bbox, input_img, input_points, input_bbox):
    result_img = draw(src_img.copy())  # create image to warp to
    src_delaunay = Delaunay(src_points)  # create Delaunay triangles to warp to

    triangle_affines = np.array(list(get_affine_transform(src_delaunay.simplices, input_points, src_points)))
    # create transform matrices to warp input points to source triangles

    src_bbox_points = np.array([(x, y) for x in range(src_bbox[0][0], src_bbox[3][0] + 1)
                                for y in range(src_bbox[0][1], src_bbox[3][1] + 1)])
    # create an array of all coordinates in source face area

    src_indicies = src_delaunay.find_simplex(src_bbox_points)  # returns triangle index for each point, -1 for none

    for triangle_index in range(len(src_delaunay.simplices)):  # for each triangle
        triangle_points = src_bbox_points[src_indicies == triangle_index]  # for the points in the triangle
        num_points = len(triangle_points)  # get the number of points
        out_points = np.dot(triangle_affines[triangle_index], np.vstack((triangle_points.T, np.ones(num_points))))
        # perform affine transform T = M.[x,y,1]^T to create triangles of source in the input

        x, y = triangle_points.T  # transpose [[x1,y1], [x2,y2], ...] to [x1, x2, ...], [y1, y2, ...]
        result_img[y, x] = bilinear_interpolate(input_img, out_points)  # interpolate between input and source
        # cv2.imshow("result_img", result_img)
        # cv2.waitKey(10)  # these show the process for each section

    return result_img


def get_affine_transform(input_simplices, input_points, src_points):
    for triangle in input_simplices:  # for each triangle
        src_triangle = np.float32(src_points[triangle])
        input_triangle = np.float32(input_points[triangle])
        mat = cv2.getAffineTransform(src_triangle, input_triangle)  # get the transform matrix
        yield mat


def bilinear_interpolate(img, points):
    int_points = np.int32(points)
    x0, y0 = int_points
    dx, dy = points - int_points

    q11 = img[y0, x0]  # 4 Neighbour pixels
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    bottom = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + bottom * (1 - dy)

    return inter_pixel.T


def draw(img):  # draws facial points, Delaunay triangles and bounding box

    points, bbox, _ = find(img)

    if points is None or bbox is None:
        print("no face to draw")
        return img

    cv2.line(img, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[0]), tuple(bbox[2]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[2]), tuple(bbox[3]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[1]), tuple(bbox[3]), (0, 0, 255), 2)

    for point in points:
        cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)

    triangles = Delaunay(points)

    for triangle in points[triangles.simplices]:
        cv2.line(img, tuple(triangle[0]), tuple(triangle[1]), (255, 255, 255), 1)
        cv2.line(img, tuple(triangle[1]), tuple(triangle[2]), (255, 255, 255), 1)
        cv2.line(img, tuple(triangle[2]), tuple(triangle[0]), (255, 255, 255), 1)

    return img








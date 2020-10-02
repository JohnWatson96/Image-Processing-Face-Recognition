# -*- coding: utf-8 -*-
"""
face recognition defines using dlib

@author: John Watson and Damen Kelly
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../dlib-models/shape_predictor_68_face_landmarks.dat")
recogniser = dlib.face_recognition_model_v1("../dlib-models/dlib_face_recognition_resnet_model_v1.dat")


def find(img):

    detections = detector(img)  # detect faces and store rectangles
    if len(detections) == 0:
        print("No face detected")
        return None, None
    shape = predictor(img, detections[0])  # find the shape of the face (use first detected face)
    points = np.array(list([point.x, point.y] for point in shape.parts()))
    bbox = [[min(points[:, 0]), min(points[:, 1])], [max(points[:, 0]), min(points[:, 1])],
            [min(points[:, 0]), max(points[:, 1])], [max(points[:, 0]), max(points[:, 1])]]
    return points, bbox


def recognise(img):
    pos, shape = find(img)
    descriptor = recogniser.compute_face_descriptor(img, shape)
    return descriptor


def draw(img):  # draws facial points, Delaunay triangles and bounding box

    points, bbox = find(img)

    if points is None or bbox is None:
        return img

    triangles = Delaunay(points)

    cv2.line(img, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[0]), tuple(bbox[2]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[2]), tuple(bbox[3]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[1]), tuple(bbox[3]), (0, 0, 255), 2)

    for point in points:
        cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)

    for triangle in points[triangles.simplices]:
        cv2.line(img, tuple(triangle[0]), tuple(triangle[1]), (255, 255, 255), 1)
        cv2.line(img, tuple(triangle[1]), tuple(triangle[2]), (255, 255, 255), 1)
        cv2.line(img, tuple(triangle[2]), tuple(triangle[0]), (255, 255, 255), 1)

    cv2.line(img, tuple(points[triangles.simplices][10][0]), tuple(points[triangles.simplices][10][1]), (255, 0, 0), 2)
    cv2.line(img, tuple(points[triangles.simplices][10][1]), tuple(points[triangles.simplices][10][2]), (255, 0, 0), 2)
    cv2.line(img, tuple(points[triangles.simplices][10][2]), tuple(points[triangles.simplices][10][0]), (255, 0, 0), 2)

    return img


def get_affine_transform(input_simplices, input_points, src_points):

    ones = [1, 1, 1]
    for triangle in input_simplices:
        input_triangle = np.vstack((input_points[triangle].T, ones))
        src_triangle = np.vstack((src_points[triangle].T, ones))
        print("input_triangle", input_triangle)
        print("src_triangle", src_triangle)
        mat = np.dot(input_triangle, np.linalg.inv(src_triangle))[:2]  # dot product and inverse
        yield mat


def bilinear_interpolate(img, coords):
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T


def swap(src_img, input_img, input_points, input_bbox):

    src_points, src_bbox = find(src_img)

    if src_points is None or src_bbox is None:  # no face detected
        print("No face to swap")
        return src_img

    result_img = src_img.copy()  # create destination face
    input_delaunay = Delaunay(input_points)
    triangle_affines = np.array(list(get_affine_transform(input_delaunay.simplices, input_points, src_points)))

    print("triangle_affines", triangle_affines)

    xmin = np.min(src_points[:, 0])
    xmax = np.max(src_points[:, 0]) + 1
    ymin = np.min(src_points[:, 1])
    ymax = np.max(src_points[:, 1]) + 1

    roi_coords = np.asarray([(x, y) for y in range(ymin, ymax)
                            for x in range(xmin, xmax)], np.uint32)

    print("roi_coords", roi_coords)

    roi_tri_indices = input_delaunay.find_simplex(roi_coords)

    print("roi_tri_indices", roi_tri_indices)

    for simplex_index in range(len(input_delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(triangle_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    cv2.imshow("result_img", result_img)

    return result_img
'''    input_img = input_img[input_bbox[0][1]:input_bbox[3][1], input_bbox[0][0]:input_bbox[3][0]]  # crop using slicing
    result_img = cv2.resize(result_img, (src_bbox[3][0] - src_bbox[0][0],
                                       src_bbox[3][1] - src_bbox[0][1]))  # size to source
    result_img[src_bbox[0][1]:src_bbox[3][1], src_bbox[0][0]:src_bbox[3][0]] = input_img  # paste'''

# old triangle finding function that uses cv2's subdivision
def old_draw(img):

    points, bbox = find(img)
    sub_div = cv2.Subdiv2D((bbox[0][0], bbox[0][1], bbox[3][0], bbox[3][1]))

    cv2.line(img, tuple(bbox[0]), tuple(bbox[1]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[0]), tuple(bbox[2]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[2]), tuple(bbox[3]), (0, 0, 255), 2)
    cv2.line(img, tuple(bbox[1]), tuple(bbox[3]), (0, 0, 255), 2)

    for point in points:
        cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)
        sub_div.insert((point[0], point[1]))

    triangles = sub_div.getTriangleList()

    for triangle in triangles:
        point1 = (triangle[0], triangle[1])
        point2 = (triangle[2], triangle[3])
        point3 = (triangle[4], triangle[5])

        cv2.line(img, point1, point2, (255, 255, 255), 1)
        cv2.line(img, point2, point3, (255, 255, 255), 1)
        cv2.line(img, point3, point1, (255, 255, 255), 1)

    return img

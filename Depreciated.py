# old triangle finding function that uses cv2's subdivision

import face_functions as face
import cv2

def old_draw(img):

    points, bbox = face.find(img)
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

'''    input_img = input_img[input_bbox[0][1]:input_bbox[3][1], input_bbox[0][0]:input_bbox[3][0]]  # crop using slicing
    result_img = cv2.resize(result_img, (src_bbox[3][0] - src_bbox[0][0],
                                       src_bbox[3][1] - src_bbox[0][1]))  # size to source
    result_img[src_bbox[0][1]:src_bbox[3][1], src_bbox[0][0]:src_bbox[3][0]] = input_img  # paste'''
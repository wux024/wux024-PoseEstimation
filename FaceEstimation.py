#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/20 10:44
@description: Use dlib library to realize face key point detection (5 points or 68 points)
"""
import cv2
import dlib
import numpy as np


def get_face_point_pairs():
    point_pairs = []
    for point in range(16):
        point_pairs.append([point, point + 1])
    for point in range(17, 21):
        point_pairs.append([point, point + 1])
    for point in range(22, 26):
        point_pairs.append([point, point + 1])
    for point in range(27, 30):
        point_pairs.append([point, point + 1])
    for point in range(31, 35):
        point_pairs.append([point, point + 1])
    for point in range(36, 41):
        point_pairs.append([point, point + 1])
    point_pairs.append([36, 41])
    for point in range(42, 47):
        point_pairs.append([point, point + 1])
    point_pairs.append([42, 47])
    for point in range(48, 59):
        point_pairs.append([point, point + 1])
    point_pairs.append([48, 59])
    for point in range(60, 67):
        point_pairs.append([point, point + 1])
    point_pairs.append([60, 67])
    return point_pairs

img = cv2.imread('TestImage/psb_3.jpg')
h, w, _ = img.shape
# img = cv2.resize(img, (w//4,h//4))
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("DlibModelC/shape_predictor_68_face_landmarks.dat")
count = 68
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detector(gray)
np.random.seed(200)
point_pairs = get_face_point_pairs()
for i,face in enumerate(faces):
    face_landmarks = dlib_facelandmark(gray, face)
    face_points = []
    for n in range(count):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        if x and y:
            face_points.append((x, y))
        else:
            face_points.append(None)
    np.random.seed(i)
    COLORS = np.random.randint(0, 255, size=(count, 3), dtype="uint8")
    for idx,face_point in enumerate(face_points):
        color = [int(c) for c in COLORS[idx]]
        if face_point:
            cv2.circle(img,
                       face_point,
                       4,
                       color,
                       thickness=-1,
                       lineType=cv2.FILLED)
    # for point in point_pairs:
    #     for idx, pair in enumerate(point_pairs):
    #         partA = pair[0]
    #         partB = pair[1]
    #         if face_points[partA] and face_points[partB]:
    #             cv2.line(img,
    #                      face_points[partA],
    #                      face_points[partB],
    #                      (255,255,255),
    #                      1,
    #                      cv2.LINE_AA)
cv2.imshow("Face Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
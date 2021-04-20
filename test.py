#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/19 20:08
@description:
"""
import cv2
from PoseEstimationOpenPose import detection_key_point

bodymodelpath = ["CaffeModels/prototxt/pose_deploy_linevec_coco.prototxt",
                 "CaffeModels/caffemodel/pose_iter_440000.caffemodel"]
facemodelpath = ["CaffeModels/prototxt/pose_deploy_face.prototxt",
                 "CaffeModels/caffemodel/pose_iter_116000.caffemodel"]
handsmodelpath = ["CaffeModels/prototxt/pose_deploy_hand.prototxt",
                  "CaffeModels/caffemodel/pose_iter_102000.caffemodel"]

img = cv2.imread("TestImage/wux_bule.JPG")
# h, w, _ = img.shape
# img = cv2.resize(img,(w//4,h//4))
model = detection_key_point(facemodelpath=facemodelpath,face=True,face_connection=False)
# output = model.predict(img.copy())
# print(output.shape)
_,face_points,_ = model.predict(img.copy())
img = model.visualizepose(img, face_points=face_points)
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/4/19 10:56
@description:Use openpose's pre-trained model to achieve pose estimation,
including human body, face, and hands.
BODYMPI =  {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
            "Background": 15 }
BODYCOCO = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
BODY25 =   {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip":8,"RHip": 9, "RKnee": 10,
            "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14, "REye": 15,
            "LEye": 16, "REar": 17, "LEar": 18, "LbigToe":19,"LSmallToe":20,
            "LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24,"Background": 25}
"""
import cv2
import numpy as np



class detection_key_point(object):
    def __init__(self,bodymodelpath = None, facemodelpath = None, handsmodelpath = None,
                 body = False, face = False, hands = False,net_width = 368, net_height = 368,
                 body_th=0.2, face_th = 0.1 , hands_th = 0.2, bodymode='BODY25',
                 body_connection = True, face_connection = True, hands_connection = True):
        self.bodymodelpath = bodymodelpath
        self.facemodelpath = facemodelpath
        self.handsmodelpath = handsmodelpath

        self.bodymode = bodymode

        self.body = body
        self.face = face
        self.hands = hands

        self.net_width = net_width
        self.net_height = net_height

        self.body_th = body_th
        self.face_th = face_th
        self.hands_th = hands_th

        self.body_point_pairs = None
        self.face_point_pairs = None
        self.hands_point_pairs = None

        self.body_num_points = None
        self.face_num_points = None
        self.hands_num_points = None

        self.body_connection = body_connection
        self.face_connection = face_connection
        self.hands_connection = hands_connection

        if self.body and self.bodymodelpath is not None:
            self.body_model = self.get_body_model()
            self.get_body_point_pairs()
        if self.face and self.facemodelpath is not None:
            self.face_model = self.get_face_model()
            self.get_face_point_pairs()
        if self.hands and self.handsmodelpath is not None:
            self.hands_model = self.get_hands_model()
            self.get_hands_point_pairs()

    def get_body_model(self):
        net = cv2.dnn.readNetFromCaffe(self.bodymodelpath[0],self.bodymodelpath[1])
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def get_face_model(self):
        net = cv2.dnn.readNetFromCaffe(self.facemodelpath[0], self.facemodelpath[1])
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def get_hands_model(self):
        net = cv2.dnn.readNetFromCaffe(self.handsmodelpath[0], self.handsmodelpath[1])
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def get_body_point_pairs(self):
        if self.bodymode == 'BODY25':
            self.body_point_pairs = [[1, 0], [1, 2], [1, 5],
                                     [2, 3], [3, 4], [5, 6],
                                     [6, 7], [0, 15], [15, 17],
                                     [0, 16], [16, 18], [1, 8],
                                     [8, 9], [9, 10], [10, 11],
                                     [11, 22], [22, 23], [11, 24],
                                     [8, 12], [12, 13], [13, 14],
                                     [14, 19], [19, 20], [14, 21]]
            self.body_num_points = 25
        elif self.bodymode == 'BODYMPI':
            self.body_point_pairs = [[0, 1], [1, 2], [2, 3],
                                     [3, 4], [1, 5], [5, 6],
                                     [6, 7], [1, 14], [14, 8],
                                     [8, 9], [9, 10], [14, 11],
                                     [11, 12], [12, 13]]
            self.body_num_points = 15

        elif self.bodymode == 'BODYCOCO':
            self.body_point_pairs = [[1, 0], [1, 2], [1, 5],
                                     [2, 3], [3, 4], [5, 6],
                                     [6, 7], [1, 8], [8, 9],
                                     [9, 10], [1, 11], [11, 12],
                                     [12, 13], [0, 14], [0, 15],
                                     [14, 16], [15, 17]]
            self.body_num_points = 18
        else:
            return

    def get_face_point_pairs(self):
        point_pairs = []
        for point in range(16):
            point_pairs.append([point,point+1])
        for point in range(17,21):
            point_pairs.append([point,point+1])
        for point in range(22,26):
            point_pairs.append([point,point+1])
        for point in range(27,30):
            point_pairs.append([point,point+1])
        for point in range(31,35):
            point_pairs.append([point,point+1])
        for point in range(36,41):
            point_pairs.append([point,point+1])
        point_pairs.append([36,41])
        for point in range(42,47):
            point_pairs.append([point,point+1])
        point_pairs.append([42,47])
        for point in range(48,59):
            point_pairs.append([point,point+1])
        point_pairs.append([48, 59])
        for point in range(60,67):
            point_pairs.append([point,point+1])
        point_pairs.append([60, 67])
        self.face_point_pairs = point_pairs
        self.face_num_points = 70

    def get_hands_point_pairs(self):
        point_pairs = []
        for point in range(4):
            point_pairs.append([point,point+1])

        point_pairs.append([0,5])
        for point in range(5,8):
            point_pairs.append([point, point + 1])

        point_pairs.append([0, 9])
        for point in range(9,12):
            point_pairs.append([point,point+1])

        point_pairs.append([0, 13])
        for point in range(13,16):
            point_pairs.append([point,point+1])
        point_pairs.append([0, 17])
        for point in range(17,20):
            point_pairs.append([point,point+1])
        self.hands_point_pairs = point_pairs
        self.hands_num_points = 21

    def predict(self, img):
        body_points = None
        face_points = None
        hands_points = None
        img_height, img_width, _ = img.shape
        in_width = int((self.net_height / img_height) * img_width)
        if self.body:
            inpBlob = cv2.dnn.blobFromImage(img,
                                            1.0 / 255,
                                            (in_width, self.net_height),
                                            (0, 0, 0),
                                            swapRB=False,
                                            crop=False)
            self.body_model.setInput(inpBlob)
            output = self.body_model.forward()
            H = output.shape[2]
            W = output.shape[3]
            points = []
            for idx in range(self.body_num_points):
                probMap = output[0, idx, :, :]  # confidence map.
                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                # Scale the point to fit on the original image
                x = (img_width * point[0]) / W
                y = (img_height * point[1]) / H
                if prob > self.body_th:
                    points.append((int(x), int(y)))
                else:
                    points.append(None)
            body_points = points
        if self.face:
            in_width = int((self.net_height / img_height) * img_width)
            inpBlob = cv2.dnn.blobFromImage(img,
                                            1.0 / 255,
                                            (in_width, self.net_height),
                                            (0, 0, 0),
                                            swapRB=False,
                                            crop=False)

            self.face_model.setInput(inpBlob)
            output = self.face_model.forward()
            H = output.shape[2]
            W = output.shape[3]
            points = []
            for idx in range(self.face_num_points):
                probMap = output[0, idx, :, :]  # confidence map.
                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                # Scale the point to fit on the original image
                x = (img_width * point[0]) / W
                y = (img_height * point[1]) / H
                if prob > self.face_th:
                    points.append((int(x), int(y)))
                else:
                    points.append(None)
            face_points = points
        if self.hands:
            inpBlob = cv2.dnn.blobFromImage(img,
                                            1.0 / 255,
                                            (in_width, self.net_height),
                                            (0, 0, 0),
                                            swapRB=False,
                                            crop=False)

            self.hands_model.setInput(inpBlob)
            output = self.hands_model.forward()
            H = output.shape[2]
            W = output.shape[3]
            points = []
            for idx in range(self.hands_num_points):
                probMap = output[0, idx, :, :]  # confidence map.
                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                # Scale the point to fit on the original image
                x = (img_width * point[0]) / W
                y = (img_height * point[1]) / H
                if prob > self.hands_th:
                    points.append((int(x), int(y)))
                else:
                    points.append(None)
            hands_points = points
        return body_points,face_points,hands_points
    def visualizepose(self, img, body_points=None, face_points=None, hands_points=None):
        if self.body:
            np.random.seed(100)
            COLORS = np.random.randint(0, 255, size=(self.body_num_points, 3), dtype="uint8")
            for idx in range(len(body_points)):
                color = [int(c) for c in COLORS[idx]]
                if body_points[idx]:
                    cv2.circle(img,
                               body_points[idx],
                               4,
                               color,
                               thickness=-1,
                               lineType=cv2.FILLED)
                    # cv2.putText(img,
                    #             "{}".format(idx),
                    #             body_points[idx],
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1,
                    #             (0, 0, 255),
                    #             2,
                    #             lineType=cv2.LINE_AA)
            if self.body_connection:
                np.random.seed(200)
                COLORS = np.random.randint(0, 255, size=(len(self.body_point_pairs), 3), dtype="uint8")
                for idx,pair in enumerate(self.body_point_pairs):
                    partA = pair[0]
                    partB = pair[1]
                    color = [int(c) for c in COLORS[idx]]
                    if body_points[partA] and body_points[partB]:
                        cv2.line(img,
                                 body_points[partA],
                                 body_points[partB],
                                 color, 2, cv2.LINE_AA)
                        # cv2.circle(img,
                        #            body_points[partA],
                        #            8,
                        #            (0, 0, 255),
                        #            thickness=-1,
                        #            lineType=cv2.FILLED)
        if self.face:
            np.random.seed(300)
            COLORS = np.random.randint(0, 255, size=(self.face_num_points, 3), dtype="uint8")
            for idx in range(len(face_points)):
                color = [int(c) for c in COLORS[idx]]
                if face_points[idx]:
                    cv2.circle(img,
                               face_points[idx],
                               4,
                               color,
                               thickness=-1,
                               lineType=cv2.FILLED)
                    # cv2.putText(img,
                    #             "{}".format(idx),
                    #             body_points[idx],
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1,
                    #             (0, 0, 255),
                    #             2,
                    #             lineType=cv2.LINE_AA)
            if self.face_connection:
                np.random.seed(400)
                COLORS = np.random.randint(0, 255, size=(len(self.face_point_pairs), 3), dtype="uint8")
                for idx, pair in enumerate(self.face_point_pairs):
                    partA = pair[0]
                    partB = pair[1]
                    color = [int(c) for c in COLORS[idx]]
                    if face_points[partA] and face_points[partB]:
                        cv2.line(img,
                                 face_points[partA],
                                 face_points[partB],
                                 color, 2, cv2.LINE_AA)
                        # cv2.circle(img,
                        #            body_points[partA],
                        #            8,
                        #            (0, 0, 255),
                        #            thickness=-1,
                        #            lineType=cv2.FILLED)
        if self.hands:
            np.random.seed(500)
            COLORS = np.random.randint(0, 255, size=(self.hands_num_points, 3), dtype="uint8")
            for idx in range(len(hands_points)):
                color = [int(c) for c in COLORS[idx]]
                if hands_points[idx]:
                    cv2.circle(img,
                               hands_points[idx],
                               4,
                               color,
                               thickness=-1,
                               lineType=cv2.FILLED)
                    # cv2.putText(img,
                    #             "{}".format(idx),
                    #             body_points[idx],
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1,
                    #             (0, 0, 255),
                    #             2,
                    #             lineType=cv2.LINE_AA)
            if self.hands_connection:
                np.random.seed(600)
                COLORS = np.random.randint(0, 255, size=(len(self.hands_point_pairs), 3), dtype="uint8")
                for idx, pair in enumerate(self.hands_point_pairs):
                    partA = pair[0]
                    partB = pair[1]
                    color = [int(c) for c in COLORS[idx]]
                    if hands_points[partA] and hands_points[partB]:
                        cv2.line(img,
                                 hands_points[partA],
                                 hands_points[partB],
                                 color, 2, cv2.LINE_AA)
                        # cv2.circle(img,
                        #            body_points[partA],
                        #            8,
                        #            (0, 0, 255),
                        #            thickness=-1,
                        #            lineType=cv2.FILLED)

        return img



import numpy as np
import cv2
import math

from vinopy.model.model import Model


class ModelFace(Model):
    def get_box(self, face, frame):
        frame_h, frame_w = frame.shape[:2]
        box = face[3:7] * np.array([frame_w,
                                    frame_h,
                                    frame_w,
                                    frame_h])
        xmin, ymin, xmax, ymax = box.astype("int")
        return xmin, ymin, xmax, ymax

    def _in_frame(self, frame, n, c, h, w):
        """
        transform frame for input data 
        """
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        return in_frame

    def crop_face_frame(self, frame, xmin, ymin, xmax, ymax):
        face_frame = frame[ymin:ymax, xmin:xmax]
        return face_frame

    def get_frame_shape(self, frame):
        self.frame_h, self.frame_w = frame.shape[:2]


class ModelDetectFace(ModelFace):
    def get_face_pos(self, init_frame):
        frame = init_frame.copy()

        n, c, h, w = self.net.inputs[self.input_blob].shape
        self.shapes = (n, c, h, w)
        scale = 640 / frame.shape[1]
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

        self.frame_h, self.frame_w = frame.shape[:2]

        in_frame = self._in_frame(frame, n, c, h, w)
        # res's shape: [1, 1, 200, 7]
        self.exec_net.start_async(request_id=0, inputs={
                                  self.input_blob: in_frame})

        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs[self.out_blob]
            # prob threshold : 0.5
            faces = res[0][:, np.where(res[0][0][:, 2] > 0.5)]
        return faces

    def detect_face(self, init_frame):
        frame = init_frame.copy()
        faces = self.get_face_pos(frame)
        for face in faces[0][0]:
            xmin, ymin, xmax, ymax = self.get_box(face, frame)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        return frame


class ModelEstimateHeadpose(ModelFace):
    def _build_camera_matrix(self, center_of_face, focal_length):

        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1

        return camera_matrix

    def _draw_axes(self, frame, center_of_face, yaw, pitch, roll, scale, focal_length):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0

        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1,                0,                               0],
                       [0,                math.cos(pitch),  -math.sin(pitch)],
                       [0,                math.sin(pitch),   math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw),    0,                  -math.sin(yaw)],
                       [0,                1,                               0],
                       [math.sin(yaw),    0,                   math.cos(yaw)]])
        Rz = np.array([[math.cos(roll),   -math.sin(roll),                 0],
                       [math.sin(roll),   math.cos(roll),                  0],
                       [0,                0,                               1]])

        # R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # R = np.dot(Rz, np.dot(Ry, Rx))
        R = Rz @ Ry @ Rx
        # print(R)
        camera_matrix = self._build_camera_matrix(center_of_face, focal_length)

        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]

        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o

        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)

        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)

        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)

        return frame

    def get_axis(self, face_frame):
        n, c, h, w = self.net.inputs[self.input_blob].shape
        in_frame = self._in_frame(face_frame, n, c, h, w)
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})
        if self.exec_net.requests[0].wait(-1) == 0:
            yaw = .0  # Axis of rotation: y
            pitch = .0  # Axis of rotation: x
            roll = .0  # Axis of rotation: z
            # Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitсh or roll).
            yaw = self.exec_net.requests[0].outputs['angle_y_fc'][0][0]
            pitch = self.exec_net.requests[0].outputs['angle_p_fc'][0][0]
            roll = self.exec_net.requests[0].outputs['angle_r_fc'][0][0]
        return yaw, pitch, roll


    def get_center_face(self, face_frame, xmin, ymin):
        if self.exec_net.requests[0].wait(-1) == 0:
            center_of_face = (xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0)
        return center_of_face


    def estimate_headpose(self, frame, faces):
        # 4. Create Async Request
        scale = 50
        focal_length = 950.0
        if len(faces) > 0:
            for face in faces[0][0]:
                xmin, ymin, xmax, ymax = self.get_box(face, frame)
                face_frame = frame[ymin:ymax, xmin:xmax]

                if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                    continue
                
                yaw, pitch, roll = self.get_axis(face_frame)
                center_of_face = self.get_center_face(face_frame, xmin, ymin)
                self._draw_axes(frame, center_of_face, yaw,
                                pitch, roll, scale, focal_length)

        return frame


class ModelEmotionRecognition(ModelFace):
    def __init__(self, task):
        super().__init__(task)
        self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')

    def get_emotion(self, face_frame):
        n, c, h, w = self.net.inputs[self.input_blob].shape
        in_frame = self._in_frame(face_frame, n, c, h, w)
        self.exec_net.start_async(request_id=0, inputs={
            self.input_blob: in_frame})

        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs[self.out_blob]
            emotion = self.label[np.argmax(res[0])]
        return emotion

    def emotion_recognition(self, init_frame, faces, rect):  # 4. Create Async Request
        frame = init_frame.copy()
        for face in faces[0][0]:
            xmin, ymin, xmax, ymax = self.get_box(face, frame)
            face_frame = self.crop_face_frame(frame, xmin, ymin, xmax, ymax)
            
            if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                continue

            emotion = self.get_emotion(face_frame)

            if rect:
                frame = cv2.rectangle(frame,
                                      (xmin, ymin), (xmax, ymax),
                                      (0, 255, 0), 2)

            cv2.putText(frame, emotion,
                        (int(xmin + (xmax - xmin) / 2), int(ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2, cv2.LINE_AA)

        return frame

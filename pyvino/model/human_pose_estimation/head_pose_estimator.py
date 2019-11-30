import cv2
import math
import numpy as np

from ...util import get_logger
from ..object_detection.object_detector import ObjectDetector
from ..object_detection.face_detector import FaceDetector


logger = get_logger(__name__)


class HeadPoseDetector(ObjectDetector):
    def __init__(self):
        self.task = 'estimate_headpose'
        super().__init__(self.task)
        self.scale = 50
        self.focal_length = 950.0
        self.detector_face = FaceDetector()

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
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(pitch), -math.sin(pitch)],
                       [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                       [0, 1, 0],
                       [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                       [math.sin(roll), math.cos(roll), 0],
                       [0, 0, 1]])

        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        R = Rz @ Ry @ Rx

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
        result = self.get_result(face_frame)
        # Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitсh or roll).
        yaw = result['angle_y_fc'][0][0]
        pitch = result['angle_p_fc'][0][0]
        roll = result['angle_r_fc'][0][0]
        return yaw, pitch, roll

    def get_center_face(self, face_frame, xmin, ymin):
        center_of_face = (xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0)
        return center_of_face

    def compute(self, init_frame, pred_flag=True, frame_flag=True):
        frame = init_frame.copy()
        faces = self.detector_face.get_pos(frame)
        results = {}
        results['preds'] = {}
        for face_num, face in enumerate(faces):
            xmin, ymin, xmax, ymax = self.get_box(face, frame)
            face_frame = frame[ymin:ymax, xmin:xmax]
            yaw, pitch, roll = self.get_axis(face_frame)
            center_of_face = self.get_center_face(face_frame, xmin, ymin)
            if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                continue
            if pred_flag:
                results['preds'][face_num] = {'bbox': (xmin, ymin, xmax, ymax),
                                     'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                     'center_of_face': center_of_face}
            if frame_flag:
                scale = (face_frame.shape[0] ** 2 + face_frame.shape[1] ** 2) ** 0.5 / 2
                self._draw_axes(frame, center_of_face, yaw,
                                pitch, roll, scale, self.focal_length)
        if frame_flag:
            results['frame'] = frame
        return results



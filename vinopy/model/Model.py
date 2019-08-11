
import os
import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
import sys
import math

from vinopy.util.config import CONFIG

# '/opt/intel/openvino_2019.2.242/python/python3.7' doesn't work
sys.path[1] = '/opt/intel/openvino_2019.2.242/python/python3.6'

DEVICE = CONFIG["MODEL"]["DEVICE"]
MODEL_DIR = CONFIG["MODEL"]["MODEL_DIR"]
CPU_EXTENSION = CONFIG["MODEL"]["CPU_EXTENSION"]

if DEVICE == "CPU":
    MODEL_FP = 'FP32'
else:
    raise NotImplementedError

TASKS = {'detect_face': 'face-detection-adas-0001',
         'emotion_recognition': 'emotions-recognition-retail-0003',
         'estimate_headpose': 'head-pose-estimation-adas-0001',
         'detect_person': 'person-detection-retail-0002'}


class Model(object):
    # TODO: load from config
    def __init__(self, task):
        self.task = task
        self.device = DEVICE
        self._set_model_path()
        # Read IR
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        # Load Model
        self._set_ieplugin()
        self._get_io_blob()

    def _set_ieplugin(self):
        plugin = IEPlugin(device=self.device, plugin_dirs=None)
        if DEVICE == "CPU":
            plugin.add_cpu_extension(CPU_EXTENSION)
        self.exec_net = plugin.load(network=self.net, num_requests=2)

    def _set_model_path(self):
        model_name = TASKS[self.task]
        path_model_dir = os.path.join(MODEL_DIR, model_name, MODEL_FP)

        self.model_xml = os.path.join(
            path_model_dir, model_name + '.xml')
        self.model_bin = os.path.join(
            path_model_dir, model_name + ".bin")

    def _get_io_blob(self):
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

    def _in_frame(self, frame, n, c, h, w):
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        return in_frame


class ModelDetectFace(Model):
    # TODO: load from config
    def get_face_pos(self, frame):
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

    def detect_face(self, frame):
        faces = self.get_face_pos(frame)

        # frame = init_frame.copy()
        for face in faces[0][0]:
            box = face[3:7] * np.array([self.frame_w,
                                        self.frame_h,
                                        self.frame_w,
                                        self.frame_h])
            (xmin, ymin, xmax, ymax) = box.astype("int")
            """
            xmin = int(face[3] * frame_w)
            ymin = int(face[4] * frame_h)
            xmax = int(face[5] * frame_w)
            ymax = int(face[6] * frame_h)
            """
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return frame


class ModelEstimateHeadpose(Model):
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

        #R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        #R = np.dot(Rz, np.dot(Ry, Rx))
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

    def estimate_headpose(self, frame, faces):
        # 4. Create Async Request
        scale = 50
        focal_length = 950.0
        frame_h, frame_w = frame.shape[:2]

        n, c, h, w = self.net.inputs[self.input_blob].shape

        if len(faces) > 0:
            for face in faces[0][0]:
                box = face[3:7] * \
                    np.array([frame_w, frame_h, frame_w, frame_h])
                (xmin, ymin, xmax, ymax) = box.astype("int")
                face_frame = frame[ymin:ymax, xmin:xmax]

                if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                    continue
                in_frame = self._in_frame(frame, n, c, h, w)
                self.exec_net.start_async(request_id=0, inputs={
                                          self.input_blob: in_frame})
                if self.exec_net.requests[0].wait(-1) == 0:
                    yaw = .0  # Axis of rotation: y
                    pitch = .0  # Axis of rotation: x
                    roll = .0  # Axis of rotation: z
                    # Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitсh or roll).
                    yaw = self.exec_net.requests[0].outputs['angle_y_fc'][0][0]
                    pitch = self.exec_net.requests[0].outputs['angle_p_fc'][0][0]
                    roll = self.exec_net.requests[0].outputs['angle_r_fc'][0][0]
                    # print("yaw:{:f}, pitch:{:f}, roll:{:f}".format(yaw, pitch, roll))
                    center_of_face = (
                        xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0)
                    self._draw_axes(frame, center_of_face, yaw,
                                    pitch, roll, scale, focal_length)
        else:
            pass

        return frame

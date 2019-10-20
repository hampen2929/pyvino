import os
import numpy as np
import pandas as pd
import math
import cv2
import urllib.request
import time

from openvino.inference_engine import IENetwork, IEPlugin
from pyvino.util.config import (TASKS, load_config)
from pyvino.detector.visualizer import Visualizer
from pyvino.util.config import COCO_LABEL
from pyvino.util.util import get_logger
import platform


logger = get_logger(__name__)


class Detector(object):
    def __init__(self, task, device=None,
                 model_fp=None, model_dir=None,
                 cpu_extension=None, path_config=None):
        """

        Args:
            task (str):  task for select model
            path_config (str): path to config for constructing model
            device (str): which device to use
            model_dir (str): path intel_models dir. if None,  ~/.pyvino/intel_models/
            model_fp (str): FP16 and FP32 are supported.
        """
        self.task = task
        self.model_name = TASKS[self.task]
        self.config = self._load_config(path_config)
        self._set_from_config(device, model_fp, model_dir, cpu_extension)
        self._set_model_path()
        # Read IR
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        # Load Model
        self._set_ieplugin()
        self._get_io_blob()
        self._get_shape()

    def _load_config(self, path_config):
        """

        Args:
            path_config (str): path to config file

        Returns:load_config(path_config)

        """
        if path_config is None:
            # path_config = os.path.join(os.path.expanduser('~'), '.pyvino', 'config.ini')
            config = None
        else:
            if not os.path.exists(path_config):
                raise FileNotFoundError("Not exists config file. {}".format(path_config))
            else:
                config = load_config(path_config)
        return config

    def _set_from_config(self, device, model_fp, model_dir, cpu_extension):
        if self.config is None:
            self.device = self._set_device(device)
            self.model_fp = self._set_model_fp(model_fp)
            self.model_dir = self._set_model_dir(model_dir)
            self.cpu_extension = self._set_cpu_extension_path(cpu_extension)
        else:
            self.device = self.config["MODEL"]["DEVICE"]
            self.model_fp = self.config["MODEL"]["MODEL_DIR"]
            self.model_dir = self.config["MODEL"]["MODEL_FP"]
            self.cpu_extension = self.config["MODEL"]["CPU_EXTENSION"]

    def _set_device(self, device):
        if device is None:
            device = "CPU"
        if device not in ['CPU', 'GPU']:
            raise NotImplementedError("Only CPU and GPU is supported")
        return device

    def _set_model_fp(self, model_fp):
        if model_fp is None:
            model_fp = "FP32"
        if model_fp not in ["FP32", "FP16"]:
            raise NotImplementedError("Only FP32 and FP16 are supported")
        return model_fp

    def _set_model_dir(self, model_dir):
        if model_dir is None:
            model_dir = os.path.join(os.path.expanduser('~'), '.pyvino', 'intel_models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info("made directory for intel models. {}".format(model_dir))
        return model_dir

    def _set_cpu_extension_path(self, cpu_extension):
        if (self.device == 'CPU') and (cpu_extension is None):
            pf = platform.system()
            message = "{} on {}".format(self.model_name, pf)
            logger.info(message)
            if pf == 'Windows':
                cpu_extension = r"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"
            elif pf == 'Darwin':
                cpu_extension = r"/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.dylib"
            elif pf == 'Linux':
                cpu_extension = r"/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx2.so"
            else:
                raise NotImplementedError
            logger.info("The path to cpu_extension is {}".format(cpu_extension))        
            
            if not os.path.exists(cpu_extension):
                raise FileNotFoundError(cpu_extension)
        return cpu_extension

    def _download_model(self, format_type):
        """

        Args:
            model_name (str): model name
            format_type (str): format_type should be xml or bin
            model_fp: fp should be FP32 or FP16

        Returns:

        """
        if format_type not in ["xml", "bin"]:
            raise ValueError("format_type should be xml or bin")

        base_url = "https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/{}/{}/{}"
        path_save_dir = self.model_dir

        model_name_format = "{}.{}".format(self.model_name, format_type)
        url = base_url.format(self.model_name, self.model_fp, model_name_format)

        # save path
        path_model_fp_dir = os.path.join(path_save_dir, self.model_name, self.model_fp)

        # download
        if not os.path.exists(path_model_fp_dir):
            os.makedirs(path_model_fp_dir)
            logger.info("make config directory for saving file. Path: {}".format(path_model_fp_dir))

        path_save = os.path.join(path_model_fp_dir, model_name_format)
        if not os.path.exists(path_save):
            urllib.request.urlretrieve(url, path_save)
            logger.info("download {} successfully.".format(model_name_format))

    def _set_model_path(self):
        """

        Args:
            model_dir (str): path to intel_models directory
            model_fp (str): FP16 and FP32 are supported.

        Returns:

        """
        path_model_dir = os.path.join(self.model_dir, self.model_name, self.model_fp)
        self.model_xml = os.path.join(
            path_model_dir, self.model_name + '.xml')
        self.model_bin = os.path.join(
            path_model_dir, self.model_name + ".bin")
        self._download_model('xml')
        self._download_model("bin")

    def _set_ieplugin(self):
        """

        """
        plugin = IEPlugin(device=self.device, plugin_dirs=None)

        if self.device == "CPU":
            plugin.add_cpu_extension(self.cpu_extension)
        else:
            raise NotImplementedError("Now, Only CPU is supported")
        self.exec_net = plugin.load(network=self.net, num_requests=2)

    def _get_io_blob(self):
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

    def _get_shape(self):
        """get shape for network input

        """
        self.shape = self.net.inputs[self.input_blob].shape

    def _in_frame(self, frame):
        """transform frame for network input shape

        Args:
            frame (np.ndarray): input frame

        Returns (np.ndarray): transformed input frame

        """
        n, c, h, w = self.shape
        in_frame = cv2.resize(frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        return in_frame

    def get_result(self, frame):
        """

        Args:
            frame: input frame transformed for network

        Returns (np.ndarray): computed results

        """
        in_frame = self._in_frame(frame)
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})
        if self.exec_net.requests[0].wait(-1) == 0:
            result = self.exec_net.requests[0].outputs
        return result


class DetectorObject(Detector):
    def __init__(self, task, device=None,
                 model_fp=None, model_dir=None,
                 cpu_extension=None, path_config=None):
        self.task = task
        super().__init__(task, device,
                         model_fp, model_dir,
                         cpu_extension, path_config)

    def get_box(self, bbox, frame):
        frame_h, frame_w = frame.shape[:2]
        box = bbox[3:7] * np.array([frame_w,
                                    frame_h,
                                    frame_w,
                                    frame_h])
        xmin, ymin, xmax, ymax = box.astype("int")
        self.xmin = xmin
        self.ymin = ymin
        return xmin, ymin, xmax, ymax

    def crop_bbox_frame(self, frame, xmin, ymin, xmax, ymax):
        bbox_frame = frame[ymin:ymax, xmin:xmax]
        return bbox_frame

    def get_frame_shape(self, frame):
        self.frame_h, self.frame_w = frame.shape[:2]

    def get_pos(self, frame, max_bbox_num=False, th=0.5):
        result = self.get_result(frame)[self.out_blob]
        # prob threshold : 0.5
        bboxes = result[0][:, np.where(result[0][0][:, 2] > th)][0][0]
        if max_bbox_num:
            bbox_sizes = np.zeros((len(bboxes)))
            for bbox_num, bbox in enumerate(bboxes):
                xmin, ymin, xmax, ymax = self.get_box(bbox, frame)
                bbox_size = self.get_bbox_size(xmin, ymin, xmax, ymax)
                bbox_sizes[bbox_num] = bbox_size
            df = pd.DataFrame(bbox_sizes, columns=['bbox_size'])
            target_bbox_nums = df.sort_values(ascending=False, by='bbox_size')[0: max_bbox_num]['bbox_size'].index
            bboxes = bboxes[target_bbox_nums]
        return bboxes

    def get_bbox_size(self, xmin, ymin, xmax, ymax):
        bbox_size = (xmax - xmin) * (ymax - ymin)
        return bbox_size

    def add_bbox_margin(self, xmin, ymin, xmax, ymax, bbox_margin):
        """add margin to bbox

        Args:
            xmin:
            ymin:
            xmax:
            ymax:
            bbox_margin: margin ratio

        Returns:

        """
        xmin = int(xmin * (1 - bbox_margin))
        ymin = int(ymin * (1 - bbox_margin))
        xmax = int(xmax * (1 + bbox_margin))
        ymax = int(ymax * (1 + bbox_margin))
        return xmin, ymin, xmax, ymax

    def compute(self, init_frame, pred_flag=False, frame_flag=False, 
                max_bbox_num=None, bbox_margin=False):
        # copy frame to prevent from overdraw results
        frame = init_frame.copy()
        bboxes = self.get_pos(frame)
        results = {}
        results['preds'] = {}
        for bbox_num, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = self.get_box(bbox, frame)
            if bbox_margin:
                xmin, ymin, xmax, ymax = self.add_bbox_margin(xmin, ymin, xmax, ymax, bbox_margin=bbox_margin)
            bbox_size = self.get_bbox_size(xmin, ymin, xmax, ymax)
            if pred_flag:
                results['preds'][bbox_num] = {'label': bbox[1],
                                              'conf': bbox[2],
                                              'bbox': (xmin, ymin, xmax, ymax),
                                              'bbox_size': bbox_size}
                if max_bbox_num:
                    df = pd.DataFrame(results['preds'])
                    target_bbox_nums = list(df.loc['bbox_size'].sort_values(ascending=False)[0: max_bbox_num].index)
                    keys = list(results['preds'].keys())
                    removed_nums = list(set(target_bbox_nums) ^ set(keys))
                    for removed_num in removed_nums:
                        results['preds'].pop(removed_num)

            if frame_flag:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if frame_flag:
            results['frame'] = frame
        return results


class DetectorFace(DetectorObject):
    def __init__(self, device=None,
                 model_fp=None, model_dir=None,
                 cpu_extension=None, path_config=None):
        self.task = 'detect_face'
        super().__init__(self.task, device,
                         model_fp, model_dir,
                         cpu_extension, path_config)


class DetectorBody(DetectorObject):
    def __init__(self):
        self.task = 'detect_body'
        super().__init__(self.task)


class DetectorHeadpose(DetectorObject):
    def __init__(self):
        self.task = 'estimate_headpose'
        super().__init__(self.task)
        self.scale = 50
        self.focal_length = 950.0
        self.detector_face = DetectorFace()

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

    def compute(self, init_frame, pred_flag=False, frame_flag=False):
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


class DetectorEmotion(DetectorObject):
    def __init__(self):
        self.task = 'emotion_recognition'
        super().__init__(self.task)
        self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')
        self.detector_face = DetectorFace()

    def get_emotion(self, face_frame):
        # TODO: paste face_frame to canvas and compute. Like humanpose estiamtion.
        in_frame = self._in_frame(face_frame)
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})
        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs[self.out_blob]
            emotion = self.label[np.argmax(res[0])]
        return emotion

    def compute(self, init_frame, pred_flag=False, frame_flag=False, rect=True):
        assert isinstance(init_frame, np.ndarray)
        frame = init_frame.copy()
        faces = self.detector_face.get_pos(frame)
        results = {}
        results['preds'] = {}
        for face_num, face in enumerate(faces):
            xmin, ymin, xmax, ymax = self.get_box(face, frame)
            face_frame = self.crop_bbox_frame(frame, xmin, ymin, xmax, ymax)
            emotion = self.get_emotion(face_frame)
            if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                continue
            if pred_flag:
                results['preds'][face_num] = {'bbox': (xmin, ymin, xmax, ymax),
                                              'emotion': emotion}
            if frame_flag:
                cv2.putText(frame, emotion,
                            (int(xmin + (xmax - xmin) / 2), int(ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2, cv2.LINE_AA)
                if rect:
                    frame = cv2.rectangle(frame,
                                          (xmin, ymin), (xmax, ymax),
                                          (0, 255, 0), 2)
        if frame_flag:
            results['frame'] = frame
        return results


class DetectorHumanPose(Detector):
    """
    https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
    """

    def __init__(self):
        self.task = 'estimate_humanpose'
        super().__init__(self.task)
        self.thr_point = 0.1
        self.detector_body = DetectorBody()
        self.segmentor = Segmentor()
        self.BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                           "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                           "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                           "LEye": 15, "REar": 16, "LEar": 17}

        self.POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                           ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                           ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                           ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                           ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

        self.POSE_PARTS_FLATTEN = ['Nose_x', 'Nose_y', 'Neck_x', 'Neck_y', 'RShoulder_x', 'RShoulder_y',
                                   'RElbow_x', 'RElbow_y', 'RWrist_x', 'RWrist_y', 'LShoulder_x',
                                   'LShoulder_y', 'LElbow_x', 'LElbow_y', 'LWrist_x', 'LWrist_y', 'RHip_x',
                                   'RHip_y', 'RKnee_x', 'RKnee_y', 'RAnkle_x', 'RAnkle_y', 'LHip_x',
                                   'LHip_y', 'LKnee_x', 'LKnee_y', 'LAnkle_x', 'LAnkle_y', 'REye_x',
                                   'REye_y', 'LEye_x', 'LEye_y', 'REar_x', 'REar_y', 'LEar_x', 'LEar_y']

    def _get_heatmaps(self, frame):
        result = self.get_result(frame)
        # pairwise_relations = result['Mconv7_stage2_L1']
        heatmaps = result['Mconv7_stage2_L2']
        return heatmaps

    def get_points(self, frame):
        """get one person's pose points.

        Args:
            frame (np.ndarray): image include only one person. other part should be masked.

        Returns (np.ndarray): human jointt points

        """
        assert isinstance(frame, np.ndarray)
        heatmaps = self._get_heatmaps(frame)
        points = np.zeros((len(self.BODY_PARTS), 2))
        for num_parts in range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = heatmaps[0, num_parts, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            x, y = np.nan, np.nan
            # Add a point if it's confidence is higher than threshold.
            if conf > self.thr_point:
                x = int((frame_width * point[0]) / heatmaps.shape[3])
                y = int((frame_height * point[1]) / heatmaps.shape[2])
            points[num_parts] = x, y
        assert isinstance(points, np.ndarray)
        return points

    def draw_pose(self, init_frame, points):
        """draw pose points and line to frame

        Args:
            init_frame: frame to draw
            points: joints position values for all person

        Returns:

        """
        frame = init_frame.copy()
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in self.BODY_PARTS)
            assert (partTo in self.BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if not (np.isnan(points[idFrom][0]) or np.isnan(points[idTo][1])):
                points_from = tuple(points[idFrom].astype('int64'))
                points_to = tuple(points[idTo].astype('int64'))
                cv2.line(frame, points_from, points_to, (0, 255, 0), 3)
                cv2.ellipse(frame, points_from, (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points_to, (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        return frame

    def _filter_points(self, points, xmin, ymin, xmax, ymax):
        x = points.T[0]
        y = points.T[1]

        x = np.where(x < xmin, np.nan, x)
        x = np.where(x > xmax, np.nan, x)
        y = np.where(y < ymin, np.nan, y)
        y = np.where(y > ymax, np.nan, y)

        filtered_points = np.asarray([x, y]).T
        filtered_points[(np.isnan(filtered_points.prod(axis=1)))] = np.nan

        return filtered_points

    def _normalize_points(self, points, xmin, ymin, xmax, ymax):
        x = points.T[0]
        y = points.T[1]

        values_x = (x - xmin) / (xmax - xmin)
        values_y = (y - ymin) / (ymax - ymin)

        norm_points = np.asarray([values_x, values_y]).T

        return norm_points

    def generate_canvas(self, xmin, ymin, xmax, ymax, height, width):
        ratio_frame = width / height # 16 / 9

        height_bbox = ymax - ymin
        width_bbox = xmax - xmin
        ratio_bbox = width_bbox / height_bbox

        if ratio_bbox < ratio_frame:
            width_canvas = int(height_bbox * ratio_frame)
            canvas_org = np.zeros((height_bbox, width_canvas, 3), np.uint8)
        elif ratio_bbox > ratio_frame:
            height_canvas = int(width_bbox / ratio_frame)
            canvas_org = np.zeros((height_canvas, width_bbox, 3), np.uint8)
        elif ratio_bbox == ratio_frame:
            canvas_org = np.zeros((height_bbox, width_bbox, 3), np.uint8)
        else:
            raise ValueError
        return canvas_org

    def re_position(self, points, xmin, ymin):
        points.T[0] = points.T[0] + xmin
        points.T[1] = points.T[1] + ymin
        return points

    def compute(self, init_frame, pred_flag=False, frame_flag=False,
                normalize_flag=False, max_bbox_num=False, mask_flag=False, bbox_margin=False):
        """ frame include multi person.

        Args:
            init_frame (np.ndarray): frame
            pred_flag (bool): whether return predicted results
            frame_flag (bool): whether return frame

        Returns (dict): detected human pose points and drawn frame selectively.

        """
        frame = init_frame.copy()
        height, width, _ = frame.shape
        # canvas_org = np.zeros((height, width, 3), np.uint8)
        bboxes = self.detector_body.get_pos(frame, max_bbox_num)
        results = {}
        results['preds'] = {}
        for bbox_num, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = self.detector_body.get_box(bbox, frame)
            if bbox_margin:
                xmin, ymin, xmax, ymax = self.detector_body.add_bbox_margin(xmin, ymin, xmax, ymax, bbox_margin)
            bbox_frame = self.detector_body.crop_bbox_frame(frame, xmin, ymin, xmax, ymax)
            if (bbox_frame.shape[0] == 0) or (bbox_frame.shape[1] == 0):
                continue
            if mask_flag:
                results_mask = self.segmentor.compute(bbox_frame, pred_flag=True, max_mask_num=1)
                if len(results_mask['masks']) > 0:
                    mask = results_mask['masks'][0]
                    mask_canvas = np.zeros((3, mask.shape[0], mask.shape[1]))
                    for num in range(len(mask_canvas)):
                        mask_canvas[num] = mask
                    mask_canvas = mask_canvas.transpose(1, 2, 0)
                    bbox_frame = (bbox_frame * mask_canvas).astype(int)
                else:
                    logger.info('no mask')

            canvas = self.generate_canvas(xmin, ymin, xmax, ymax, height, width)
            canvas[0:bbox_frame.shape[0], 0:bbox_frame.shape[1]] = bbox_frame

            # get points can detect only one person.
            # exception of detected area should be 0
            points = self.get_points(canvas)
            points = self.re_position(points, xmin, ymin)
            filtered_points = self._filter_points(points, xmin, ymin, xmax, ymax)

            if pred_flag:
                results['preds'][bbox_num] = {'points': filtered_points, 'bbox': (xmin, ymin, xmax, ymax)}
                if normalize_flag:
                    norm_points = self._normalize_points(filtered_points, xmin, ymin, xmax, ymax)
                    results['preds'][bbox_num]['norm_points'] = norm_points

            if frame_flag:
                frame = self.draw_pose(frame, filtered_points)
        if frame_flag:
            results['frame'] = frame
        return results


class Segmentor(Detector):
    """Detector for segmentation.

    """
    def __init__(self):
        self.task = 'detect_segmentation'
        super().__init__(self.task)
        self.prob_threshold = 0.5
        # 81 classes for segmentation
        self.class_labels = COCO_LABEL

    def expand_box(self, box, scale):
        w_half = (box[2] - box[0]) * .5
        h_half = (box[3] - box[1]) * .5
        x_c = (box[2] + box[0]) * .5
        y_c = (box[3] + box[1]) * .5
        w_half *= scale
        h_half *= scale
        box_exp = np.zeros(box.shape)
        box_exp[0] = x_c - w_half
        box_exp[2] = x_c + w_half
        box_exp[1] = y_c - h_half
        box_exp[3] = y_c + h_half
        return box_exp

    def segm_postprocess(self, box, raw_cls_mask, im_h, im_w):
        # Add zero border to prevent upsampling artifacts on segment borders.
        raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
        extended_box = self.expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
        w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
        x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

        raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
        mask = raw_cls_mask.astype(np.uint8)
        # Put an object mask in an image mask.
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                                (x0 - extended_box[0]):(x1 - extended_box[0])]
        return im_mask

    def _input_image(self, frame, scale):
        """transform frame to proper size for network

        Args:
            frame (np.ndarray): frame
            scale: specify how much resize according to frame shape

        Returns (np.ndarray): input image and input image info

        """
        n, c, h, w = self.shape
        input_image = cv2.resize(frame, None, fx=scale, fy=scale)
        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                           (0, w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w)).astype(np.float32)
        input_image_info = np.asarray([[input_image_size[0], input_image_size[1], scale]], dtype=np.float32)
        return input_image, input_image_info

    def compute(self, init_frame, pred_flag=False, frame_flag=False,
                show_boxes=False, show_scores=False, max_mask_num=False):
        """

        Args:
            init_frame (np.ndarray): input frame
            pred_flag (bool): whether return pred result
            frame_flag (bool): whether return frame
            show_boxes (bool): whether draw boxes
            show_scores (bool): whether draw scores

        Returns (dict): predicted results

        """
        frame = init_frame.copy()
        required_input_keys = {'im_data', 'im_info'}
        assert required_input_keys == set(self.net.inputs.keys()), \
            'Demo supports only topologies with the following input keys: {}'.format(', '.join(required_input_keys))
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks'}
        assert required_output_keys.issubset(self.net.outputs.keys()), \
            'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

        n, c, h, w = self.shape
        assert n == 1, 'Only batch 1 is supported by the demo application'

        logger.info('Starting inference...')

        # Resize the image to leave the same aspect ratio and to fit it to a window of a target size.
        scale = min(h / frame.shape[0], w / frame.shape[1])
        input_image, input_image_info = self._input_image(frame, scale)

        # Run the net.
        inf_start = time.time()
        outputs = self.exec_net.infer({'im_data': input_image, 'im_info': input_image_info})
        inf_end = time.time()
        det_time = inf_end - inf_start

        # Parse detection results of the current request
        boxes = outputs['boxes'] / scale
        scores = outputs['scores']
        classes = outputs['classes'].astype(np.uint32)
        masks = []
        for box, cls, raw_mask in zip(boxes, classes, outputs['raw_masks']):
            raw_cls_mask = raw_mask[cls, ...]
            mask = self.segm_postprocess(box, raw_cls_mask, frame.shape[0], frame.shape[1])
            masks.append(mask)

        # Filter out detections with low confidence.
        detections_filter = scores > self.prob_threshold
        boxes = boxes[detections_filter].astype(int)

        if max_mask_num and len(boxes) > 0:
            boxes_df = pd.DataFrame(boxes)
            boxes_df['bbox_size'] = boxes_df.T.apply(lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            larger_nums = boxes_df.sort_values(ascending=False, by='bbox_size')[0: max_mask_num].index

            boxes = boxes[larger_nums]
            scores = scores[detections_filter][larger_nums]
            classes = classes[detections_filter][larger_nums]
            labels = np.asarray([self.class_labels[class_] for class_ in classes])
            masks = np.asarray(list(segm for segm, is_valid in zip(masks, detections_filter) if is_valid))[larger_nums]
        else:
            scores = scores[detections_filter]
            classes = classes[detections_filter]
            labels = [self.class_labels[class_] for class_ in classes]
            masks = list(segm for segm, is_valid in zip(masks, detections_filter) if is_valid)

        results = {}
        if pred_flag:
            results['scores'] = scores
            results['classes'] = classes
            results['labels'] = labels
            results['boxes'] = boxes
            results['masks'] = np.asarray(masks)

        if frame_flag:
            # Visualize masks
            visualizer = Visualizer(self.class_labels, show_boxes=show_boxes, show_scores=show_scores)
            frame = visualizer(frame, boxes, classes, scores, masks)
            results['frame'] = frame
        return results


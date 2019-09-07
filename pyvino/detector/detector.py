import os
import numpy as np
import math
import cv2
import urllib.request
from openvino.inference_engine import IENetwork, IEPlugin
from pyvino.util.config import (TASKS, load_config)
from pyvino.util.util import get_logger
import platform


logger = get_logger(__name__)


class Detector(object):
    def __init__(self, task, device=None, model_fp=None, model_dir=None, cpu_extension=None, path_config=None):
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
        if cpu_extension is None:
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
    def __init__(self, task):
        self.task = task
        super().__init__(self.task)

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

    def get_pos(self, frame):
        result = self.get_result(frame)[self.out_blob]
        # prob threshold : 0.5
        bboxes = result[0][:, np.where(result[0][0][:, 2] > 0.5)]
        return bboxes

    def get_bbox_size(self, xmin, ymin, xmax, ymax):
        bbox_size = (xmax - xmin) * (ymax - ymin)
        return bbox_size

    def compute(self, init_frame, pred_flag=False, frame_flag=False):
        # copy frame to prevent from overdraw results
        frame = init_frame.copy()
        bboxes = self.get_pos(frame)
        results = {}
        results['preds'] = {}
        for bbox_num, bbox in enumerate(bboxes[0][0]):
            xmin, ymin, xmax, ymax = self.get_box(bbox, frame)
            bbox_size = self.get_bbox_size(xmin, ymin, xmax, ymax)
            if pred_flag:
                results['preds'][bbox_num] = {'label': bbox[1],
                                              'conf': bbox[2],
                                              'bbox': (xmin, ymin, xmax, ymax),
                                              'bbox_size': bbox_size}
            if frame_flag:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if frame_flag:
            results['frame'] = frame
        return results


class DetectorFace(DetectorObject):
    def __init__(self):
        self.task = 'detect_face'
        super().__init__(self.task)


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
        for face_num, face in enumerate(faces[0][0]):
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
        for face_num, face in enumerate(faces[0][0]):
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


    def compute(self, init_frame, pred_flag=False, frame_flag=False, normalize_flag=False):
        """ frame include multi person.

        Args:
            init_frame (np.ndarray): frame
            pred_flag (bool): whether return predicted results
            frame_flag (bool): whether return frame

        Returns (dict): detected human pose points and drawn frame selectively.

        """
        frame = init_frame.copy()
        height, width, _ = frame.shape
        canvas_org = np.zeros((height, width, 3), np.uint8)
        bboxes = self.detector_body.get_pos(frame)
        results = {}
        results['preds'] = {}
        for bbox_num, bbox in enumerate(bboxes[0][0]):
            xmin, ymin, xmax, ymax = self.detector_body.get_box(bbox, frame)
            bbox_frame = self.detector_body.crop_bbox_frame(frame, xmin, ymin, xmax, ymax)
            if (bbox_frame.shape[0] == 0) or (bbox_frame.shape[1] == 0):
                continue
            canvas = canvas_org.copy()
            canvas[ymin:ymax, xmin:xmax] = bbox_frame
            # get points can detect only one person.
            # exception of detected area should be 0
            points = self.get_points(canvas)
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

    def compute_single(self, frame, pred_flag=False, frame_flag=False):
        """frame include only one person. Deteced person by DetectBody.

        Args:
            frame:
            pred_flag:
            frame_flag:

        Returns:

        """
        points = self.get_points(frame)
        points = np.asarray(points)
        results = {}
        if pred_flag:
            results['points'] = points
        if frame_flag:
            frame = self.draw_pose(frame, points)
            results['frame'] = frame
        return results

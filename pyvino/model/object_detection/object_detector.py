import os
import numpy as np
import pandas as pd

import cv2
import urllib.request


from openvino.inference_engine import IENetwork, IEPlugin
from ...util.config import (TASKS, load_config)
from ...util.image import generate_canvas
from ...util import get_logger
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
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
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


class ObjectDetector(Detector):
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

    def get_pos(self, frame, max_bbox_num=False, conf_th=0.8):
        result = self.get_result(frame)[self.out_blob]
        # prob threshold : 0.5
        bboxes = result[0][:, np.where(result[0][0][:, 2] > conf_th)][0][0]
        if max_bbox_num:
            bbox_size = (bboxes[:, 5] - bboxes[:, 3]) * (bboxes[:, 6] - bboxes[:, 4])
            idxs = np.argsort(bbox_size)[::-1]
            bboxes = bboxes[idxs[0: max_bbox_num]]            
        else:
            pass
        return bboxes

    def get_bbox_size(self, xmin, ymin, xmax, ymax):
        bbox_size = (xmax - xmin) * (ymax - ymin)
        return bbox_size
    
    def filter_bbox(self, xmin, ymin, xmax, ymax):
        xmin = np.maximum(0, xmin)
        ymin = np.maximum(0, ymin)
        xmax = np.minimum(self.width, xmax)
        ymax = np.minimum(self.height, ymax)
        return xmin, ymin, xmax, ymax

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

    def compute(self, init_frame, pred_flag=True, frame_flag=True, 
                max_bbox_num=None, bbox_margin=False, conf_th=0.8):
        # copy frame to prevent from overdraw results
        frame_org = init_frame.copy()
        
        self.height, self.width, _ = frame_org.shape
        frame = generate_canvas(0, 0, self.width, self.height)
        frame[0:self.height, 0:self.width] = frame_org
                
        bboxes = self.get_pos(frame, max_bbox_num, conf_th)
        results = {}
        results['preds'] = {}
        for bbox_num, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = self.get_box(bbox, frame)
            if bbox_margin:
                xmin, ymin, xmax, ymax = self.add_bbox_margin(xmin, ymin, xmax, ymax, bbox_margin=bbox_margin)
            xmin, ymin, xmax, ymax = self.filter_bbox(xmin, ymin, xmax, ymax)
            bbox_size = self.get_bbox_size(xmin, ymin, xmax, ymax)
            if pred_flag:
                results['preds'][bbox_num] = {'label': bbox[1],
                                              'conf': bbox[2],
                                              'bbox': (xmin, ymin, xmax, ymax),
                                              'bbox_size': bbox_size}
            if frame_flag:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if frame_flag:
            results['frame'] = frame[0:self.height, 0:self.width]
        
        return results

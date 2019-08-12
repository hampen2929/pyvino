
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

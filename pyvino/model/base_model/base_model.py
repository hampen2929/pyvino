import os

import cv2
import urllib.request
import platform
from google_drive_downloader import GoogleDriveDownloader as gdd

from openvino.inference_engine import IENetwork, IECore

from ...util.config import (TASKS, load_config)
from ...util import get_logger


logger = get_logger(__name__)


class BaseModel(object):
    def __init__(self, task, device=None,
                 model_fp=None, model_dir=None,
                 cpu_extension=None, path_config=None,
                 openvino_ver=None):
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
        # import pdb; pdb.set_trace()
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

    def _download_model(self, format_type, openvino_ver=None):
        """

        Args:
            model_name (str): model name
            format_type (str): format_type should be xml or bin
            model_fp: fp should be FP32 or FP16

        Returns:

        """
        if format_type not in ["xml", "bin"]:
            raise ValueError("format_type should be xml or bin")
        
        model_name_format = "{}.{}".format(self.model_name, format_type)
        
        if self.task=="estimate_humanpose_3d":
            # download from google drive
            if format_type == "bin":
                url = "19t_NLi8-nS1PcPJsNOgsgmMpwb5vI9TD"
            elif format_type == "xml":
                url = "10-kGlhO2KB3umqnApZOAE_Rxz8wavUOv"
            else:
                raise ValueError()
        else:
            # download from official url
            if openvino_ver is None:
                # 2019 R2
                base_url = "https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/{}/{}/{}"
            elif openvino_ver == 'R3':
                # 2019 R3
                base_url = "https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/{}/{}/{}"
            elif openvino_ver == 'R4':
                # 2019 R4
                base_url = "https://download.01.org/opencv/2019/open_model_zoo/R4/20191121_190000_models_bin/{}/{}/{}"
            else:
                raise ValueError
            url = base_url.format(self.model_name, self.model_fp, model_name_format)

        # save path
        path_model_fp_dir = os.path.join(self.model_dir, self.model_name, self.model_fp)

        # download
        if not os.path.exists(path_model_fp_dir):
            os.makedirs(path_model_fp_dir)
            logger.info("make config directory for saving file. Path: {}".format(path_model_fp_dir))

        path_save = os.path.join(path_model_fp_dir, model_name_format)
        if not os.path.exists(path_save):
            self._download(url, path_save)
            logger.info("download {} successfully.".format(model_name_format))

    def _download(self, url, path_save):
        if self.task=='estimate_humanpose_3d':
            gdd.download_file_from_google_drive(file_id=url,
                                                dest_path=path_save)
        else:
            urllib.request.urlretrieve(url, path_save)
    
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
        self.ie = IECore()

        if self.device == "CPU":
            pass
            #plugin.add_cpu_extension(self.cpu_extension)
        else:
            raise NotImplementedError("Now, Only CPU is supported")
        # self.exec_net = plugin.load(network=self.net, num_requests=2)
        
        self.exec_net = self.ie.load_network(network=self.net,
                                             num_requests=1,
                                             device_name=self.device)

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
    
    def post_process(self):
        raise NotImplementedError
    
    def pre_process(self):
        raise NotImplementedError
    
    def compute(self):
        raise NotImplementedError
    
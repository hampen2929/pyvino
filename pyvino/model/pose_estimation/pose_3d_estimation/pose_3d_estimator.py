from time import time
from math import exp as exp

import cv2
import numpy as np

from .modules.input_reader import VideoReader, ImageReader
from .modules.draw import Plotter3d, draw_poses
from .modules.parse_poses import parse_poses

from ...openvino_model.openvino_model import OpenVinoModel
from ....util.logger import get_logger
from ....util.image import gen_nan_mat


logger = get_logger(__name__)


class Pose3DEstimator(OpenVinoModel):
    model_name = 'human-pose-estimation-3d'    
    model_loc = 'google'
    xml_url = '1Vt9P8HRjmq4fE1RX0xHQGSBTSmUMPfjD'
    bin_url = '1IAUlpqcKtBRg6rbTzfKdcIbd-mhdzyc0'

    stride = 8
    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    fx = -1
    height_size = 256
    is_video = False
    
    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)
        self.base_height = self.height_size
    
    def compute(self, frames):        
        if isinstance(frames, np.ndarray):
            frames = [frames]
        results = {}
        for request_id, frame in enumerate(frames):
            results[request_id] = {}
            result = self._compute(frame, request_id)            
            results[request_id] = result
        return results

    def _compute(self, frame, request_id):
        in_frame = frame.copy()
        pre_frame = self._pre_process(in_frame, request_id)
        self._infer(pre_frame, request_id)
        result = self._post_process(frame, request_id)
        return result
            
    def _pre_process(self, frame, cur_request_id=0):
        logger.info("Starting inference...")
        print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
        print("To switch between sync/async modes, press TAB key in the output window")
        
        self.input_scale = self.base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=self.input_scale, fy=self.input_scale)
        img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % self.stride)]
        # resize input_frame to network size
        if self.fx < 0:  # Focal length is unknown
            self.fx = np.float32(0.8 * frame.shape[1])
        
        if self.h != img.shape[0] or self.w != img.shape[1]:
            self.net.reshape({self.input_blob: (self.n, self.c, img.shape[0], img.shape[1])})
            self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=self.args.device)

        img = np.transpose(img, (2, 0, 1))[None, ]
        logger.info('resized_shape is {}'.format(img.shape))
        return img
    
    def _infer(self, frame, request_id=0):
        # Start inference
        start_time = time()
        self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: frame})
        det_time = time() - start_time
        det_time = round(float(det_time), 3) * 1000
        logger.info('det time {} ms'.format(det_time))
                    
    def _post_process(self, frame, cur_request_id=0):
        # Collecting object detection results
        outputs = {}
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            infer_rets = self.exec_net.requests[cur_request_id].outputs                        
            infer_rets = (infer_rets['features'][0], infer_rets['heatmaps'][0], infer_rets['pafs'][0])
            poses_3d, poses_2d = parse_poses(infer_rets, 
                                            self.input_scale, 
                                            self.stride, self.fx, 
                                            self.is_video)
        if self.args.draw:
            draw_poses(frame, poses_2d)
            outputs['image'] = frame
        
        pose_2d_arrs = []
        pose_3d_arrs = []
        for pose_id, pose in enumerate(zip(poses_2d, poses_3d)):
            pose_2d, pose_3d = pose
            
            pose_2d = pose_2d[0:-1].reshape(-1,3)
            pose_3d = pose_3d.reshape((-1, 4))
            
            pose_2d_arrs.append(pose_2d)
            pose_3d_arrs.append(pose_3d)
        
        pose_2d_arrs = np.array(pose_2d_arrs)
        pose_3d_arrs = np.array(pose_3d_arrs)

        if len(poses_2d) == 0:
            poses_2d = gen_nan_mat((1,58))
            poses_3d = gen_nan_mat((1,76))

        outputs['poses_2d'] = pose_2d_arrs
        outputs['poses_3d'] = pose_3d_arrs
        return outputs

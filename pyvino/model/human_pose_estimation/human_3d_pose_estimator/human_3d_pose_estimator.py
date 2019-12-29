import cv2
import numpy as np
from copy import copy

from openvino.inference_engine import IECore

from ....util import get_logger
from ...base_model.base_model import BaseModel

from .modules.draw import Plotter3d, draw_poses
from .modules.parse_poses import parse_poses


logger = get_logger(__name__)


class Human3DPoseDetector(BaseModel):
    BODY_PARTS = {
        'neck': 0, 'nose': 1, 'center_balance': 2, 'l_sho': 3, 'l_elb': 4,
        'l_wri': 5, 'l_hip': 6, 'l_knee': 7, 'l_ank': 8, 'r_sho': 9, 
        'r_elb': 10, 'r_wri': 11, 'r_hip': 12, 'r_knee': 13, 'r_ank': 14,
        'r_eye': 15, 'l_eye': 16, 'r_ear': 17,'l_ear': 18
    }
    
    BODY_EDGES = np.array(
    [[0, 1],  # neck - nose
     [1, 16], [16, 18],  # nose - l_eye - l_ear
     [1, 15], [15, 17],  # nose - r_eye - r_ear
     [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
     [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
     [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
     [0, 12], [12, 13], [13, 14]])  # neck - r_hip - r_knee - r_ankle
    
    base_height = 256
    stride = 8

    # canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas_3d = np.zeros((360, 640, 3), dtype=np.uint8)

    plotter = Plotter3d(canvas_3d.shape[:2])
    # canvas_3d_window_name = 'Canvas 3D'
    # cv2.namedWindow(canvas_3d_window_name)
    # cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)
    
    def __init__(self, model_dir=None):
        self.task = 'estimate_humanpose_3d'
        super().__init__(self.task, model_dir=model_dir)
        self.thr_point = 0.1
        self.fx = -1
        self._set_data()
            
    def _set_data(self):
        self.R = [
            [0.1656794936, 0.0336560618, -0.9856051821],
            [-0.09224101321, 0.9955650135, 0.01849052095],
            [0.9818563545, 0.08784972047, 0.1680491765]
            ]
        self.t = [
            [17.76193366],
            [126.741365],
            [286.3860507]
            ]
        
    def get_result(self, img):
        """
        img: cropped as a human image
        """
        self.ie = IECore()

        input_layer = next(iter(self.net.inputs))
        n, c, h, w = self.net.inputs[input_layer].shape
        if h != img.shape[0] or w != img.shape[1]:
            self.net.reshape({input_layer: (n, c, img.shape[0], img.shape[1])})
            self.exec_net = self.ie.load_network(network=self.net,
                                                 num_requests=1,
                                                 device_name=self.device)
        img = np.transpose(img, (2, 0, 1))[None, ]

        inference_result = self.exec_net.infer(inputs={'data': img})
        return inference_result

    def pre_process(self, inference_result):
        inference_result = (inference_result['features'][0],
                            inference_result['heatmaps'][0], inference_result['pafs'][0])
        return inference_result
    
    def rotate_poses(self, poses_3d, R, t):
        R_inv = np.linalg.inv(R)
        for pose_id in range(len(poses_3d)):
            pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
            pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
            poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
        return poses_3d
    
    def resize_input(self, frame):
        self.input_scale = self.base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None,
                                fx=self.input_scale,
                                fy=self.input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % self.stride)]  # better to pad, but cut out for demo
        return scaled_img
    
    def compute(self, frame, theta=3.1415/4, phi=-3.1415/6):
        """
        frame: cropped as a human image
        """
        scaled_img = self.resize_input(frame)        
        
        if self.fx < 0:  # Focal length is unknown
            self.fx = np.float32(0.8 * frame.shape[1])

        inference_result = self.get_result(scaled_img)
        
        # TODO: if batch inference, fix
        inference_result = (inference_result['features'][0],
                            inference_result['heatmaps'][0],
                            inference_result['pafs'][0],)
        
        poses_3d, poses_2d = parse_poses(inference_result, self.input_scale, 
                                         self.stride, self.fx, is_video=False)
        edges = []
        if len(poses_3d):
            poses_3d = self.rotate_poses(poses_3d, self.R, self.t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        self.plotter.plot(self.canvas_3d, poses_3d, edges, theta, phi)
        frame_draw = draw_poses(frame, poses_2d)
        
        if len(poses_2d) > 0:        
            poses_2d = np.array(poses_2d[0][0:-1]).reshape((-1, 3)).transpose().T
            poses_2d = np.where(poses_2d < 0, np.nan, poses_2d)
            
            # norm
            height, width, _ = frame.shape
            pose_2d_norm = copy(poses_2d)
            pose_2d_norm[:, 0] = poses_2d[:, 0] / width
            pose_2d_norm[:, 1] = poses_2d[:, 1] / height
            
            # L2 norm
            l2_norm = np.linalg.norm(pose_2d_norm, ord=2)
            pose_2d_l2_norm = pose_2d_norm / l2_norm
            
        else:
            pose_2d_norm = []
            pose_2d_l2_norm = []
            
        
        preds = {'pose_2d': poses_2d, 'pose_2d_norm': pose_2d_norm, 'pose_2d_l2_norm': pose_2d_l2_norm,
                 'pose_3d': poses_3d, 'edges': edges}
        results = {'preds': preds, 'frame': frame_draw, 'frame_3d': self.canvas_3d}
        return results
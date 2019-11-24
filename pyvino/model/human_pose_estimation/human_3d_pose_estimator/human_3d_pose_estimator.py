import os
import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

from ....util import get_logger
from ...object_detection.object_detector import Detector
from ...object_detection.body_detector import BodyDetector
from ...instance_segmentation.instance_segmentor import InstanceSegmentor

from .modules.draw import Plotter3d, draw_poses
from .modules.parse_poses import parse_poses


logger = get_logger(__name__)


class Human3DPoseDetector(Detector):
    BODY_PARTS = {
        'neck': 0, 'nose': 1, 'l_sho': 2, 'l_elb': 3, 'l_wri': 4, 'l_hip': 5,
        'l_knee': 6, 'l_ank': 7, 'r_sho': 8, 'r_elb': 9, 'r_wri': 10,
        'r_hip': 11, 'r_knee': 12, 'r_ank': 13, 'r_eye': 14, 'l_eye': 15,
        'r_ear': 16,'l_ear': 17
    }
    
    base_height = 256
    stride = 8

    # canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas_3d = np.zeros((360, 640, 3), dtype=np.uint8)

    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)
    
    # def __init__(self, device='CPU', net_model_xml_path='human-pose-estimation-3d.xml'):
    #     self.task = 'estimate_humanpose_3d'
    #     super().__init__(self.task)
    #     self.thr_point = 0.1
    #     self.detector_body = BodyDetector()
    #     self.segmentor = InstanceSegmentor()
    #     self.fx = -1
    #     self._set_data()

    #     ##########
    #     self.device = device

    #     net_model_bin_path = os.path.splitext(net_model_xml_path)[0] + '.bin'
    #     self.net = IENetwork(model=net_model_xml_path, weights=net_model_bin_path)
    #     required_input_key = {'data'}
    #     assert required_input_key == set(self.net.inputs.keys()), \
    #         'Demo supports only topologies with the following input key: {}'.format(', '.join(required_input_key))
    #     required_output_keys = {'features', 'heatmaps', 'pafs'}
    #     assert required_output_keys.issubset(self.net.outputs.keys()), \
    #         'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

    #     self.ie = IECore()
    #     self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=device)
        
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
        
    def infer(self, img):
        """
        img: cropped as a human image
        """
        input_layer = next(iter(self.net.inputs))
        n, c, h, w = self.net.inputs[input_layer].shape
        if h != img.shape[0] or w != img.shape[1]:
            self.net.reshape({input_layer: (n, c, img.shape[0], img.shape[1])})
            self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=self.device)
        img = np.transpose(img, (2, 0, 1))[None, ]

        inference_result = self.exec_net.infer(inputs={'data': img})

        inference_result = (inference_result['features'][0],
                            inference_result['heatmaps'][0], inference_result['pafs'][0])
        return inference_result
    
    def pre_process(self):
        pass
    
    def rotate_poses(self, poses_3d, R, t):
        R_inv = np.linalg.inv(R)
        for pose_id in range(len(poses_3d)):
            pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
            pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
            poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
        return poses_3d
    
    def compute(self, frame, theta=3.1415/4, phi=-3.1415/6):
        """
        frame: cropped as a human image
        """
        
        input_scale = self.base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % self.stride)]  # better to pad, but cut out for demo
        if self.fx < 0:  # Focal length is unknown
            self.fx = np.float32(0.8 * frame.shape[1])
        
        inference_result = self.infer(scaled_img)
        
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, 
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
        draw_poses(frame, poses_2d)
        
        preds = {'pose_2d': poses_2d, 'pose_3d': poses_3d, 'edges': edges}
        results = {'preds': preds, 'frame': frame, 'frame_3d': self.canvas_3d}
        
        return results
import cv2
import numpy as np
from copy import copy

from ...util import get_logger
from ..object_detection.object_detector import Detector
from ..object_detection.body_detector import BodyDetector
from ..instance_segmentation.instance_segmentor import InstanceSegmentor


logger = get_logger(__name__)


class HumanPoseDetector(Detector):
    """
    https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
    """

    def __init__(self):
        self.task = 'estimate_humanpose'
        super().__init__(self.task)
        self.thr_point = 0.1
        self.detector_body = BodyDetector()
        self.segmentor = InstanceSegmentor()
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
    
    def mask_compute(self, bbox_frame):
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
        return bbox_frame

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

    def generate_canvas(self, xmin, ymin, xmax, ymax, ratio_frame=16/9):
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
    
    def extract_human_pose_points(self, xmin, ymin, xmax, ymax, bbox_frame):
        canvas = self.generate_canvas(xmin, ymin, xmax, ymax)
        canvas[0:bbox_frame.shape[0], 0:bbox_frame.shape[1]] = bbox_frame
        
        points = self.get_points(canvas)
        points = self.re_position(points, xmin, ymin)
        filtered_points = self._filter_points(points, xmin, ymin, xmax, ymax)
        return filtered_points

    def compute(self, init_frame, pred_flag=False, frame_flag=False,
                normalize_flag=False, max_bbox_num=False, mask_flag=False, 
                bbox_margin=False, save_dir=False):
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
            self.human_body_image = copy(bbox_frame)
            if (bbox_frame.shape[0] == 0) or (bbox_frame.shape[1] == 0):
                continue            
            if mask_flag:
                bbox_frame = self.mask_compute(bbox_frame)
                self.human_body_masked_image = copy(bbox_frame)

            # get points can detect only one person.
            # exception of detected area should be 0
            filtered_points = self.extract_human_pose_points(xmin, ymin, xmax, ymax, bbox_frame)

            if pred_flag:
                results['preds'][bbox_num] = {'points': filtered_points, 'bbox': (xmin, ymin, xmax, ymax)}
                if normalize_flag:
                    norm_points = self._normalize_points(filtered_points, xmin, ymin, xmax, ymax)
                    results['preds'][bbox_num]['norm_points'] = norm_points

            if frame_flag:
                frame = self.draw_pose(frame, filtered_points)
                self.human_body_masked_image = frame[ymin:ymax, xmin:xmax]
                
        if frame_flag:
            results['frame'] = frame
        return results
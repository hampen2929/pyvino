
import logging as log
import numpy as np
import math
import cv2
import time

from .visualizer import Visualizer
from pyvino.util.config import COCO_LABEL
from pyvino.detector.detector import Detector


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

    def compute(self, init_frame, pred_flag=False, frame_flag=False, show_boxes=False, show_scores=False):
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

        log.info('Starting inference...')

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
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        labels = [self.class_labels[class_] for class_ in classes]
        boxes = boxes[detections_filter].astype(int)
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

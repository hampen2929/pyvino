
import logging as log
import numpy as np
import math
import cv2
import time

from .visualizer import Visualizer
from vinopy.util.config import COCO_LABEL

from vinopy.detector.detector import Detector


class DetectorSegmentation(Detector):
    def __init__(self):
        self.task = 'detect_segmentation'
        super().__init__(self.task)
        self.class_labels = COCO_LABEL
        self.prob_threshold = 0.5

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

    def get_mask(self, frame):
        result = self.get_result(frame)
        mask = result

    def comnpute(self, frame):
        required_input_keys = {'im_data', 'im_info'}
        assert required_input_keys == set(self.net.inputs.keys()), \
            'Demo supports only topologies with the following input keys: {}'.format(', '.join(required_input_keys))
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks'}
        assert required_output_keys.issubset(self.net.outputs.keys()), \
            'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

        n, c, h, w = self.net.inputs['im_data'].shape
        assert n == 1, 'Only batch 1 is supported by the demo application'

        log.info('Loading IR to the plugin...')
        # exec_net = ie.load_network(network=self.net, device_name=args.device, num_requests=2)

        tracker = None
        # assert (args.video is None) != (args.images is None), \
        #     'Please specify either --video or --images command line argument'
        # if args.video is not None:
        #     try:
        #         video = int(args.video)
        #     except ValueError:
        #         video = args.video
        #     log.info('Using video "{}" as an image source...'.format(video))
        #     cap = cv2.VideoCapture(video)
        #     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        #     delay = 1
        #     tracker = StaticIOUTracker()
        # if args.images is not None:
        #     log.info('Using images "{}" as an image source...'.format(args.images))
        #     cap = ImagesCapture(args.images, skip_non_images=True)
        #     delay = 0

        visualizer = Visualizer(self.class_labels, show_boxes=True, show_scores=True)

        render_time = 0

        log.info('Starting inference...')
        print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

        #######

        # Resize the image to leave the same aspect ratio and to fit it to a window of a target size.
        scale = min(h / frame.shape[0], w / frame.shape[1])
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
        boxes = boxes[detections_filter]
        masks = list(segm for segm, is_valid in zip(masks, detections_filter) if is_valid)

        render_start = time.time()

        # Get instance track IDs.
        masks_tracks_ids = None
        if tracker is not None:
            masks_tracks_ids = tracker(masks, classes)

        # Visualize masks.
        frame = visualizer(frame, boxes, classes, scores, masks, masks_tracks_ids)

        # Draw performance stats.
        inf_time_message = 'Inference time: {:.3f} ms'.format(det_time * 1000)
        render_time_message = 'OpenCV rendering time: {:.3f} ms'.format(render_time * 1000)
        cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        return frame
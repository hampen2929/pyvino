import time

import cv2
import numpy as np

from ...openvino_model.basic_model import BasicModel
from .visualizer import Visualizer
from ....util.logger import get_logger


logger = get_logger(__name__)


class InstanceSegmentation(BasicModel):
    class_labels = ['__background__','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','trafficlight','firehydrant',
              'stopsign','parkingmeter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
              'handbag','tie','suitcase','frisbee','skis','snowboard','sportsball','kite','baseballbat','baseballglove','skateboard','surfboard',
              'tennisracket','bottle','wineglass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
              'hotdog','pizza','donut','cake','chair','couch','pottedplant','bed','diningtable','toilet','tv','laptop','mouse','remote','keyboard',
              'cellphone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddybear','hairdrier','toothbrush']
    
    def __init__(self, xml_path=None, fp=None, draw=False, prob_threshold=0.6):
        super().__init__(xml_path=xml_path, fp=fp, draw=draw)
        self.visualizer = Visualizer(self.class_labels, show_boxes=True, show_scores=True)

    def _compute(self, frame, request_id):
        in_frame = frame.copy()
        input_image, input_image_info = self._pre_process(in_frame, request_id)
        self._infer(input_image, input_image_info, request_id)
        result = self._post_process(frame, request_id)
        return result

    def _pre_process(self, frame, cur_request_id=0):
        # n, c, h, w = self.shape
        self.scale = min(self.h / frame.shape[0], self.w / frame.shape[1])
        input_image = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, self.h - input_image_size[0]),
                                           (0, self.w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((self.n, self.c, self.h, self.w)).astype(np.float32)
        input_image_info = np.asarray([[input_image_size[0], input_image_size[1], self.scale]], dtype=np.float32)
        return input_image, input_image_info

    def _infer(self, frame, input_image_info, request_id=0):
        self.exec_net.start_async(request_id=request_id, inputs={'im_data': frame, 'im_info': input_image_info})

    def _post_process(self, frame, cur_request_id=0):
        # Collecting object detection results
        results = {}
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            outputs = self.exec_net.requests[cur_request_id].outputs

            # Parse detection results of the current request
            boxes = outputs['boxes'] / self.scale
            scores = outputs['scores']
            classes = outputs['classes'].astype(np.uint32)
            masks = []
            for box, cls, raw_mask in zip(boxes, classes, outputs['raw_masks']):
                raw_cls_mask = raw_mask[cls, ...]
                mask = self.segm_postprocess(box, raw_cls_mask, frame.shape[0], frame.shape[1])
                masks.append(mask)

            # Filter out detections with low confidence.
            detections_filter = scores > self.args.prob_threshold
            scores = scores[detections_filter]
            classes = classes[detections_filter]
            boxes = boxes[detections_filter]
            masks = list(segm for segm, is_valid in zip(masks, detections_filter) if is_valid)

            results['scores'] = scores
            results['classes'] = classes
            results['boxes'] = boxes
            results['masks'] = masks

            render_start = time.time()

            if len(boxes) and self.args.raw_output_message:
                log.info('Detected boxes:')
                log.info('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
                for box, cls, score, mask in zip(boxes, classes, scores, masks):
                    log.info('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

            # Visualize masks.
            if self.args.draw:
                frame = self.visualizer(frame, boxes, classes, scores, masks)
                results['image'] = frame

        return results

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
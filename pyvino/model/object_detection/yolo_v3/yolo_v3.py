from time import time
from math import exp as exp

import cv2
import numpy as np

from ...openvino_model.openvino_model import OpenVinoModel
from ....util.logger import get_logger


logger = get_logger(__name__)


class YoloV3(OpenVinoModel):
    model_loc = 'google'
    model_name = 'yolo_v3'
    xml_url = '1NluXuPF5qAAjs6HYxl8IHp4uXjmp3sSX'
    bin_url = '1dPj5gx04qlDb5_AS7RjzuD2AeN4dM06z'
        
    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)
        # logger = get_logger(self.__class__.__name__)
    
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
        result = self._post_process(frame, pre_frame, request_id)
        return result
            
    def _pre_process(self, frame, cur_request_id=0):
        logger.debug("Starting inference...")
        print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
        print("To switch between sync/async modes, press TAB key in the output window")
        
        request_id = cur_request_id
        in_frame = cv2.resize(frame, (self.w, self.h))

        # resize input_frame to network size
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        return in_frame
    
    def _infer(self, frame, request_id=0):
        # Start inference
        start_time = time()
        self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: frame})
        det_time = time() - start_time
                    
    def _post_process(self, frame, in_frame, cur_request_id=0):
        # Collecting object detection results
        result = {}
        objects = list()
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            output = self.exec_net.requests[cur_request_id].outputs
            start_time = time()
            for layer_name, out_blob in output.items():
                out_blob = out_blob.reshape(self.net.layers[self.net.layers[layer_name].parents[0]].shape)
                layer_params = YoloParams(self.net.layers[layer_name].params, out_blob.shape[2])
                logger.debug("Layer {} parameters: ".format(layer_name))
                layer_params.log_params()
                objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                             frame.shape[:-1], layer_params,
                                             self.args.prob_threshold)
            parsing_time = time() - start_time

        # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if intersection_over_union(objects[i], objects[j]) > self.args.iou_threshold:
                    objects[j]['confidence'] = 0

        # Drawing objects with respect to the --prob_threshold CLI parameter
        objects = [obj for obj in objects if obj['confidence'] >= self.args.prob_threshold]

        if len(objects) and self.args.raw_output_message:
            logger.debug("\nDetected boxes for batch {}:".format(1))
            logger.debug(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

        origin_im_size = frame.shape[:-1]
        for obj in objects:
            # Validation bbox of detected object
            obj['xmin'] = int(np.maximum(obj['xmin'], 0))
            obj['ymin'] = int(np.maximum(obj['ymin'], 0))
            obj['xmax'] = int(np.minimum(obj['xmax'], origin_im_size[1]))
            obj['ymax'] = int(np.minimum(obj['ymax'], origin_im_size[0]))

            if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                logger.info('invalid bbox {}'.format(obj))
                continue
            color = (int(min(obj['class_id'] * 12.5, 255)),
                     min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
            det_label = self.labels_map[obj['class_id']] if self.labels_map and len(self.labels_map) >= obj['class_id'] else \
                str(obj['class_id'])

            if self.args.raw_output_message:
                logger.debug(
                    "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                              obj['ymin'], obj['xmax'], obj['ymax'],
                                                                              color))
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            cv2.putText(frame,
                        "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                        (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        # Draw performance stats over frame
#         inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1e3)
#         render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
#         async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id)
#         parsing_message = "YOLO parsing time is {:.3f} ms".format(parsing_time * 1e3)

#         cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
#         cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
#         cv2.putText(frame, async_mode_message, (10, int(origin_im_size[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
#                     (10, 10, 200), 1)
#         cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
        
        result = {'output': objects, 'image': frame}
        
        return result


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [logger.debug("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topologgery we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

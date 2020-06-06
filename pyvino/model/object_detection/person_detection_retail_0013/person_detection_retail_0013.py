from time import time

import cv2
import numpy as np

from ...openvino_model.openvino_model import OpenVinoModel
from ....util.logger import get_logger


logger = get_logger(__name__)


class PersonDetectorRetail0013(OpenVinoModel):
    model_name = 'person-detection-retail-0013'
    model_loc = 'intel'
    xml_url = None
    bin_url = None
    
    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)

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
        logger.info('infer time {}'.format(det_time))
                    
    def _post_process(self, frame, cur_request_id=0):
        # Collecting object detection results
        result = {}
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            detections = self.exec_net.requests[cur_request_id].outputs
            # input is only one image here
            detections = detections['detection_out'][0][0]
            output = self._decode_detections(detections, frame.shape)
            result['output'] = output
        
        if self.args.draw:
            for obj in output:
                class_id = obj[1]
                confidence = obj[2]
                xmin, ymin, xmax, ymax = obj[3:7].astype(int)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(frame,
                            "#" + str(class_id) + ' ' + str(round(confidence * 100, 1)) + ' %',
                            (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
            result['image'] = frame
        
        return result
    
    def _decode_detections(self, detection, frame_shape):
        """
        detection: [image_id, class_id, confidence, xmin, ymin, xmax, ymax, ]
        """
        print(frame_shape)
        # for detection in detections:
        detection = detection[detection[:, 2] > self.args.conf]
                    
        detection[:, 3] = (np.maximum(detection[:, 3], 0) * frame_shape[1]).astype(int)
        detection[:, 4] = (np.maximum(detection[:, 4], 0) * frame_shape[0]).astype(int)
        detection[:, 5] = (np.minimum(detection[:, 5], frame_shape[1]) * frame_shape[1]).astype(int)
        detection[:, 6] = (np.minimum(detection[:, 6], frame_shape[0]) * frame_shape[0]).astype(int)
        
        # if len(detection) > 1:
        #     detection.sort(key=lambda x: x[1], reverse=True)

        return detection
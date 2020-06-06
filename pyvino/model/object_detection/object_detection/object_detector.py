import cv2
import numpy as np

from ...openvino_model.basic_model import BasicModel
from ....util.logger import get_logger


logger = get_logger(__name__)


class ObjectDetector(BasicModel):
    
    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)

    def _post_process(self, frame, cur_request_id=0):
        # Collecting object detection results
        result = {}
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            detections = self.exec_net.requests[cur_request_id].outputs
            # input is only one image here
            detections = detections[self.output_blob][0][0]
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
        # for detection in detections:
        detection = detection[detection[:, 2] > self.args.conf]
                    
        detection[:, 3] = (np.maximum(detection[:, 3], 0) * frame_shape[1]).astype(int)
        detection[:, 4] = (np.maximum(detection[:, 4], 0) * frame_shape[0]).astype(int)
        detection[:, 5] = (np.minimum(detection[:, 5], frame_shape[1]) * frame_shape[1]).astype(int)
        detection[:, 6] = (np.minimum(detection[:, 6], frame_shape[0]) * frame_shape[0]).astype(int)
        
        return detection

import cv2
import numpy as np

from ...openvino_model.basic_model import BasicModel
from ....util.logger import get_logger


logger = get_logger(__name__)


class FacialLandmark(BasicModel):
    model_name = 'facial-landmarks-35-adas-0002'
    model_loc = 'intel'
    label = ('neutral', 'happy', 'sad', 'surprise', 'anger')

    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)
    
    def _post_process(self, frame, cur_request_id=0):
        # Collecting object detection results
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            detections = self.exec_net.requests[cur_request_id].outputs
            
            normed_landmarks = detections['align_fc3'].flatten()

            pos = []
            for num, i in enumerate(range(int(normed_landmarks.size / 2))):
                normed_x = normed_landmarks[2 * i]
                normed_y = normed_landmarks[2 * i + 1]
                x_lm = frame.shape[1] * normed_x
                y_lm = frame.shape[0] * normed_y
                
                pos.append([int(x_lm), int(y_lm)])

                if self.args.draw:
                    cv2.circle(frame, (int(x_lm), int(y_lm)), 1 + int(0.03 * frame.shape[1]), (255, 255, 0), -1)
            
            result = {}
            result['normed_landmarks'] = normed_landmarks
            result['pos'] = pos
            result['image'] = frame

        return result


import numpy as np

from ...openvino_model.basic_model import BasicModel
from ....util.logger import get_logger


logger = get_logger(__name__)


class FaceAgeGenderRecognition(BasicModel):
    model_name = 'age-gender-recognition-retail-0013'
    model_loc = 'intel'
    gender_label = ('Female', 'Male')

    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)
    
    def _post_process(self, frame, cur_request_id=0):
        # Collecting object detection results
        result = {}
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            detections = self.exec_net.requests[cur_request_id].outputs
            
            age = (detections['age_conv3'].flatten()[0]) * 100
            gender_plobs = detections['prob'].flatten()
            gender_id = int(np.argmax(gender_plobs))
            gender = self.gender_label[gender_id]
            gender_plob = gender_plobs[gender_id]

            result['age'] = round(age, 1)
            result['gender'] = gender
            result['gender_plob'] = gender_plob
                    
        return result


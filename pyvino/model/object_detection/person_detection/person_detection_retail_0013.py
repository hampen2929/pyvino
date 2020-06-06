from .person_detector import PersonDetector
from ....util.logger import get_logger


logger = get_logger(__name__)


class PersonDetectorRetail0013(PersonDetector):
    model_name = 'person-detection-retail-0013'
    model_loc = 'intel'
    
    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)

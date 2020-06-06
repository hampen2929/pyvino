from .face_detector import FaceDetector
from ....util.logger import get_logger


logger = get_logger(__name__)


class FaceDetector0100(FaceDetector):
    model_name = 'face-detection-0100'
    model_loc = 'intel'
    
    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)

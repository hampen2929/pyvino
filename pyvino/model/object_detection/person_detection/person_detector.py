from ..object_detection.object_detector import ObjectDetector
from ....util.logger import get_logger


logger = get_logger(__name__)


class PersonDetector(ObjectDetector):
    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)

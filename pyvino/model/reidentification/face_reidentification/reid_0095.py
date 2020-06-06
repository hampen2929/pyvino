from ..reidentification.reidentification import Reid
from ....util.logger import get_logger


logger = get_logger(__name__)


class FaceReid0095(Reid):
    model_name = 'face-reidentification-retail-0095'
    model_loc = 'intel'
    dim = 256
    
    def __init__(self, xml_path=None, fp=None, draw=False):
        super().__init__(xml_path=xml_path, fp=fp, draw=draw)

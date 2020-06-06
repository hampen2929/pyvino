from .instance_segmentation_security import InstanceSegmentation
from ....util.logger import get_logger


logger = get_logger(__name__)


class InstanceSegmentation0010(InstanceSegmentation):
    model_name = 'instance-segmentation-security-0010'
    model_loc = 'intel'
    
    def __init__(self, xml_path=None, fp=None, draw=False):
        super().__init__(xml_path=xml_path, fp=fp, draw=draw)

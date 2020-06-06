from ...openvino_model.basic_model import BasicModel
from ....util.logger import get_logger


logger = get_logger(__name__)


class Reid(BasicModel):
    
    def __init__(self, xml_path=None, fp=None, draw=False):
        super().__init__(xml_path=xml_path, fp=fp, draw=draw)

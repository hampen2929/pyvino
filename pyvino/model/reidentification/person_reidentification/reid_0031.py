from ..reidentification.reidentification import Reid
from ....util.logger import get_logger


logger = get_logger(__name__)


class PersonReid0031(Reid):
    model_name = 'person-reidentification-retail-0031'
    model_loc = 'intel'
    xml_url = None
    bin_url = None
    dim = 256
    
    def __init__(self, xml_path=None, fp=None, draw=False):
        super().__init__(xml_path=xml_path, fp=fp, draw=draw)

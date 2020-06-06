import cv2
import numpy as np

from .openvino_model import OpenVinoModel
from ...util.logger import get_logger


logger = get_logger(__name__)


class BasicModel(OpenVinoModel):
    model_name = ''
    model_loc = ''
    xml_url = None
    bin_url = None

    def __init__(self, xml_path=None, fp=None, conf=0.6, draw=False):
        super().__init__(xml_path, fp, conf, draw)

    def compute(self, frames):        
        if isinstance(frames, np.ndarray):
            frames = [frames]
        results = {}
        for request_id, frame in enumerate(frames):
            results[request_id] = {}
            result = self._compute(frame, request_id)            
            results[request_id] = result
        return results

    def _compute(self, frame, request_id):
        in_frame = frame.copy()
        pre_frame = self._pre_process(in_frame, request_id)
        self._infer(pre_frame, request_id)
        result = self._post_process(frame, request_id)
        return result
            
    def _pre_process(self, frame, cur_request_id=0):        
        in_frame = cv2.resize(frame, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        return in_frame
    
    def _infer(self, frame, request_id=0):
        self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: frame})
                    
    def _post_process(self, frame, cur_request_id=0):
        # Collecting object detection results
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            infer_rets = self.exec_net.requests[cur_request_id].outputs
            ########### Implement here ###########
            infer_rets = infer_rets[self.output_blob][0]
            ######################################
        results = {'output': infer_rets}
        return results

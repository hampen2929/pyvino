
import numpy as np
from PIL.JpegImagePlugin import JpegImageFile

from ..detector.detector import *
from ..segmentor.segmentor import *


class Model(object):
    """ include all model.
    
    Raises:
        NotImplementedError: No task.

    """
    def __init__(self, task):
        """construct models by specified task.
        
        Args:
            task (str): task for constructing model
        """
        assert isinstance(task, str)
        self.task = task
        self._set_detector()

    def _set_detector(self):
        """set detectors according to task

        Returns:

        """
        if self.task == 'detect_face':
            self.detector = DetectorFace()
        elif self.task == 'detect_body':
            self.detector = DetectorBody()
        elif self.task == 'emotion_recognition':
            self.detector = DetectorEmotion()
        elif self.task == 'estimate_headpose':
            self.detector = DetectorHeadpose()
        elif self.task == 'estimate_humanpose':
            self.detector = DetectorHumanPose()
        elif self.task == 'detect_segmentation':
            self.detector = Segmentor()
        else:
            raise NotImplementedError
    
    def _validate_frame(self, frame):
        """validate frame
        
        Args:
            frame (any): input frame
        
        Returns:
            [np.ndarray]: frame should be np.ndarray before computed
        """
        if isinstance(frame, JpegImageFile):
            frame = np.asarray(frame)
        elif type(frame) == np.ndarray:
            pass
        # TODO: implement mode frame type
        assert isinstance(frame, np.ndarray)
        return frame

    def predict(self, frame, frame_flag=False):
        """predict on input-frame. return predict results as dict.

        Args:
            frame (np.ndarray): frame which include something to detect
            frame_flag (bool): whether return frame

        Returns (dict): predicted results


        """
        frame = self._validate_frame(frame)
        preds = self.detector.compute(frame, pred_flag=True, frame_flag=frame_flag)
        assert isinstance(preds, dict)
        return preds

    def compute(self, frame):
        """predict and draw to frame
        
        Args:
            frame (np.ndarray): frame to compute
        
        Returns (np.ndarray): frame which estimated results is drawn
        """
        frame = self._validate_frame(frame)
        frame = self.detector.compute(frame, frame_flag=True)['frame']
        assert isinstance(frame, np.ndarray)
        return frame

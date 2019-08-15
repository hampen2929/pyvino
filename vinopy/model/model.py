
import numpy as np
import PIL

from ..detector.detector_human import (DetectorFace,
                                       DetectorBody,
                                       DetectorHeadpose,
                                       DetectorEmotion)


class Model(object):
    """ include model for detection
    
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
        if self.task == 'detect_face':
            self.detector = DetectorFace()
        elif self.task == 'detect_body':
            self.detector = DetectorBody()
        elif self.task == 'emotion_recognition':
            self.detector = DetectorEmotion()
        elif self.task == 'estimate_headpose':
            self.detector = DetectorHeadpose()
        else:
            raise NotImplementedError
    
    def _validate_frame(self, frame):
        """validate frame
        
        Args:
            frame (any): input frame
        
        Returns:
            [np.ndarray]: frame should be np.ndarray before computed
        """
        if type(frame) == PIL.JpegImagePlugin.JpegImageFile:
            frame = np.asarray(frame)
        elif type(frame) == np.ndarray:
            pass
        # TODO: implement mode frame type
        assert isinstance(frame, np.ndarray)
        return frame

    def predict(self, frame):
        preds = self.detector.compute(frame, pred_flag=True)
        assert type(preds) == dict
        return preds

    def compute(self, frame):
        """predict and draw to frame
        
        Args:
            frame (np.asarray): frame to compute
        
        Returns:
            [np.asarray]: frame with estimated results
        """
        frame = self._validate_frame(frame)
        frame = self.detector.compute(frame, frame_flag=True)
        return frame

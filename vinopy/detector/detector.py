
from ..model.model import (ModelDetectFace,
                           ModelEstimateHeadpose,
                           ModelEmotionRecognition)
from vinopy.util.config import CONFIG


class Detector(object):
    def __init__(self, task='detect_face', rect=True):
        self.task = task
        self.rect = rect
        self._set_model()

    def _set_model(self):
        if self.task == 'detect_face':
            self.model_df = ModelDetectFace('detect_face')
        elif self.task == 'emotion_recognition':
            self.model_df = ModelDetectFace('detect_face')
            self.model_er = ModelEmotionRecognition('emotion_recognition')
        elif self.task == 'estimate_headpose':
            self.model_df = ModelDetectFace('detect_face')
            self.model_eh = ModelEstimateHeadpose('estimate_headpose')
        else:
            raise NotImplementedError

    def compute(self, frame):
        if self.task == 'detect_face':
            frame = self.model_df.detect_face(frame)
        elif self.task == 'emotion_recognition':
            faces = self.model_df.get_face_pos(frame)
            frame = self.model_er.emotion_recognition(frame, faces, self.rect)
        elif self.task == 'estimate_headpose':
            faces = self.model_df.get_face_pos(frame)
            frame = self.model_eh.estimate_headpose(frame, faces)
        else:
            raise NotImplementedError

        return frame

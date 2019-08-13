
from ..model.model_detect import (ModelDetectFace,
                                  ModelDetectBody,
                                  ModelEstimateHeadpose,
                                  ModelEmotionRecognition)


class Detector(object):
    def __init__(self, task, rect=True):
        self.task = task
        self.rect = rect
        self._set_model()

    def _set_model(self):
        if self.task == 'detect_face':
            self.model_df = ModelDetectFace()
        elif self.task == 'detect_body':
            self.model_df = ModelDetectBody()
        elif self.task == 'emotion_recognition':
            self.model_df = ModelDetectFace()
            self.model_er = ModelEmotionRecognition()
        elif self.task == 'estimate_headpose':
            self.model_df = ModelDetectFace()
            self.model_eh = ModelEstimateHeadpose()
        else:
            raise NotImplementedError

    def compute(self, frame):
        if self.task == 'detect_face':
            frame = self.model_df.detect(frame)
        elif self.task == 'detect_body':
            frame = self.model_df.detect(frame)
        elif self.task == 'emotion_recognition':
            faces = self.model_df.get_pos(frame)
            frame = self.model_er.emotion_recognition(frame, faces, rect=self.rect)
        elif self.task == 'estimate_headpose':
            faces = self.model_df.get_pos(frame)
            frame = self.model_eh.estimate_headpose(frame, faces)
        else:
            raise NotImplementedError

        return frame

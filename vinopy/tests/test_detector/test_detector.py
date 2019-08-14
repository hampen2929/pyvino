import pytest
from vinopy.detector.detector import Detector
from vinopy.model.model_detect import (ModelDetectFace,
                                       ModelDetectBody,
                                       ModelEstimateHeadpose,
                                       ModelEmotionRecognition)
from PIL import Image
import numpy as np
# import pytest

TEST_FACE = './data/test/face.jpg'
TEST_BODY = './data/test/person2.jpg'


class TestDetector(object):
    def load_image(self, path_image=TEST_FACE):
        frame = np.array(Image.open(path_image))
        return frame

    @pytest.mark.parametrize('task', ['detect_face', 
                                      'detect_body',
                                      'emotion_recognition',
                                      'estimate_headpose'])
    def test_compute(self, task):
        detector = Detector(task)
        frame = self.load_image()
        detector.predict(frame)
    
    @pytest.mark.parametrize('task', ['detect_face', 
                                      'detect_body',
                                      'emotion_recognition',
                                      'estimate_headpose'])
    def test_predict(self, task):
        detector = Detector(task)
        frame = self.load_image()
        detector.predict(frame)

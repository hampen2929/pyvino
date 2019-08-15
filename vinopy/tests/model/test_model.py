import pytest
from vinopy.model.model import Model
from vinopy.detector.detector_human import (DetectorFace,
                                            DetectorBody,
                                            # DetectorHeadpose,
                                            DetectorEmotion)
from PIL import Image
import numpy as np
# import pytest

TEST_FACE = './data/test/face.jpg'
TEST_BODY = './data/test/person2.jpg'


class TestModel(object):
    def load_image(self, path_image=TEST_FACE):
        frame = np.array(Image.open(path_image))
        return frame

    @pytest.mark.parametrize('task', ['detect_face', 
                                      'detect_body',
                                      'emotion_recognition',
                                      'estimate_headpose'])
    def test_predict(self, task):
        model = Model(task)
        frame = self.load_image()
        model.predict(frame)
    
    @pytest.mark.parametrize('task', ['detect_face', 
                                      'detect_body',
                                      'emotion_recognition',
                                      'estimate_headpose'])
    def test_compute(self, task):
        model = Model(task)
        frame = self.load_image()
        model.compute(frame)

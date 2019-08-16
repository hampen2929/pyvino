import pytest
from vinopy.model.model import Model
from PIL import Image
import numpy as np


TEST_FACE = './data/test/face.jpg'
TEST_BODY = './data/test/person2.jpg'


class TestModel(object):
    def load_image(self, path_image=TEST_FACE):
        frame = np.array(Image.open(path_image))
        return frame

    @pytest.mark.parametrize('task', ['detect_face', 
                                      'detect_body',
                                      'emotion_recognition',
                                      'estimate_headpose',
                                      'estimate_humanpose'])
    def test_predict(self, task):
        model = Model(task)
        frame = self.load_image()
        results = model.predict(frame)
        assert isinstance(results, dict)
    
    @pytest.mark.parametrize('task', ['detect_face', 
                                      'detect_body',
                                      'emotion_recognition',
                                      'estimate_headpose',
                                      'estimate_humanpose'])
    def test_compute(self, task):
        model = Model(task)
        frame = self.load_image()
        results = model.compute(frame)
        assert isinstance(results, np.ndarray)

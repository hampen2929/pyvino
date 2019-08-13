
from vinopy.detector.detector import Detector
from vinopy.model.model_detect import (ModelDetectFace,
                                       ModelDetectBody,
                                       ModelEstimateHeadpose,
                                       ModelEmotionRecognition)
import cv2
from PIL import Image
import numpy as np
# import pytest

TEST_FACE = './data/test/face.jpg'
TEST_BODY = './data/test/person2.jpg'


class TestDetector(object):
    def load_image(self, path_image=TEST_FACE):
        frame = np.array(Image.open(path_image))
        # resize image with keeping frame width
        return frame

    def test_compute(self):
        frame = self.load_image()
        model = ModelDetectFace()
        faces = model.get_pos(frame)
        
        detector = Detector('detect_face')
        detector.compute(frame)

        detector = Detector('detect_body')
        detector.compute(frame)

        detector = Detector('emotion_recognition')
        detector.compute(frame)

        detector = Detector('estimate_headpose')
        detector.compute(frame)

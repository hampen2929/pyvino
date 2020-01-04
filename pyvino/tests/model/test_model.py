import os
import pytest
from pyvino.model import (FaceDetector, 
                          BodyDetector,
                          HumanPoseDetector,
                          Human3DPoseDetector,
                          HeadPoseDetector,
                          EmotionRecognizer,
                          InstanceSegmentor,
                          PersonReidentifier,
                          FaceReidentifier,)
import cv2


TEST_FACE = './data/test/face.jpg'
TEST_BODY = './data/test/person1.jpg'


class TestModel(object):
    def load_image(self, path_image=TEST_FACE):
        frame = cv2.imread(path_image)
        return frame

    @pytest.mark.parametrize('Model', [FaceDetector,
                                       BodyDetector,
                                       HumanPoseDetector,
                                       Human3DPoseDetector,
                                       HeadPoseDetector,
                                       EmotionRecognizer,
                                       InstanceSegmentor,
                                     ])
    def test_compute(self, Model):
        model = Model()
        input_frame = self.load_image(TEST_BODY)
        results = model.compute(input_frame)
        output_frame = results['frame']
        save_path = os.path.join('pyvino', 'tests', 'data', '{}.png'.format(model.task))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, output_frame)
        
    @pytest.mark.parametrize('Model', [PersonReidentifier,
                                       FaceReidentifier,
                                     ])
    def test_compute_vector(self, Model):
        model = Model()
        input_frame = self.load_image(TEST_BODY)
        outputs = model.compute(input_frame)
        dim = outputs.shape[0]
        assert dim == 256
    
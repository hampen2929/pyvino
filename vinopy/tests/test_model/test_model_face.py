

from vinopy.model.model_face import ModelDetectFace, ModelEmotionRecognition
import cv2
from PIL import Image
import numpy as np
import pytest
import sys
sys.path[1] = '/opt/intel/openvino_2019.2.242/python/python3.6'

TEST_DATA = './data/test/test.jpg'


class TestModelFace(object):
    def load_image(self):
        frame = np.array(Image.open(TEST_DATA))
        # resize image with keeping frame width
        return frame


class TestModelDetectFace(TestModelFace):
    @pytest.mark.parametrize('task', ['detect_face'])
    def test_get_face_pos(self, task):
        frame = self.load_image()
        model = ModelDetectFace(task)
        faces = model.get_face_pos(frame)

        faces_exp = np.array([[[[0.,  1.,  0.99999917,  0.8040396,  0.50772989,
                                 0.93906057,  0.74512625],
                                [0.,  1.,  0.99999273,  0.67415386,  0.20316961,
                                 0.81081396,  0.42160091],
                                [0.,  1.,  0.99998677,  0.34577927,  0.13760452,
                                 0.47722587,  0.39774776],
                                [0.,  1.,  0.99937373,  0.06222285,  0.2267226,
                                 0.22237313,  0.46430075]]]], dtype=np.float32)

        np.testing.assert_almost_equal(faces, faces_exp)


class TestModelEmotionRecognition(TestModelFace):
    def test_get_face_pos(self):
        frame = self.load_image()
        model_df = ModelDetectFace('detect_face')
        model_er = ModelEmotionRecognition('emotion_recognition')

        model_df.get_frame_shape(frame)
        faces = model_df.get_face_pos(frame)
        emotion_exp = ['happy', 'happy', 'happy', 'happy']
        for face, emotion_exp in zip(faces[0][0], emotion_exp):
            xmin, ymin, xmax, ymax = model_df.get_box(face, frame)
            face_frame = model_df.crop_face_frame(frame,
                                                  xmin, ymin, xmax, ymax)
            emotion = model_er.get_emotion(face_frame)

            assert emotion == emotion_exp

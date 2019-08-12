

from vinopy.model.model_face import (ModelDetectFace,
                                     ModelEstimateHeadpose,
                                     ModelEmotionRecognition)
from PIL import Image
import numpy as np
import pytest
import sys

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


class TestModelEstimateHeadpose(TestModelFace):
    def test_get_axis(self):
        frame = self.load_image()

        model_df = ModelDetectFace('detect_face')
        model_df.get_frame_shape(frame)
        faces = model_df.get_face_pos(frame)

        model_es = ModelEstimateHeadpose('estimate_headpose')
        headpose_exps = [(-7.8038163, 15.785929, -3.390882), 
                         (-12.603701, 9.402246, -11.0962925), 
                         (-5.01876, 23.120262, -1.7416985), 
                         (2.898665, 26.77724, 15.251921)]
        for face, headpose_exp in zip(faces[0][0], headpose_exps):
            xmin, ymin, xmax, ymax = model_df.get_box(face, frame)
            face_frame = model_df.crop_face_frame(frame,
                                                  xmin, ymin, xmax, ymax)
            headpose = model_es.get_axis(face_frame)
            headpose = np.asarray(headpose).astype(np.float32)
            headpose_exp = np.asarray(headpose_exp).astype(np.float32)
            
            np.testing.assert_almost_equal(headpose, headpose_exp)

    
    def test_get_center_face(self):
        pass


class TestModelEmotionRecognition(TestModelFace):
    def test_get_face_pos(self):
        frame = self.load_image()

        model_df = ModelDetectFace('detect_face')
        model_df.get_frame_shape(frame)
        faces = model_df.get_face_pos(frame)

        model_er = ModelEmotionRecognition('emotion_recognition')

        emotions_exp = ['happy', 'happy', 'happy', 'happy']
        for face, emotion_exp in zip(faces[0][0], emotions_exp):
            xmin, ymin, xmax, ymax = model_df.get_box(face, frame)
            face_frame = model_df.crop_face_frame(frame,
                                                  xmin, ymin, xmax, ymax)
            emotion = model_er.get_emotion(face_frame)

            assert emotion == emotion_exp

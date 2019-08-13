
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

class TestModelDetect(object):
    def load_image(self, path_image=TEST_FACE):
        frame = np.array(Image.open(path_image))
        # resize image with keeping frame width
        return frame


class TestModelDetectFace(TestModelDetect):
    def test_get_pos(self):
        frame = self.load_image()
        model = ModelDetectFace()
        faces = model.get_pos(frame)
        detector = Detector('detect_face')
        detector.compute(frame)

        faces_exp = np.array([[[[0.,  1.,  0.99999917,  0.8040396,  0.50772989,
                                 0.93906057,  0.74512625],
                                [0.,  1.,  0.99999273,  0.67415386,  0.20316961,
                                 0.81081396,  0.42160091],
                                [0.,  1.,  0.99998677,  0.34577927,  0.13760452,
                                 0.47722587,  0.39774776],
                                [0.,  1.,  0.99937373,  0.06222285,  0.2267226,
                                 0.22237313,  0.46430075]]]], dtype=np.float32)

        np.testing.assert_almost_equal(faces, faces_exp)

class TestModelDetectBody(TestModelDetect):
    def test_get_pos(self):
        frame = self.load_image(TEST_BODY)
        model = ModelDetectBody()
        bboxes = model.get_pos(frame)

        bboxes_exp = np.array([[[[0.       , 1.       , 0.9991504, 0.6607301,
                                  0.2990044, 0.7968546, 0.8540128],
                                 [0.       , 1.       , 0.9973748, 0.4410298,
                                  0.3112628, 0.6126808, 0.8403662]]]])
        
        np.testing.assert_almost_equal(bboxes, bboxes_exp)

class TestModelEstimateHeadpose(TestModelDetect):
    def test_get_axis(self):
        frame = self.load_image()

        model_df = ModelDetectFace()
        model_df.get_frame_shape(frame)
        faces = model_df.get_pos(frame)

        model_es = ModelEstimateHeadpose()
        headpose_exps = [(-7.8038163, 15.785929, -3.390882), 
                         (-12.603701, 9.402246, -11.0962925), 
                         (-5.01876, 23.120262, -1.7416985), 
                         (2.898665, 26.77724, 15.251921)]
        for face, headpose_exp in zip(faces[0][0], headpose_exps):
            xmin, ymin, xmax, ymax = model_df.get_box(face, frame)
            face_frame = model_df.crop_bbox_frame(frame,
                                                  xmin, ymin, xmax, ymax)
            headpose = model_es.get_axis(face_frame)
            headpose = np.asarray(headpose).astype(np.float32)
            headpose_exp = np.asarray(headpose_exp).astype(np.float32)
            
            np.testing.assert_almost_equal(headpose, headpose_exp)

    def test_get_center_face(self):
        pass

class TestModelEmotionRecognition(TestModelDetect):
    def test_get_emotion(self):
        frame = self.load_image()

        model_df = ModelDetectFace()
        model_df.get_frame_shape(frame)
        faces = model_df.get_pos(frame)

        model_er = ModelEmotionRecognition()

        emotions_exp = ['happy', 'happy', 'happy', 'happy']
        for face, emotion_exp in zip(faces[0][0], emotions_exp):
            xmin, ymin, xmax, ymax = model_df.get_box(face, frame)
            face_frame = model_df.crop_bbox_frame(frame,
                                                  xmin, ymin, xmax, ymax)
            emotion = model_er.get_emotion(face_frame)

            assert emotion == emotion_exp

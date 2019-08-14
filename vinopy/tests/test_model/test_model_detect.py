
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

        faces_exp = np.array([[[[0.        , 1.        , 0.99999917, 0.8044468 , 0.50868136, 0.9387975 , 0.74597126],
                                [0.        , 1.        , 0.99999475, 0.6740358 , 0.20301963, 0.81081235, 0.42199725],
                                [0.        , 1.        , 0.99998975, 0.34619942, 0.13755499, 0.47750208, 0.3995008 ],
                                [0.        , 1.        , 0.999305  , 0.06173299, 0.2270025 , 0.22192575, 0.46406496]]]], dtype=np.float32)
        
        np.testing.assert_almost_equal(faces, faces_exp)
    
    def test_predict(self):
        frame = self.load_image()
        model = ModelDetectFace()
        preds = model.predict(frame)
        preds_exp = {0: {'label': 1.0, 'conf': 0.99999917, 'bbox': (1206, 586, 1408, 860)}, 
                     1: {'label': 1.0, 'conf': 0.9999927, 'bbox': (1011, 234, 1216, 486)}, 
                     2: {'label': 1.0, 'conf': 0.99998677, 'bbox': (518, 158, 715, 459)}, 
                     3: {'label': 1.0, 'conf': 0.99937373, 'bbox': (93, 261, 333, 536)}}
        #for factor, factor_exp in (preds.items(), preds_exp.items()):
            #assert 1 == 2

        

class TestModelDetectBody(TestModelDetect):
    def test_get_pos(self):
        frame = self.load_image(TEST_BODY)
        model = ModelDetectBody()
        bboxes = model.get_pos(frame)

        bboxes_exp = np.array([[[[0.        , 1.        , 0.9991472 , 0.6621982 , 0.30289605, 0.7962606 , 0.855718  ],
                                 [0.        , 1.        , 0.9978035 , 0.4420939 , 0.3098352 , 0.6119692 , 0.84347534]]]])
                          
        np.testing.assert_almost_equal(bboxes, bboxes_exp)


class TestModelEstimateHeadpose(TestModelDetect):
    def test_get_axis(self):
        frame = self.load_image()

        model_df = ModelDetectFace()
        model_df.get_frame_shape(frame)
        faces = model_df.get_pos(frame)

        model_es = ModelEstimateHeadpose()
        headpose_exps = [(-5.459803 , 17.332203 , -2.9661326), 
                         (-11.929161,   9.150341, -10.437834), 
                         (-5.246365, 22.493275, -2.398564), 
                         (2.904601, 24.449804, 14.927055)]
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

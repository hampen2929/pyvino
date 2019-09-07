
from pyvino.detector.detector import *
from pyvino.util.tester import TestDetector
import numpy as np
import pandas as pd

TEST_FACE = './data/test/face.jpg'
TEST_BODY = './data/test/person2.jpg'


class TestDetector_(TestDetector):
    def test_init(self):
        Detector(task='detect_face')


class TestDetectorFace(TestDetector):
    def test_get_pos(self):
        frame = self.load_image(TEST_FACE)
        detector = DetectorFace()
        faces = detector.get_pos(frame)

        faces_exp = np.array([[[[0.        , 1.        , 0.99999917, 0.8044468 , 0.50868136, 0.9387975 , 0.74597126],
                                [0.        , 1.        , 0.99999475, 0.6740358 , 0.20301963, 0.81081235, 0.42199725],
                                [0.        , 1.        , 0.99998975, 0.34619942, 0.13755499, 0.47750208, 0.3995008 ],
                                [0.        , 1.        , 0.999305  , 0.06173299, 0.2270025 , 0.22192575, 0.46406496]]]], dtype=np.float32)
        
        np.testing.assert_almost_equal(faces, faces_exp)
    
    def test_compute_pred(self):
        frame = self.load_image(TEST_FACE)
        detector = DetectorFace()
        preds = detector.compute(frame, pred_flag=True)['preds']
        preds_exp = {0: {'label': 1.0, 'conf': 0.99999917, 'bbox': (1206, 587, 1408, 861)}, 
                     1: {'label': 1.0, 'conf': 0.99999475, 'bbox': (1011, 234, 1216, 487)}, 
                     2: {'label': 1.0, 'conf': 0.99998975, 'bbox': (519, 158, 716, 461)}, 
                     3: {'label': 1.0, 'conf': 0.999305,   'bbox': (92, 262, 332, 535)}}
        for num in range(len(preds)):
            assert preds[num]['label'] == preds_exp[num]['label']
            np.testing.assert_almost_equal(preds[num]['conf'], preds_exp[num]['conf'])
            np.testing.assert_almost_equal(preds[num]['bbox'], preds_exp[num]['bbox'])
        

class TestDetectorBody(TestDetector):
    def test_get_pos(self):
        frame = self.load_image(TEST_BODY)
        detector = DetectorBody()
        bboxes = detector.get_pos(frame)

        bboxes_exp = np.array([[[[0.        , 1.        , 0.9991472 , 0.6621982 , 0.30289605, 0.7962606 , 0.855718  ],
                                 [0.        , 1.        , 0.9978035 , 0.4420939 , 0.3098352 , 0.6119692 , 0.84347534]]]])
                          
        np.testing.assert_almost_equal(bboxes, bboxes_exp)

    def test_compute_pred(self):
        frame = self.load_image(TEST_FACE)
        detector = DetectorFace()
        preds = detector.compute(frame, pred_flag=True)['preds']
        preds_exp = {0: {'label': 1.0, 'conf': 0.99999917, 'bbox': (1206, 587, 1408, 861)}, 
                     1: {'label': 1.0, 'conf': 0.99999475, 'bbox': (1011, 234, 1216, 487)}, 
                     2: {'label': 1.0, 'conf': 0.99998975, 'bbox': (519, 158, 716, 461)}, 
                     3: {'label': 1.0, 'conf': 0.999305, 'bbox': (92, 262, 332, 535)}}
        
        for num in range(len(preds)):
            assert preds[num]['label'] == preds_exp[num]['label']
            np.testing.assert_almost_equal(preds[num]['conf'], preds_exp[num]['conf'])
            np.testing.assert_almost_equal(preds[num]['bbox'], preds_exp[num]['bbox'])


class TestDetectorHeadpose(TestDetector):
    def test_get_axis(self):
        frame = self.load_image(TEST_FACE)

        detector_face = DetectorFace()
        detector_face.get_frame_shape(frame)
        faces = detector_face.get_pos(frame)

        detector_headpose = DetectorHeadpose()
        headpose_exps = [(-5.459803 , 17.332203 , -2.9661326), 
                         (-11.929161,   9.150341, -10.437834), 
                         (-5.246365, 22.493275, -2.398564), 
                         (2.904601, 24.449804, 14.927055)]
        for face, headpose_exp in zip(faces[0][0], headpose_exps):
            xmin, ymin, xmax, ymax = detector_face.get_box(face, frame)
            face_frame = detector_face.crop_bbox_frame(frame,
                                                  xmin, ymin, xmax, ymax)
            headpose = detector_headpose.get_axis(face_frame)
            headpose = np.asarray(headpose).astype(np.float32)
            headpose_exp = np.asarray(headpose_exp).astype(np.float32)
            
            np.testing.assert_almost_equal(headpose, headpose_exp)

    def test_get_center_face(self):
        frame = self.load_image(TEST_FACE)

        detector_face = DetectorFace()
        detector_face.get_frame_shape(frame)
        faces = detector_face.get_pos(frame)

        detector_headpose = DetectorHeadpose()
        exps = [(1307.0, 724.0, 0), 
                (1113.5, 360.5, 0), 
                (617.5, 309.5, 0), 
                (212.0, 398.5, 0)]
        for face, exp in zip(faces[0][0], exps):
            xmin, ymin, xmax, ymax = detector_face.get_box(face, frame)
            face_frame = detector_face.crop_bbox_frame(frame,
                                                  xmin, ymin, xmax, ymax)
            center = detector_headpose.get_center_face(face_frame, xmin, ymin)
            
            np.testing.assert_almost_equal(center, exp)    


class TestDetectorEmotion(TestDetector):
    def test_get_emotion(self):
        frame = self.load_image(TEST_FACE)
        detector_face = DetectorFace()
        detector_face.get_frame_shape(frame)
        faces = detector_face.get_pos(frame)
        detector_emotion = DetectorEmotion()
        emotions_exp = ['happy', 'happy', 'happy', 'happy']
        for face, emotion_exp in zip(faces[0][0], emotions_exp):
            xmin, ymin, xmax, ymax = detector_face.get_box(face, frame)
            face_frame = detector_face.crop_bbox_frame(frame,
                                                  xmin, ymin, xmax, ymax)
            emotion = detector_emotion.get_emotion(face_frame)
            assert emotion == emotion_exp

    def test_compute(self):
        frame = self.load_image(TEST_FACE)
        detector = DetectorEmotion()
        results = detector.compute(frame, pred_flag=True)
        results = pd.DataFrame(results['preds']).T['emotion'].values
        exps = ['happy', 'happy', 'happy', 'happy']
        for result, exp in zip(results, exps):
            assert result == exp


class TestDetectorHumanPose(TestDetector):
    def test_compute(self):
        frame = self.load_image(TEST_BODY)
        detector = DetectorHumanPose()
        results = detector.compute(frame, pred_flag=True, normalize_flag=True)
        exps = np.asarray([[[768., 275.],[768., 325.],[730., 325.],[712., 375.],[730., 400.],[805., 325.],[824., 375.],[805., 400.],[749., 450.],
                            [749., 550.],[749., 625.],[786., 450.],[786., 525.],[768., 625.],[768., 250.],[768., 250.],[749., 275.],[786., 250.],[  np.nan,   np.nan]],
                           [[562., 300.],[562., 325.],[524., 325.],[487., 375.],[487., 450.],[618., 325.],[618., 400.],[618., 450.],[524., 450.],
                            [524., 525.],[543., 625.],[580., 450.],[580., 525.],[580., 600.],[562., 275.],[562., 275.],[543., 275.],[580., 275.],[np.nan, np.nan]]])
        exps_norm = np.asarray([[[0.07176471, 0.04817518],
                               [0.07176471, 0.12116788],
                               [0.02705882, 0.12116788],
                               [0.00588235, 0.19416058],
                               [0.02705882, 0.23065694],
                               [0.11529412, 0.12116788],
                               [0.13764706, 0.19416058],
                               [0.11529412, 0.23065694],
                               [0.04941177, 0.30364963],
                               [0.04941177, 0.44963503],
                               [0.04941177, 0.5591241 ],
                               [0.09294118, 0.30364963],
                               [0.09294118, 0.4131387 ],
                               [0.07176471, 0.5591241 ],
                               [0.07176471, 0.01167883],
                               [0.07176471, 0.01167883],
                               [0.04941177, 0.04817518],
                               [0.09294118, 0.01167883],
                               [    np.nan,     np.nan]],
                              [[0.13782542, 0.07703704], [0.13782542, 0.11407407], [0.07963247, 0.11407407],
                               [0.0229709 , 0.18814815], [0.0229709 , 0.29925926], [0.22358346, 0.11407407],
                               [0.22358346, 0.22518519], [0.22358346, 0.29925926], [0.07963247, 0.29925926],
                               [0.07963247, 0.41037037], [0.10872894, 0.55851852], [0.16539051, 0.29925926],
                               [0.16539051, 0.41037037], [0.16539051, 0.52148148], [0.13782542, 0.04      ],
                               [0.13782542, 0.04      ], [0.10872894, 0.04      ], [0.16539051, 0.04      ],
                               [    np.nan,     np.nan]]])
        for num, exp in zip(results['preds'], exps):
            np.testing.assert_almost_equal(results['preds'][num]['points'], exp)

        for num, exp_norm in zip(results['preds'], exps_norm):
            np.testing.assert_almost_equal(results['preds'][num]['norm_points'].astype(np.float32), exp_norm.astype(np.float32))


from vinopy.segmentor.segmentor import *
from vinopy.util.tester import TestDetector
import numpy as np

TEST_BODY = './data/test/face.jpg'


class TestSegmentor(TestDetector):
    def test_compute(self):
        frame = self.load_image(TEST_BODY)
        detector = Segmentor()
        results = detector.compute(frame, pred_flag=True)
        exps = {'scores': np.array([0.9995578 , 0.99924695, 0.9989293 , 0.9912441 ], dtype=np.float32), 'classes': np.array([1, 1, 1, 1], dtype=np.uint32),
                'labels': ['person', 'person', 'person', 'person'],
                'boxes': np.array([[1011,  481, 1496, 1148],
                                   [ 301,  111,  934, 1151],
                                   [   0,  219,  463, 1153],
                                   [ 891,  231, 1368, 1125]])}
        np.testing.assert_almost_equal(results['scores'], exps['scores'])
        assert results['labels'] == exps['labels']
        np.testing.assert_almost_equal(results['boxes'], exps['boxes'])

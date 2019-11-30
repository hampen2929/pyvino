import cv2
import numpy as np

from ...util import get_logger
from ..object_detection.object_detector import ObjectDetector
from ..object_detection.face_detector import FaceDetector


logger = get_logger(__name__)


class EmotionRecognizer(ObjectDetector):
    def __init__(self):
        self.task = 'emotion_recognition'
        super().__init__(self.task)
        self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')
        self.detector_face = FaceDetector()

    def get_emotion(self, face_frame):
        # TODO: paste face_frame to canvas and compute. Like humanpose estiamtion.
        in_frame = self._in_frame(face_frame)
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})
        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs[self.out_blob]
            emotion = self.label[np.argmax(res[0])]
        return emotion

    def compute(self, init_frame, pred_flag=True, frame_flag=True, rect=True):
        assert isinstance(init_frame, np.ndarray)
        frame = init_frame.copy()
        faces = self.detector_face.get_pos(frame)
        results = {}
        results['preds'] = {}
        for face_num, face in enumerate(faces):
            xmin, ymin, xmax, ymax = self.get_box(face, frame)
            face_frame = self.crop_bbox_frame(frame, xmin, ymin, xmax, ymax)
            emotion = self.get_emotion(face_frame)
            if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                continue
            if pred_flag:
                results['preds'][face_num] = {'bbox': (xmin, ymin, xmax, ymax),
                                              'emotion': emotion}
            if frame_flag:
                cv2.putText(frame, emotion,
                            (int(xmin + (xmax - xmin) / 2), int(ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2, cv2.LINE_AA)
                if rect:
                    frame = cv2.rectangle(frame,
                                          (xmin, ymin), (xmax, ymax),
                                          (0, 255, 0), 2)
        if frame_flag:
            results['frame'] = frame
        return results




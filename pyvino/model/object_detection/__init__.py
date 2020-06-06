from .yolo_v3.yolo_v3 import YoloV3
from .person_detection.person_detection_retail_0013 import PersonDetectorRetail0013
from .face_detection.face_detector_0100 import FaceDetector0100
from .face_detection.face_detector_0104 import FaceDetector0104


__model_factory = {
    'yolo_v3': YoloV3,
    'person_detector': PersonDetectorRetail0013,
    'face_detector_0100': FaceDetector0100,
    'face_detector_0104': FaceDetector0104,
}


def build_object_detection_model(name='yolo_v3', xml_path=None, fp=None, conf=0.6, draw=False):
    """A function wrapper for building a model.
    Args:
        name (str): model name.
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](xml_path, fp, conf, draw)


def show_avai_models():
    """Displays available models.
    """
    print(list(__model_factory.keys()))

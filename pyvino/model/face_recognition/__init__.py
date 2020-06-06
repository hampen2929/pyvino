from .face_age_gender_recognition.face_age_gender_recognition import FaceAgeGenderRecognition
from .emotion_recognition.emotion_recognition import EmotionRecognition
from .facial_landmark.facial_landmark import FacialLandmark
from .head_pose_estimation.head_pose_estimation import HedPoseEstimation

__model_factory = {
    'face_age_gender': FaceAgeGenderRecognition,
    'emotion': EmotionRecognition,
    'facial_landmark': FacialLandmark,
    'head_pose': HedPoseEstimation,
}

def build_face_recognition_model(name='face_age_gender', xml_path=None, fp=None, conf=0.6, draw=False):
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

from .person_reidentification.reid_0031 import PersonReid0031
from .person_reidentification.reid_0248 import PersonReid0248
from .face_reidentification.reid_0095 import FaceReid0095


__model_factory = {
    'person_reid_0031': PersonReid0031,
    'person_reid_0248': PersonReid0248,
    'face_reid_0095': FaceReid0095,
}


def build_person_reidentification_model(name='person_reid_0031', xml_path=None, fp=None, draw=False):
    """A function wrapper for building a model.
    Args:
        name (str): model name.
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](xml_path, fp, draw)


def show_avai_models():
    """Displays available models.
    """
    print(list(__model_factory.keys()))

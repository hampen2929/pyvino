from .instance_segmentation_security.instance_segmentation_security_0010 import InstanceSegmentation0010
from .instance_segmentation_security.instance_segmentation_security_1025 import InstanceSegmentation1025


__model_factory = {
    'instance_segmentation_0010': InstanceSegmentation0010,
    'instance_segmentation_1025': InstanceSegmentation1025,
}


def build_instance_segmentation_model(name='instance_segmentation_0010', xml_path=None, fp=None, draw=False):
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

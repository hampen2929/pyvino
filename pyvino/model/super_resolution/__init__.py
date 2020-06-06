from .image_super_resolution.image_super_resolution import ImageSuperResolution


__model_factory = {
    'image_super_resolution': ImageSuperResolution,
}


def build_super_resolution_model(name='image_super_resolution', xml_path=None, fp=None, draw=False):
    """A functaion wrapper for building a model.
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

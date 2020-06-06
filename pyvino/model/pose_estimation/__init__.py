from .pose_3d_estimation.pose_3d_estimator import Pose3DEstimator


__model_factory = {
    'pose_3d': Pose3DEstimator,
}


def build_pose_estimation_model(name='pose_3d', xml_path=None, fp=None, conf=0.6, draw=False):
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

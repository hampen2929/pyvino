import os
import urllib.request
import logging


logger = logging.getLogger("download_models")
logger.setLevel(20)
sh = logging.StreamHandler()
logger.addHandler(sh)


intel_models = ["face-detection-adas-0001",
                "emotions-recognition-retail-0003",
                "head-pose-estimation-adas-0001",
                "person-detection-retail-0013",
                "human-pose-estimation-0001",
                "instance-segmentation-security-0050"]

base_url = "https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/{}/{}/{}"
path_save_dir = os.path.join(os.path.expanduser('~'), '.pyvino', 'intel_models')
fp = 'FP32'

for intel_model in intel_models:
    model_bin = "{}.bin".format(intel_model)
    model_xml = "{}.xml".format(intel_model)

    url_bin = base_url.format(intel_model, fp, model_bin)
    url_xml = base_url.format(intel_model, fp, model_xml)
    
    # save path
    path_model_fp_dir = os.path.join(path_save_dir, intel_model, fp)

    # download
    if not os.path.exists(path_model_fp_dir):
        os.makedirs(path_model_fp_dir)
        logger.info("make config directory for saving file. Path: {}".format(path_model_fp_dir))

    path_save_bin = os.path.join(path_model_fp_dir, model_bin)
    path_save_xml = os.path.join(path_model_fp_dir, model_xml)

    urllib.request.urlretrieve(url_bin, path_save_bin)
    urllib.request.urlretrieve(url_xml, path_save_xml)

    logger.info("download {} successfully.".format(intel_model))

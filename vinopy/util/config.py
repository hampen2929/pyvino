
import configparser
import json

def load_config():
    config = configparser.ConfigParser()
    config.read('./config.ini', 'UTF-8')
    return config


def device_type(device):
    if device == "CPU":
        model_fp = 'FP32'
    else:
        raise NotImplementedError
    return model_fp


def load_json(file_name):
    f = open(file_name, 'r')
    data = json.load(f)
    f.close()
    return data


def load_task():
    tasks = load_json('./tasks.json')
    return tasks


def load_txt(path_file_name):
    with open(path_file_name, 'rt') as file:
        data = file.read().splitlines()
    return data

def load_labels(path_file_name):
    data = load_txt(path_file_name)
    return data


DEVICE = "CPU"
MODEL_FP = 'FP32'
CPU_EXTENSION = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.dylib"
# CPU_EXTENSION = C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll

# path to intel models
MODEL_DIR = "/Users/yuya/src/openvino/intel_models/"

MODEL_ZOO = "https://download.01.org/opencv/2019/open_model_zoo/R2/20190716_170000_models_bin/"
TASKS = {
  "detect_face": "face-detection-adas-0001",
  "emotion_recognition": "emotions-recognition-retail-0003",
  "estimate_headpose": "head-pose-estimation-adas-0001",
  "detect_body": "person-detection-retail-0013",
  "estimate_humanpose": "human-pose-estimation-0001",
  "detect_segmentation": "instance-segmentation-security-0050"
}

COCO_LABEL = ['__background__','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','trafficlight','firehydrant',
              'stopsign','parkingmeter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
              'handbag','tie','suitcase','frisbee','skis','snowboard','sportsball','kite','baseballbat','baseballglove','skateboard','surfboard',
              'tennisracket','bottle','wineglass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
              'hotdog','pizza','donut','cake','chair','couch','pottedplant','bed','diningtable','toilet','tv','laptop','mouse','remote','keyboard',
              'cellphone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddybear','hairdrier','toothbrush']

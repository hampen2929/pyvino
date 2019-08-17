
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


CONFIG = load_config()
DEVICE = CONFIG["MODEL"]["DEVICE"]
MODEL_DIR = CONFIG["MODEL"]["MODEL_DIR"]
MODEL_FP = device_type(DEVICE)
CPU_EXTENSION = CONFIG["MODEL"]["CPU_EXTENSION"]

TASKS = load_task()

COCO_LABEL = load_labels("./coco_labels.txt")

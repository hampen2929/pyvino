
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


CONFIG = load_config()
DEVICE = CONFIG["MODEL"]["DEVICE"]
MODEL_DIR = CONFIG["MODEL"]["MODEL_DIR"]
MODEL_FP = device_type(DEVICE)
CPU_EXTENSION = CONFIG["MODEL"]["CPU_EXTENSION"]

TASKS = load_task()

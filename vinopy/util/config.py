
import configparser


def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini', 'UTF-8')
    return config


CONFIG = read_config()

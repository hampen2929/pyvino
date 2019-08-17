
from PIL import Image
import numpy as np


class TestDetector(object):
    def load_image(self, path_image):
        frame = np.array(Image.open(path_image))
        return frame

from skimage import io
import numpy as np


class TiffImage(object):

    def __init__(self, path):
        self.path = path

    def convertToNumpyArray(self):
        im = io.imread(self.path)
        return im


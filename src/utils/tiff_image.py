from skimage import io


class TiffImage(object):

    def __init__(self, path):
        self.path = path

    def convert_to_numpy_array(self):
        im = io.imread(self.path)
        return im

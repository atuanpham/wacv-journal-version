import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Plotting(object):
    """
    :type image: numpy.ndarray
    """

    def __init__(self, image):
        self.image = image.reshape(image.shape[0], image.shape[1], image.shape[2])  # remove channel


    def plot_3d(self):
        verts, faces, *_ = measure.marching_cubes_lewiner(self.image)

        fig = plt.figure(figsize=(10, 10))  # type: Figure
        ax = fig.add_subplot(111, projection='3d')  # type: Axes3D

        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, self.image.shape[0])
        ax.set_ylim(0, self.image.shape[1])
        ax.set_zlim(0, self.image.shape[2])

        plt.show()


    def plot_slice(self, slice_index, cmap=None):
        plt.imshow(self.image[slice_index, :, :], cmap=cmap)
        plt.show()


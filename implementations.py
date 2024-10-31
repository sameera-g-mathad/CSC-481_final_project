import cv2 as cv
import os
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Image(ABC):

    def __init__(self):
        self.images = []

    @abstractmethod
    def read_images(self, filepath: str):
        pass

    @abstractmethod
    def resize(self, size: int):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class RGB(Image):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        pass

    def resize(self, widht: int, height: int):
        pass

    def read_images(self, filepath):
        pass


class Grayscale(Image):
    def __init__(self):
        super().__init__()

    def read_images(self, filepath):
        try:
            os.chdir(filepath)
            for file in os.listdir(filepath):
                image = cv.imread(f"./{file}", cv.IMREAD_GRAYSCALE)
                self.images.append(image)

        except FileNotFoundError as error:
            print(error)

    def resize(self, size: int = 300):
        """
        @param size: Size in int to reshape the image. The image will be reshaped into size x size pixels.
        """
        for index, image in enumerate(self.images):
            self.images[index] = cv.resize(self.images[index], (size, size))  # type: ignore

    def __repr__(self):
        return f"Grayscale - {len(self.images)} images"


class SimpleImageFactory:
    def type(self, format: str) -> Image:
        format = format.lower()
        match (format):
            case "gray":
                return Grayscale()
            case _:
                return RGB()


class HOG:
    def __init__(self, imageObj) -> None:
        self.images = imageObj.images
        # print(len(self.images))

    def calc_gradients(self, image):
        grad_x = np.zeros(image.shape, dtype=np.float32)
        grad_y = np.zeros(image.shape, dtype=np.float32)
        image = image.astype(np.float32)
        width, height = image.shape
        print(width, height)
        for row in range(width):
            for col in range(height):
                # calculating grad_x
                if col == 0:  # handles leftmost pixel
                    left = 0
                    right = image[row, col + 1]
                elif col == height - 1:  # handles rightmost pixel
                    left = image[row, col - 1]
                    right = 0
                else:
                    left = image[row, col - 1]
                    right = image[row, col + 1]

                grad_x[row, col] = right - left

                # calculating grad_y
                if row == 0:  # handles upmost pixel
                    up = 0
                    down = image[row + 1, col]
                elif row == width - 1:  # handles downmost pixel
                    up = image[row - 1, col]
                    down = 0
                else:
                    up = image[row - 1, col]
                    down = image[row + 1, col]

                grad_y[row, col] = up - down
        return grad_x, grad_y

    def normalize(self, histogram):
        return histogram / (np.sqrt(np.sum(np.pow(histogram, 2))) + 1e-6)

    def compute_hist(self, gx, gy):
        total_angle = 180
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gx, gy) * (180 / np.pi)
        direction[direction < 0] += total_angle

        num_bins = 9
        histogram = [0] * num_bins
        bin_width = total_angle // num_bins
        for index in range(len(direction)):
            bin_number = int(direction[index] / bin_width)
            bin_number %= num_bins
            histogram[bin_number] += magnitude[index]

        return self.normalize(histogram)

    def process(self, **kwargs):
        self.hog = []
        block_x, block_y = (8, 8)
        for i, image in enumerate(self.images):
            histogram = []
            gx, gy = self.calc_gradients(image)
            magnitude = np.sqrt(gx**2 + gy**2)
            width, height = image.shape
            for w_region in range(0, width, block_x):
                for h_region in range(0, height, block_y):
                    region_grad_x = gx[
                        w_region : w_region + block_x, h_region : h_region + block_y
                    ]
                    region_grad_y = gy[
                        w_region : w_region + block_x, h_region : h_region + block_y
                    ]
                    histogram.extend(
                        self.compute_hist(
                            region_grad_x.flatten(), region_grad_y.flatten()
                        )
                    )
            self.hog.append(histogram)


def to_hog(imageObj: Image, **kwargs) -> list:
    """
    To compute Histogram of gradients of an image.
    :param imageObj: The image object that contains all the images to be converted to HOG features.

    :return: Returns list of HOG features of images.
    """
    hog = HOG(imageObj)
    hog.process(**kwargs)
    return hog.hog


def read_images(filepath: str, format: str):
    """
    :param filepath: Relative filepath to the directory of images. Have to read images separately incase of multiple directories.
    :param format: Format to read the image. Currently only ['gray'], i.e one format is supported.

    :return Image: Instance of an image. Currently *Grayscale* is returned.

    """
    image: Image = SimpleImageFactory().type(format)
    image.read_images(filepath)
    return image

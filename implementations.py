import cv2 as cv
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Image(ABC):

    def __init__(self):
        self.images = []
        self.location = []

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
                self.location.append(f"{filepath}/{file}")
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

    def rotate(self):
        rotations = [
            cv.ROTATE_90_CLOCKWISE,
            cv.ROTATE_90_COUNTERCLOCKWISE,
            cv.ROTATE_180,
        ]
        for index, image in enumerate(self.images):
            self.images[index] = cv.rotate(image, np.random.choice(rotations))

    def __repr__(self):
        for i in range(2):
            index = np.random.randint(0, len(self.images))
            plt.imshow(self.images[index], cmap="gray")
            plt.axis("off")
            plt.show()
        return f"Grayscale - {len(self.images)} images"


class SimpleImageFactory:
    def type(self, format: str) -> Image:
        format = format.lower()
        match (format):
            case "gray":
                return Grayscale()
            case _:
                return RGB()


class Descriptors:
    def __init__(self, imageObj: Image) -> None:
        self.images = imageObj.images
        self.location = imageObj.location

    def normalize(self, histogram) -> list[float]:
        return histogram / (np.sqrt(np.sum(np.pow(histogram, 2))) + 1e-6)

    def compute_hist(self, gx, gy) -> list[float]:
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

    @abstractmethod
    def process(
        self,
        add_image_location: bool,
        append_value: int,
        to_append: bool,
        block: int = 8,
    ) -> list:
        pass


class HOG(Descriptors):
    def __init__(self, imageObj: Image) -> None:
        super().__init__(imageObj)

    def calc_gradients(self, image):
        grad_x = np.zeros(image.shape, dtype=np.float32)
        grad_y = np.zeros(image.shape, dtype=np.float32)
        image = image.astype(np.float32)
        width, height = image.shape
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

    def process(
        self,
        add_image_location: bool,
        append_value: int,
        to_append: bool,
        block: int = 8,
    ) -> list:
        self.hog = []
        block_x, block_y = (block, block)
        for i, image in enumerate(self.images):
            histogram = [self.location[i]] if add_image_location else []
            gx, gy = self.calc_gradients(image)
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
            if to_append:
                histogram.append(append_value)
            self.hog.append(histogram)
        return self.hog


class SIFT(Descriptors):
    def __init__(self, imageObj) -> None:
        super().__init__(imageObj)

    def calc_scales(
        self,
        image: np.ndarray,
        num_scale=5,
        std=1.0,
        ksize: tuple[int, int] = (3, 3),
        scale_factor=2,
    ) -> list[np.ndarray]:
        blurred_images = []
        for i in range(num_scale):
            blurred_image = cv.GaussianBlur(image, ksize, std)
            blurred_images.append(blurred_image)
            std *= scale_factor
        return blurred_images

    def detect_keypoints(self, blurred_images: list[np.ndarray], threshold=0.2):
        keypoints = []
        for index in range(1, len(blurred_images) - 1):
            # blurred_images[index + 1] = blurred_images[index + 1].astype(float) / 255
            # blurred_images[index - 1] = blurred_images[index - 1].astype(float) / 255
            diff_of_grad_images = blurred_images[index + 1] - blurred_images[index - 1]
            diff_of_grad = (diff_of_grad_images > threshold) | (
                diff_of_grad_images < -threshold
            )
            for x in range(1, len(diff_of_grad_images)):
                for y in range(1, len(diff_of_grad_images)):
                    if diff_of_grad[x, y] == 1:
                        keypoints.append((x, y, index))
        return keypoints

    def descriptors(
        self, image, keypoints: list[tuple[int, int, int]], block, clip: int
    ):
        size = block // 2
        _descriptors = []
        for x, y, scale in keypoints:
            window = image[abs(x - size) : x + size, abs(y - size) : y + size]
            grad_x = np.gradient(window, axis=1)
            grad_y = np.gradient(window, axis=0)
            _descriptors.extend(self.compute_hist(np.ravel(grad_x), np.ravel(grad_y)))

        if len(_descriptors) < clip:
            _descriptors = np.pad(
                _descriptors,
                (0, clip - len(_descriptors)),
                mode="constant",
                constant_values=(0, 0),
            ).tolist()
        return _descriptors[:clip]

    def process(
        self,
        add_image_location: bool,
        append_value: int,
        to_append: bool,
        block: int = 8,
    ) -> list:
        self.sift = []
        for i, image in enumerate(self.images[:100]):
            histogram = [self.location[i]] if add_image_location else []
            # first calculate all the scales using gaussian blur that detects different images.
            blurred_images = self.calc_scales(image)
            keypoints = self.detect_keypoints(blurred_images)
            descriptors = self.descriptors(image, keypoints, block, 1000)
            histogram.extend(descriptors)
            if to_append:
                histogram.append(append_value)
            self.sift.append(histogram)
        return self.sift


class ImagePacker:
    def __init__(self, data: list[Image], resize: int = 200) -> None:
        self.data = []
        self.data = self.add_data(data, resize)

    def add_data(self, data: list[Image], resize):
        output = []
        for imageObj in data:
            imageObj.resize(resize)
            output.append(imageObj)
        return output

    def image_to_descriptor(self, data: list[Image], cls, **kwargs) -> list[float]:
        _features = []
        append_label = kwargs.get("append_label", [])
        block_size = kwargs.get("block_size", 8)
        add_image_location = kwargs.get("add_image_location", True)
        value = 0
        to_append = False
        if type(append_label).__name__ == "list" and len(append_label) > 0:
            to_append = True
        for index, imageObj in enumerate(data):
            if to_append:
                value = append_label[index]
            descriptor = cls(imageObj)
            _features.extend(
                descriptor.process(add_image_location, value, to_append, block_size)
            )
        return _features

    def rotate_images(self):
        for index, imageObj in enumerate(self.data):
            imageObj.rotate()

    def to_hog(self, **kwargs) -> pd.DataFrame | list:
        """
        To compute Histogram of gradients of an image.
        :param imageObj: The image object that contains all the images to be converted to HOG features.

        :return: Returns list of HOG features of images.
        """
        hog = self.image_to_descriptor(self.data, HOG, **kwargs)
        return hog

    def to_sift(self, **kwargs) -> pd.DataFrame | list:
        """
        To compute Histogram of gradients of an image.
        :param imageObj: The image object that contains all the images to be converted to HOG features.

        :return: Returns list of HOG features of images.
        """
        sift = self.image_to_descriptor(self.data, SIFT, **kwargs)
        return sift

    def descriptor_to_df(self, data, shuffle=True, target_col_name="target"):
        df = pd.DataFrame(data)
        df = df.rename(columns={df.columns[-1]: target_col_name})
        if shuffle:
            df = df.sample(frac=1)
        return df


def read_images(filepath: str, format: str):
    """
    :param filepath: Relative filepath to the directory of images. Have to read images separately incase of multiple directories.
    :param format: Format to read the image. Currently only ['gray'], i.e one format is supported.

    :return Image: Instance of an image. Currently *Grayscale* is returned.

    """
    image: Image = SimpleImageFactory().type(format)
    image.read_images(filepath)
    return image

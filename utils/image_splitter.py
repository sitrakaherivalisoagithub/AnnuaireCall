import cv2
from typing import List
import numpy as np


class ImageSplitter:

    def __init__(self, filename: str):
        self.image = cv2.imread(filename)
        self.inverted_gray_image = None

    def gray_scale_image(self):
        """Convert the image to grayscale"""
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Invert the grayscale image
        self.inverted_gray_image = cv2.bitwise_not(gray_image)

    def split_image(self):
        self.gray_scale_image()

        margin = self.inverted_gray_image.shape[1] // 10
        y_max = self.inverted_gray_image.shape[1]

        self.inverted_gray_image[:, 0:margin] = 255
        self.inverted_gray_image[:, y_max - margin: y_max] = 255

        line_median = self.inverted_gray_image.shape[1] // 2
        first_gap = np.argmin(self.inverted_gray_image[:, 0:line_median].sum(axis=0))
        second_gap = np.argmin(self.inverted_gray_image[:, line_median: y_max].sum(axis=0))

        second_gap = second_gap + line_median
        image1 = self.image[:, 0: first_gap]
        image2 = self.image[:, first_gap: second_gap]
        image3 = self.image[:, second_gap: self.image.shape[1]]

        # save image
        cv2.imwrite('image1.jpg', image1)
        cv2.imwrite('image2.jpg', image2)
        cv2.imwrite('image3.jpg', image3)

        return image1, image2, image3

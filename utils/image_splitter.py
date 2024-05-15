import cv2
import numpy as np
from sklearn.cluster import DBSCAN


class ImageSplitter:
    def __init__(self, filename: str, naive_method: bool = False):
        self.image = cv2.imread(filename)
        self.inverted_gray_image = None
        self.gaps = []
        self.naive_method = naive_method

    def convert_to_gray(self):
        """Convert the image to grayscale"""
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.inverted_gray_image = cv2.bitwise_not(gray_image)
        self._ignore_border()

    def _ignore_border(self):
        """Ignore border pixels in the grayscale image"""
        margin = self.inverted_gray_image.shape[1] // 10
        y_max = self.inverted_gray_image.shape[1]
        self.inverted_gray_image[:, 0:margin] = 255
        self.inverted_gray_image[:, y_max - margin: y_max] = 255

    def split_image(self):
        self.convert_to_gray()

        if self.naive_method:
            self._naive_algorithm()
        else:
            self._sophisticated_algorithm()

    def _sophisticated_algorithm(self):
        sorted_array = np.sort(self.inverted_gray_image.sum(axis=0))
        index_sorted_array = np.argsort(self.inverted_gray_image.sum(axis=0))
        data = index_sorted_array[(sorted_array / self.inverted_gray_image.shape[0]) < 1]
        X = data.reshape(-1, 1)
        dbscan = DBSCAN(eps=10, min_samples=2)
        dbscan.fit(X)
        labels = dbscan.labels_
        unique_labels = np.unique(labels)
        groups = {}
        for label in unique_labels:
            if label == -1:
                print("Noise:", data[labels == label])
            else:
                groups[f"Group{label}"] = data[labels == label]
        self._calculate_gaps(groups)
        self._save_images()

    def _calculate_gaps(self, groups):
        for value in groups.values():
            self.gaps.append(int(np.median(value)))
        self.gaps = sorted(self.gaps)

    def _save_images(self):
        images = []
        previous_left = 0
        for i, gap in enumerate(self.gaps):
            images.append(self.image[:, previous_left: gap])
            previous_left = gap
        images.append(self.image[:, previous_left: self.inverted_gray_image.shape[1]])
        for i, img in enumerate(images):
            cv2.imwrite(f"image{i+1}.jpg", img)

    def _naive_algorithm(self):
        y_max = self.inverted_gray_image.shape[1]
        line_median = y_max // 2
        first_gap = np.argmin(self.inverted_gray_image[:, 0:line_median].sum(axis=0))
        second_gap = np.argmin(self.inverted_gray_image[:, line_median:y_max].sum(axis=0)) + line_median
        self._save_individual_images(first_gap, second_gap)

    def _save_individual_images(self, first_gap, second_gap):
        image1 = self.image[:, 0: first_gap]
        image2 = self.image[:, first_gap: second_gap]
        image3 = self.image[:, second_gap: self.image.shape[1]]
        cv2.imwrite('image1.jpg', image1)
        cv2.imwrite('image2.jpg', image2)
        cv2.imwrite('image3.jpg', image3)

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


class ImageSplitter:

    def __init__(self, filename: str, naive_method: bool = False):
        self.image = cv2.imread(filename)
        self.inverted_gray_image = None
        self.gaps = []
        self.naive_method = naive_method

    def gray_scale_image(self):
        """Convert the image to grayscale"""
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Invert the grayscale image
        self.inverted_gray_image = cv2.bitwise_not(gray_image)

        # Ignore border
        margin = self.inverted_gray_image.shape[1] // 10
        y_max = self.inverted_gray_image.shape[1]

        self.inverted_gray_image[:, 0:margin] = 255
        self.inverted_gray_image[:, y_max - margin: y_max] = 255

    def split_image(self):
        self.gray_scale_image()

        if self.naive_method:
            self.naive_algorithm()
        else:
            self.sophisticated_algorithm()

    def sophisticated_algorithm(self):

        sorted_array = np.sort(self.inverted_gray_image.sum(axis=0))
        index_sorted_array = np.argsort(self.inverted_gray_image.sum(axis=0))

        # Index Array
        data = index_sorted_array[(sorted_array / self.inverted_gray_image.shape[0]) < 1]

        # Réorganiser les données en colonne
        X = data.reshape(-1, 1)

        # Utiliser DBSCAN pour trouver les clusters
        dbscan = DBSCAN(eps=10, min_samples=2)
        dbscan.fit(X)

        # Obtenir les labels des clusters
        labels = dbscan.labels_

        # Extraire les valeurs uniques de chaque cluster
        unique_labels = np.unique(labels)

        groups = {}
        # Afficher les blocs trouvés
        for label in unique_labels:
            if label == -1:
                print("Noise:", data[labels == label])
            else:
                groups["Group{}".format(label)] = data[labels == label]

        for value in groups.values():
            self.gaps.append(int(np.median(value)))

        # important
        self.gaps = sorted(self.gaps)

        images = []
        previous_left = 0

        for i, gap in enumerate(self.gaps):
            images.append(self.image[:, previous_left: gap])
            previous_left = gap
        images.append(self.image[:, previous_left: self.inverted_gray_image.shape[1]])

        # save image
        for i, im in enumerate(images):
            cv2.imwrite('image' + str(i+1) + '.jpg', im)

    def naive_algorithm(self):

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

        # return image1, image2, image3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from more_itertools import locate


class Visualizer:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def image_count(self):
        d = {}
        class_ids = list(set(self.y))
        for class_id in class_ids:
            d[class_id] = self.y.count(class_id)
        return d

    def data_distribution_barplot(self):
        unique, count = np.unique(self.y, return_counts=True)
        figure = plt.figure(figsize=(20, 20))
        sn.barplot(unique, count).set_title('Data Distribution')
        plt.show()

    def display_one_image(self):
        idx = []
        class_ids = list(set(self.y))
        for class_id in class_ids:
            idx.append(self.x[self.y.index(class_id)])
        cols = 4
        if len(class_ids) % cols == 0:
            rows = int(len(class_ids)/cols)
        else:
            rows = int(len(class_ids)/cols) + 1
        figure = plt.figure(figsize=(20, 20))
        for i in range(len(idx)):
            plt.subplot(rows, cols, i)
            plt.imshow(idx[i])
            plt.axis('off')
            plt.title(class_ids[i])
        plt.show()

    def display_images_of_category(self, n, class_id):
        idx = list(locate(self.y, lambda cat: cat == class_id))
        cols = 4
        idx = idx[:n]
        if n % cols == 0:
            rows = int(n/cols)
        else:
            rows = int(n/cols) + 1
        figure = plt.figure(figsize=(20, 20))
        for i in range(n):
            plt.subplot(rows, cols, i)
            plt.imshow(self.x[idx[i]])
            plt.axis('off')
        plt.show()

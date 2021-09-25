import cv2
import numpy as np
import pandas as pd
import csv
import os


class DatasetLoader:
    def __init__(self, flag, parameters):
        self.flag = flag
        if flag == 'image_from_directory':
            self.datadir = parameters[0]
            self.height = parameters[1]
            self.width = parameters[2]
            self.channels = parameters[3]

        elif flag == 'image_from_csv':
            self.file_path = parameters[0]
            self.height = parameters[1]
            self.width = parameters[2]
            self.channels = parameters[3]
            self.original_shape = parameters[4]

        elif flag == 'dataset_from_csv':
            self.file_path = parameters[0]

    def load_images_from_directory(self):
        if self.flag == 'image_from_directory':
            x = []
            y = []
            for file in os.listdir(self.datadir):
                path = os.path.join(self.datadir, file)
                for img in os.listdir(path):
                    image = cv2.imread(os.path.join(path, img))
                    if self.channels == 1 and image.shape[2] != 1:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (self.height, self.width))
                    x.append(image)
                    y.append(file)
            return x, y

        elif self.flag == 'image_from_csv':
            x = []
            y = []
            with open(self.file_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    label = row[0]
                    img = np.array([int(a) for a in row[1:]], dtype='uint8')
                    img = img.reshape(self.original_shape)
                    if self.channels == 1 and img.shape[2] != 1:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    img = cv2.resize(img, (self.height, self.width))
                    x.append(img)
                    y.append(label)
            return x, y

        elif self.flag == 'dataset_from_csv':
            df = pd.read_csv(self.file_path)
            return df

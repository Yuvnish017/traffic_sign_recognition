import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from more_itertools import locate
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


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

    def predict_on_image(self, model, image):
        try:
            image = np.array(image)
            if image.shape == 3:
                image = np.expand_dims(image, axis=-1)
            prediction = model.predict(image)
            prediction = prediction.tolist()
            m = max(prediction[0])
            class_id = prediction[0].index(m)
            figure = plt.figure(figsize=(5, 5))
            plt.imshow(image)
            plt.axis('off')
            plt.title(class_id)
            plt.show()
        except:
            print('Image shape inconsistent with model input shape')

    def classification_report(self, model):
        x = np.array(self.x)
        y = np.array(self.y)
        y = to_categorical(y)
        y_pred = model.predict(x)
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        print(classification_report(y, y_pred))

    def confusion_matrix(self, model):
        x = np.array(self.x)
        y = np.array(self.y)
        y_pred = model.predict(x)
        matrix = confusion_matrix(y, y_pred)
        df = pd.DataFrame(matrix, index=list(set(self.y)), columns=list(set(self.y)))
        figure = plt.figure(figsize=(10, 10))
        sn.heatmap(df, annot=True, fmt='d')
        plt.show()

    def accuracy_and_loss_plots(self, model):
        plt.plot(model.history.history['accuracy'], label='Train_accuracy')
        plt.plot(model.history.history['val_accuracy'], label='Test_accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper left")
        plt.show()

        plt.plot(model.history.history['loss'], label='Train_loss')
        plt.plot(model.history.history['val_loss'], label='Test_loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        plt.show()

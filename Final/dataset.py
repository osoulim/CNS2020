import cv2
import glob as gb
import random


class Dataset:
    def __init__(self, dataset_directory, subjects, image_size=(30, 30), filters=None):
        self.data = []
        for index, subject in enumerate(subjects):
            for image_name in gb.glob("%s/%s/*.jpg" % (dataset_directory, subject)):
                img = cv2.imread(image_name)
                img = cv2.resize(img, image_size)
                for image_filter in filters:
                    img = image_filter(img)
                self.data.append((img, index))  # (image, label)

    def get_data(self, train_percentage=0.7):
        random.shuffle(self.data)
        train_size = int(len(self.data) * train_percentage)
        train_data, train_label = [[i for i, j in self.data[0: train_size]], [j for i, j in self.data[0: train_size]]]
        test_data, test_label = [[i for i, j in self.data[train_size:]], [j for i, j in self.data[train_size:]]]
        return train_data, train_label, test_data, test_label


if __name__ == "__main__":
    bgr2gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    dataset = Dataset('data', ['ant', 'camera'], (100, 100), (bgr2gray, ))
    cv2.imshow("sag", dataset.get_data()[2][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

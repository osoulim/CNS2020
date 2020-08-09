import cv2
import glob as gb
import random


class Dataset:
    def __init__(self, dataset_directory, subjects, image_size=(30, 30)):
        self.data = []
        for index, subject in enumerate(subjects):
            for image_name in gb.glob("%s/%s/*.jpg" % (dataset_directory, subject)):
                img = cv2.imread(image_name)
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.data.append((img, index))  # (image, label)

    def get_data(self, filters, train_percentage=0.7):
        random.shuffle(self.data)
        batch_data = []
        for image, label in self.data:
            temp = []
            for image_filter in filters:
                temp.append(image_filter(image))
            batch_data.append((temp, label))

        train_size = int(len(batch_data) * train_percentage)
        train_labeled_data = sorted(batch_data[0: train_size], key=lambda x: x[1])
        train_data, train_label = [[i for i, j in train_labeled_data], [j for i, j in train_labeled_data]]
        test_data, test_label = [[i for i, j in batch_data[train_size:]], [j for i, j in batch_data[train_size:]]]
        return train_data, train_label, test_data, test_label


if __name__ == "__main__":
    bgr2gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    dataset = Dataset('data', ['ant', 'camera'], (100, 100))
    cv2.imshow("sag", dataset.get_data([bgr2gray, ])[2][1][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

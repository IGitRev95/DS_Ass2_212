import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt
import DataLoader


def main():
    (x_train, y_train) = retrieveNecessaryData()




    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    # for i in range(0, 5):
    #     r = random.randint(1, 10000)
    #     images_2_show.append(x_test[r])
    #     titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

    DataLoader.show_images(images_2_show, titles_2_show)


def retrieveNecessaryData():
    cwd = os.getcwd()
    input_path = cwd + '\\MNIST'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte\\train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte\\train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte\\t10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte')

    dtLd = DataLoader.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                      test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = dtLd.load_data()
    return x_train, y_train


if __name__ == '__main__':
    main()

import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt
import DataLoader


def main():
    (image_train_list, label_train_list) = retrieve_necessary_data()
    # TODO change reshape size from 200 to 60000
    image_train_normalize = np.array(image_train_list).reshape(200, 28 ** 2)
    image_with_label_list = [[label_train_list[i],image_train_normalize[i]] for i in range(200)]
    kmeans(10, image_with_label_list)


def kmeans(k, image_with_label_list):
    # initialize k random clusters, assigned with empty sets
    centers_array = [[x, []] for x in np.random.rand(k, 28 ** 2)]  # array of [center,bounded images]
    for image in image_with_label_list:
        centers_array[find_closest_center_index(centers_array, image[1])][1].append([image])
        # bound image to closest center

    # for cluster in centers_array.filter():
    #     image_pool = map(lambda image_with_label: image_with_label[1], cluster[1])
    #     cluster



    print(centers_array)




def find_closest_center_index(center_array, data_unit):
    centers = map(lambda center_set_tuple: center_set_tuple[0], center_array)
    distances = map(lambda cent: np.linalg.norm(data_unit - cent), centers)
    x = [np.linalg.norm(data_unit-center) for center in centers]
    return np.argmin(x)

    # images_2_show = []
    # titles_2_show = []
    # for i in range(0, 10):
    #     r = random.randint(1, 60000)
    #     images_2_show.append(x_train[r])
    #     titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))
    #
    # # for i in range(0, 5):
    # #     r = random.randint(1, 10000)
    # #     images_2_show.append(x_test[r])
    # #     titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))
    #
    # DataLoader.show_images(images_2_show, titles_2_show)


def retrieve_necessary_data():
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

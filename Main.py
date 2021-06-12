import numpy as np
import struct
from array import array
from os.path import join
import os
import random
import matplotlib.pyplot as plt
import DataLoader


def main():
    (image_train_list, label_train_list), (image_test_list, label_test_list) = retrieve_data()
    # TODO change reshape size from 200 to 60000
    image_train_normalize = np.array(image_train_list).reshape(60000, 28 ** 2)
    image_with_label_list_train = [[label_train_list[i], image_train_normalize[i]] for i in range(60000)]
    laybeled_centers = kmeans(10, image_with_label_list_train)

    # TODO change reshape size from 200 to 60000
    image_test_normalize = np.array(image_test_list).reshape(10000, 28 ** 2)
    image_with_label_list_test = [[label_test_list[i], image_test_normalize[i]] for i in range(10000)]

    testing_kmeans(laybeled_centers, image_with_label_list_test)


def kmeans(k, image_with_label_list):
    # initialize k random clusters, assigned with empty sets
    centers_array = [[x, []] for x in np.random.rand(k, 28 ** 2)]  # array of [center,bounded images]
    # centers_array = [[look_for_label(x, image_with_label_list), []] for x in
    #                  range(10)]  # array of [center,bounded images]
    counter = 0
    while True:
        counter = counter + 1
        print("iteration number {}\n".format(counter))
        cur_centers = [center[0] for center in centers_array]

        #  Assigning step
        for image in image_with_label_list:
            centers_array[find_closest_center_index(centers_array, image[1])][1].append([image])
            # assign image to closest center

        #  Centers updating step
        for cluster in filter(lambda claster: len(claster[1]) > 0, centers_array):
            image_pool = [img[0][1] for img in cluster[1]]  # extracting all assigned images values
            cluster[0] = np.average(image_pool, axis=0)  # compute the average vector to be the new center

        if np.array_equal(cur_centers, [center[0] for center in centers_array]):
            break

        #  Assigning reset
        for cluster in centers_array:
            cluster[1] = []

    return [[common_label(cluster), cluster[0]] for cluster in centers_array]


def testing_kmeans(laybeled_centers, image_with_label_list):
    centers_array = [[x[1], x[0]] for x in laybeled_centers]
    succes_counter = 0
    fail_counter = 0
    #  Assigning step
    for image in image_with_label_list:
        if centers_array[find_closest_center_index(centers_array, image[1])][1] == image[0]:
            succes_counter = succes_counter + 1
        else:
            fail_counter = fail_counter + 1
    print("suc: {}, fail: {} ,precentage: {}".format(succes_counter, fail_counter,
                                                     succes_counter / len(image_with_label_list)))


def look_for_label(label, data_set):
    for img in data_set:
        if img[0] == label:
            return img[1]


def common_label(cluster):
    label_pool = [img[0][0] for img in cluster[1]]  # extracting all assigned images values
    return np.bincount(label_pool).argmax()
    # print(np.argmax(counts))


def find_closest_center_index(center_array, data_unit):
    centers = map(lambda center_set_tuple: center_set_tuple[0], center_array)
    distances = map(lambda cent: np.linalg.norm(data_unit - cent), centers)
    x = [np.linalg.norm(data_unit - center) for center in centers]
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


def retrieve_data():
    cwd = os.getcwd()
    input_path = cwd + '\\MNIST'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte\\train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte\\train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte\\t10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte')

    dtld = DataLoader.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                      test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = dtld.load_data()
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    main()

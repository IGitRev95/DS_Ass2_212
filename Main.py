import numpy as np
from os.path import join
import os
import DataLoader


def main():
    (image_train_list, label_train_list), (image_test_list, label_test_list) = retrieve_data()
    # reshaping the data stream for flattening the images matrices
    # 60000 mat 28X28 -> 60000 vector in size 28**2
    image_train_normalize = np.array(image_train_list).reshape(60000, 28 ** 2)
    image_with_label_list_train = [[label_train_list[i], image_train_normalize[i]] for i in
                                   range(len(label_train_list))]
    image_test_normalize = np.array(image_test_list).reshape(10000, 28 ** 2)
    image_with_label_list_test = [[label_test_list[i], image_test_normalize[i]] for i in range(len(label_test_list))]

    # kmeans returns the clusters centers bounded to their common label
    labeled_centers = random_init_kmeans(10, image_with_label_list_train)

    testing_kmeans(labeled_centers, image_with_label_list_test)

    labeled_centers = non_random_init_kmeans(10, image_with_label_list_train)

    testing_kmeans(labeled_centers, image_with_label_list_test)


def random_init_kmeans(k, image_with_label_list):
    # initialize k random clusters, assigned with empty sets
    centers_array = [[x, []] for x in np.random.rand(k, 28 ** 2)]  # array of [center,bounded images]
    iteration_counter = 0
    cur_centers = [center[0] for center in centers_array]  # will keep current centers for comparison

    while True:
        iteration_counter = iteration_counter + 1
        print("iteration number {}\n".format(iteration_counter))

        #  Assigning step
        for image in image_with_label_list:
            centers_array[find_closest_center_index(centers_array, image[1])][1].append([image])
            # assign image to closest center

        #  Centers updating step
        for cluster in filter(lambda claster: len(claster[1]) > 0, centers_array):
            image_pool = [img[0][1] for img in cluster[1]]  # extracting all assigned images values
            cluster[0] = np.average(image_pool, axis=0)  # compute the average vector to be the new center
        new_centers = [center[0] for center in centers_array]
        #  check if the centers have changed in the past iteration - termination condition
        if np.array_equal(cur_centers, new_centers):
            break

        cur_centers = new_centers  # keeping current centers for comparison

        #  Images to clusters assigning reset
        for cluster in centers_array:
            cluster[1] = []

    return [[common_label(cluster), cluster[0]] for cluster in centers_array]  # returns [label, cluster]


def non_random_init_kmeans(k, image_with_label_list, random_init):
    # initialize clusters with concrete sample of each cluster class, assigned with empty sets
    centers_array = [[look_for_label(x, image_with_label_list), []] for x in
                     range(10)]  # array of [center,bounded images]
    iteration_counter = 0
    cur_centers = [center[0] for center in centers_array]  # will keep current centers for comparison

    while True:
        iteration_counter = iteration_counter + 1
        print("iteration number {}\n".format(iteration_counter))

        #  Assigning step
        for image in image_with_label_list:
            centers_array[find_closest_center_index(centers_array, image[1])][1].append([image])
            # assign image to closest center

        #  Centers updating step
        for cluster in filter(lambda claster: len(claster[1]) > 0, centers_array):
            image_pool = [img[0][1] for img in cluster[1]]  # extracting all assigned images values
            cluster[0] = np.average(image_pool, axis=0)  # compute the average vector to be the new center
        new_centers = [center[0] for center in centers_array]
        #  check if the centers have changed in the past iteration - termination condition
        if np.array_equal(cur_centers, new_centers):
            break

        cur_centers = new_centers  # keeping current centers for comparison

        #  Images to clusters assigning reset
        for cluster in centers_array:
            cluster[1] = []

    return [[common_label(cluster), cluster[0]] for cluster in centers_array]  # returns [label, cluster]


def testing_kmeans(labeled_centers, image_with_label_list):
    centers_array = [[x[1], x[0]] for x in labeled_centers]  # make an centers - labels array
    success_counter = 0
    fail_counter = 0
    #  Assigning step like, each comparing the assigned image label to the cluster pre-given label by the k-means
    for image in image_with_label_list:
        if centers_array[find_closest_center_index(centers_array, image[1])][1] == image[0]:
            success_counter = success_counter + 1
        else:
            fail_counter = fail_counter + 1
    print("suc: {}, fail: {} ,success rate: {}".format(success_counter, fail_counter,
                                                       success_counter / len(image_with_label_list)))


def look_for_label(label, image_set):
    for img in image_set:
        if img[0] == label:
            return img[1]


#  Looking for the common label in a cluster
def common_label(cluster):
    if len(cluster[1]) == 0:
        return -1  # default value for a cluster which no image was assigned to
    label_pool = [img[0][0] for img in cluster[1]]  # extracting all assigned images values
    return np.bincount(label_pool).argmax()  # counting and returning the common label


def find_closest_center_index(center_array, data_unit):
    centers = map(lambda center_set_tuple: center_set_tuple[0], center_array)  # extract current centers
    squared_norm = lambda x: np.inner(x, x)
    x = [squared_norm(data_unit - center) for center in centers]  # compute norm distance from each center
    return np.argmin(x)  # return the index of the closest center


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

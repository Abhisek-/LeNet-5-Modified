#!/usr/bin/python
# -*- coding: utf-8 -*-

from mnist import MNIST
import numpy as np

def create_one_hot(array):
    '''
    Create one hot encoding
    '''

    array_length = len(array)
    array = np.array(array)
    array_final = np.zeros((array_length, 10))
    array_final[np.arange(array_length), array] = 1

    return array_final

def read_image(directory):
    '''
    Read MNIST dataset

    :param directory: path where MNIST dataset present
    '''
    mndata = MNIST(directory)

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    num_training = len(train_images)
    num_testing = len(test_images)

    # Reshape images to make it 2D
    train_images = np.reshape(np.array(train_images), [num_training, 28, 28, 1])
    test_images = np.reshape(np.array(test_images), [num_testing, 28, 28, 1])

    # normalize the data
    train_images = np.divide(train_images, 255)
    test_images = np.divide(test_images, 255)

    # Create one hot encoding
    train_labels = create_one_hot(train_labels)
    test_labels = create_one_hot(test_labels)

    return (train_images, train_labels,
            test_images, test_labels)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mndir')
    args = parser.parse_args()
    read_image(args.mndir)

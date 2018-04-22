#!/usr/bin/python
# -*- coding: utf-8 -*-

from mnist import MNIST

def read_image(directory):
    '''
    Read MNIST dataset

    :param directory: path where MNIST dataset present
    '''
    mndata = MNIST(directory)

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    return (train_images, train_labels,
            test_images, test_labels)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mndir')
    args = parser.parse_args()
    read_image(args.mndir)

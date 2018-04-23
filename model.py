#!/usr/bin/python
# -*- coding: utf-8 -*-

from preprocess import read_image

import tensorflow as tf

def get_hyperparameters():
    '''
    Create hyperparameters for LeNet-5 model
    '''
    filters = [5, 2, 5, 2]
    strides = [1, 2, 1, 2]
    channels = [None, 6, 6, 16, 16]
    fc_units = [120, 84, 10]
    return filters, strides, channels, fc_units

def create_model(learning_rate):
    '''
    Create Modified LeNet-5 model
    '''

    filters, strides, channels, fc_units = get_hyperparameters()

    input_shape = [None, 28, 28, 1]
    output_shape = [None, 10]

    # Input
    input_image = tf.placeholder(dtype=tf.float32, name='input', shape=input_shape)
    labels = tf.placeholder(dtype=tf.float32, name='labels', shape=output_shape)

    # Convolution Layer 1
    filter1 = tf.Variable(tf.truncated_normal(
        [filters[0], filters[0], input_shape[-1], channels[1]],
        dtype = tf.float32, stddev = 1e-1), name = 'filter1')
    conv1 = tf.nn.conv2d(input_image,
                         filter1,
                         [strides[0], strides[0], strides[0], strides[0]],
                         padding = 'VALID',
                         name = 'conv1')

    # Max Pooling layer 1
    pool1 = tf.layers.max_pooling2d(conv1,
                                    [filters[1], filters[1]],
                                    [strides[1], strides[1]],
                                    padding = 'VALID',
                                    name = 'pool1')

    # Convolution Layer 2
    filter2 = tf.Variable(tf.truncated_normal(
        [filters[2], filters[2], channels[2], channels[3]],
        dtype = tf.float32, stddev = 1e-1), name = 'filter2')
    conv2 = tf.nn.conv2d(pool1,
                         filter2,
                         [strides[2], strides[2], strides[2], strides[2]],
                         padding = 'VALID',
                         name = 'conv2')

    # Max Pooling layer 2
    pool2 = tf.layers.max_pooling2d(conv2,
                                    [filters[3], filters[3]],
                                    [strides[3], strides[3]],
                                    padding = 'VALID',
                                    name = 'pool2')

    # Flatten
    flatten = tf.layers.flatten(pool2, name = 'flatten')

    # Fully Connected layer 3
    FC3 = tf.contrib.layers.fully_connected(flatten, fc_units[0])

    # Fully Connected layer 4
    FC4 = tf.contrib.layers.fully_connected(FC3, fc_units[1])

    # Fully Connected layer 5
    output = tf.contrib.layers.fully_connected(FC4, 10)

    # softmax
    softmax = tf.nn.softmax_cross_entropy_with_logits(labels = labels,
                                                      logits = output,
                                                      name = 'softmax')

    loss = tf.reduce_mean(softmax)

    correct = tf.equal(tf.argmax(labels, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return input_image, labels, accuracy, loss, optimizer

def train(mndir, epochs, batch_size):
    '''
    Train LeNet-5 model
    '''

    (train_images, train_labels,
     test_images, test_labels) = read_image(mndir)

    train_len = train_images.shape[0]

    (input_image, labels, accuracy,
     loss, optimizer) = create_model(0.0001)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            accuracies = []
            losses = []
            for batch_i in range(train_len // batch_size):
                start_idx = batch_size * batch_i
                end_idx = start_idx + batch_size
                _, _loss, acc = sess.run([optimizer, loss, accuracy],
                                    feed_dict = {input_image: train_images[start_idx: end_idx,],
                                                 labels: train_labels[start_idx: end_idx,]})

                accuracies.append(acc)
                losses.append(_loss)

                if batch_i % 20 == 0:
                    print('Epoch: {}/{} Batch: {}/{} Loss: {} Accuracy: {}'.format(
                        epoch, epochs,
                        batch_i, train_len // batch_size,
                        sum(losses) / len(losses),
                        sum(accuracies) / len(accuracies)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mndir')
    args = parser.parse_args()
    train(args.mndir, 5, 128)

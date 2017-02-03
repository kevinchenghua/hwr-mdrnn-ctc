from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import numpy as np
import tensorflow as tf


character_to_index_dict = {' ': 0, '!': 1, '"': 2, '#': 3, '&': 4,
                           "'": 5, '(': 6, ')': 7, '*': 8, '+': 9,
                           ',': 10, '-': 11, '.': 12, '/': 13, '0': 14,
                           '1': 15, '2': 16, '3': 17, '4': 18, '5': 19,
                           '6': 20, '7': 21, '8': 22, '9': 23, ':': 24,
                           ';': 25, '?': 26, 'A': 27, 'B': 28, 'C': 29,
                           'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34,
                           'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39,
                           'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44,
                           'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49,
                           'X': 50, 'Y': 51, 'Z': 52, 'a': 53, 'b': 54,
                           'c': 55, 'd': 56, 'e': 57, 'f': 58, 'g': 59,
                           'h': 60, 'i': 61, 'j': 62, 'k': 63, 'l': 64,
                           'm': 65, 'n': 66, 'o': 67, 'p': 68, 'q': 69,
                           'r': 70, 's': 71, 't': 72, 'u': 73, 'v': 74,
                           'w': 75, 'x': 76, 'y': 77, 'z': 78, '\n': 79}


index_to_character_dict = {v: k for k, v in character_to_index_dict.items()}


def convert_character_to_index(c):
    """Convert character to defined index"""
    return character_to_index_dict[c]


def convert_index_to_character(i):
    """Convert defined index to character"""
    return index_to_character_dict[i]


def inputs(alignment_file, data_dir, num_epochs=None):
    """The input pipeline for reading images and labels

    There should be two directories under `data_dir`:

        /data_dir/images
        /data_dir/labels

    This dataset is from IAM handwriting database, and we choose to use the
    text lines image. The files in the images directory are `.png` files.
    And the files in the labels directory are `.xml` files.

    Args:
        alignment_file: path to the file containing labeled images list
        data_dir: the path to the data directory

    Returns:
        image: image tensor, 3D, num_raws x num_cols x cnm_channels
        label: lable SparseTensor<2>, 2D, 1 x target_length
    """

    # read the alignment file
    image_list, label_list = read_labeled_image_list(alignment_file, data_dir)

    images = tf.convert_to_tensor(image_list, dtype=tf.string, name='images')
    labels = tf.convert_to_tensor(label_list, dtype=tf.string, name='labels')

    # create input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True,
                                                name='input_queue')

    # read image and label from input_queue
    image, label = read_images_and_labels_from_disk(input_queue)

    return image, label


def read_labeled_image_list(alignment_file, data_dir):
    """Read the file containing alignment of image files and labels

    There should be two directories under `data_dir`:

        /data_dir/images
        /data_dir/labels

    Args:
        alignment_file: path to the file containing labeled images list
        data_dir: the path to the data directory

    Returns:
        image_files(list): list of image files
        labels(list): list of labels with index corresponding to image_files
    """

    with open(alignment_file, 'r') as f:
        image_files = []
        labels = []
        for line in f:
            image_file, label = line.split('||')
            image_files.append(os.path.join(data_dir, 'images/', image_file))
            labels.append(label)
    return image_files, labels


def read_images_and_labels_from_disk(input_queue):
    """Consume a single input_queue containing a file and a label

    Args:
        input_queue [tensor, tensor]: a list of image and label tensor

    Returns:
        example: image tensor, 3D, num_raws x num_cols x cnm_channels
        label: lable SparseTensor<2>, 2D, 1 x target_length
    """
    table = create_lookup_table(character_to_index_dict)
    table.init.run()
    label = tf.string_split(tf.expand_dims(input_queue[1], 0), delimiter="")
    label = table.lookup(label)
    file_contents = tf.read_file(input_queue[0])
    example = tf.to_float(tf.image.decode_png(file_contents, 1))
    return example, label


def create_lookup_table(d):
    """Create a tensorflow dictionary lookup"""
    keys, values = zip(*d.items())
    keys = tf.constant(keys)
    values = tf.constant(values, dtype=tf.int64)
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1)
    return table


def test_inputs():
    sess = tf.Session()
    with sess.as_default():
        with tf.name_scope('inputs'):
            _, label = inputs('alignment.txt', '../data/')
        init_op = tf.initialize_all_variables()

        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('../log/unit_test/', sess.graph)

        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(label.eval())
        summary = sess.run(merged)
        summary_writer.add_summary(summary, 1)

        coord.request_stop()
        coord.join(threads)

        sess.close()


def convert_strings_to_sparse_tensor(string_list):
    """Create a very simple SparseTensor with dimensions (num_example, time).

    Args:
        string_list: a list of string
    Returns:
        label_tensor: lable_index and label_value, the indices and values
                      of the SparseTensor<2>.
    """
    label_index = []
    label_value = []
    for s_i, s in enumerate(string_list):
        for c_i, c in enumerate(s):
            label_index.append([s_i, c_i])
            label_value.append(convert_character_to_index(c))
    label_shape = [len(string_list), np.asarray(label_index).max(0)[1] + 1]
    label_index = tf.constant(label_index, tf.int64)
    label_value = tf.constant(label_value, tf.int32)
    label_shape = tf.constant(label_shape, tf.int64)
    label_tensor = tf.SparseTensor(label_index, label_value, label_shape)

    return label_tensor


def test():

    # test function read_labeled_image_list
    image_files, labels = read_labeled_image_list('alignment.txt', '../data/')
    assert(len(image_files) == len(labels) == 13353)
    assert(re.match(
        r'\.\.[/\\]data[/\\]images[/\\]\w+[/\\]\w+-\w+[/\\]\w+-\w+-\w+\.png',
        image_files[0]))

    # test function convert_strings_to_sparse_tensor
    # sess = tf.Session()
    # print(convert_strings_to_sparse_tensor(labels).eval(session=sess))

    # test inputs
    test_inputs()


test()

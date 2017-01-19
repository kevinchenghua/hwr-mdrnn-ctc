from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf


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
    """

    # read the alignment file
    image_list, label_list = read_labeled_image_list(alignment_file, data_dir)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.string)

    # create input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)

    # read image and label from input_queue
    image, label = read_images_and_labels_from_disk(input_queue)

    # create summary op
    tensor_name = image.op.name
    tf.image_summary(tensor_name + 'images', image)

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
        example: image tensor
        label: string tensor
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents)
    return example, label


def test():

    # test function read_labeled_image_list
    image_files, labels = read_labeled_image_list('alignment.txt', '../data/')
    assert(len(image_files) == len(labels) == 13353)
    assert(re.match(
        r'\.\.[/\\]data[/\\]images[/\\]\w+[/\\]\w+-\w+[/\\]\w+-\w+-\w+\.png',
        image_files[0]))

    # test inputs
    _, label = inputs('alignment.txt', '../data/')

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(label.eval())

        coord.request_stop()
        coord.join(threads)

        sess.close()


test()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import input
import gridlstm


class Model():
    """Model build with GridLSTM4DLayer and patch
        GridLSTM4DLayer_output_last
        ^
        patch_last
        ^
        .
        .
        .
        ^
        GridLSTM4DLayer_output_2
        ^
        patch_2
        ^
        GridLSTM4DLayer_output_1
        ^
        patch_1
    """

    def __init__(self, input_size, output_size, window_shape):
        """Initialize the parameters of the model

        Args:
            input_size: int, the size of input image channel
            output_size: list of int, indicate the output_size for each
                         direction of GridLSTM4DLayer
            window_shape: list of tuple, the tuple (window_height,
                          window_width) for each GridLSTM4DLayer
        """

        self.layers = self._parse_conf(input_size, output_size, window_shape)

    def __call__(self, input):
        """Build the computation graph of the model

        Args:
            input: input tensor, 3D, num_raws x num_cols x input_size
        Returns:
            layers_output: list of layer output 3D tensor,
                           num_raws x num_cols x (4*output_size)
        """
        layers_output = []
        for i in range(len(self.layers)):
            with tf.variable_scope("GridLSTM_" + str(i + 1)):
                output = self.layers[i].build(input)
                layers_output.append(tf.squeeze(output, axis=[2]))
                input = output
        return layers_output

    def _parse_conf(self, input_size, output_size, window_shape):
        """This method parse the argument of __init__ and create layers

        Args:
            input_size: int, the size of input image channel
            output_size: list of int, indicate the output_size for each
                         direction of GridLSTM4DLayer
            window_shape: list of tuple, the tuple (window_height,
                          window_width) for each GridLSTM4DLayer
        Returns:
            layers: list of `layer`, with index 0 indicate the first layer
        """

        # check the format
        if not isinstance(input_size, int):
            raise TypeError("input_size should be an instance of `int`.")
        if not isinstance(output_size, list):
            raise TypeError("output_size should be an instance of `list`.")
        if not isinstance(window_shape, list):
            raise TypeError("window_shape should be an instance of `list`.")
        if not all(isinstance(elem, int) for elem in output_size):
            raise TypeError("The elements of output_size should all be " +
                            "instances of `int`.")
        if not all(isinstance(elem, (tuple, list)) and len(elem) == 2 for
                   elem in window_shape):
            raise TypeError("The elements of window_shape should all be " +
                            "instances of `tuple` with length 2.")
        if len(output_size) != len(window_shape):
            raise ValueError("output_size and window_shape must have the " +
                             "same length.")

        # create layers
        layers = []
        for i in range(len(output_size)):
            layers.append(self.layer(input_size,
                                     output_size[i],
                                     window_shape[i],
                                     first_layer=(i == 0)))
            input_size = 4 * output_size[i]

        return layers

    class layer():
        """This is a class to gather parameters of layer and build it

        Attrs:
            input_size: int, the size of input image channel
            ksizes: list of ints with length 4, the parameters for patching
            strides: list of ints with length 4, the parameters for patching
            rates: list of ints with length 4, the parameters for patching
            gird_input_size: int, the input size for gridlstm after patching
            grid_output_channel: int, the output size for each direction of
                                 GridLSTM4DLayer
            grid_output_size: int, the output size after gathering each
                              direction of GridLSTM4DLayer
            first_layer: bool, indicate whether this layer the first layer
        """

        def __init__(self, input_size, output_size, window_shape, first_layer):
            """Initialize the parameters of layer

            Args:
                input_size: int, the size of input image channel
                output_size: int, indicate the output_size for each
                             direction of GridLSTM4DLayer
                window_shape: tuple, the tuple (window_height, window_width)
                              for patching
                first_layer: bool, indicate whether this layer the first layer
            """
            self.input_size = input_size
            self.ksizes = [1, window_shape[0], window_shape[1], 1]
            self.strides = self.ksizes
            self.rates = [1, 1, 1, 1]
            self.grid_input_size = input_size * self.ksizes[1] * self.ksizes[2]
            self.grid_output_channel = output_size
            self.grid_output_size = output_size * 4
            self.first_layer = first_layer

        def build(self, input):
            """This method build the computation graph of the layer

            Args:
                input: input Tensor, 3D or 4D,
                    if `self.first_layer` is True, 3D, with shape:
                        num_raws x num_cols x input_size,
                    if `self.first_layer` is False, 4D with shape:
                        num_raws x num_cols x batch_size x input_size
            Returns:
                grid_lstm_output: output Tensor, 4D,
                    num_raws x num_cols x batch_size x (4 * output_size)
            """
            if self.first_layer:
                grid_lstm_input = tf.expand_dims(input, 0)
            else:
                grid_lstm_input = tf.transpose(input, [2, 0, 1, 3])
            grid_lstm_input = tf.extract_image_patches(grid_lstm_input,
                                                       ksizes=self.ksizes,
                                                       strides=self.strides,
                                                       rates=self.rates,
                                                       padding="SAME",
                                                       name="input_patch")
            grid_lstm_input = tf.transpose(grid_lstm_input, [1, 2, 0, 3])
            grid_lstm = gridlstm.GridLSTM4DLayer(self.grid_input_size,
                                                 self.grid_output_channel)
            grid_lstm_output = grid_lstm(grid_lstm_input)

            return grid_lstm_output


def test():
    model = Model(1, [2, 10, 50], [(4, 3), (4, 3), (4, 3)])
    with tf.variable_scope("inputs"):
        image, label = input.inputs("alignment.txt", "../data/")
    layers_output = model(image)

    init_op = tf.global_variables_initializer()

    sess = tf.Session()

    tf.summary.FileWriter('../log/unit_test/model/', sess.graph)
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # sess.run([a, b])
    print(sess.run(layers_output[-1]))

    coord.request_stop()
    coord.join(threads)


# test()

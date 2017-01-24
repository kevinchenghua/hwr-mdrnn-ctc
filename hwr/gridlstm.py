from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import tensorflow as tf
from tensorflow.contrib import grid_rnn


class GridLSTMLayer():
    """2D Grid LSTM layer scan from top-left to bottom-left

    This implementation is based on:

        http://arxiv.org/pdf/1507.01526v3.pdf

    And the lstm cell Grid3LSTM is from:

        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
                contrib/grid_rnn/python/ops/grid_rnn_cell.py
    """

    def __init__(self,
                 input_size,
                 output_size,
                 output_active_fn=None):
        """Initialize the parameters of GridLSTMLayer

        Args:
            input_size: int, the number of input feature/channel size
            output_size: int, the number of output feature/channel size
            output_active_fn: a tensorflow Op that will be the transfer
                function of the output, default to be `tensorflow.nn.relu`
        """
        self.input_size = input_size
        self.output_size = output_size
        self.output_active_fn = output_active_fn \
            if output_active_fn is not None else tf.nn.relu

    def __call__(self,
                 input,
                 initializer=tf.random_normal_initializer(),
                 scope=None):
        """Build the computation graph of GridLSTMLayer

        Args:
            input: input Tensor, 4D,
                   num_raws x num_cols x batch_size x input_size
            initializer: tensorflow variables initializer for cell parameters
            scope: VariableScope for the created sub graph,
                   defaults to "GridLSTMLayer"

        Returns:
            output: output Tensor, 4D,
                    num_raws x num_cols x batch_size x output_size
        """
        # check the input tensor
        assert(len(input.get_shape().as_list()) == 4)
        assert(input.get_shape().as_list()[2] == 1)
        assert(input.get_shape().as_list()[3] == self.input_size)

        with tf.variable_scope(scope or type(self).__name__,
                               initializer=initializer):
            # initialize the raw_states, col_states, grid_lstm_cell
            input_shape = tf.shape(input)
            init_raw_states = tf.zeros(
                [input_shape[1], 1, self.output_size * 2],
                name="init_raw_states")
            init_col_states = tf.zeros(
                [input_shape[0], 1, self.output_size * 2],
                name="init_col_states")
            grid_lstm_cell = grid_rnn.Grid3LSTMCell(
                self.output_size, non_recurrent_fn=self.output_active_fn)

            # build the computation graph, and using cpu is faster
            with tf.device('/cpu:0'):
                output = _scan_grid(
                    grid_lstm_cell, input, init_raw_states, init_col_states)

        return output


def _scan_grid(cell, input, init_raw_states, init_col_states):
    """Function for scaning grid

    Note:
        cell_memory_size = cell_hidden_size = cell_output_size = num_units
        cell_state_size = cell_memory_size + cell_hidden_size = 2 * num_units
        state_size = 2 * cell_state_size (from up and left)
        output_size = cell_output_size
    Args:
        cell: grid LSTM cell
        input: input Tensor, 4D, num_raws x num_cols x batch_size x input_size
        init_raw_states: raw_states initializer Tensor, 3D,
                         num_cols x batch_size x cell_state_size
        init_col_states: column_states initializer Tensor, 3D,
                         num_raws x batch_size x cell_state_size
    Returns:
        output: output Tensor, 4D,
                num_raws x num_cols x batch_size x output_size
    """
    def loop(init, elem):
        """Loop function for scan

        Args:
            elem: (raw_input, init_col_states_elem)
                raw_input: 3D Tensor of raw_input element,
                           num_cols x batch_size x input_size
                init_col_states_elem: column state initializer Tensor, 2D,
                                      batch_size x cell_state_size
            init: (raw_output, init_raw_states)
                raw_output: 3D Tensor of raw_output,
                            num_cols x batch_size x output_size
                init_raw_states: raw_states initializer Tensor, 3D,
                                 num_cols x batch_size x cell_state_size
        Returns: (output_elem, raw_elem, col_init)
            raw_output: 3D Tensor of raw_output,
                        num_cols x batch_size x output_size
            raw_states: raw_states Tensor, 3D,
                        num_cols x batch_size x cell_state_size
        """

        # build the computation graph of the raw
        raw_output, raw_states = _scan_grid_raw(
            cell, elem[0], init[1], elem[1])

        return raw_output, raw_states

    # prepare the initializer tensors for the scan function
    raw_states_shape = tf.shape(init_raw_states)
    raw_output_shape = [raw_states_shape[0],
                        raw_states_shape[1],
                        raw_states_shape[2] // 2]
    init_raw_output = tf.zeros(raw_output_shape, name="init_raw_output")

    # scan the grid
    output, _ = tf.scan(
        loop,
        elems=(input, init_col_states),
        initializer=(init_raw_output, init_raw_states),
        name="scan_raw_elems")

    return output


def _scan_grid_raw(cell, raw_input, prev_raw_states, init_col_state):
    """Function for scaning raw of grid

    Note:
        cell_memory_size = cell_hidden_size = cell_output_size = num_units
        cell_state_size = cell_memory_size + cell_hidden_size = 2 * num_units
        state_size = 2 * cell_state_size (from up and left)
        output_size = cell_output_size
    Args:
        cell: grid LSTM cell
        raw_input: raw_input Tensor, 3D, num_cols x batch_size x input_size
        prev_raw_states: prev raw Tensor, 3D,
                         num_cols x batch_size x cell_state_size
        init_col_state: column state initializer Tensor, 2D,
                         batch_size x cell_state_size
    Returns:
        raw_output: raw_output Tensor, 3D, num_cols x batch_size x output_size
        raw_states: raw_state Tensor, 3D,
                    num_cols x batch_size x cell_state_size
    """
    def loop(init, elem):
        """Loop function for scan

        Args:
            elem: (raw_elem_input, prev_raw_states_elem)
                raw_input_elem: 2D Tensor of raw_input element,
                             batch_size x input_size
                prev_raw_states_elem: 2D Tensor of prev_raw_states element,
                               batch_size x cell_state_size
            init: (raw_output_elem, raw_states_elem, init_col_state)
                raw_output_elem: 2D Tensor of raw_output element,
                                 batch_size x output_size
                raw_states_elem: 2D Tensor of raw_states element,
                          batch_size x cell_state_size
                init_col_state: column state initializer Tensor, 2D,
                          batch_size x cell_state_size
        Returns: (raw_output_elem, raw_states_elem, init_col_state)
            raw_output_elem: 2D Tensor of raw_output element,
                             batch_size x output_size
            raw_states_elem: 2D Tensor of raw_states element,
                             batch_size x cell_state_size
            init_col_state: 2D Tensor, batch_size x cell_state_size
        """

        # prepare for the inputs of gridlstm cell
        cell_input = elem[0]
        cell_states = tf.concat(1, [init[2], elem[1]])

        # build the computation graph of the cell
        raw_output_elem, states = cell(cell_input, cell_states)

        # split the cell ouput `states` to `col_state` and `raw_state`
        init_col_state, raw_states_elem = tf.split(1, 2, states)

        return raw_output_elem, raw_states_elem, init_col_state

    # prepare the initializer tensors for the scan function
    raw_states_elem_shape = tf.shape(init_col_state)
    raw_output_elem_shape = [raw_states_elem_shape[0],
                             raw_states_elem_shape[1] // 2]
    init_raw_states_elem = tf.zeros(raw_states_elem_shape,
                                    name="init_raw_states_elem")
    init_raw_output_elem = tf.zeros(raw_output_elem_shape,
                                    name="init_raw_output_elem")

    # scan the raw
    output, raw_states, _ = tf.scan(
        loop,
        elems=(raw_input, prev_raw_states),
        initializer=(init_raw_output_elem,
                     init_raw_states_elem,
                     init_col_state),
        name="scan_raw")

    return output, raw_states


def test_scan():
    # ------------------------------ build graph ------------------------------

    sess = tf.Session()

    input_h = tf.placeholder(tf.float32, [None, None, 1, 200])
    raw_h = tf.placeholder(tf.float32, [None, 1, 100])
    col_h = tf.placeholder(tf.float32, [None, 1, 100])
    gl = grid_rnn.Grid3LSTMCell(50, non_recurrent_fn=tf.nn.relu)

    # build forward computation, the initializer is needed
    with tf.variable_scope("test",
                           initializer=tf.random_normal_initializer()):
        # cpu is faster
        with tf.device('/cpu:0'):
            output = _scan_grid(gl, input_h, raw_h, col_h)

    # build backward computation, cpu is faster
    with tf.device('/cpu:0'):
        t = time.time()
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
            tf.reduce_mean(output))
        print("backward building time:" + str(time.time() - t))

    # varables initializer
    init_op = tf.global_variables_initializer()
    tf.summary.FileWriter('../log/unit_test/gridlstm/', sess.graph)

    # --------------------------- run session -------------------------------

    sess.run(init_op)

    # input
    input = np.zeros([33, 500, 1, 200])
    raw = np.zeros([500, 1, 100])
    col = np.zeros([33, 1, 100])

    # forward computation test
    t = time.time()
    sess.run(output, feed_dict={input_h: input, raw_h: raw, col_h: col})
    print("forward computation time:" + str(time.time() - t))

    # backward computation test
    t = time.time()
    sess.run(train_step, feed_dict={input_h: input, raw_h: raw, col_h: col})
    print("backward computation time:" + str(time.time() - t))


def test_layer():
    # ------------------------------ build graph ------------------------------

    sess = tf.Session()

    # build forward computation
    layer = GridLSTMLayer(200, 50)
    input_h = tf.placeholder(tf.float32, [None, None, 1, 200])
    output = layer(input_h)

    # varables initializer
    init_op = tf.global_variables_initializer()
    tf.summary.FileWriter('../log/unit_test/gridlstm/', sess.graph)

    # --------------------------- run session -------------------------------

    sess.run(init_op)

    # input
    input = np.zeros([33, 500, 1, 200])

    # forward computation test
    t = time.time()
    sess.run(output, feed_dict={input_h: input})
    print("forward computation time:" + str(time.time() - t))


def test():
    test_layer()


test()

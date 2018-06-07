import tensorflow as tf
import data
import numpy as np

def linear_layer(inputs, output_size, is_bias=True, name="linear"):
    # get params
    input_size = inputs.get_shape()[-1].value
    dtype = inputs.dtype

    # calculate
    W = tf.get_variable(name=name+"_w", shape=[input_size, output_size], dtype=dtype)
    b = 0
    if is_bias:
        b = tf.get_variable(name=name+"_b", shape=[output_size], dtype=dtype)
    outputs =  tf.tensordot(inputs, W, [[-1],[0]])+ b
    return outputs


def _copy_through(time, length, output, new_output):
    copy_cond = (time >= length)
    if isinstance(new_output, tf.nn.rnn_cell.LSTMStateTuple):
        c, h = output
        c_new, h_new = new_output
        c_out = tf.where(copy_cond, c, c_new)
        h_out = tf.where(copy_cond, h, h_new)
        return tf.nn.rnn_cell.LSTMStateTuple(c_out, h_out)
    return tf.where(copy_cond, output, new_output)


def lstm_layer(cell, inputs, inputs_length, initial_state=None):
    # get params
    output_size = cell.output_size
    batch = tf.shape(inputs)[0]
    time_steps = tf.shape(inputs)[1]
    dtype = inputs.dtype

    # prepare tensor for iteration
    zero_output = tf.zeros([batch, output_size], dtype)
    if initial_state is None: initial_state = cell.zero_state(batch, dtype)
    input_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="input_array")
    output_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="output_array")
    input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))
    time = tf.constant(0, dtype=tf.int32, name="time")
    loop_vars = (time, output_ta, initial_state)

    def loop_func(t, out_ta, state):
        inp_t = input_ta.read(t)
        cell_output, new_state = cell(inp_t, state)
        cell_output = _copy_through(t, inputs_length, zero_output, cell_output)
        new_state = _copy_through(t, inputs_length, state, new_state)
        out_ta = out_ta.write(t, cell_output)
        return t + 1, out_ta, new_state

    outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
                            loop_vars, parallel_iterations=32,
                            swap_memory=True)

    output_final_ta = outputs[1]
    final_state = outputs[2]

    all_output = output_final_ta.stack()
    all_output.set_shape([None, None, output_size])
    all_output = tf.transpose(all_output, [1, 0, 2])
    return all_output, final_state
#


def model_x(hidden_size, dropout, inputs, inputs_length, model, params):
    if int(model)==1:
        # build cell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell")
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1-dropout)

        # build layer
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, sequence_length=inputs_length, dtype=inputs.dtype)

    elif int(model)==2:
        # build cell
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell_1")
        lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, output_keep_prob=1-dropout)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell_2")
        lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_2, output_keep_prob=1-dropout)
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])

        # build layer
        outputs, final_state = tf.nn.dynamic_rnn(multi_lstm_cell, inputs, sequence_length=inputs_length, dtype=inputs.dtype)

    elif int(model)==3:
        # build cell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell")
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1-dropout)

        # build layer
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, sequence_length=inputs_length, dtype=inputs.dtype)

        # outputs [batch_size, time_steps, hidden_size] -> [batch_size, hight, width, channel]
        # filter [hight, width, inchannel, outchannel]
        # stride [1, stride_hight, stride_width, 1]
        outputs = tf.expand_dims(outputs,-1)
        filters = tf.get_variable("filter", [params.window_size*2+1, 1, 1, 1], tf.float32,
                                  tf.ones_initializer,trainable=False)
        strides = [1, 1, 1, 1]
        outputs = tf.nn.conv2d(outputs, filters, strides, padding="SAME", data_format="NHWC")
        outputs = tf.squeeze(outputs, -1)

    elif int(model)==4:
        lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell_1")
        lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, output_keep_prob=1-dropout)
        lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell_2")
        lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_2, output_keep_prob=1-dropout)
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])

        # build layer
        outputs, final_state = tf.nn.dynamic_rnn(multi_lstm_cell, inputs, sequence_length=inputs_length, dtype=inputs.dtype)

        # outputs [batch_size, time_steps, hidden_size]
        # channel = inputs.get_shape()[1].value
        outputs = tf.expand_dims(outputs,-1)
        filters = tf.get_variable("filter", [params.window_size*2+1, 1, 1, 1], tf.float32,
                                  tf.ones_initializer,trainable=False)
        strides = [1, 1, 1, 1]
        outputs = tf.nn.conv2d(outputs, filters, strides, padding="SAME", data_format="NHWC")
        outputs = tf.squeeze(outputs, -1)

    elif int(model)==5:
        lstm_cell_f = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell_f")
        lstm_cell_f = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_f, output_keep_prob=1-dropout)
        lstm_cell_b = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell_b")
        lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_b, output_keep_prob=1-dropout)

        # build layer
        outputs, final_state = tf.nn.bidirectional_dynamic_rnn(lstm_cell_f,lstm_cell_b, inputs,
                                                               sequence_length=inputs_length, dtype=inputs.dtype)
        outputs = tf.concat(outputs,-1)
        outputs = linear_layer(outputs,hidden_size,name="fuse_f_b")

        # outputs [batch_size, time_steps, hidden_size]
        # channel = inputs.get_shape()[1].value
        outputs = tf.expand_dims(outputs,-1)
        filters = tf.get_variable("filter", [params.window_size*2+1, 1, 1, 1], tf.float32,
                                  tf.ones_initializer,trainable=False)
        strides = [1, 1, 1, 1]
        outputs = tf.nn.conv2d(outputs, filters, strides, padding="SAME", data_format="NHWC")
        outputs = tf.squeeze(outputs, -1)
    else:
        raise NotImplementedError()
    return outputs


def loss_x(params, features, logits, char_mask):
    if params.loss_criterion=="cross-entropy":
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=features["tags"])
        loss = tf.reduce_sum(ce * char_mask) / tf.reduce_sum(char_mask)
    elif params.loss_criterion=="max-margin":
        loss = None

    else:
        raise NotImplementedError()
    return loss



def model_graph(params, features, mode="instantiate"):
    # get params
    char_size = len(params.vocab)
    hidden_size = params.hidden_size
    dropout = params.dropout
    tag_types = len(params.tag)

    # get tensors
    chars = features["chars"]
    char_mask = (1-tf.sequence_mask(features["start"],maxlen=tf.shape(chars)[1],dtype=tf.float32))* \
                tf.sequence_mask(features["end"],maxlen=tf.shape(chars)[1],dtype=tf.float32)

    # build embeding
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)
    embedding = tf.get_variable("source_embedding",
                           [char_size, hidden_size],
                           initializer=initializer)

    # convert to input
    inputs = tf.gather(embedding, chars) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(char_mask, -1)
    inputs_length = features["end"]+features["start"]

    outputs = model_x(hidden_size, dropout, inputs, inputs_length, params.model, params)

    # build linear layer
    logits = linear_layer(outputs, tag_types)

    if mode=="instantiate":
        return None

    if mode == "inference":
        logits = logits * tf.expand_dims(char_mask, -1)
        logprobs = logits - tf.reduce_logsumexp(logits, axis=2, keepdims=True)
        return logprobs

    # build softmax layer
    loss = loss_x(params, features, logits, char_mask)

    # for monitor
    features["char_mask"]=char_mask
    features["logits"]=logits
    features["prediction"]=tf.argmax(logits,axis=-1)

    if mode=="train" or mode=="validation":
        return loss


class ModelManger:
    def __init__(self, base_params, scope=None):
        self.base_params = base_params
        self.model = base_params.model
        self.scope = scope or "model_"+self.model

    @staticmethod
    def get_name(model):
        return "model_"+model

    def instantiate(self, features):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            model_graph(self.base_params, features)

    def get_training_func(self, params=None):
        scope = self.scope
        params = params or self.base_params

        def fn(features):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                return model_graph(params,features,mode="train")

        return fn

    def get_validation_fn(self, params=None):
        scope = self.scope
        params = params or self.base_params
        params.dropout = 0.0

        def fn(features):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                return model_graph(params,features,mode="validation")

        return fn

    def get_inference_func(self, params=None):
        scope = self.scope
        params = params or self.base_params
        params.dropout = 0.0

        def fn(features):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                return model_graph(params,features,mode="inference")

        return fn



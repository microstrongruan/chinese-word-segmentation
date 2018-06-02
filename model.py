import tensorflow as tf


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


# def _copy_through(time, length, output, new_output):
#     copy_cond = (time >= length)
#     if isinstance(new_output, tf.nn.rnn_cell.LSTMStateTuple):
#         c, h = output
#         c_new, h_new = new_output
#         c_out = tf.where(copy_cond, c, c_new)
#         h_out = tf.where(copy_cond, h, h_new)
#         return tf.nn.rnn_cell.LSTMStateTuple(c_out, h_out)
#     return tf.where(copy_cond, output, new_output)
#
#
# def lstm_layer(cell, inputs, inputs_length, initial_state=None):
#     # get params
#     output_size = cell.output_size
#     batch = tf.shape(inputs)[0]
#     time_steps = tf.shape(inputs)[1]
#     dtype = inputs.dtype
#
#     # prepare tensor for iteration
#     zero_output = tf.zeros([batch, output_size], dtype)
#     if initial_state is None: initial_state = cell.zero_state(batch, dtype)
#     input_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="input_array")
#     output_ta = tf.TensorArray(dtype, time_steps, tensor_array_name="output_array")
#     input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))
#     time = tf.constant(0, dtype=tf.int32, name="time")
#     loop_vars = (time, output_ta, initial_state)
#
#     def loop_func(t, out_ta, state):
#         inp_t = input_ta.read(t)
#         cell_output, new_state = cell(inp_t, state)
#         cell_output = _copy_through(t, inputs_length, zero_output, cell_output)
#         new_state = _copy_through(t, inputs_length, state, new_state)
#         out_ta = out_ta.write(t, cell_output)
#         return t + 1, out_ta, new_state
#
#     outputs = tf.while_loop(lambda t, *_: t < time_steps, loop_func,
#                             loop_vars, parallel_iterations=32,
#                             swap_memory=True)
#
#     output_final_ta = outputs[1]
#     final_state = outputs[2]
#
#     all_output = output_final_ta.stack()
#     all_output.set_shape([None, None, output_size])
#     all_output = tf.transpose(all_output, [1, 0, 2])
#     return all_output, final_state
#


def model_x(hidden_size, dropout, inputs, inputs_length, model):
    if int(model)==1:
        # build cell
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell")
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1-dropout)

        # build layer
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, sequence_length=inputs_length, dtype=inputs.dtype)
        # outputs, final_state = lstm_layer(lstm_cell, inputs, inputs_length)
    elif int(model)==2:
        outputs = None
    else:
        raise NotImplementedError()
    return outputs


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

    outputs = model_x(hidden_size, dropout, inputs, inputs_length, params.model)

    # build linear layer
    logits = linear_layer(outputs, tag_types)

    if mode == "inference":
        logits = logits * tf.expand_dims(char_mask, -1)
        return logits

    # build softmax layer
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=features["tags"])

    # build loss
    loss = tf.reduce_sum(ce * char_mask) / tf.reduce_sum(char_mask)

    if mode=="train":
        return loss


class ModelManger:
    def __init__(self, base_params, scope=None):
        self.base_params = base_params
        self.model = base_params.model
        self.scope = scope or "model_"+self.model

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

    def get_inference_func(self, params):
        scope = self.scope
        params = params or self.base_params
        params.dropout = 0.0

        def fn(features):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                return model_graph(params,features,mode="inference")

        return fn



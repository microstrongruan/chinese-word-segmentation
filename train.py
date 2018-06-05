import argparse
import os
import json
import tensorflow as tf
import data
import model
import hook


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training cws",
        usage="train.py [<args>] [-h | --help]"
    )

    parser.add_argument("--input", type=str, required=True,
                        help="")
    parser.add_argument("--output", type=str, required=True,
                        help="")
    parser.add_argument("--validation", type=str,
                        help="")
    parser.add_argument("--reference", type=str,
                        help="")
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    return parser.parse_args(args)


def build_params(args):
    def default_parameters():
        params = tf.contrib.training.HParams(
            # args
            input=None,
            output=None,
            validation=None,
            reference=None,
            model=None,

            # data
            vocab=None,
            vocabback=None,
            transition=None,
            tag={"B": 0, "M": 1, "E": 2, "S": 3},
            num_threads=6,
            buffer_size=512,
            window_size=3,
            batch_size=32,

            # model
            hidden_size=620,
            dropout=0.2,

            # opt
            learning_rate=1e-4,
            learning_rate_decay="noam",  # noam, piecewise_constant, none
            warmup_steps=1000,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            clip_grad_norm=5.0,

            # train
            train_steps=300000,
            save_checkpoint_steps=1000,
            save_checkpoint_secs=None,
            keep_checkpoint_max=5,
            device_list=[0],

            # validation
            validate_steps=1000,
            validate_secs=None,
            validate_max_to_keep=10,

        )
        return params

    def import_parameters(import_dir):
        import_dir = os.path.abspath(import_dir)
        params_file = os.path.join(import_dir, "params.json")
        params = tf.contrib.training.HParams()

        if not os.path.exists(params_file):
            return params

        with open(params_file, "r") as f:
            tf.logging.info("Restoring parameters from %s", params_file)
            restore = json.load(f)
            for k, v in restore.items():
                params.add_hparam(k, v)
        return params

    def merge_parameters(params1, params2): # params2 have SWANhigher prior
        params = tf.contrib.training.HParams()
        for (k, v) in params1.values().items():
            params.add_hparam(k, v)

        params_dict = params.values()

        for (k, v) in params2.values().items():
            if k in params_dict:
                # Override
                setattr(params, k, v)
            else:
                params.add_hparam(k, v)

        return params

    def overwrite_parameters(params, args):
        params.input = args.input
        params.output = args.output
        params.validation = args.validation or params.validation
        params.reference = args.validation or params.reference
        params.model = args.model
        params.parse(args.parameters)
        return params

    def export_parameters(export_dir, params):
        export_dir = os.path.abspath(export_dir)
        if not tf.gfile.Exists(export_dir):
            tf.gfile.MakeDirs(export_dir)

        params_file = os.path.join(export_dir, "params.json")
        with tf.gfile.Open(params_file, "w") as f:
            tf.logging.info("Exporting parameters to %s", params_file)
            f.write(params.to_json())

    tf.logging.info("build params")
    params1 = default_parameters()
    params2 = import_parameters(args.output)
    params = merge_parameters(params1, params2)
    params = overwrite_parameters(params, args)
    if params.vocab is None:
        params.vocab, params.vocabback, params.transtion = data.build_vocab_trans(params.input, params.tag)
    export_parameters(args.output, params)

    return params


def build_learning_rate(params, global_step):
    if params.learning_rate_decay == "noam":
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return tf.convert_to_tensor(params.learning_rate * decay,tf.float32)
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.convert_to_tensor(tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values),
                                   tf.float32)
    elif params.learning_rate_decay == "none":
        return tf.convert_to_tensor(params.learning_rate, tf.float32)
    else:
        raise ValueError("Unknown learning_rate_decay")


def build_opt(params, learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate,
                                 beta1=params.adam_beta1,
                                 beta2=params.adam_beta2,
                                 epsilon=params.adam_epsilon)
    return opt


def build_train_op(params, loss, global_step, learning_rate, opt):
    train_op = tf.contrib.layers.optimize_loss(
        name="training",
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        clip_gradients=params.clip_grad_norm or None,
        optimizer=opt,
        colocate_gradients_with_ops=True
    )
    return train_op


def build_train_hooks(params, global_step, loss, features, validate_initializer, validate_op):
    train_hooks = [
        tf.train.StopAtStepHook(last_step=params.train_steps),
        tf.train.NanTensorHook(loss),
        tf.train.LoggingTensorHook(
            {
                "step": global_step,
                "loss": loss,
                "chars": tf.shape(features["chars"]),
            },
            every_n_iter=1
        ),
        tf.train.CheckpointSaverHook(
            checkpoint_dir=params.output,
            save_secs=params.save_checkpoint_secs or None,
            save_steps=params.save_checkpoint_steps or None,
            saver=tf.train.Saver(
                max_to_keep=params.keep_checkpoint_max,
                sharded=False
            )
        )
    ]
    if params.validation and params.reference:
        train_hooks.append(hook.ValidationHook(validate_initializer=validate_initializer,
                                               validate_op=validate_op,
                                               max_to_keep=params.validate_max_to_keep,
                                               output=params.output,
                                               validate_secs=params.validate_secs or None,
                                               validate_steps=params.validate_steps or None))
    return train_hooks


def build_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    # build params && build vocab and vocabback
    params = build_params(args)

    with tf.Graph().as_default():
        # build data pipe
        features = data.get_trainning_input(params)

        # build model_manager
        model_manager = model.ModelManger(params)
        model_manager.instantiate(features)

        # # build loss under single or multi GPU
        loss =  model_manager.get_training_func(params)(features)

        # build global step
        global_step = tf.train.get_or_create_global_step()

        # build learning rate
        learning_rate = build_learning_rate(params, global_step)

        # build opt
        opt = build_opt(params, learning_rate)

        # build train op
        train_op = build_train_op(params, loss, global_step, learning_rate, opt)

        # build validate op
        validate_initializer, validate_features = data.get_validation_input(params)
        validate_op =  model_manager.get_validation_fn(params)(validate_features)

        # build hooks
        train_hooks = build_train_hooks(params, global_step, loss, features, validate_initializer, validate_op)

        # build config
        config = build_config(params)

        # start session
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            while not sess.should_stop():
                sess.run(train_op)
                # res = sess.run(features)
                # print('------one res----')
                # for k,v in res.items():
                #     print("k", k)
                #     print("v", v)
                # x = input()


if __name__ == "__main__":
    main(parse_args())
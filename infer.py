import argparse
import os
import json
import tensorflow as tf
import data
import model
import search

def parse_args():
    parser = argparse.ArgumentParser(
        description="infer cws",
        usage="infer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--json", type=str, required=True,
                        help="Path of source and target vocabulary")
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")
    return parser.parse_args()


def build_params(args):
    def default_parameters():
        params = tf.contrib.training.HParams(
            # args
            input=None,
            output=None,
            validation=None,
            reference=None,
            model=None,
            checkpoints=None,

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

            # validate
            validate_steps=1000,
            validate_secs=None,

            # infer
            search_policy="viterbi",
        )
        return params

    def import_parameters(json_file):
        params_file = os.path.abspath(json_file)
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
        params.model = args.model
        params.checkpoints = args.checkpoints
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
    params2 = import_parameters(args.json)
    params = merge_parameters(params1, params2)
    params = overwrite_parameters(params, args)
    if params.vocab is None or params.vocabback is None:
        tf.logging.error("json needs to provide vocab")
    # export_parameters(args.output, params)
    return params


def build_checkpoints(params):
    model_var_lists = []
    for i, checkpoint in enumerate(params.checkpoints):
        tf.logging.info("Loading %s" % checkpoint)
        var_list = tf.train.list_variables(checkpoint)
        values = {}
        reader = tf.train.load_checkpoint(checkpoint)

        for (name, shape) in var_list:
            if not name.startswith(model.ModelManger.get_name(params.model)):
                continue

            if name.find("losses_avg") >= 0:
                continue

            tensor = reader.get_tensor(name)
            values[name] = tensor

        model_var_lists.append(values)
    return model_var_lists


def build_models(params, features):
    model_fns = []
    for i in range(len(params.checkpoints)):
        model_manager = model.ModelManger(params, model.ModelManger.get_name(params.model)+"_%d"%i)
        model_manager.instantiate(features)
        model_fns.append(model_manager.get_inference_func())
    return model_fns


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.info("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break

    return ops


def build_assign(params, model_var_lists):
    assign_ops = []
    all_var_list = tf.trainable_variables()
    for i in range(len(params.checkpoints)):
        un_init_var_list = []
        name = model.ModelManger.get_name(params.model)

        for v in all_var_list:
            if v.name.startswith(name + "_%d" % i):
                un_init_var_list.append(v)

        ops = set_variables(un_init_var_list, model_var_lists[i], name + "_%d" % i)
        assign_ops.extend(ops)

    assign_op = tf.group(*assign_ops)
    return assign_op


def build_predictions(model_fns, features):
    # predictions = search(model_fns, features)
    predictions = []
    for model_fn in model_fns:
        predictions.append(model_fn(features))
    return tf.add_n(predictions)/len(predictions)


def build_session_creator(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    sess_creator = tf.train.ChiefSessionCreator(config=config)
    return sess_creator


def write(params, newlines):
    output_file = os.path.abspath(params.output)
    output_dir = os.path.split(output_file)[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_file, "a") as f:
        for line in newlines:
            f.writelines(line)


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    # build params
    params = build_params(args)

    # Build Graph
    with tf.Graph().as_default():
        # build data_pipe
        features = data.get_inference_input(params)

        # build checkpoints
        model_var_lists = build_checkpoints(params)

        # build models
        model_fns = build_models(params, features)

        # build assign
        assign_op = build_assign(params, model_var_lists)

        # build logits
        logprobs = build_predictions(model_fns, features)
        features["logprobs"] = logprobs

        # build config
        session_creater = build_session_creator(params)

        # Create session
        with tf.train.MonitoredSession(session_creator=session_creater) as sess:
            # Restore variables
            sess.run(assign_op)

            while not sess.should_stop():
                # result = sess.run(features)
                result = sess.run(features)
                newlines = search.recover_features(params, result)
                write(params, newlines)

                print('------one res----')
                for k,v in result.items():
                    print("k", k)
                    if k=="origin":
                        print("v", [vv.decode("utf-8") for vv in v])
                    else:
                        print("v", v)
                    print("v_len", v.shape)
                print("newlines", newlines)
                x = input()


if __name__ == "__main__":
    main(parse_args())

import os
import datetime
import operator
import tensorflow as tf


def _get_saver():
    # Get saver from the SAVERS collection if present.
    collection_key = tf.GraphKeys.SAVERS
    savers = tf.get_collection(collection_key)

    if not savers:
        raise RuntimeError("No items in collection {}. "
                           "Please add a saver to the collection ")
    elif len(savers) > 1:
        raise RuntimeError("More than one item in collection")

    return savers[0]


def _save_log(filename, result):
    metric, global_step, score = result

    with open(filename, "a") as fd:
        time = datetime.datetime.now()
        msg = "%s: %s at step %d: %f\n" % (time, metric, global_step, score)
        fd.write(msg)


def _read_checkpoint_def(filename):
    records = []

    with tf.gfile.GFile(filename) as fd:
        fd.readline()

        for line in fd:
            records.append(line.strip().split(":")[-1].strip()[1:-1])

    return records


def _save_checkpoint_def(filename, checkpoint_names):
    keys = []

    for checkpoint_name in checkpoint_names:
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, checkpoint_name))

    sorted_names = sorted(keys, key=operator.itemgetter(0),
                          reverse=True)

    with tf.gfile.GFile(filename, "w") as fd:
        fd.write("model_checkpoint_path: \"%s\"\n" % checkpoint_names[0])

        for checkpoint_name in sorted_names:
            checkpoint_name = checkpoint_name[1]
            fd.write("all_model_checkpoint_paths: \"%s\"\n" % checkpoint_name)


def _read_score_record(filename):
    # "checkpoint_name": score
    records = []

    if not tf.gfile.Exists(filename):
        return records

    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            name, score = line.strip().split(":")
            name = name.strip()[1:-1]
            score = float(score)
            records.append([name, score])

    return records


def _save_score_record(filename, records):
    keys = []

    for record in records:
        checkpoint_name = record[0]
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, record))

    sorted_keys = sorted(keys, key=operator.itemgetter(0),
                         reverse=True)
    sorted_records = [item[1] for item in sorted_keys]

    with tf.gfile.GFile(filename, "w") as fd:
        for record in sorted_records:
            checkpoint_name, score = record
            fd.write("\"%s\": %f\n" % (checkpoint_name, score))


def _add_to_record(records, record, max_to_keep):
    added = None
    removed = None
    models = {}

    for (name, score) in records:
        models[name] = score

    if len(records) < max_to_keep:
        if record[0] not in models:
            added = record[0]
            records.append(record)
    else:
        sorted_records = sorted(records, key=lambda x: -x[1])
        worst_score = sorted_records[-1][1]
        current_score = record[1]

        if current_score >= worst_score:
            if record[0] not in models:
                added = record[0]
                removed = sorted_records[-1][0]
                records = sorted_records[:-1] + [record]

    # Sort
    records = sorted(records, key=lambda x: -x[1])

    return added, removed, records


class ValidationHook(tf.train.SessionRunHook):
    """validate a validate set every N steps or seconds."""

    def __init__(self,
                 validate_initializer,
                 validate_op,
                 output,
                 max_to_keep,
                 validate_secs=None,
                 validate_steps=None,
                 steps_per_run=1,
                 metric="validation loss"):

        tf.logging.info("Create ValidationHook.")
        self._validate_initializer = validate_initializer
        self._validate_op = validate_op
        self._global_step = None
        self._max_to_keep = max_to_keep
        self._base_dir = output
        self._save_dir = os.path.join(output, "validation")
        self._log_name = os.path.join(self._save_dir,"log")
        self._record_name = os.path.join(self._save_dir, "record")
        self._timer = tf.train.SecondOrStepTimer(every_secs=validate_secs,
                                                 every_steps=validate_steps)
        self._steps_per_run = steps_per_run
        self._metric = metric

    def begin(self):
        global_step = tf.train.get_global_step()
        self._global_step = global_step
        if not os.path.exists(self._save_dir):
            tf.logging.info("Making dir: %s" % self._save_dir)
            tf.gfile.MakeDirs(self._save_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results

        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step)

            # Get the real value
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                # Save model
                save_path = os.path.join(self._base_dir, "model.ckpt")
                saver = _get_saver()
                tf.logging.info("Saving checkpoints for %d into %s." %
                                (global_step, save_path))
                saver.save(run_context.session,
                           save_path,
                           global_step=global_step)
                # Do validation here
                tf.logging.info("Validating model at step %d" % global_step)
                # score = _evaluate(self._eval_fn, self._eval_input_fn,
                #                   self._eval_decode_fn,
                #                   self._base_dir,
                #                   self._session_config)
                score = self._validate(run_context.session)
                tf.logging.info("%s at step %d: %f" %
                                (self._metric, global_step, score))

                _save_log(self._log_name, (self._metric, global_step, score))

                checkpoint_filename = os.path.join(self._base_dir,
                                                   "checkpoint")
                all_checkpoints = _read_checkpoint_def(checkpoint_filename)
                records = _read_score_record(self._record_name)
                latest_checkpoint = all_checkpoints[-1]
                record = [latest_checkpoint, score]
                added, removed, records = _add_to_record(records, record,
                                                         self._max_to_keep)

                if added is not None:
                    old_path = os.path.join(self._base_dir, added)
                    new_path = os.path.join(self._save_dir, added)
                    old_files = tf.gfile.Glob(old_path + "*")
                    tf.logging.info("Copying %s to %s" % (old_path, new_path))

                    for o_file in old_files:
                        n_file = o_file.replace(old_path, new_path)
                        tf.gfile.Copy(o_file, n_file, overwrite=True)

                if removed is not None:
                    filename = os.path.join(self._save_dir, removed)
                    tf.logging.info("Removing %s" % filename)
                    files = tf.gfile.Glob(filename + "*")

                    for name in files:
                        tf.gfile.Remove(name)

                _save_score_record(self._record_name, records)
                checkpoint_filename = checkpoint_filename.replace(
                    self._base_dir, self._save_dir
                )
                _save_checkpoint_def(checkpoint_filename,
                                     [item[0] for item in records])

                best_score = records[0][1]
                tf.logging.info("Best score at step %d: %f" %
                                (global_step, best_score))







        # stale_global_step = run_values.results
        # if self._timer.should_trigger_for_step(
        #         stale_global_step + self._steps_per_run):
        #     # get the real value after train op.
        #     global_step = run_context.session.run(self._global_step)
        #     if self._timer.should_trigger_for_step(global_step):
        #         self._timer.update_last_triggered_step(global_step)
        #
        #         # Save model
        #         save_file = os.path.join(self._base_dir, "model.ckpt")
        #         saver = _get_saver()
        #         tf.logging.info("Saving checkpoints for %d into %s." %
        #                         (global_step, save_file))
        #         saver.save(run_context.session, save_file, global_step=global_step)
        #
        #         # Validate
        #         tf.logging.info("Validating model at step %d" % global_step)
        #         score = self._validate(run_context.session)
        #         tf.logging.info("%s at step %d: %f" %
        #                         (self._metric, global_step, score))
        #
        #         _save_log(self._log_name, (self._metric, global_step, score))
        #
        #         checkpoint_filename = os.path.join(self._base_dir, "checkpoint")
        #         all_checkpoints = _read_checkpoint_def(checkpoint_filename)
        #         records = _read_score_record(self._record_name)
        #         latest_checkpoint = all_checkpoints[-1]
        #         record = [latest_checkpoint, score]
        #         added, removed, records = _add_to_record(records, record,
        #                                                  self._max_to_keep)
        #
        #         if added is not None:
        #             old_path = os.path.join(self._base_dir, added)
        #             new_path = os.path.join(self._save_dir, added)
        #             old_files = tf.gfile.Glob(old_path + "*")
        #             tf.logging.info("Copying %s to %s" % (old_path, new_path))
        #
        #             for o_file in old_files:
        #                 n_file = o_file.replace(old_path, new_path)
        #                 tf.gfile.Copy(o_file, n_file, overwrite=True)
        #
        #         if removed is not None:
        #             filename = os.path.join(self._save_dir, removed)
        #             tf.logging.info("Removing %s" % filename)
        #             files = tf.gfile.Glob(filename + "*")
        #
        #             for name in files:
        #                 tf.gfile.Remove(name)
        #
        #         _save_score_record(self._record_name, records)
        #         checkpoint_filename = checkpoint_filename.replace(self._base_dir, self._save_dir)
        #         _save_checkpoint_def(checkpoint_filename,[item[0] for item in records])
        #
        #         best_score = records[0][1]
        #         tf.logging.info("Best score at step %d: %f" %(global_step, best_score))

    def end(self, session):
        pass

    def _validate(self, session):
        # features = data.get_validation_input(self._params)
        total_loss = []

        session.run(self._validate_initializer)
        while True:
            try:
                total_loss.append(session.run(self._validate_op))
            except tf.errors.OutOfRangeError:
                break

        return -sum(total_loss)/len(total_loss)


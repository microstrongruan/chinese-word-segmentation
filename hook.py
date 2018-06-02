import os
import tensorflow as tf


class ValidationHook(tf.train.SessionRunHook):
    """validate a validate set every N steps or seconds."""

    def __init__(self,
                 validateion,
                 reference,
                 output,
                 validate_secs=None,
                 validate_steps=None,
                 steps_per_run=1):

        tf.logging.info("Create ValidationHook.")
        self._validation=validateion
        self._reference=reference
        self._global_step = None
        self._save_dir = os.path.join(output,"validation")
        self._timer = tf.train.SecondOrStepTimer(every_secs=validate_secs,
                                                 every_steps=validate_steps)
        self._steps_per_run = steps_per_run

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
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                # self._validate(run_context.session, global_step)
                print(run_context)
                print(run_context.session)

    def end(self, session):
        pass


# coding=utf-8
"""main function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import data_reader
import cnn_modeling
import optimization
import tokenization_ch
import time
from  datetime import  datetime
from custom_export_output import CustomClassificationOutput
from custom_export_output import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from custom_export_output import SIGNATURE_OUTPUT_NAME

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "model_config_file", None,
    "The config json file corresponding to the model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("online_signature_export", False, "Whether to export signature for online")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("log_frequency", 1000,
                     "How many steps to make log.")

flags.DEFINE_bool("extract_feature", False, "Whether to just extract features.")


def create_model(bert_config, is_training, input_ids, labels, num_labels):
    """Creates a classification model."""
    model = cnn_modeling.TextCNNModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        num_labels=num_labels)

    # In the demo, we are doing a simple classification task on the entire
    # segment.

    with tf.variable_scope("loss"):

        logits = model.get_logits()
        probabilities = tf.nn.softmax(logits, dim=-1)
        log_probs = tf.nn.log_softmax(logits, dim=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, logits, probabilities


def _def_logger_hook(loss, train_accuracy):
    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""
        def __init__(self):
            self.avg_loss = None
            self.avg_accu = None
            self.decay = 0.9999

        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs([loss, train_accuracy])  # Asks for value.

        def after_run(self, run_context, run_values):
            loss_value, acc_value = run_values.results

            # running avg
            if self.avg_loss is None:
                self.avg_loss = loss_value
            else:
                self.avg_loss = self.avg_loss * self.decay + (1 - self.decay) * loss_value

            if self.avg_accu is None:
                self.avg_accu = acc_value
            else:
                self.avg_accu = self.avg_accu * self.decay + (1 - self.decay) * acc_value

            if self._step % FLAGS.log_frequency == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                examples_per_sec = FLAGS.log_frequency * FLAGS.train_batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)
                format_str = ('%s: step %d, loss = %.3f, train_accu: %.3f,'
                              ' (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), self._step, loss_value,
                                    acc_value, examples_per_sec, sec_per_batch))

    return _LoggerHook


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, extract_feature):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        # when invoking in online signature_export, features is a dict (name, tensor)
        # when invoking in other method(train/test), features is a DataSet object
        if FLAGS.online_signature_export and params[FLAGS.online_signature_export]:
            for name in sorted(features.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            input_ids = features["input_ids"]
            label_ids = features["label_ids"]
        else:
            for name in sorted(features.output_shapes.keys()):
                tf.logging.info("  name = %s, shape = %s" % (name, features.output_shapes[name]))

            # tf 1.4 features is a Dataset which needs initializer
            if mode == tf.estimator.ModeKeys.TRAIN:
                # make_initializable_iterator for train
                batched_iter = features.make_initializable_iterator()
                init_train = batched_iter.initializer
            else:
                # make_one_shot_iterator for eval
                batched_iter = features.make_one_shot_iterator()

            iterator = batched_iter.get_next()
            input_ids = iterator["input_ids"]
            label_ids = iterator["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, label_ids,
            num_labels)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None

        # tf 1.4 add iterator to scaffold
        if mode == tf.estimator.ModeKeys.TRAIN:
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            train_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, label_ids), tf.float32))
            scaffold_fn = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(), init_train))


        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = cnn_modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, extract_feature)

            _LoggerHook = _def_logger_hook(total_loss, train_accuracy)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[_LoggerHook()],
                scaffold=scaffold_fn
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                eval_accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)
                return {
                    "eval_accuracy": eval_accuracy,
                    "eval_loss": loss,
                }

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=metric_fn(per_example_loss, label_ids, logits),
                scaffold=scaffold_fn
            )

        else:
            # model to Predict
            # get export_outputs
            export_outputs = CustomClassificationOutput(scores={SIGNATURE_OUTPUT_NAME: probabilities})

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=probabilities,
                scaffold=scaffold_fn,
                export_outputs={DEFAULT_SERVING_SIGNATURE_DEF_KEY: export_outputs})

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "yesno": data_reader.YesNoDataProcessor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    model_config = cnn_modeling.TextCNNConfig.from_json_file(FLAGS.model_config_file)

    if FLAGS.max_seq_length > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, model_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization_ch.ChTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # for tf 1.4
    run_confg = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=model_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        extract_feature=FLAGS.extract_feature)

    params = {}
    if FLAGS.do_train:
        params["batch_size"] = FLAGS.train_batch_size
    elif FLAGS.do_eval:
        params["batch_size"] = FLAGS.eval_batch_size
    else:
        params["batch_size"] = FLAGS.predict_batch_size

    if FLAGS.online_signature_export:
        params["online_signature_export"] = FLAGS.online_signature_export

    # for tf 1.4
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_confg,
        params=params
    )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        data_reader.file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = data_reader.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        data_reader.file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        eval_drop_remainder = False
        eval_input_fn = data_reader.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        data_reader.file_based_convert_examples_to_features(predict_examples, label_list,
                                                            FLAGS.max_seq_length, tokenizer, predict_file)

        predict_input_fn = data_reader.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    if FLAGS.online_signature_export:
        def serving_input_fn():
            input_ids = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length], name="input_ids")
            input_mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length], name="input_mask")
            segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length], name="segment_ids")
            label_ids = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids")

            features = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids,
                        'label_ids': label_ids}
            receiver_tensors = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}

            return tf.estimator.export.ServingInputReceiver(
                features=features, receiver_tensors=receiver_tensors, receiver_tensors_alternatives={})

        tf.logging.info("***** Starting to export *****")
        estimator.export_savedmodel(FLAGS.output_dir, serving_input_fn)
        tf.logging.info("***** Ending export *****")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

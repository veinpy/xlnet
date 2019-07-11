from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import math
import json
import time
import numpy as np

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

import data_utils
import model_utils
from gpu_utils import assign_to_gpu, average_grads_and_vars
import function_builder


# GPU config
flags.DEFINE_integer("num_hosts", default=1,
      help="Number of hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")
flags.DEFINE_bool("use_tpu", default=False,
      help="Whether to use TPUs for training.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_integer("num_passes", default=1,
      help="Number of passed used for training.")
flags.DEFINE_string("record_info_dir", default=None,
      help="Path to local directory containing `record_info-lm.json`.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_string("init_checkpoint", default=None,
      help="checkpoint path for initializing the model.")

# Optimization config
flags.DEFINE_float("learning_rate", default=1e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=1.0,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.001,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")
flags.DEFINE_float("adam_epsilon", default=1e-8,
      help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly",
      help="poly or cos")
flags.DEFINE_float("weight_decay", default=0.0,
      help="weight decay")

# Training config
flags.DEFINE_integer("train_batch_size", default=16,
      help="Size of train batch.")
flags.DEFINE_integer("train_steps", default=100000,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=1000,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=None,
      help="number of steps for model checkpointing.")

# Data config
flags.DEFINE_integer('seq_len', default=0,
      help='Sequence length for pretraining.')
flags.DEFINE_integer('reuse_len', default=0,
      help="How many tokens to be reused in the next batch. "
      "Could be half of seq_len")
flags.DEFINE_bool("bi_data", default=True,
      help="Use bidirectional data streams, i.e., forward & backward.")
flags.DEFINE_integer("mask_alpha", default=6,
      help="How many tokens to form a group.")
flags.DEFINE_integer("mask_beta", default=1,
      help="How many tokens to mask within each group.")
flags.DEFINE_integer("num_predict", default=None,
      help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer('perm_size', default=None,
  help='perm size.')
flags.DEFINE_bool("uncased", False,
      help="Use uncased inputs or not.")
flags.DEFINE_integer("n_token", 32000, help="Vocab size")

# Model config
flags.DEFINE_integer("mem_len", default=0,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=32,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=8,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=32,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.0,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_string("ff_activation", default="relu",
      help="Activation type used in position-wise feed-forward.")
flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")


FLAGS = flags.FLAGS

def create_mems_tf(bsz_per_core):
  mems = [tf.placeholder(dtype=tf.float32,
                         shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model])
          for layer in range(FLAGS.n_layer)]

  return mems


def get_model_fn():
  def model_fn(features, labels, mode, params):
      is_training = (mode == tf.estimator.ModeKeys.TRAIN)



      if mode == tf.estimator.ModeKeys.PREDICT:
          # prediction dataset
          bsz_per_core = FLAGS.train_batch_size // FLAGS.num_core_per_host
          mems = {}
          if FLAGS.mem_len:
              mems['mems'] = create_mems_tf(bsz_per_core)
          output = function_builder.get_loss(FLAGS, features,labels,mems, is_training)
          predictions = {
              "outputs": output
          }
          #### load pretrained models
          scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

          output_spec = tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
          return output_spec
      if mode == tf.estimator.ModeKeys.TRAIN:
          # train dataset
          pass

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    train_input_fn, record_info_dict = data_utils.get_input_fn(
        tfrecord_dir=FLAGS.record_info_dir,
        split="train",
        bsz_per_host=FLAGS.train_batch_size,
        seq_len=FLAGS.seq_len,
        reuse_len=FLAGS.reuse_len,
        bi_data=FLAGS.bi_data,
        num_hosts=1,
        num_core_per_host=1,  # set to one no matter how many GPUs
        perm_size=FLAGS.perm_size,
        mask_alpha=FLAGS.mask_alpha,
        mask_beta=FLAGS.mask_beta,
        uncased=FLAGS.uncased,
        num_passes=FLAGS.num_passes,
        use_bfloat16=FLAGS.use_bfloat16,
        num_predict=FLAGS.num_predict)

    tf.logging.info("num of batches {}".format(record_info_dict["num_batch"]))

    bsz_per_core = FLAGS.train_batch_size // FLAGS.num_core_per_host
    params = {
        "batch_size": FLAGS.train_batch_size  # the whole batch
    }
    train_set = train_input_fn(params)
    example = train_set.make_one_shot_iterator().get_next()

    run_config = model_utils.configure_tpu(FLAGS)
    model_fn = get_model_fn()
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    all_preds = []

    gpu_options = tf.GPUOptions(allow_growth=True)
    for result in estimator.predict(input_fn=train_input_fn,yield_single_examples=True):
        all_preds.append(result)

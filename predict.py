"""Pretraining on GPUs."""
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
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("xlData/sellpoint/m_corpus_v2.model")

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
flags.DEFINE_bool("do_train", default=False,
      help="whether train ")
flags.DEFINE_bool("do_eval", default=False,
      help="whether eval ")
flags.DEFINE_bool("do_predict", default=False,
      help="whether predict ")
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


def visualize_infer_result():
    """
    visualize one batch of infer result
    :return:
    """



def get_model_fn():
  def model_fn(features, labels, mems, mode):
    is_training=(mode==tf.estimator.ModeKeys.TRAIN)

    if mode==tf.estimator.ModeKeys.TRAIN:
        #### Get loss from inputs
        total_loss, new_mems, monitor_dict = function_builder.get_loss(
            FLAGS, features, labels, mems, is_training)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # GPU
        assert is_training
        all_vars = tf.trainable_variables()
        grads = tf.gradients(total_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))

        return total_loss, new_mems, grads_and_vars
    elif mode==tf.estimator.ModeKeys.EVAL:
        raise NotImplementedError()
    elif mode==tf.estimator.ModeKeys.PREDICT:
        outputs = function_builder.get_loss(
            FLAGS, features, labels, mems, is_training,mode=mode)
        return outputs
  return model_fn


def single_core_graph(mode, features, mems):
  model_fn = get_model_fn()

  model_ret = model_fn(
      features=features,
      labels=None,
      mems=mems,
      mode=mode)

  return model_ret


def create_mems_tf(bsz_per_core):
  mems = [tf.placeholder(dtype=tf.float32,
                         shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model])
          for layer in range(FLAGS.n_layer)]

  return mems


def initialize_mems_np(bsz_per_core):
  mems_np = [np.zeros(shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model],
                      dtype=np.float32)
             for layer in range(FLAGS.n_layer)]

  return mems_np


def predict(ps_device):
  ##### Get input function and model function
  train_input_fn, record_info_dict = data_utils.get_input_fn(
      tfrecord_dir=FLAGS.record_info_dir,
      split="train",
      bsz_per_host=FLAGS.train_batch_size,
      seq_len=FLAGS.seq_len,
      reuse_len=FLAGS.reuse_len,
      bi_data=FLAGS.bi_data,
      num_hosts=1,
      num_core_per_host=1, # set to one no matter how many GPUs
      perm_size=FLAGS.perm_size,
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      uncased=FLAGS.uncased,
      num_passes=FLAGS.num_passes,
      use_bfloat16=FLAGS.use_bfloat16,
      num_predict=FLAGS.num_predict)

  # for key, info in record_info_dict.items():
  tf.logging.info("num of batches {}".format(record_info_dict["num_batch"]))

  ##### Create input tensors / placeholders
  bsz_per_core = FLAGS.train_batch_size // FLAGS.num_core_per_host

  params = {
      "batch_size": FLAGS.train_batch_size # the whole batch
  }
  train_set = train_input_fn(params)

  example = train_set.make_one_shot_iterator().get_next()
  examples = [example]

  ##### Create computational graph
  tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

  mems_i = {}
  if FLAGS.mem_len:
    mems_i["mems"] = create_mems_tf(bsz_per_core)

  _predicts_hid, new_mems_i,_predicts = single_core_graph(
      mode="infer",
      features=examples[0],
      mems=mems_i)
  tower_mems.append(mems_i)
  tower_new_mems.append(new_mems_i)


  ## average losses and gradients across towers
  # if len(tower_losses) > 1:
  #   loss = tf.add_n(tower_losses) / len(tower_losses)
  #   grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
  # else:
  #   loss = tower_losses[0]
  #   grads_and_vars = tower_grads_and_vars[0]
  #
  # ## get train op
  # train_op, learning_rate, gnorm = model_utils.get_train_op(FLAGS, None,
  #     grads_and_vars=grads_and_vars)
  # global_step = tf.train.get_global_step()

  ##### Training loop
  # initialize mems
  tower_mems_np = []
  for i in range(FLAGS.num_core_per_host):
    mems_i_np = {}
    for key in tower_mems[i].keys():
      mems_i_np[key] = initialize_mems_np(bsz_per_core)
    tower_mems_np.append(mems_i_np)

  gpu_options = tf.GPUOptions(allow_growth=True)

  model_utils.init_from_checkpoint(FLAGS, global_vars=True)

  all_predictions = []
  all_inputs = []

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
      gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    fetches = [_predicts, tower_new_mems, examples[0]]

    # if not FLAGS.init_checkpoint:
    #     total_loss, prev_step = 0., -1
    # else:
    #     total_loss, prev_step = 0., sess.run(global_step,{})
    #     FLAGS.train_steps += prev_step + FLAGS.train_steps

    cnt =0
    while True:
      feed_dict = {}
      for key in tower_mems_np[0].keys():
        for m, m_np in zip(tower_mems[i][key], tower_mems_np[0][key]):
          feed_dict[m] = m_np

      fetched = sess.run(fetches, feed_dict=feed_dict)
      # loss_np, tower_mems_np, curr_step = fetched[:3]
      # total_loss += loss_np
      predict = fetched[0]
      tower_mems_np = fetched[1]
      input_data = fetched[2]

      all_predictions.append(predict)
      all_inputs.append(input_data)

      cnt+=1
      if cnt==1:
          import ipdb;ipdb.set_trace()

      if cnt >=int(9346/FLAGS.train_batch_size):
        import ipdb;ipdb.set_trace()
      print("??")


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get corpus info
  FLAGS.n_token = data_utils.VOCAB_SIZE
  tf.logging.info("n_token {}".format(FLAGS.n_token))

  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)

  predict("/gpu:0")


if __name__ == "__main__":
  tf.app.run()

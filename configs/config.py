from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

##########################
#                  Model and summary
##########################
tf.app.flags.DEFINE_string(
    'train_dir', './output/summary/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_model', './output/models/scene_recognition_final.ckpt',
    'Path to checkpoint model')

tf.app.flags.DEFINE_string(
    'latest_ckpt', './output/training',
    'Path to latest checkpoint model')


##########################
#                  dataset
##########################
tf.app.flags.DEFINE_string(
    'dataset_dir', 'data',
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'meta/train.txt',
    'The name of the train/test/val split.')

tf.app.flags.DEFINE_string(
    'surfix', 'train',
    'surfix name')

tf.app.flags.DEFINE_integer(
    'num_classes', 365,
    'The number of classes for training.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'max_iters', 3000000,
    'max iterations')

tf.app.flags.DEFINE_integer(
    'num_epochs', 200,
    'the maximum number of epoch')

tf.app.flags.DEFINE_integer(
    'image_min_size', 224,
    'resize image so that the min edge equals to image_min_size')


######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.app.flags.FLAGS

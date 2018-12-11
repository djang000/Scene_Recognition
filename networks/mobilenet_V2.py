import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import namedtuple

class MobileNet_V2(object):
    def __init__(self, input, num_classes=1000, weight_decay=0.0004, is_training=True):
        self.imgs      = input
        self.depth_multiplier = 1.0
        self.min_depth = 8
        self.num_classes = num_classes
        self.up_sample=6
        self.weight_decay = weight_decay
        self.is_training = is_training
        self.end_points = {}
        self.build_model()

    def build_model(self):
        # channels = [32, 16, 24, 32,64, 96, 160, 320, 1280]
        with tf.variable_scope('MobilenetV2') as sc:
            depth = lambda d: max(int(d*self.depth_multiplier), self.min_depth)
            with slim.arg_scope(training_scope(is_training=self.is_training, weight_decay=self.weight_decay)):
                x = slim.conv2d(self.imgs, depth(32), [3, 3], stride=2, activation_fn=tf.nn.relu6)
                self.end_points['Conv2d'] = x
                x = self.InvertedBottleneck(x, up_sample=1, out_ch=16, stride=1, scope='expanded_conv')

                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=24, stride=2, scope='expanded_conv_1')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=24, stride=1, scope='expanded_conv_2')

                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=32, stride=2, scope='expanded_conv_3')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=32, stride=1, scope='expanded_conv_4')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=32, stride=1, scope='expanded_conv_5')

                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=64, stride=2, scope='expanded_conv_6')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=64, stride=1, scope='expanded_conv_7')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=64, stride=1, scope='expanded_conv_8')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=64, stride=1, scope='expanded_conv_9')

                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=96, stride=1, scope='expanded_conv_10')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=96, stride=1, scope='expanded_conv_11')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=96, stride=1, scope='expanded_conv_12')

                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=160, stride=2, scope='expanded_conv_13')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=160, stride=1, scope='expanded_conv_14')
                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=160, stride=1, scope='expanded_conv_15')

                x = self.InvertedBottleneck(x, up_sample=self.up_sample, out_ch=320, stride=1, scope='expanded_conv_16')

                x = slim.conv2d(x, depth(1280), [1, 1], stride=1, activation_fn=tf.nn.relu6)
                self.end_points['Conv2d_1'] = x

                x = slim.avg_pool2d(x, [7, 7], scope='avg_pool')

        logits = slim.conv2d(x, depth(self.num_classes), [1, 1], activation_fn=None, normalizer_fn=None, scope='Logists/Conv2d_1c_1x1')
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        predictions = slim.softmax(logits, scope='Predictions')

        self.end_points['logits'] = logits
        self.end_points['predictions'] = predictions

    def InvertedBottleneck(self, input_tensor, up_sample, out_ch, stride, scope):
        with tf.variable_scope(scope):
            in_ch = input_tensor.get_shape().as_list()[-1]
            inner_size = up_sample * in_ch
            x = input_tensor
            if inner_size > in_ch:
                x = slim.conv2d(x, inner_size, [1, 1], stride=1, activation_fn=tf.nn.relu6, scope='expand')
                self.end_points[scope+'/expand'] = x
            x = slim.separable_conv2d(x, num_outputs=None,
                                      kernel_size=[3, 3],
                                      depth_multiplier=self.depth_multiplier,
                                      stride=stride,
                                      activation_fn=tf.nn.relu6,
                                      scope='depthwise')
            self.end_points[scope + '/depthwise'] = x

            out_tensor = slim.conv2d(x, out_ch, [1, 1], stride=1, activation_fn=tf.identity, scope='project')
            self.end_points[scope + '/project'] = out_tensor

            if stride == 1 and in_ch == out_ch:
                out_tensor += input_tensor
                out_tensor = tf.identity(out_tensor, name='output')

            return out_tensor


    def add_summary(self, x):
        tf.summary.histogram(x.op.name+'/activations', x)
        tf.summary.scalar(x.op.name+'/sparsity', tf.nn.zero_fraction(x))

def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8):
    """Defines Mobilenet training scope.
    Usage:
        with tf.contrib.slim.arg_scope(mobilenet.training_scope()):
        logits, endpoints = mobilenet_v2.mobilenet(input_tensor)
        # the network created will be trainble with dropout/batch norm
        # initialized appropriately.
    Args:
        is_training: if set to False this will ensure that all customizations are
            set to non-training mode. This might be helpful for code that is reused
            across both training/evaluation, but most of the time training_scope with
            value False is not needed. If this is set to None, the parameters is not
            added to the batch_norm arg_scope.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: Standard deviation for initialization, if negative uses xavier.
        dropout_keep_prob: dropout keep probability (not set if equals to None).
        bn_decay: decay for the batch norm moving averages (not set if equals to None).
    Returns:
        An argument scope to use via arg_scope.
    """
    # Note: do not introduce parameters that would change the inference
    # model here (for example whether to use bias), modify conv_def instead.
    batch_norm_params = {
        'decay': 0.997,
        'is_training': is_training
    }
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
        weights_initializer=weight_intitializer,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as sc:
                    return sc


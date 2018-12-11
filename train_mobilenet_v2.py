import os
from time import gmtime, strftime
import random
import tensorflow as tf
import configs.config as cfg
import networks.mobilenet_V2 as model
import data.datapipe as datasets
import tensorflow.contrib.slim as slim

class_to_labels = {}
labels_to_class = {}

def _get_learning_rate(num_sample_per_epoch, global_step):
    decay_step = int((num_sample_per_epoch / cfg.FLAGS.batch_size) * cfg.FLAGS.num_epochs_per_decay)
    return tf.train.exponential_decay(cfg.FLAGS.learning_rate,
                                      global_step,
                                      decay_step,
                                      cfg.FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')

def generate_dir_file_map(image_dir, train_info, class_to_labels):
    files = []
    labels = []

    with open(train_info, 'r') as txt:
        datas = [l.strip() for l in txt.readlines()]
        for f in datas:
            dir_name, id = f.split('/')
            l = class_to_labels[dir_name]
            files.append(image_dir+f+'.jpg' )
            labels.append(l)

        shuffled_index = list(range(len(files)))

        random.seed(12345)

        random.shuffle(shuffled_index)

        shuffled__files = [files[i] for i in shuffled_index]

        shuffled_labels = [labels[i] for i in shuffled_index]

        return shuffled__files, shuffled_labels


def train():
    num_classes = cfg.FLAGS.num_classes
    batch_size = cfg.FLAGS.batch_size
    surfix = cfg.FLAGS.surfix
    num_train_images = 1803460

    # Create global_step
    with tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()

    """ load data """
    with tf.device('/cpu:0'):
        image, target_labels = datasets.get_dataset(surfix, is_training=True, im_batch=batch_size)

    labels_onehot = tf.one_hot(target_labels, depth=num_classes, on_value=1.0, off_value=0.0)

    nets = model.MobileNet_V2(image,
                           num_classes=num_classes,
                           is_training=True,
                           weight_decay=cfg.FLAGS.weight_decay)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    logits = nets.end_points['logits']
    print(target_labels.shape, labels_onehot.shape, logits.shape)
    tf.losses.softmax_cross_entropy(onehot_labels=labels_onehot,
                                    logits=logits, weights=1.0)

    """ compute total loss """
    all_losses = []
    cross_entropy = tf.get_collection(tf.GraphKeys.LOSSES)
    cross_loss = tf.add_n(cross_entropy, name='cross_loss')
    all_losses.append(cross_loss)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses, name='sum_regularization_loss')
    all_losses.append(regularization_loss)

    total_loss = tf.add_n(all_losses)

    """ Configure the optimization procedure. """
    with tf.device('/cpu:0'):
        learning_rate = _get_learning_rate(num_train_images, global_step)
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              decay=cfg.FLAGS.rmsprop_decay,
                                              momentum=cfg.FLAGS.rmsprop_momentum,
                                              epsilon=cfg.FLAGS.opt_epsilon)

    """ Variables to train """
    train_vars = tf.trainable_variables()
    grad_op = optimizer.minimize(total_loss, global_step=global_step, var_list=train_vars)
    update_ops.append(grad_op)
    update_op = tf.group(*update_ops)

    """ estimate Accurancy """
    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(target_labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    pred_cls = tf.cast(tf.argmax(nets.end_points['predictions'], axis=1), tf.int32)
    correct_predictio = tf.equal(pred_cls, target_labels)
    accurancy = tf.reduce_mean(tf.cast(correct_predictio, tf.float32))

    """ set Summary and log info """
    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    """ Add summaries for total loss """
    summaries.add(tf.summary.scalar('total_loss', total_loss))
    tf.summary.scalar('regularization_loss', regularization_loss)

    """ Add summaries for variables. """
    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))

    summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    logdir = os.path.join(cfg.FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.Session().graph)

    """ create saver and initialize variables """
    saver = tf.train.Saver(max_to_keep=20)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(cfg.FLAGS.latest_ckpt)
        """ resotre checkpoint of Backbone network """
        if ckpt is not None:
            lastest_ckpt = tf.train.latest_checkpoint(cfg.FLAGS.latest_ckpt)
            print('lastest', lastest_ckpt)
            re_saver = tf.train.Saver()
            re_saver.restore(sess, lastest_ckpt)
        else:
            variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MobilenetV2')
            re_saver = tf.train.Saver(var_list=variables_to_restore)
            re_saver.restore(sess, 'data/pretrained_models/mobilenet_v2_1.0_224.ckpt')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            print ('Training...')
            while not coord.should_stop():
                _, current_step, losses, acc, gt_lables, pred_labels = sess.run([update_op, global_step, total_loss, accurancy, target_labels, pred_cls])
                epoch = int(current_step / (num_train_images / cfg.FLAGS.batch_size))
                print(""" iter %d / %d: total_loss %.4f, accuracy %.4f """ %(current_step, epoch, losses, acc))

                """ write summary """
                if current_step % 500 == 0:
                    """ write summary """
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, current_step)
                    print('gt_lables', gt_lables)
                    print('pred_labels', pred_labels)


                """ save trained model datas """
                if current_step % 1000 == 0:
                    saver.save(sess, cfg.FLAGS.training_model, global_step=current_step)

                if current_step == cfg.FLAGS.max_iters:
                    print('step is reached the maximum iteration')
                    print('Done training!!!!!')
        except tf.errors.OutOfRangeError:
            print('Error is occured and stop the training')
        finally:
            saver.save(sess, cfg.FLAGS.checkpoint_model, write_meta_graph=False)
            coord.request_stop()

        coord.join(threads)

    print('Done.')

if __name__ == "__main__":
    train()
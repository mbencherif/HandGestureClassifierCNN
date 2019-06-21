import tensorflow as tf

from data import DataPipeline
from model import FuseModel

import argparse
import os
import sys
import cv2
import numpy as np
import time


def build_model(images_input, num_classes, batch_size):
    fused_model = FuseModel(images_input, output_size=num_classes, batch_size=batch_size)
    return fused_model.prediction


def get_loss(y_pred_logits, y_true):
    loss = tf.losses.softmax_cross_entropy(y_true, y_pred_logits, label_smoothing=0.01)
    return loss


def get_optimizer(loss, lr=0.001, opt='adam', **kwargs):
    optimizers = {
        'adadelta': tf.train.AdadeltaOptimizer,
        'adam': tf.train.AdamOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'gradient descent': tf.train.GradientDescentOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer
    }
    optimizer = optimizers[opt.lower()](learning_rate=lr, **kwargs)
    train_op = optimizer.minimize(loss, name='optimizer_update_op')
    return train_op


def get_accuracy(y_pred_logits, y_true, val=False):
    name_prefix = 'val_' if val else 'train_'
    acc, acc_update_op = tf.metrics.accuracy(tf.math.argmax(y_true, axis=-1), tf.math.argmax(y_pred_logits, axis=-1),
                                             name='{}accuracy'.format(name_prefix))
    initializer = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='{}accuracy'.format(name_prefix))
    initializer = tf.variables_initializer(var_list=initializer)
    return (acc, acc_update_op), initializer


def get_top_k_accuracy(y_pred_logits, y_true, k=5, val=False):
    name_prefix = 'val_' if val else 'train_'
    targets = tf.math.argmax(y_true, axis=-1)
    top_ks = tf.math.in_top_k(y_pred_logits, targets, k, name='{}in_top_{}'.format(name_prefix, k))
    acc, acc_update_op = tf.metrics.mean(top_ks, name='{}top_{}_accuracy'.format(name_prefix, k))
    initializer = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='{}top_{}_accuracy'.format(name_prefix, k))
    initializer = tf.variables_initializer(var_list=initializer)
    return (acc, acc_update_op), initializer


def correct_pred_val(y_pred_logits, y_true, val=False):
    name_prefix = 'val_' if val else 'train_'
    y_pred = tf.nn.softmax(y_pred_logits, axis=-1, name='{}y_pred_softmax'.format(name_prefix))
    diff = tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=-1, name='reduce_sum_diff')
    avg_max_diff, avg_max_diff_update = tf.metrics.mean(diff, name='{}avg_max_diff'.format(name_prefix))
    initializer = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='{}avg_max_diff'.format(name_prefix))
    initializer = tf.variables_initializer(var_list=initializer)
    return (avg_max_diff, avg_max_diff_update), initializer


def get_dataset(generator, batch_size=1, val=False):
    d = tf.data.Dataset.from_generator(generator,
                                       output_types=(tf.float32, tf.float32))
    if val:
        d = d.repeat()
    d = d.batch(batch_size)
    if val:
        iterator = d.make_one_shot_iterator()
        return iterator.get_next()
    else:
        iterator = d.make_initializable_iterator()
        return iterator.initializer, iterator.get_next()


def start_training(data_location='/Users/Yuhan', log_dir='log1', save_dir='saved_models', model_name=None,
                   steps_per_epoch=2000, val_steps=16, start_epoch=0, epochs=100, global_step=0,
                   summary_update_freq=10, val_freq=200, save_freq=200,
                   batch_size=1):
    if model_name is None:
        model_name = 'default_model'

    save_dir = os.path.join(model_name, save_dir)
    log_dir = os.path.join(model_name, log_dir)

    # Default: 144 x 256
    # [Rows, Columns]
    image_shape = [144, 256]

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    d = None
    dataset_init = None
    dataset_images = None
    dataset_label = None
    val_dataset_images = None
    val_dataset_label = None
    try:
        print('creating dataset pipeline')
        d = DataPipeline(sess, image_shape, home_path=data_location)
        num_classes = d.num_classes()
        dataset_init, (dataset_images, dataset_label) = get_dataset(d.generator, batch_size=batch_size)
        val_dataset_images, val_dataset_label = get_dataset(d.val_generator, batch_size=batch_size,
                                                            val=True)
        dataset_images.set_shape([batch_size, None] + image_shape + [3])
        val_dataset_images.set_shape([batch_size, None] + image_shape + [3])
    except FileNotFoundError:
        print("Training files not found. Executing under fake dataset")
        num_classes = 27

    print('creating placeholders')
    images_placeholder = tf.placeholder(tf.float32, shape=[batch_size, None] + image_shape + [3],
                                        name='images_placeholder')
    prediction_placeholder = tf.placeholder(tf.float32, shape=[batch_size, num_classes],
                                            name='prediction_placeholder')
    y_actual_placeholder = tf.placeholder(tf.float32, shape=[batch_size, num_classes],
                                          name='y_actual_placeholder')

    print('constructing fused model')
    model_output = build_model(images_placeholder, num_classes, batch_size)

    print('creating loss')
    loss_op = get_loss(model_output, y_actual_placeholder)

    print('creating optimizers')
    optimizer = get_optimizer(loss_op, lr=0.001, opt='adam')

    print('creating metrics')
    metric_calc_ops = list()
    metric_update_ops = list()
    metric_reset_ops = list()
    metric_names = list()

    metrics = {
        'accuracy': get_accuracy(prediction_placeholder, y_actual_placeholder),
        'top_2_accuracy': get_top_k_accuracy(prediction_placeholder, y_actual_placeholder, k=2),
        'top_5_accuracy': get_top_k_accuracy(prediction_placeholder, y_actual_placeholder, k=5),
        'correct_pred_val': correct_pred_val(prediction_placeholder, y_actual_placeholder)
    }

    for name, ((calc, update), reset) in metrics.items():
        metric_names.append(name)
        metric_calc_ops.append(calc)
        metric_update_ops.append(update)
        metric_reset_ops.append(reset)

    val_metric_calc_ops = list()
    val_metric_update_ops = list()
    val_metric_reset_ops = list()
    val_metric_names = list()

    val_metrics = {
        'accuracy': get_accuracy(prediction_placeholder, y_actual_placeholder, val=True),
        'top_2_accuracy': get_top_k_accuracy(prediction_placeholder, y_actual_placeholder, k=2, val=True),
        'top_5_accuracy': get_top_k_accuracy(prediction_placeholder, y_actual_placeholder, k=5, val=True),
        'correct_pred_val': correct_pred_val(prediction_placeholder, y_actual_placeholder, val=True)
    }

    for name, ((calc, update), reset) in val_metrics.items():
        val_metric_names.append(name)
        val_metric_calc_ops.append(calc)
        val_metric_update_ops.append(update)
        val_metric_reset_ops.append(reset)

    print('creating summary writers')
    summary_placeholder_dict = dict()
    loss_placeholder = tf.placeholder(tf.float32, shape=[],
                                      name='loss_placeholder')
    tf.summary.scalar(name='loss', tensor=loss_placeholder)
    summary_placeholder_dict[loss_placeholder] = 'loss'
    for name in metric_names:
        placeholder = tf.placeholder(tf.float32, shape=[],
                                     name='{}_placeholder'.format(name))
        tf.summary.scalar(name=name, tensor=placeholder)
        summary_placeholder_dict[placeholder] = name
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), graph=sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(log_dir, 'validation'))

    print('initializing all variables')
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    print('creating savers')
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2, reshape=True)
    last_save = tf.train.latest_checkpoint(save_dir)
    if last_save is None:
        print('found no previous save; will start fresh')
    else:
        last_component = last_save.split('/')[-1]
        last_global_step = last_component.split('-')[-1]
        try:
            global_step = int(last_global_step) + 1
        except ValueError:
            print("couldn't find last global step; will continue from step: {}".format(global_step))
        print('found previous save; will continue from global step: {}'.format(global_step))
        saver.restore(sess, last_save)
    save_location = os.path.join(save_dir, '{}.meta'.format(model_name))
    saver.export_meta_graph(save_location, clear_devices=True)

    if d is None:
        print("Cannot train because we don't have the dataset")
        exit(0)

    print('testing speed')
    num_frames = 0
    durr = 10.0**-9
    sess.run(dataset_init)
    images, label = sess.run([dataset_images, dataset_label])
    num_frames += images.shape[1] * images.shape[0]
    start_time = time.time()
    sess.run(model_output, feed_dict={images_placeholder: images})
    durr += time.time() - start_time
    print('processed {} frames in {} seconds; {:.01f} fps'.format(num_frames, durr, num_frames / durr))

    print('starting training')
    for epoch in range(start_epoch, start_epoch + epochs):
        sess.run([dataset_init, metric_reset_ops])
        for train_step in range(1, steps_per_epoch + 1):
            # fetch the next batch of features and labels
            images, label = sess.run([dataset_images, dataset_label])

            # run a single iteration of training, loss, and optimization
            _, loss, prediction = sess.run([optimizer, loss_op, model_output],
                                           feed_dict={images_placeholder: images,
                                                      y_actual_placeholder: label})

            # evaluate the metrics
            metrics = sess.run(metric_update_ops, feed_dict={prediction_placeholder: prediction,
                                                             y_actual_placeholder: label})
            metric_name_val = list(zip(metric_names, metrics))
            metric_string = '  '.join(map(lambda x: '{} - {:.3f}'.format(*x), metric_name_val))
            print('Epoch {} step {}/{}\tloss - {:.2f}, {}'.format(epoch, train_step, steps_per_epoch,
                                                                  loss, metric_string))

            if global_step > 0 and global_step % save_freq == 0:
                # save model
                print('saving model...')
                save_location = os.path.join(save_dir, model_name)
                saver.save(sess, save_path=save_location, global_step=global_step, write_meta_graph=False)

            if global_step == 0 or global_step % summary_update_freq == 0 and train_step > 0:
                # evaluate summaries
                summary_values = dict(metric_name_val + [('loss', loss)])
                feed_dict = dict()
                for placeholder, name in summary_placeholder_dict.items():
                    feed_dict[placeholder] = summary_values[name]
                summary = sess.run(summaries, feed_dict=feed_dict)
                train_writer.add_summary(summary, global_step=global_step)
                train_writer.flush()

            if global_step % val_freq == 0:
                print('running validation step')
                # evaluate our data on the validation set
                sess.run(val_metric_reset_ops)
                for val_step in range(val_steps):
                    # fetch the next batch of features and labels
                    images, label = sess.run([val_dataset_images, val_dataset_label])

                    # run a single iteration of training, loss, and optimization
                    loss, prediction = sess.run([loss_op, model_output],
                                                feed_dict={images_placeholder: images,
                                                           y_actual_placeholder: label})

                    # evaluate the metrics
                    metrics = sess.run(val_metric_update_ops, feed_dict={prediction_placeholder: prediction,
                                                                         y_actual_placeholder: label})

                metric_name_val = list(zip(metric_names, metrics))
                metric_string = '  '.join(map(lambda x: '{} - {:.4f}'.format(*x), metric_name_val))
                print('Validation: Epoch {} step {}/{}\tloss - {:.2f}, {}'.format(epoch, train_step, steps_per_epoch,
                                                                                  loss, metric_string))

                # evaluate val summaries
                summary_values = dict(metric_name_val + [('loss', loss)])
                feed_dict = dict()
                for placeholder, name in summary_placeholder_dict.items():
                    feed_dict[placeholder] = summary_values[name]
                summary = sess.run(summaries, feed_dict=feed_dict)
                val_writer.add_summary(summary, global_step=global_step)
                val_writer.flush()

            global_step += 1


if __name__ == '__main__':
    print(sys.path)
    for path in sys.path:
        if '/Users/Yuhan' in path:
            base_location = '/Users/Yuhan'
            break
    else:
        base_location = '/home/yliu102199'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', dest='batch_size', type=int,
                        default=8)
    results = parser.parse_args()
    print("Batch size set to: {}".format(results.batch_size))
    start_training(data_location=base_location, log_dir='log', save_dir='saved_models', model_name='model_1',
                   steps_per_epoch=20000, val_steps=32, start_epoch=0, epochs=1000, global_step=0,
                   summary_update_freq=30, val_freq=200, save_freq=500,
                   batch_size=results.batch_size)

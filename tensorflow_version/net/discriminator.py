import tensorflow as tf
from net.ops import dis_spectralconv, flatten
# from net.loss import *
from util.util import f2uint
from functools import partial, reduce

conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')

def wgan_patch_discriminator(x, mask, d_cnum, reuse=False):
    cnum = d_cnum
    with tf.variable_scope('discriminator_local', reuse=reuse):
        h, w = mask.get_shape().as_list()[1:3]
        x = conv5_ds(x, filters=cnum, name='conv1')
        x = conv5_ds(x, filters=cnum * 2, name='conv2')
        x = conv5_ds(x, filters=cnum * 4, name='conv3')
        x = conv5_ds(x, filters=cnum * 8, name='conv4')
        x = tf.layers.conv2d(x, kernel_size=5, strides=2, filters=1, activation=None, name='conv5', padding='SAME')

        mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
        mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
        mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
        mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
        mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')

        x = x * mask
        x = tf.reduce_sum(x, axis=[1, 2, 3]) / tf.reduce_sum(mask, axis=[1, 2, 3])
        mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True)
        return x, mask_local


def wgan_local_discriminator(x, d_cnum, reuse=False):
    cnum = d_cnum
    with tf.variable_scope('disc_local', reuse=reuse):
        x = conv5_ds(x, filters=cnum, name='conv1')
        x = conv5_ds(x, filters=cnum * 2, name='conv2')
        x = conv5_ds(x, filters=cnum * 4, name='conv3')
        x = conv5_ds(x, filters=cnum * 8, name='conv4')
        x = conv5_ds(x, filters=cnum * 4, name='conv5')
        x = conv5_ds(x, filters=cnum * 2, name='conv6')

        x = tf.layers.flatten(x, name='flatten')
        return x


def wgan_global_discriminator(x, d_cnum, reuse=False):
    cnum = d_cnum
    with tf.variable_scope('disc_global', reuse=reuse):
        x = conv5_ds(x, filters=cnum, name='conv1')
        x = conv5_ds(x, filters=cnum * 2, name='conv2')
        x = conv5_ds(x, filters=cnum * 4, name='conv3')
        x = conv5_ds(x, filters=cnum * 4, name='conv4')
        x = tf.layers.flatten(x, name='flatten')
        return x


def wgan_discriminator(batch_local, batch_global, d_cnum, mask=None, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        dlocal = wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
        dglobal = wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
        dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
        dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
        return dout_local, dout_global


def wgan_mask_discriminator(batch_global, mask, d_cnum, batch_local=None, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        dglobal = wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
        dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
        dout_local, mask_local = wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
    return dout_local, dout_global


def build_sn_patch_gan_discriminator(x, reuse=False, training=True):
    with tf.variable_scope('sn_patch_gan', reuse=reuse):
        cnum = 64
        x = dis_spectralconv(x, cnum, name='conv1', training=training)
        x = dis_spectralconv(x, cnum*2, name='conv2', training=training)
        x = dis_spectralconv(x, cnum*4, name='conv3', training=training)
        x = dis_spectralconv(x, cnum*4, name='conv4', training=training)
        x = dis_spectralconv(x, cnum*4, name='conv5', training=training)
        x = dis_spectralconv(x, cnum*4, name='conv6', training=training)
        x = flatten(x, name='flatten')
        return x


def build_gan_discriminator(
        batch, reuse=False, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        d = build_sn_patch_gan_discriminator(
            batch, reuse=reuse, training=training)
        return d

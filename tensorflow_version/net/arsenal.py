import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from functools import partial

epslion = 1e-2


# mask: 1 for unknown and 0 for known
def context_normalization(x, mask, alpha=0.5, eps=1e-5):
    h, w = x.get_shape().as_list()[1:3]
    mask_s = tf.image.resize_nearest_neighbor(1 - mask[:, :, :, 0:1], [h, w])
    x_known_cnt = tf.maximum(eps, tf.reduce_sum(mask_s, [1, 2], keep_dims=True))
    x_known_mean = tf.reduce_sum(x * mask_s, [1, 2], keep_dims=True) / x_known_cnt
    x_known_variance = tf.reduce_sum((x * mask_s - x_known_mean) ** 2, [1, 2], keep_dims=True) / x_known_cnt

    mask_s_rev = 1 - mask_s
    x_unknown_cnt = tf.maximum(eps, tf.reduce_sum(mask_s_rev, [1, 2], keep_dims=True))
    x_unknown_mean = tf.reduce_sum(x * mask_s_rev, [1, 2], keep_dims=True) / x_unknown_cnt
    x_unknown_variance = tf.reduce_sum((x * mask_s_rev - x_unknown_mean) ** 2, [1, 2],
                                       keep_dims=True) / x_unknown_cnt
    x_unknown = alpha * tf.nn.batch_normalization(x * mask_s_rev, x_unknown_mean, x_unknown_variance, x_known_mean,
                                                  tf.sqrt(x_known_variance), eps) + (1 - alpha) * x * mask_s_rev
    x = x_unknown * mask_s_rev + x * mask_s
    return x


def squeeze_and_excitation(x, ratio=8, name='se', reuse=False):
    h, w, ch = x.get_shape().as_list()[1:]
    with tf.variable_scope(name, reuse=reuse):
        x_bar = tf.squeeze(tf.nn.avg_pool2d(x, (h, w), 1))
        alpha = tf.layers.fully_connected(x_bar, num_outputs=ch // ratio, activation_fn=tf.nn.relu, reuse=reuse,
                                          name='fc1')
        alpha = tf.layers.fully_connected(alpha, num_outputs=ch, activation=tf.nn.sigmoid, reuse=reuse,
                                          name='fc2')
        alpha = tf.expand_dims(alpha, axis=[1, 2])
    return alpha


def context_normalization_se(x, mask, ratio=8, eps=1e-5, reuse=False, name='cn'):
    h, w, ch = x.get_shape().as_list()[1:]
    mask_s = tf.image.resize_nearest_neighbor(1 - mask[:, :, :, 0:1], [h, w])
    x_known_cnt = tf.maximum(eps, tf.reduce_sum(mask_s, [1, 2], keep_dims=True))
    x_known_mean = tf.reduce_sum(x * mask_s, [1, 2], keep_dims=True) / x_known_cnt
    x_known_variance = tf.reduce_sum((x * mask_s - x_known_mean) ** 2, [1, 2], keep_dims=True) / x_known_cnt

    mask_s_rev = 1 - mask_s
    x_unknown_cnt = tf.maximum(eps, tf.reduce_sum(mask_s_rev, [1, 2], keep_dims=True))
    x_unknown_mean = tf.reduce_sum(x * mask_s_rev, [1, 2], keep_dims=True) / x_unknown_cnt
    x_unknown_variance = tf.reduce_sum((x * mask_s_rev - x_unknown_mean) ** 2, [1, 2],
                                       keep_dims=True) / x_unknown_cnt

    alpha = squeeze_and_excitation(x, ratio, name+'_se', reuse)

    x_unknown = alpha * tf.nn.batch_normalization(x * mask_s_rev, x_unknown_mean, x_unknown_variance, x_known_mean,
                                                  tf.sqrt(x_known_variance), eps) + (1 - alpha) * x * mask_s_rev
    x = x_unknown * mask_s_rev + x * mask_s
    return x


@add_arg_scope
def context_resblock(x, mask=None, cnum=32, ksize=3, stride=1, rate=1, padding='SAME', activation=tf.nn.elu,
                     reuse=False, name='crb', debug='old', alpha=0.5):
    with tf.variable_scope(name, reuse=reuse):
        if debug == 'old':
            x_b = tf.layers.conv2d(x, cnum, ksize, strides=stride, dilation_rate=rate,
                                   activation=activation, padding=padding, name='c1', reuse=reuse)
            x_b = tf.layers.conv2d(x_b, cnum, ksize, strides=stride, dilation_rate=rate,
                                   activation=None, padding=padding, name='c2', reuse=reuse)
            if mask is not None:
                x_b = context_normalization(x_b, mask, alpha=alpha)
            x_b = activation(x_b)
            x = x_b
        else:
            x_b = tf.layers.conv2d(x, cnum, ksize, strides=stride, dilation_rate=rate,
                                   activation=None, padding=padding, name='c1', reuse=reuse)
            # todo: remove one cn, it is quite time consuming.
            # if mask is not None:
            #     x_b = context_normalization(x_b, mask)
            x_b = activation(x_b)
            x_b = tf.layers.conv2d(x_b, cnum, ksize, strides=1, dilation_rate=rate,
                                   activation=None, padding=padding, name='c2', reuse=reuse)
            if mask is not None:
                if debug == 'se':
                    x_b = context_normalization_se(x_b, mask, reuse=reuse)
                else:
                    x_b = context_normalization(x_b, mask, alpha=alpha)
            x = tf.layers.conv2d(x, cnum, 1, strides=stride, dilation_rate=1,
                                 activation=None, padding=padding, name='c3', reuse=reuse)
            x = x + x_b
            x = activation(x)
    return x


@add_arg_scope
def resblock(x, cnum=32, ksize=3, stride=1, rate=1, padding='SAME', activation=tf.nn.elu,
             reuse=False, name='rb'):
    with tf.variable_scope(name, reuse=reuse):
        x_b = tf.layers.conv2d(x, cnum, ksize, strides=stride, dilation_rate=rate,
                               activation=None, padding=padding, name='c1', reuse=reuse)
        x_b = activation(x_b)
        x_b = tf.layers.conv2d(x_b, cnum, ksize, strides=1, dilation_rate=rate,
                               activation=None, padding=padding, name='c2', reuse=reuse)
        x = tf.layers.conv2d(x, cnum, 1, strides=stride, dilation_rate=1,
                             activation=None, padding=padding, name='c3', reuse=reuse)
        x = x + x_b
        x = activation(x)
    return x

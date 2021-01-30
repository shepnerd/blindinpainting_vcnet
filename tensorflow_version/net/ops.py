import cv2
import numpy as np
import tensorflow as tf
from logging import exception
import math
import scipy.stats as st
import os
import urllib
import scipy
from scipy import io
from functools import partial
from tensorflow.contrib.framework.python.ops import add_arg_scope

np.random.seed(2018)

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


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v**2)**0.5+eps)


def spectral_normed_weight(weights, num_iters=1, update_collection=None, with_sigma=False):
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])
    u = tf.get_variable('u', [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_ = u
    v_ = None
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))
    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


def snconv2d(x, output_dim, kh=3, kw=3, dh=2, dw=2, sn_iters=1, update_collection=None, name='snconv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kh, kw, x.get_shape().as_list()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
        conv = tf.nn.conv2d(x, w_bar, strides=[1, dh, dw, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, biases)
        return conv


def sndeconv2d(x, output_shape, kh=3, kw=3, dh=2, dw=2, stddev=0.02, init_bias=0,
               sn_iters=1, update_collection=None, name='sndeconv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kh, kw, x.get_shape().as_list()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
        deconv = tf.nn.conv2d_transpose(x, w_bar, output_shape=output_shape, strides=[1, dh, dw, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(init_bias))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv.shape.assert_is_compatible_with(output_shape)
        return deconv


def snlinear(x, output_dim, bias_start=0.0, sn_iters=1, update_collection=None, name='snlinear'):
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        matrix = tf.get_variable('matrix', [shape[1], output_dim], tf.float32, tf.contrib.layers.xavier_initializer())
        matrix_bar = spectral_normed_weight(matrix, num_iters=sn_iters, update_collection=update_collection)
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(bias_start))
        out = tf.matmul(x, matrix_bar) + bias
        return out


"""
https://arxiv.org/abs/1806.03589
"""
def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def free_form_mask_tf(parts, maxVertex=16, maxLength=60, maxBrushWidth=14, maxAngle=360, im_size=(256, 256), name='fmask'):
    # mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    with tf.variable_scope(name):
        mask = tf.Variable(tf.zeros([1, im_size[0], im_size[1], 1]), name='free_mask')
        maxVertex = tf.constant(maxVertex, dtype=tf.int32)
        maxLength = tf.constant(maxLength, dtype=tf.int32)
        maxBrushWidth = tf.constant(maxBrushWidth, dtype=tf.int32)
        maxAngle = tf.constant(maxAngle, dtype=tf.int32)
        h = tf.constant(im_size[0], dtype=tf.int32)
        w = tf.constant(im_size[1], dtype=tf.int32)
        for i in range(parts):
            p = tf.py_func(np_free_form_mask, [maxVertex, maxLength, maxBrushWidth, maxAngle, h, w], tf.float32)
            p = tf.reshape(p, [1, im_size[0], im_size[1], 1])
            mask = mask + p
        mask = tf.minimum(mask, 1.0)
    return mask


def gauss_kernel(size=21, sigma=3):
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2, sigma+interval/2, size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((size, size, 1, 1))
    return out_filter


def tf_make_guass_var(size, sigma):
    kernel = gauss_kernel(size, sigma)
    var = tf.Variable(tf.convert_to_tensor(kernel))
    return var


def priority_loss_mask(mask, hsize=64, sigma=1/40, iters=7):
    kernel = tf_make_guass_var(hsize, sigma)
    init = 1-mask
    mask_priority = tf.ones_like(mask)
    for i in range(iters):
        mask_priority = tf.nn.conv2d(init, kernel, strides=[1,1,1,1], padding='SAME')
        mask_priority = mask_priority * mask
        init = mask_priority + (1-mask)
    return mask_priority


def priority_loss_mask_cp(mask, hsize=64, sigma=1.0/40, iters=9):
    eps = 1e-5
    kernel = tf_make_guass_var(hsize, sigma)
    init = 1-mask
    mask_priority = None
    mask_priority_pre = None
    for i in range(iters):
        mask_priority = tf.nn.conv2d(init, kernel, strides=[1,1,1,1], padding='SAME')
        mask_priority = mask_priority * mask
        if i == iters-2:
            mask_priority_pre = mask_priority
        init = mask_priority + (1-mask)
    mask_priority = mask_priority_pre / (mask_priority+eps)
    return mask_priority


def random_interpolates(x, y, alpha=None):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def random_bbox(config):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.img_shapes
    img_height = img_shape[0]
    img_width = img_shape[1]
    if config.random_mask is True:
        maxt = img_height - config.margins[0] - config.mask_shapes[0]
        maxl = img_width - config.margins[1] - config.mask_shapes[1]
        t = tf.random_uniform(
            [], minval=config.margins[0], maxval=maxt, dtype=tf.int32)
        l = tf.random_uniform(
            [], minval=config.margins[1], maxval=maxl, dtype=tf.int32)
    else:
        t = config.mask_shapes[0]//2
        l = config.mask_shapes[1]//2
    h = tf.constant(config.mask_shapes[0])
    w = tf.constant(config.mask_shapes[1])
    return (t, l, h, w)


def bbox2mask(bbox, config, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.img_shapes
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             config.max_delta_shapes[0], config.max_delta_shapes[1]],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def local_patch(x, bbox):
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def gradients_penalty(x, y, mask=None, norm=1.):
    """Improved Training of Wasserstein GANs

    - https://arxiv.org/abs/1704.00028
    """
    gradients = tf.gradients(y, x)[0]
    
    if mask is None:
        mask = tf.ones_like(gradients)
    
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))


def gan_wgan_loss(pos, neg, name='gan_wgan_loss'):
    """
    wgan loss function for GANs.

    - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    """
    with tf.variable_scope(name):
        d_loss = tf.reduce_mean(neg-pos)
        g_loss = -tf.reduce_mean(neg)
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('pos_value_avg', tf.reduce_mean(pos))
        tf.summary.scalar('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss


def average_gradients(tower_grads):
    """ Calculate the average gradient for each shared variable across
    all towers.

    **Note** that this function provides a synchronization point
    across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples.
            The outer list is over individual gradients. The inner list is
            over the gradient calculation for each tower.

    Returns:
        List of pairs of (gradient, variable) where the gradient
            has been averaged across all towers.

    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        v = grad_and_vars[0][1]
        # sum
        grad = tf.add_n([x[0] for x in grad_and_vars])
        # average
        grad = grad / float(len(tower_grads))
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


"""
Partial Convolution
"""
@add_arg_scope
def sparse_invariant_conv(x, mask, cnum, ksize, stride=1, rate=1, name='sparse_invariant_conv',
                 padding='SAME', activation=None, reuse=False):
    '''

    :param x_mask: (x, mask). In mask, 1 indicates valid pixel while 0 means unknown pixel
    :param cnum:
    :param ksize:
    :param stride:
    :param rate:
    :param name:
    :param padding:
    :param activation:
    :param reuse:
    :return:
    '''
    mask_ch = mask.get_shape().as_list()[-1]
    eplison = 1e-5
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate * (ksize - 1) / 2)
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
        padding = 'VALID'

    sum_kernel_np = np.ones((ksize, ksize, mask_ch, mask_ch), dtype=np.float32)
    sum_kernel = tf.Variable(tf.convert_to_tensor(sum_kernel_np))

    x = x * mask / (tf.nn.conv2d(mask, sum_kernel, stride, padding) + eplison)

    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name, reuse=reuse)
    mask = tf.layers.max_pooling2d(mask, ksize, stride, padding=padding)
    return x, mask


@add_arg_scope
def partial_conv(x, mask, cnum, ksize, stride=1, rate=1, name='partial_conv',
                 padding='SAME', activation=None, reuse=False):
    mask_ch = mask.get_shape().as_list()[-1]
    x_ch = x.get_shape().as_list()[-1]
    assert mask_ch == x_ch
    eplison = 1e-5
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate * (ksize - 1) / 2)
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
        padding = 'VALID'

    x = x * mask
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name, reuse=reuse, use_bias=False)
    out_ch = x.get_shape().as_list()[-1]
    mask = tf.tile(mask[:, :, :, 0:1], [1, 1, 1, out_ch])
    ones = tf.ones_like(mask, dtype=tf.float32)

    sum_kernel_np = np.ones((ksize, ksize, out_ch, out_ch), dtype=np.float32)
    sum_kernel = tf.Variable(tf.convert_to_tensor(sum_kernel_np))

    coef = tf.nn.conv2d(ones, sum_kernel,
                        [1, stride, stride, 1], padding) / (tf.nn.conv2d(mask,
                                                                         sum_kernel,
                                                                         [1, stride, stride, 1],
                                                                         padding) + eplison)
    biases = tf.get_variable('{}/biases'.format(name), [out_ch], initializer=tf.zeros_initializer())
    x = tf.nn.bias_add(x * coef, biases)
    mask = tf.layers.max_pooling2d(mask, ksize, (stride, stride), padding=padding)
    return x, mask


def resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
           func=tf.image.resize_bilinear, name='resize'):
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x


def random_interpolates(x, y, alpha=None):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize


@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True, reuse=False):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name, reuse=reuse)
    return x


@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True, reuse=False):
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training, reuse=reuse)
    return x


@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True):
    x = tf.layers.conv2d(x, cnum, ksize, stride, 'SAME', name=name)
    x = tf.nn.leaky_relu(x)
    return x


@add_arg_scope
def gen_gatedconv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)
    if cnum == 3 or activation is None:
        # conv for output
        return x
    x, y = tf.split(x, 2, 3) # value, num_or_size_splits, axis
    x = activation(x)
    y = tf.nn.sigmoid(y)
    x = x * y
    return x


@add_arg_scope
def gen_degatedconv(x, cnum, name='upsample', padding='SAME', training=True):
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


def kernel_spectral_norm(kernel, iteration=1, name='kernel_sn'):
    # spectral_norm
    def l2_norm(input_x, epsilon=1e-12):
        input_x_norm = input_x / (tf.reduce_sum(input_x**2)**0.5 + epsilon)
        return input_x_norm
    with tf.variable_scope(name) as scope:
        w_shape = kernel.get_shape().as_list()
        w_mat = tf.reshape(kernel, [-1, w_shape[-1]])
        u = tf.get_variable(
            'u', shape=[1, w_shape[-1]],
            initializer=tf.truncated_normal_initializer(),
            trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite+1

        u_hat, v_hat,_ = power_iteration(u, iteration)
        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
        w_mat = w_mat / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


class Conv2DSepctralNorm(tf.layers.Conv2D):
    def build(self, input_shape):
        super(Conv2DSepctralNorm, self).build(input_shape) # is it a bug or just complier problem.
        self.kernel = kernel_spectral_norm(self.kernel)


def conv2d_spectral_norm(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    layer = Conv2DSepctralNorm(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs)


@add_arg_scope
def dis_spectralconv(x, cnum, ksize=5, stride=2, name='conv', training=True):
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    x = conv2d_spectral_norm(x, cnum, ksize, stride, 'SAME', name=name)
    x = tf.nn.leaky_relu(x)
    return x


def gan_hinge_loss(pos, neg, value=1., name='gan_hinge_loss'):
    """
    gan with hinge loss:
    https://github.com/pfnet-research/sngan_projection/blob/c26cedf7384c9776bcbe5764cb5ca5376e762007/updater.py
    """
    with tf.variable_scope(name):
        hinge_pos = tf.reduce_mean(tf.nn.relu(1-pos))
        hinge_neg = tf.reduce_mean(tf.nn.relu(1+neg))
        d_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)
        g_loss = -tf.reduce_mean(neg)
    return g_loss, d_loss


def flatten(x, name='flatten'):
    """Flatten wrapper.

    """
    with tf.variable_scope(name):
        return tf.contrib.layers.flatten(x)


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img

"""
contextual attention layer
"""
def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse: # fs[1]: 32, fs[2]: 32, bs[1]: 64, bs[2]: 64
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
    return y, flow
import numpy as np
import cv2
import math
import time
import shutil
import os
import re
import tensorflow as tf
from net.ops import np_free_form_mask

def f2uint(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value((x+1)*127.5, 0, 255), tf.uint8)
    else:
        return np.clip((x+1)*127.5, 0, 255).astype(np.uint8)


def generate_mask_rect(im_shapes, mask_shapes, rand=True):
    mask = np.zeros((im_shapes[0], im_shapes[1])).astype(np.float32)
    if rand:
        of0 = np.random.randint(0, im_shapes[0]-mask_shapes[0])
        of1 = np.random.randint(0, im_shapes[1]-mask_shapes[1])
    else:
        of0 = (im_shapes[0]-mask_shapes[0])//2
        of1 = (im_shapes[1]-mask_shapes[1])//2
    mask[of0:of0+mask_shapes[0], of1:of1+mask_shapes[1]] = 1
    mask = np.expand_dims(mask, axis=2)
    return mask


def generate_mask_stroke(im_size, parts=16, maxVertex=24, maxLength=100, maxBrushWidth=24, maxAngle=360):
    h, w = im_size[:2]
    mask = np.zeros((h, w, 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w)
    mask = np.minimum(mask, 1.0)
    return mask


# import matplotlib
# import matplotlib.cm
#
#
# def colorize_im(value, vmin=None, vmax=None, cmap=None):
#     # normalize
#     vmin = tf.reduce_min(value) if vmin is None else vmin
#     vmax = tf.reduce_max(value) if vmax is None else vmin
#     value = (value - vmin) / (vmax - vmin)
#
#     # squeeze last dim if it exists
#     value = tf.squeeze(value)
#
#     # quantize
#     indices = tf.to_int32(tf.round(value * 255))
#
#     # gather
#     cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'jet')
#     colors = cm(np.arange(256))[:, :3]
#     colors = tf.constant(colors, dtype=tf.float32)
#     value = tf.gather(colors, indices)
#
#     return value
#
#
# def colorize_tensor(tensor, vmin=None, vmax=None, cmap=None):
#     # slice the tensor along the batch dimension
#     b = tensor.get_shape().as_list()[0]
#     imgs = tf.split(tensor, b, 0)
#     imgs_colorized = list(map(lambda x: colorize_im(x[0], vmin=vmin, vmax=vmax, cmap=cmap), imgs))
#     imgs_colorized = list(map(lambda x: tf.expand_dims(x, 0), imgs_colorized))
#     return tf.concat(imgs_colorized, 0)

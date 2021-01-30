import numpy as np
import cv2
import os
import subprocess
import glob
import tensorflow as tf
from options.test_options import TestOptions
from net.model import Blindinpaint_model
from util.util import generate_mask_rect, generate_mask_stroke

import numpy as np
import time

os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
        ))

def crop_image(image, shapes):
    h, w = image.shape[:2]
    if h >= shapes[0] and w >= shapes[1]:
        h_start = (h - shapes[0]) // 2
        w_start = (w - shapes[1]) // 2
        image = image[h_start: h_start + shapes[0], w_start: w_start + shapes[1], :]
    else:
        t = min(h, w)
        image = image[(h - t) // 2:(h - t) // 2 + t, (w - t) // 2:(w - t) // 2 + t, :]
        image = cv2.resize(image, (shapes[1], shapes[0]))
    return image

def vis(x):
    return tf.cast(tf.clip_by_value((x + 1)*127.5, 0, 255), tf.uint8)

def model_capacity(net_vars):
    total_parameters = 0
    for variable in net_vars:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

config = TestOptions().parse()

if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)
total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

if os.path.isfile(config.data_noise):
    noise_pathfile = open(config.data_noise, 'rt').read().splitlines()
elif os.path.isdir(config.data_noise):
    noise_pathfile = glob.glob(os.path.join(config.data_noise, '*.png'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)
np.random.shuffle(noise_pathfile)
if len(noise_pathfile) >= test_num:
    noise_pathfile = noise_pathfile[:test_num]
else:
    times = test_num // len(noise_pathfile) + 1
    noise_pathfile = noise_pathfile * times
    noise_pathfile = noise_pathfile[:test_num]

noise_total_number = len(noise_pathfile)

model = Blindinpaint_model(config)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False

with tf.Session(config=sess_config) as sess:

    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    input_noise_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

    output, complete, mask_pred, logit, input = model.evaluate(input_image_tf, input_noise_tf, input_mask_tf)

    output, input = vis(output), vis(input)
    if complete is not None:
        complete = vis(complete)
    if mask_pred is not None:
        mask_pred = tf.cast(tf.clip_by_value(mask_pred * 255, 0, 255), tf.uint8)
    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                          vars_list))
    sess.run(assign_ops)
    g_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
    total_parameters = model_capacity(g_vars)
    print('model capacity: {}'.format(total_parameters))
    
    print('Model loaded.')
    total_time = 0

    if config.random_mask:
        np.random.seed(config.seed)
    time_total = 0
    for i in range(test_num):
        if config.mask_type == 'rect':
            mask = generate_mask_rect(config.img_shapes, config.mask_shapes, config.random_mask)
        else:
            mask = generate_mask_stroke(im_size=(config.img_shapes[0], config.img_shapes[1]),
                                        parts=8, maxBrushWidth=20, maxLength=80, maxVertex=16)

        if config.use_noise == 0:
            mask = np.zeros((config.img_shapes[0], config.img_shapes[1], 1))
            config.use_blend = 0

        if config.save_intermediate is True:
            cv2.imwrite(os.path.join(config.saving_path, 'mask_{:03d}.png'.format(i)),
                        np.tile(mask*255, [1, 1, 3]).astype(np.uint8))

        image = cv2.imread(pathfile[i])[:, :, ::-1] # rgb
        image = crop_image(image, config.img_shapes)

        if config.save_intermediate is True:
            cv2.imwrite(os.path.join(config.saving_path, 'gt_{:03d}.png'.format(i)), image[:, :, ::-1].astype(np.uint8))

        noise = cv2.imread(noise_pathfile[i])[:, :, ::-1] # rgb
        noise = crop_image(noise, config.img_shapes)

        if config.save_intermediate is True:
            cv2.imwrite(os.path.join(config.saving_path, 'noise_{:03d}.png'.format(i)), noise[:, :, ::-1].astype(np.uint8))

        assert image.shape[:2] == mask.shape[:2]

        h, w = image.shape[:2]
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        noise = noise[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        noise = np.expand_dims(noise, 0)
        mask = np.expand_dims(mask, 0)
        ret_mask = None

        print(config.model)

        if config.model == 'vcn':
            start_t = time.time()
            ret_pred, ret_complete, ret_mask, ret_logit, ret_input = sess.run([output, complete, mask_pred, logit, input],
                                                                              feed_dict={input_image_tf: image,
                                                                                         input_noise_tf: noise,
                                                                                         input_mask_tf: mask})
            end_t = time.time()
        else:

            if config.model == 'ca' or config.model == 'gmcnn' or config.model == 'ed':
                start_t = time.time()
                ret_pred, ret_complete, ret_input = sess.run([output, complete, input],
                                                            feed_dict={input_image_tf: image,
                                                                        input_noise_tf: noise,
                                                                        input_mask_tf: mask})
                end_t = time.time()
        
        total_time += end_t - start_t
        cv2.imwrite(os.path.join(config.saving_path, 'pred_{:03d}.png'.format(i)), ret_pred[0][:, :, ::-1])

        if ret_mask is not None:
            cv2.imwrite(os.path.join(config.saving_path, 'md_{:03d}.png'.format(i)), ret_mask[0][:, :, ::-1])

        if config.save_intermediate is True:
            cv2.imwrite(os.path.join(config.saving_path, 'bled_{:03d}.png'.format(i)), ret_complete[0][:, :, ::-1])
            cv2.imwrite(os.path.join(config.saving_path, 'feed_{:03d}.png'.format(i)), ret_input[0][:, :, ::-1])
        print(' > {} / {}'.format(i+1, test_num))

print('total time >{}, avg > {}'.format(total_time, total_time / test_num))
print('done.')

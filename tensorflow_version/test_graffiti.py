import numpy as np
import cv2
import os
import subprocess
import glob
import tensorflow as tf
from options.test_options import TestOptions
from util.util import generate_mask_rect, generate_mask_stroke
from net.model import Blindinpaint_model
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
        ))
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def vis(x):
    return tf.cast(tf.clip_by_value((x + 1)*127.5, 0, 255), tf.uint8)

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


config = TestOptions().parse()
print(config.dataset_path)
if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
    t = []
    for x in pathfile:
        if '001.png' in x or '002.png' in x or '003.png' in x or '004.png' in x:
            t.append(x)
    pathfile = t
else:
    print('Invalid testing data file/folder path.')
    exit(1)
np.random.shuffle(pathfile)
total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

# add face
if os.path.isfile(config.data_noise):
    noise_pathfile = open(config.data_noise, 'rt').read().splitlines()
elif os.path.isdir(config.data_noise):
    noise_pathfile = glob.glob(os.path.join(config.data_noise, 'mask_*.png'))
else:
    print('Invalid testing data file/folder path.')
np.random.shuffle(noise_pathfile)
if len(noise_pathfile) >= test_num:
    noise_pathfile = noise_pathfile[:test_num]
else:
    times = test_num // len(noise_pathfile) + 1
    noise_pathfile = noise_pathfile * times
    noise_pathfile = noise_pathfile[:test_num]

noise_total_number = len(noise_pathfile)

model = Blindinpaint_model(config)

reuse = False
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False
mask_c = 8
with tf.Session(config=sess_config) as sess:
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    input_noise_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

    output, complete, _, logit, input = model.evaluate(input_image_tf, input_noise_tf, input_mask_tf)

    output, complete, input = vis(output), vis(complete), vis(input)
    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                          vars_list))
    sess.run(assign_ops)
    print('Model loaded.')
    total_time = 0

    if config.random_mask:
        np.random.seed(config.seed)

    bce_arry = np.zeros((test_num, 1))
    mask_arry = [None] * mask_c
    for i in range(test_num):
        print(pathfile[i])
        image = cv2.imread(pathfile[i])[:, :, ::-1] # rgb
        image = crop_image(image, config.img_shapes)

        if config.save_intermediate is True:
            cv2.imwrite(os.path.join(config.saving_path, 'gt_{:03d}.png'.format(i)), image[:, :, ::-1].astype(np.uint8))
        for j in range(mask_c):
            noise = cv2.imread('./imgs/masks/mask_{}.png'.format(j+1))[:, :, ::-1] # rgb
            noise = crop_image(noise, config.img_shapes)
            if mask_arry[j] is None:
                mask = np.ones((256, 256)) * 1
                idx1 = noise[:, :, 0] == 0
                
                idx2 = noise[:, :, 1] == 0
                
                idx3 = noise[:, :, 2] == 0
                idx = idx1 == idx2
                idx = idx == idx3
                
                mask[idx] = 0
                mask = np.expand_dims(mask, -1)
                mask_arry[j] = mask
            else:
                mask = mask_arry[j]

            noise = np.expand_dims(noise, 0)
            if len(mask.shape) < 4:
                mask = np.expand_dims(mask, 0)

            if j == 0:

                image = np.expand_dims(image, 0)

            if config.model == 'ca' or config.model == 'gmcnn':
                ret_pred, ret_complete, ret_input = sess.run([output, complete, input],
                                                             feed_dict={input_image_tf: image,
                                                                        input_noise_tf: noise,
                                                                        input_mask_tf: mask})
            else:
                ret_pred, ret_complete, ret_logit, ret_input = sess.run([output, complete, logit, input],
                                                                        feed_dict={input_image_tf: image,
                                                                                   input_noise_tf: noise,
                                                                                   input_mask_tf: mask})
                cv2.imwrite(os.path.join(config.saving_path, 'pred_{:03d}_{}.png'.format(i, j)), ret_pred[0][:, :, ::-1])
                if config.save_intermediate is True:
                    cv2.imwrite(os.path.join(config.saving_path, 'feed_{:03d}_{}.png'.format(i, j)), ret_input[0][:, :, ::-1])
            print('pred')
        print(' > {} / {}, bce: {}'.format(i+1, test_num, ret_logit))
        bce_arry[i] = ret_logit
print('bce > mean: {}, std: {}'.format(bce_arry.mean(), bce_arry.std()))
print('done.')

import tensorflow as tf
import numpy as np
import os
import glob


class DataLoader:
    def __init__(self, file_in, file_out, im_size, batch_size, file_aux=None, random_crop=False, paired=False):
        getFileList = lambda f: open(f, 'rt').read().splitlines() \
            if os.path.isfile(f) else glob.glob(os.path.join(f, '*.png')) + glob.glob(os.path.join(f, '*.jpg'))

        self.file_in = getFileList(file_in)
        self.file_out = getFileList(file_out)

        self.paired = paired
        if paired is False:
            np.random.shuffle(self.file_out)
        if len(self.file_out) > len(self.file_in):
            self.file_out = self.file_out[:len(self.file_in)]
        else:
            t = len(self.file_in) // len(self.file_out) + 1
            self.file_out = self.file_out * t
            self.file_out = self.file_out[:len(self.file_in)]
        if file_aux is not None:
            self.file_aux = getFileList(file_aux)
            np.random.shuffle(self.file_aux)
            if len(self.file_aux) > len(self.file_in) // 2:
                self.file_out = self.file_out[:len(self.file_in) // 2]
            else:
                self.file_out = self.file_out[:len(self.file_in)-len(self.file_aux)]
            self.file_out += self.file_aux[:len(self.file_in) - len(self.file_out)]
            np.random.shuffle(self.file_out)

        self.im_size = im_size
        self.batch_size = batch_size
        self.random_crop = random_crop

    def next(self):
        with tf.variable_scope('feed'):
            file_in_tensor = tf.convert_to_tensor(self.file_in, dtype=tf.string)
            file_out_tensor = tf.convert_to_tensor(self.file_out, dtype=tf.string)
            data_queue1, data_queue2 = tf.train.slice_input_producer([file_in_tensor, file_out_tensor])
            im_gt = tf.image.decode_image(tf.read_file(data_queue1), channels=3)
            im_gt = tf.cast(im_gt, tf.float32)
            if self.random_crop is True:
                im_gt = tf.random_crop(im_gt, [self.im_size[0], self.im_size[1], 3])
            else:
                im_gt = tf.image.resize_image_with_crop_or_pad(im_gt, self.im_size[0], self.im_size[1])

            im_noise = tf.image.decode_image(tf.read_file(data_queue2), channels=3)
            im_noise = tf.cast(im_noise, tf.float32)

            if self.random_crop is True:
                im_noise = tf.random_crop(im_noise, [self.im_size[0], self.im_size[1], 3])
            else:
                im_noise = tf.image.resize_image_with_crop_or_pad(im_noise, self.im_size[0], self.im_size[1])

            if self.paired is False:
                im_noise = tf.image.random_flip_up_down(im_noise)

            im = tf.concat([im_gt, im_noise], axis=2)

            im.set_shape([self.im_size[0], self.im_size[1], 6])

            batch_data = tf.train.batch([im], batch_size=self.batch_size, num_threads=8)
        return batch_data

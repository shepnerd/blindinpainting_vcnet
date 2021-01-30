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

class Vgg19(object):
    def __init__(self, filepath=None):
        self.mean = np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
        self.vgg_weights = filepath if filepath is not None else os.path.join('vgg19_weights', 'imagenet-vgg-verydeep-19.mat')
        if os.path.exists(self.vgg_weights) is False:
            self.vgg_weights = os.path.join('vgg19_weights', 'imagenet-vgg-verydeep-19.mat')
            if os.path.isdir('vgg19_weights') is False:
                os.mkdir('vgg19_weights')
            url = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
            print('Downloading vgg19..')
            urllib.request.urlretrieve(url, self.vgg_weights)
            print('vgg19 weights have been downloaded and stored in {}'.format(self.vgg_weights))

    def build_net(self, ntype, nin, nwb=None, name=None):
        if ntype == 'conv':
            return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
        elif ntype == 'pool':
            return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_weight_bias(self, vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        weights = tf.constant(weights)
        bias = vgg_layers[i][0][0][2][0][1]
        bias = tf.constant(np.reshape(bias, (bias.size)))
        return weights, bias

    def build_vgg19(self, input, reuse=False):
        with tf.variable_scope('vgg19', reuse=reuse):
            net = {}
            # vgg_rawnet = scipy.io.loadmat(self.vgg_weights)
            vgg_rawnet = io.loadmat(self.vgg_weights)
            vgg_layers = vgg_rawnet['layers'][0]
            net['input'] = input - self.mean
            net['conv1_1'] = self.build_net('conv', net['input'], self.get_weight_bias(vgg_layers, 0),
                                            name='vgg_conv1_1')
            net['conv1_2'] = self.build_net('conv', net['conv1_1'], self.get_weight_bias(vgg_layers, 2),
                                            name='vgg_conv1_2')
            net['pool1'] = self.build_net('pool', net['conv1_2'])
            net['conv2_1'] = self.build_net('conv', net['pool1'], self.get_weight_bias(vgg_layers, 5),
                                            name='vgg_conv2_1')
            net['conv2_2'] = self.build_net('conv', net['conv2_1'], self.get_weight_bias(vgg_layers, 7),
                                            name='vgg_conv2_2')
            net['pool2'] = self.build_net('pool', net['conv2_2'])
            net['conv3_1'] = self.build_net('conv', net['pool2'], self.get_weight_bias(vgg_layers, 10),
                                            name='vgg_conv3_1')
            net['conv3_2'] = self.build_net('conv', net['conv3_1'], self.get_weight_bias(vgg_layers, 12),
                                            name='vgg_conv3_2')
            net['conv3_3'] = self.build_net('conv', net['conv3_2'], self.get_weight_bias(vgg_layers, 14),
                                            name='vgg_conv3_3')
            net['conv3_4'] = self.build_net('conv', net['conv3_3'], self.get_weight_bias(vgg_layers, 16),
                                            name='vgg_conv3_4')
            net['pool3'] = self.build_net('pool', net['conv3_4'])
            net['conv4_1'] = self.build_net('conv', net['pool3'], self.get_weight_bias(vgg_layers, 19),
                                            name='vgg_conv4_1')
            net['conv4_2'] = self.build_net('conv', net['conv4_1'], self.get_weight_bias(vgg_layers, 21),
                                            name='vgg_conv4_2')
            net['conv4_3'] = self.build_net('conv', net['conv4_2'], self.get_weight_bias(vgg_layers, 23),
                                            name='vgg_conv4_3')
            net['conv4_4'] = self.build_net('conv', net['conv4_3'], self.get_weight_bias(vgg_layers, 25),
                                            name='vgg_conv4_4')
            net['pool4'] = self.build_net('pool', net['conv4_4'])
            net['conv5_1'] = self.build_net('conv', net['pool4'], self.get_weight_bias(vgg_layers, 28),
                                            name='vgg_conv5_1')
            net['conv5_2'] = self.build_net('conv', net['conv5_1'], self.get_weight_bias(vgg_layers, 30),
                                            name='vgg_conv5_2')
            net['conv5_3'] = self.build_net('conv', net['conv5_2'], self.get_weight_bias(vgg_layers, 32),
                                            name='vgg_conv5_3')
            net['conv5_4'] = self.build_net('conv', net['conv5_3'], self.get_weight_bias(vgg_layers, 34),
                                            name='vgg_conv5_4')
            net['pool5'] = self.build_net('pool', net['conv5_4'])
        return net

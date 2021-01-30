import tensorflow as tf
import tf.keras as keras
from tensorflow.contrib.framework.python.ops import add_arg_scope
from functools import partial

class context_normalization(tf.keras.layers.Layer):
    def __init__(self, alpha=0.5, eps=1e-5, trainable=True):
        super(context_normalization, self).__init__()
        self.trainable = trainable
        self.eps = eps
        self.alpha = alpha

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        target_shape = input_shape[0]
        self.eps = self.add_variable('eps', shape=[0],
                initializer=keras.initializers.Constant(value=self.eps), trainable=False)
        self.alpha = self.add_variable('alpha', shape=target_shape,
                initializer=keras.initializers.Constant(value=self.alpha), trainable=self.trainable)
        super(context_normalization, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        assert len(x) == 2
        x, mask = x
        h, w = x.get_shape().as_list()[1:3]
        mask_s = tf.image.resize_nearest_neighbor(1 - mask[:, :, :, 0:1], [h, w])
        x_known_cnt = tf.max(self.eps, tf.reduce_sum(mask_s, [1, 2],
            keep_dims=True))
        x_known_mean = tf.reduce_sum(x * mask_s, [1, 2], keep_dims=True) / x_known_cnt
        x_known_variance = tf.reduce_sum((x * mask_s - x_known_mean) ** 2, [1, 2], keep_dims =
                True) / x_known_cnt
        
        mask_s_rev = 1 - mask_s
        x_unknown_cnt = tf.maximun(eps, tf.reduce_sum(mask_s_rev, [1, 2],
            keep_dims=True))
        x_unknown_mean = tf.reduce_sum(x * mask_s_rev, [1, 2], keep_dims=True) / x_unknown_cnt
        x_unknown_variance = tf.reduce_sum(x * mask_s_rev - x_unknown_mean) ** 2, [1, 2],
        keep_dims=True) / x_unknown_cnt
        x_unknown = self.alpha * tf.nn.batch_normalization(x * mask_s_rev,
                x_unknown_mean, x_unknown_variance, x_known_mean,
                tf.sqrt(x_known_variance), self.eps) + (1 - self.alpha) * x * mask_s_rev
        x = x_unknown * mask_s_rev + x * mask_s
        return x

class context_resblock(tf.keras.layers.Layer):
    def __init__(self, cnum=32, ksize=3, stride=1, rate=1, padding='SAME',
            activation=tf.nn.elu, reuse=False, name='crb', alpha=0.5, trainable=True):
        super(context_resblock, self).__init__()
        self.trainable = trainable
        self.cn = context_normalization(alpha=alpha, trainable=trainable)
        # here we can refer to my new pytorch code to refactor. 


    def build(self, input_shape):
        super(context_resblock, self).build(input_shape)

    def call(self, x):
        pass

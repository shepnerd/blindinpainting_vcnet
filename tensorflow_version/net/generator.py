import tensorflow as tf
from net.ops import *
from net.loss import *
from util.util import f2uint
from functools import partial, reduce
from abc import abstractmethod, ABC as AbstractBaseClass
from tensorflow.contrib.framework.python.ops import arg_scope


class BaseNetwork(AbstractBaseClass):
    def __init__(self, config=None):
        self.config = config
        self.net = partial(self.build_net, config=config)

    @abstractmethod
    def build_net(self, x, mask, config=None, reuse=False, training=True, name='blind_inpaint_net'):
        pass

    @abstractmethod
    def evaluate(self, im, noise, mask, config, reuse=False):
        pass

    def forward(self, x, mask, reuse=False):
        return self.net(x=x, mask=mask, reuse=reuse, training=True, name=self.config.name)


class VCNModel(BaseNetwork):
    def __init__(self, config=None):
        super(VCNModel, self).__init__(config=config)

    def build_net(self, x, mask=None, reuse=False, name='blind_inpaint_net', config=None):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        xin = x
        rho = config.rho

        # network with three branches
        cnum = self.config.g_cnum

        cn_type = self.config.cn_type
        conv_3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')

        if rho is not None:
            config.rho = rho
        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(xin, cnum*2, 5, stride=2, name='mask_conv2')
            x = resblock(x, cnum*4, 3, stride=2, name='mask_conv3')
            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            xb3 = tf.image.resize_bilinear(mx_feat, [xh, xw], align_corners=True)
            x = conv_3(inputs=x, filters=cnum * 4, strides=1, name='mask_conv8')

            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            x = resblock(x, cnum * 2, 3, stride=1, name='mask_deconv9')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            x = resblock(x, cnum, 3, stride=1, name='mask_deconv10')

            x = conv_3(inputs=x, filters=cnum // 2, strides=1, name='mask_compress_conv')
            mask_logit = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None, padding='SAME',
                                          name='mask_output')
            mask_pred = tf.clip_by_value(mask_logit, 0., 1.)

            if config.use_cn is True:
                if config.phase == 'tune':
                    mask = mask_pred
            else:
                mask = None
            if config.embrace is True:
                xin = xin * (1 - mask)
            x = context_resblock(xin, mask, cnum, 5, stride=1, name='cmp_conv1', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum*2, 3, stride=2, name='cmp_conv2', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 2, 3, stride=1, name='cmp_conv21', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=2, name='cmp_conv3', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=1, name='cmp_conv31', debug=cn_type, alpha=config.rho)

            x = context_resblock(x, mask, cnum*4, 3, stride=1, rate=2, name='cmp_conv4_atrous', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=1, rate=2, name='cmp_conv5_atrous', alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=1, rate=4, name='cmp_conv6_atrous', alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=1, rate=4, name='cmp_conv7_atrous', debug=cn_type, alpha=config.rho)

            x = context_resblock(x, mask, cnum * 4, 3, stride=1, name='cmp_conv8', debug=cn_type, alpha=config.rho)

            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            x = context_resblock(x, mask, cnum * 2, 3, stride=1, name='cmp_deconv9', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 2, 3, stride=1, name='cmp_deconv91', debug=cn_type, alpha=config.rho)
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            x = context_resblock(x, mask, cnum, 3, stride=1, name='cmp_deconv10', debug=cn_type, alpha=config.rho)
            xb1 = context_resblock(x, mask, cnum, 3, stride=1, name='cmp_deconv101', debug=cn_type, alpha=config.rho)

            x = tf.concat([xb1, xb3], axis=-1)
            x = conv_3(inputs=x, filters=cnum, strides=1, name='cmp_compress_conv1')
            x = conv_3(inputs=x, filters=cnum//2, strides=1, name='cmp_compress_conv2')
            x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=3, strides=1, activation=None, padding='SAME',
                                 name='cmp_output')
            x = tf.clip_by_value(x, -1., 1.)

        return x, mask_pred, mask_logit

    def evaluate(self, im, noise, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        noise = noise / 127.5 - 1
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            im = im * (1 - mask_soft) + noise * mask_soft
        else:
            im = im * (1 - mask) + noise * mask
        batch_input = im
        # inpaint
        batch_predict, batch_mask, batch_mask_logit = self.build_net(im, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + im * (1 - batch_mask)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=batch_mask_logit))

        return batch_predict, batch_complete, batch_mask, bce, batch_input

    def de_fence(self, im, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        batch_input = im
        # inpaint
        self.config.phase = 'acc'
        batch_predict, batch_mask, batch_mask_logit = self.build_net(im, mask=mask, reuse=reuse, config=self.config)
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + im * (1 - batch_mask)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=batch_mask_logit))

        return batch_predict, batch_complete, batch_mask, bce, batch_input

    def dummy_use(self, im, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        im = im * (1 - mask)
        # inpaint
        batch_predict, batch_mask, batch_mask_logit = self.build_generator(im, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + im * (1 - batch_mask)
        # batch_complete = batch_predict
        return batch_predict, batch_complete, batch_mask


class InpaintCAModel_MEN(BaseNetwork):
    def __init__(self, config=None):
        super(InpaintCAModel_MEN, self).__init__(config)

    def build_net(self, x, mask, config=None, reuse=False, training=True, name='blind_inpaint_net'):
        xin = x
        x_one = tf.ones_like(x)[:, :, :, 0:1]
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]

        # network with three branches
        if config is None:
            cnum = self.config.g_cnum
        else:
            cnum = config.g_cnum
        conv_3 = self.conv3

        padding='SAME'

        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(xin, cnum*2, 5, stride=2, name='mask_conv2')
            x = resblock(x, cnum*4, 3, stride=2, name='mask_conv3')
            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            x = conv_3(inputs=x, filters=cnum * 4, strides=1, name='mask_conv8')

            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            x = resblock(x, cnum * 2, 3, stride=1, name='mask_deconv9')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            x = resblock(x, cnum, 3, stride=1, name='mask_deconv10')

            x = conv_3(inputs=x, filters=cnum // 2, strides=1, name='mask_compress_conv')
            mask_logit = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None, padding='SAME',
                                          name='mask_output')
            mask_pred = tf.clip_by_value(mask_logit, 0., 1.)

            if config.phase == 'tune':
                mask = mask_pred
        if config.embrace is True:
            xin = xin * (1 - mask)
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            x = tf.concat([xin, mask * x_one], axis=-1)
            # stage1
            x = gen_conv(x, cnum, 5, 1, name='conv1')
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6')
            x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
            x = gen_deconv(x, 2*cnum, name='conv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
            x = gen_deconv(x, cnum, name='conv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='conv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x

            x = x*mask + xin*(1.-mask)

            ones_x = tf.ones_like(x, dtype=tf.float32)[:, :, :, 0:1]
            xnow = tf.concat([x, ones_x*mask], axis=3)
            x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
            x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')
            x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')
            x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x
            # attention branch
            x = gen_conv(xnow, cnum, 5, 1, name='pmconv1')
            x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
            x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
                         activation=tf.nn.relu)
            mask_s = resize_mask_like(mask, x)[0:1, :, :, :]
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')
            x = gen_deconv(x, 2*cnum, name='allconv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')
            x = gen_deconv(x, cnum, name='allconv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='allconv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
            x_stage2 = tf.clip_by_value(x, -1., 1.)
        return x_stage1, x_stage2, offset_flow, mask_pred, mask_logit

    def evaluate(self, batch_data, batch_noise, masks, config=None, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        batch_pos = batch_data / 127.5 - 1.
        batch_noise = batch_noise / 127.5 - 1
        im = batch_pos
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - masks, hsize=15, iters=4) + masks
            im = im * (1 - mask_soft) + batch_noise * mask_soft
        else:
            im = im * (1 - masks) + batch_noise * masks
        # inpaint
        x1, x2, flow, mask_pred, mask_logit = self.build_net(im, masks, reuse=reuse, training=is_training, config=config)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*mask_pred + im*(1-mask_pred)

        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=masks, logits=mask_logit))
        return x2, batch_complete, mask_pred, bce, im


class NaiveED(BaseNetwork):
    def __init__(self, config=None):
        super(NaiveED, self).__init__(config)

    def build_net(self, x, mask, config=None, reuse=False, training=True, name='blind_inpaint_net'):

        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage1
            x = gen_conv(x, cnum, 5, 1, name='conv1')
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6')

            x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
            x = gen_deconv(x, 2*cnum, name='conv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
            x = gen_deconv(x, cnum, name='conv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='conv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            # x = x*mask + xin*(1.-mask)
            # x.set_shape(xin.get_shape().as_list())
            # conv branch
            # xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
            xnow = x_stage1
            x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
            x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')
            x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')
            x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')

            x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')
            x = gen_deconv(x, 2*cnum, name='allconv13_upsample')
            x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')
            x = gen_deconv(x, cnum, name='allconv15_upsample')
            x = gen_conv(x, cnum//2, 3, 1, name='allconv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
            x_stage2 = tf.clip_by_value(x, -1., 1.)
        return x_stage1, x_stage2

    def evaluate(self, batch_data, noise, mask, config=None, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        im = batch_data / 127.5 - 1.
        noise = noise / 127.5 - 1
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            im = im * (1 - mask_soft) + noise * mask_soft
        else:
            im = im * (1 - mask) + noise * mask
        # inpaint
        x1, x2 = self.build_net(im, mask, reuse=reuse, training=is_training, config=config)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict
        return batch_predict, batch_complete, None, None, im


class GMCNNModel_MEN(BaseNetwork):
    def __init__(self, config=None):
        super(GMCNNModel_MEN, self).__init__(config)

    def build_net(self, x, mask, config=None, reuse=False, training=True, name='blind_inpaint_net'):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]

        if config is not None:
            self.config = config
        # network with three branches
        cnum = self.config.g_cnum
        b_names = ['b1', 'b2', 'b3', 'merge']

        conv_7 = partial(tf.layers.conv2d, kernel_size=7, activation=tf.nn.elu, padding='SAME')
        conv_5 = partial(tf.layers.conv2d, kernel_size=5, activation=tf.nn.elu, padding='SAME')
        conv_3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')

        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(x, cnum*2, 5, stride=2, name='mask_conv2')
            x = resblock(x, cnum*4, 3, stride=2, name='mask_conv3')

            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            x = conv_3(inputs=x, filters=cnum * 4, strides=1, name='mask_conv8')

            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            x = resblock(x, cnum * 2, 3, stride=1, name='mask_deconv9')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            x = resblock(x, cnum, 3, stride=1, name='mask_deconv10')

            x = conv_3(inputs=x, filters=cnum // 2, strides=1, name='mask_compress_conv')
            mask_logit = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None, padding='SAME',
                                          name='mask_output')
            mask_pred = tf.clip_by_value(mask_logit, 0., 1.)

            # branch 3
            if config.phase == 'tune':
                mask = mask_pred
        if config.embrace is True:
            x = x * (1 - mask)
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x_w_mask = tf.concat([x, ones_x * mask], axis=3)
        with tf.variable_scope(name, reuse=reuse):
            # branch 1
            x = conv_7(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[0] + 'conv1')
            x = conv_7(inputs=x, filters=2*cnum, strides=2, name=b_names[0] + 'conv2_downsample')
            x = conv_7(inputs=x, filters=2*cnum, strides=1, name=b_names[0] + 'conv3')
            x = conv_7(inputs=x, filters=4*cnum, strides=2, name=b_names[0] + 'conv4_downsample')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv5')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv6')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=2, name=b_names[0] + 'conv7_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=4, name=b_names[0] + 'conv8_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=8, name=b_names[0] + 'conv9_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=16, name=b_names[0] + 'conv10_atrous')
            if cnum > 32:
                x = conv_7(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[0] + 'conv11_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv11')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv12')
            x_b1 = tf.image.resize_bilinear(x, [xh, xw], align_corners=True)

            # branch 2
            x = conv_5(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[1] + 'conv1')
            x = conv_5(inputs=x, filters=2 * cnum, strides=2, name=b_names[1] + 'conv2_downsample')
            x = conv_5(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv3')
            x = conv_5(inputs=x, filters=4 * cnum, strides=2, name=b_names[1] + 'conv4_downsample')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv5')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv6')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=2, name=b_names[1] + 'conv7_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=4, name=b_names[1] + 'conv8_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=8, name=b_names[1] + 'conv9_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=16, name=b_names[1] + 'conv10_atrous')
            if cnum > 32:
                x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[1] + 'conv11_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv11')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv12')
            x = tf.image.resize_nearest_neighbor(x, [xh//2, xw//2], align_corners=True)
            with tf.variable_scope(b_names[1] + 'conv13_upsample'):
                x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv13_upsample_conv')
            x = conv_5(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv14')
            x_b2 = tf.image.resize_bilinear(x, [xh, xw], align_corners=True)

            # branch 3
            x = conv_5(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[2] + 'conv1')
            x = conv_3(inputs=x, filters=2 * cnum, strides=2, name=b_names[2] + 'conv2_downsample')
            x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv3')
            x = conv_3(inputs=x, filters=4 * cnum, strides=2, name=b_names[2] + 'conv4_downsample')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv5')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv6')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=2, name=b_names[2] + 'conv7_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=4, name=b_names[2] + 'conv8_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=8, name=b_names[2] + 'conv9_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=16, name=b_names[2] + 'conv10_atrous')
            if cnum > 32:
                x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[2] + 'conv11_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv11')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv12')
            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            with tf.variable_scope(b_names[2] + 'conv13_upsample'):
                x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv13_upsample_conv')
            x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv14')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            with tf.variable_scope(b_names[2] + 'conv15_upsample'):
                x = conv_3(inputs=x, filters=cnum, strides=1, name=b_names[2] + 'conv15_upsample_conv')
            x_b3 = conv_3(inputs=x, filters=cnum//2, strides=1, name=b_names[2] + 'conv16')

            x_merge = tf.concat([x_b1, x_b2, x_b3], axis=3)

            x = conv_3(inputs=x_merge, filters=cnum // 2, strides=1, name=b_names[3] + 'conv17')
            x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=3, strides=1, activation=None, padding='SAME',
                                 name=b_names[3] + 'conv18')
            x = tf.clip_by_value(x, -1., 1.)
        return x, mask_pred, mask_logit

    def evaluate(self, im, noise, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        noise = noise / 127.5 - 1
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            im = im * (1 - mask_soft) + noise * mask_soft
        else:
            im = im * (1 - mask) + noise * mask

        # inpaint
        batch_predict, mask_pred, mask_logit = self.build_net(im, mask, config=config, reuse=reuse)
        # apply mask and reconstruct
        batch_complete = batch_predict * mask_pred + im * (1 - mask_pred)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_logit))
        return batch_predict, batch_complete, mask_pred, bce, im


class PartialConvNet(BaseNetwork):
    def __init__(self, config=None):
        super(PartialConvNet, self).__init__(config)

    def build_net(self, x, mask=None, reuse=False, name='blind_inpaint_net', config=None):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        xin = x

        # network with three branches
        cnum = self.config.g_cnum

        conv_3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(xin, cnum*2, 5, stride=2, name='mask_conv2')
            x = resblock(x, cnum*4, 3, stride=2, name='mask_conv3')

            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            x = resblock(mx_feat, cnum * 4, 3, stride=1, name='mask_conv8')
            x = conv_3(inputs=x, filters=cnum * 4, strides=1, name='mask_conv8')

            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            x = resblock(x, cnum * 2, 3, stride=1, name='mask_deconv9')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            x = resblock(x, cnum, 3, stride=1, name='mask_deconv10')

            x = conv_3(inputs=x, filters=cnum // 2, strides=1, name='mask_compress_conv')
            mask_logit = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None,
                                          padding='SAME', name='mask_output')
            mask_pred = tf.clip_by_value(mask_logit, 0., 1.)

            # branch 3
            if config.phase == 'tune' or mask is None:
                mask = mask_pred
            if config.embrace is True:
                xin = xin * (1 - mask)

            xin_ch = xin.get_shape().as_list()[-1]
            m = 1 - tf.tile(mask, [1, 1, 1, xin_ch])
            min = m
            x1, m1 = partial_conv(xin, m, cnum * 2, 7, stride=2, activation=tf.nn.relu, name='cmp_pconv1')
            x2, m2 = partial_conv(x1, m1, cnum * 4, 5, stride=2, activation=tf.nn.relu, name='cmp_pconv2')
            x3, m3 = partial_conv(x2, m2, cnum * 8, 5, stride=2, activation=tf.nn.relu, name='cmp_pconv3')
            x4, m4 = partial_conv(x3, m3, cnum * 16, 3, stride=2, activation=tf.nn.relu, name='cmp_pconv4')
            x5, m5 = partial_conv(x4, m4, cnum * 16, 3, stride=2, activation=tf.nn.relu, name='cmp_pconv5')
            x6, m6 = partial_conv(x5, m5, cnum * 16, 3, stride=2, activation=tf.nn.relu, name='cmp_pconv6')
            x, m = partial_conv(x6, m6, cnum * 16, 3, stride=2, activation=tf.nn.relu, name='cmp_pconv7')

            h, w = x.get_shape().as_list()[1:3]
            h, w = h * 2, w * 2
            x = tf.image.resize_nearest_neighbor(x, [h, w])
            m = tf.image.resize_nearest_neighbor(m, [h, w])
            x, m = partial_conv(tf.concat([x, x6], -1), tf.concat([m, m6], -1), cnum * 16, 3, stride=1,
                                activation=tf.nn.leaky_relu, name='cmp_pdconv1')
            h, w = h * 2, w * 2
            x = tf.image.resize_nearest_neighbor(x, [h, w])
            m = tf.image.resize_nearest_neighbor(m, [h, w])
            x, m = partial_conv(tf.concat([x, x5], -1), tf.concat([m, m5], -1), cnum * 16, 3, stride=1,
                                activation=tf.nn.leaky_relu, name='cmp_pdconv2')
            h, w = h * 2, w * 2
            x = tf.image.resize_nearest_neighbor(x, [h, w])
            m = tf.image.resize_nearest_neighbor(m, [h, w])
            x, m = partial_conv(tf.concat([x, x4], -1), tf.concat([m, m4], -1), cnum * 16, 3, stride=1,
                                activation=tf.nn.leaky_relu, name='cmp_pdconv3')
            h, w = h * 2, w * 2
            x = tf.image.resize_nearest_neighbor(x, [h, w])
            m = tf.image.resize_nearest_neighbor(m, [h, w])
            x, m = partial_conv(tf.concat([x, x3], -1), tf.concat([m, m3], -1), cnum * 8, 3, stride=1,
                                activation=tf.nn.leaky_relu, name='cmp_pdconv4')
            h, w = h * 2, w * 2
            x = tf.image.resize_nearest_neighbor(x, [h, w])
            m = tf.image.resize_nearest_neighbor(m, [h, w])
            x, m = partial_conv(tf.concat([x, x2], -1), tf.concat([m, m2], -1), cnum * 4, 3, stride=1,
                                activation=tf.nn.leaky_relu, name='cmp_pdconv5')
            h, w = h * 2, w * 2
            x = tf.image.resize_nearest_neighbor(x, [h, w])
            m = tf.image.resize_nearest_neighbor(m, [h, w])
            x, m = partial_conv(tf.concat([x, x1], -1), tf.concat([m, m1], -1), cnum * 2, 3, stride=1,
                                activation=tf.nn.leaky_relu, name='cmp_pdconv6')
            h, w = h * 2, w * 2
            x = tf.image.resize_nearest_neighbor(x, [h, w])
            m = tf.image.resize_nearest_neighbor(m, [h, w])
            x, _ = partial_conv(tf.concat([x, xin], -1), tf.concat([m, min], -1), 3, 3, stride=1,
                                activation=None, name='cmp_pdconv7')
            x = tf.clip_by_value(x, -1., 1.)

        return x, mask_pred, mask_logit

    def evaluate(self, im, noise, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        noise = noise / 127.5 - 1
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            im = im * (1 - mask_soft) + noise * mask_soft
        else:
            im = im * (1 - mask) + noise * mask
        # inpaint
        batch_predict, mask_pred, mask_logit = self.build_net(im, mask, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * mask + im * (1 - mask)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_logit))
        return batch_predict, batch_complete, mask_pred, bce, im


class InpaintGatedModel_MEN(BaseNetwork):
    def __init__(self, config=None):
        super(InpaintGatedModel_MEN, self).__init__(config)

    def build_net(self, x, mask, config=None, reuse=False, training=True, name='blind_inpaint_net'):
        xin = x
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        padding = 'SAME'

        if config is None:
            cnum = self.config.g_cnum
        else:
            cnum = config.g_cnum
        conv_3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        with tf.variable_scope(name, reuse=reuse):
            x = resblock(xin, cnum * 2, 5, stride=2, name='mask_conv2')
            x = resblock(x, cnum * 4, 3, stride=2, name='mask_conv3')

            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')

            x = conv_3(inputs=x, filters=cnum * 4, strides=1, name='mask_conv8')

            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            x = resblock(x, cnum * 2, 3, stride=1, name='mask_deconv9')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            x = resblock(x, cnum, 3, stride=1, name='mask_deconv10')

            x = conv_3(inputs=x, filters=cnum // 2, strides=1, name='mask_compress_conv')
            mask_logit = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None,
                                          padding='SAME',
                                          name='mask_output')
            mask_pred = tf.clip_by_value(mask_logit, 0., 1.)

            if config.phase == 'tune':
                mask = mask_pred
        if config.embrace is True:
            xin = xin * (1 - mask)

        # two stage network
        cnum = 48
        x = tf.concat([xin, ones_x, ones_x * mask], axis=3)
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage1
            x = gen_gatedconv(x, cnum, 5, 1, name='conv1')
            x = gen_gatedconv(x, 2*cnum, 3, 2, name='conv2_downsample')
            x = gen_gatedconv(x, 2*cnum, 3, 1, name='conv3')
            x = gen_gatedconv(x, 4*cnum, 3, 2, name='conv4_downsample')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='conv5')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='conv6')
            mask_s = resize_mask_like(mask, x)
            x = gen_gatedconv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
            x = gen_gatedconv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
            x = gen_gatedconv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
            x = gen_gatedconv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='conv11')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='conv12')
            x = gen_degatedconv(x, 2*cnum, name='conv13_upsample')
            x = gen_gatedconv(x, 2*cnum, 3, 1, name='conv14')
            x = gen_degatedconv(x, cnum, name='conv15_upsample')
            x = gen_gatedconv(x, cnum//2, 3, 1, name='conv16')
            x = gen_gatedconv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.nn.tanh(x)
            x_stage1 = x

            # stage2, paste result as input
            x = x*mask + xin[:, :, :, 0:3]*(1.-mask)
            x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())

            xnow = x
            x = gen_gatedconv(xnow, cnum, 5, 1, name='xconv1')
            x = gen_gatedconv(x, cnum, 3, 2, name='xconv2_downsample')
            x = gen_gatedconv(x, 2*cnum, 3, 1, name='xconv3')
            x = gen_gatedconv(x, 2*cnum, 3, 2, name='xconv4_downsample')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='xconv5')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='xconv6')
            x = gen_gatedconv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
            x = gen_gatedconv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
            x = gen_gatedconv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
            x = gen_gatedconv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x
            # attention branch
            x = gen_gatedconv(xnow, cnum, 5, 1, name='pmconv1')
            x = gen_gatedconv(x, cnum, 3, 2, name='pmconv2_downsample')
            x = gen_gatedconv(x, 2*cnum, 3, 1, name='pmconv3')
            x = gen_gatedconv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='pmconv5')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='pmconv6', activation=tf.nn.relu)
            mask_s = tf.reduce_mean(mask_s, axis=0, keep_dims=True)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='pmconv9')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='pmconv10')
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            x = gen_gatedconv(x, 4*cnum, 3, 1, name='allconv11')
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='allconv12')
            x = gen_degatedconv(x, 2*cnum, name='allconv13_upsample')
            x = gen_gatedconv(x, 2*cnum, 3, 1, name='allconv14')
            x = gen_degatedconv(x, cnum, name='allconv15_upsample')
            x = gen_gatedconv(x, cnum//2, 3, 1, name='allconv16')
            x = gen_gatedconv(x, 3, 3, 1, activation=None, name='allconv17')
            x = tf.nn.tanh(x)
            x_stage2 = x
        return x_stage1, x_stage2, offset_flow, mask_pred, mask_logit

    def build_sn_patch_gan_discriminator(self, x, reuse=False, training=True):
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
            self, batch, reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            d = self.build_sn_patch_gan_discriminator(
                batch, reuse=reuse, training=training)
            return d

    def evaluate(self, batch_data, noise, mask, config=None, reuse=False, is_training=False):

        im = batch_data / 127.5 - 1.
        noise = noise / 127.5 - 1
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1-mask, hsize=15, iters=4)+mask
            im = im * (1 - mask_soft) + noise * mask_soft
        else:
            im = im * (1 - mask) + noise * mask

        x1, x2, flow, mask_pred, mask_logit = self.build_net(im, mask, config=config, reuse=reuse, training=False)
        batch_predict = x2

        batch_complete = batch_predict * mask_pred + im * (1 - mask_pred)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_logit))

        return batch_predict, batch_complete, mask_pred, bce, im

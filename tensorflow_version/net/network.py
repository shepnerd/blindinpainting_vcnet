import tensorflow as tf
# from net.ops import random_bbox, bbox2mask, local_patch
# from net.ops import priority_loss_mask
# from net.ops import id_mrf_reg
# from net.ops import gan_wgan_loss, gradients_penalty, random_interpolates
# from net.ops import free_form_mask_tf
from net.ops import *
from net.loss import *
from util.util import f2uint
from functools import partial, reduce

from net.arsenal import g_uint, focal_uint, c_unit, harness_confidence
from net.arsenal import context_normalization, context_resblock, resblock, context_normalization_se
from tensorflow.contrib.framework.python.ops import arg_scope


class InpaintCAModel_MEN:
    def __init__(self, config=None):
        self.config = config
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')
        return

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='blind_inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        x_one = tf.ones_like(x)[:, :, :, 0:1]
        offset_flow = None
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]

        # network with three branches
        if config is None:
            cnum = self.config.g_cnum
        else:
            cnum = config.g_cnum
        conv_3 = self.conv3

        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(xin, cnum*2, 5, stride=2, name='mask_conv2')
            # x = resblock(x, cnum*2, 3, stride=1, name='mask_conv21')
            x = resblock(x, cnum*4, 3, stride=2, name='mask_conv3')
            # x = resblock(x, cnum * 4, 3, stride=1, name='mask_conv31')

            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            xb3 = tf.image.resize_bilinear(mx_feat, [xh, xw], align_corners=True)
            # x = resblock(mx_feat, cnum * 4, 3, stride=1, name='mask_conv8')
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
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)
            # x.set_shape(xin.get_shape().as_list())
            # conv branch
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

    def wgan_patch_discriminator(self, x, mask, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('discriminator_local', reuse=reuse):
            h, w = mask.get_shape().as_list()[1:3]
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum*2, name='conv2')
            x = self.conv5_ds(x, filters=cnum*4, name='conv3')
            x = self.conv5_ds(x, filters=cnum*8, name='conv4')
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

    def wgan_local_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_local', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')

            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_global_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_global', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv4')
            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_discriminator(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def wgan_mask_discriminator(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
        return dout_local, dout_global, mask_local

    def build_net(self, batch_data, batch_noise, config, training=True, summary=True, reuse=False):
        self.config = config
        batch_pos = batch_data / 127.5 - 1.
        batch_noise = batch_noise / 127.5 - 1
        # generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=8, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=20, maxLength=80, maxVertex=16)
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            batch_incomplete = batch_pos * (1. - mask_soft) + batch_noise * mask_soft
        else:
            batch_incomplete = batch_pos * (1. - mask) + batch_noise * mask
        x1, x2, offset_flow, mask_pred, mask_logit = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=reuse, training=training)
        if config.pretrain_network:
            batch_predicted = x1
        else:
            batch_predicted = x2
        losses = {}
        # apply mask and complete image
        if config.use_blend is True:
            batch_complete = batch_predicted
        else:
            batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # batch_complete = batch_predicted
        # local patches
        if config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            local_patch_mask = local_patch(mask, bbox)
            local_patch_batch_pred = local_patch(batch_predicted, bbox)
            local_patch_x1 = local_patch(x1, bbox)
            local_patch_x2 = local_patch(x2, bbox)
        else:
            local_patch_batch_pos = batch_pos
            local_patch_batch_complete = batch_complete
            local_patch_batch_pred = batch_predicted
            local_patch_x1 = x1
            local_patch_x2 = x2
        l1_alpha = config.pretrain_l1_alpha
        losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1))
        if not config.pretrain_network:
            losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2))
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1.-mask))
        if not config.pretrain_network:
            losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)

        losses['mask_loss'] = bce_weighted(mask_logit, mask, mask)
        if summary:
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
            tf.summary.scalar('losses/mask_loss', losses['mask_loss'])

            batch_mask_vis = tf.tile(mask_pred, [1, 1, 1, 3]) * 2 - 1
            mask_vis = tf.tile(mask, [config.batch_size, 1, 1, 3]) * 2 - 1

            viz_img = tf.concat([batch_pos, batch_incomplete, batch_complete, batch_mask_vis, mask_vis], axis=2)
            # viz_img = [batch_pos, batch_incomplete, batch_predicted, batch_complete]
            if offset_flow is not None:
                viz_img = tf.concat([viz_img, resize(offset_flow, scale=4,
                                                     func=tf.image.resize_nearest_neighbor)], axis=2)
            tf.summary.image('raw__incomplete__predicted__flow__mask-pred__mask', f2uint(viz_img))

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        if config.mask_type == 'rect':
            # local deterministic patch
            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
            # wgan with gradient penalty
            pos_neg_local, pos_neg_global = self.wgan_discriminator(local_patch_batch_pos_neg,
                                                                    batch_pos_neg, config.d_cnum, reuse=reuse)
        else:
            pos_neg_local, pos_neg_global, mask_local = self.wgan_mask_discriminator(batch_pos_neg,
                                                                                     mask, config.d_cnum, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        global_wgan_loss_alpha = 1.0
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        if config.mask_type == 'rect':
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            dout_local, dout_global = self.wgan_discriminator(
                interpolates_local, interpolates_global, config.d_cnum, reuse=True)
        else:
            interpolates_local = interpolates_global
            dout_local, dout_global, _ = self.wgan_mask_discriminator(interpolates_global, mask, config.d_cnum,
                                                                      reuse=True)

        # apply penalty
        if config.mask_type == 'rect':
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
        else:
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('convergence/d_loss', losses['d_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global)
            tf.summary.scalar('gan_wgan_loss/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gan_wgan_loss/gp_penalty_local', penalty_local)
            tf.summary.scalar('gan_wgan_loss/gp_penalty_global', penalty_global)

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        losses['g_loss'] += config.mask_loss_alpha * losses['mask_loss']
        ##
        if summary:
            tf.summary.scalar('G_loss', losses['g_loss'])

        print('Set L1_LOSS_ALPHA to %f' % config.l1_loss_alpha)
        print('Set GAN_LOSS_ALPHA to %f' % config.gan_loss_alpha)
        print('Set AE_LOSS_ALPHA to %f' % config.ae_loss_alpha)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

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
        x1, x2, flow, mask_pred, mask_logit = self.build_inpaint_net(im,
                                                                     masks, reuse=reuse,
                                                                     training=is_training, config=config)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*mask_pred + im*(1-mask_pred)

        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=masks, logits=mask_logit))
        return x2, batch_complete, mask_pred, bce, im


class NaiveED:
    def __init__(self, config=None):
        self.config = config
        self.conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')
        return

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='blind_inpaint_net'):

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

    def wgan_patch_discriminator(self, x, mask, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('discriminator_local', reuse=reuse):
            h, w = mask.get_shape().as_list()[1:3]
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum*2, name='conv2')
            x = self.conv5_ds(x, filters=cnum*4, name='conv3')
            x = self.conv5_ds(x, filters=cnum*8, name='conv4')
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

    def wgan_local_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_local', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')

            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_global_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_global', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv4')
            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_discriminator(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def wgan_mask_discriminator(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
        return dout_local, dout_global, mask_local

    def build_net(self, batch_data, batch_noise, config, training=True, summary=True, reuse=False):
        self.config = config
        batch_pos = batch_data / 127.5 - 1.
        batch_noise = batch_noise / 127.5 - 1.
        # generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=8, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=20, maxLength=80, maxVertex=16)
        mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask

        coin = tf.random_uniform([1], minval=0, maxval=1.0)[0]
        mask_used = tf.cond(coin > 0.5, lambda: mask_soft, lambda: mask)
        batch_incomplete = batch_pos * (1. - mask_used) + batch_noise * mask_used
        x1, x2 = self.build_inpaint_net(batch_incomplete, mask, config, reuse=reuse, training=training)
        if config.pretrain_network:
            batch_predicted = 0.5 * x2 + 0.5 * x1
        else:
            batch_predicted = x2
        losses = {}
        # apply mask and complete image
        if config.use_blend is True:
            batch_complete = batch_predicted
        else:
            batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # batch_complete = batch_predicted
        # local patches
        if config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            local_patch_mask = local_patch(mask, bbox)
            local_patch_batch_pred = local_patch(batch_predicted, bbox)
            local_patch_x1 = local_patch(x1, bbox)
            local_patch_x2 = local_patch(x2, bbox)
        else:
            local_patch_batch_pos = batch_pos
            local_patch_batch_complete = batch_complete
            local_patch_batch_pred = batch_predicted
            local_patch_x1 = x1
            local_patch_x2 = x2
        l1_alpha = config.pretrain_l1_alpha
        losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1))
        if not config.pretrain_network:
            losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2))
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1.-mask))
        if not config.pretrain_network:
            losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        if summary:
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
            viz_img = tf.concat([batch_pos, batch_incomplete, batch_complete], axis=2)
            # viz_img = [batch_pos, batch_incomplete, batch_predicted, batch_complete]
            tf.summary.image('raw__incomplete__predicted__complete', f2uint(viz_img))

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        if config.mask_type == 'rect':
            # local deterministic patch
            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
            # wgan with gradient penalty
            pos_neg_local, pos_neg_global = self.wgan_discriminator(local_patch_batch_pos_neg,
                                                                    batch_pos_neg, config.d_cnum, reuse=reuse)
        else:
            pos_neg_local, pos_neg_global, mask_local = self.wgan_mask_discriminator(batch_pos_neg,
                                                                                     mask, config.d_cnum, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        global_wgan_loss_alpha = 1.0
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        if config.mask_type == 'rect':
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            dout_local, dout_global = self.wgan_discriminator(
                interpolates_local, interpolates_global, config.d_cnum, reuse=True)
        else:
            interpolates_local = interpolates_global
            dout_local, dout_global, _ = self.wgan_mask_discriminator(interpolates_global, mask, config.d_cnum,
                                                                      reuse=True)

        # apply penalty
        if config.mask_type == 'rect':
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
        else:
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('convergence/d_loss', losses['d_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global)
            tf.summary.scalar('gan_wgan_loss/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gan_wgan_loss/gp_penalty_local', penalty_local)
            tf.summary.scalar('gan_wgan_loss/gp_penalty_global', penalty_global)

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        # losses['g_loss'] += config.mask_loss_alpha * losses['mask_loss']
        ##
        if summary:
            tf.summary.scalar('G_loss', losses['g_loss'])

        print('Set L1_LOSS_ALPHA to %f' % config.l1_loss_alpha)
        print('Set GAN_LOSS_ALPHA to %f' % config.gan_loss_alpha)
        print('Set AE_LOSS_ALPHA to %f' % config.ae_loss_alpha)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

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
        x1, x2 = self.build_inpaint_net(im, mask, reuse=reuse, training=is_training, config=config)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict
        return batch_predict, batch_complete, None, None, im


class GMCNNModel_MEN:
    def __init__(self):
        self.config = None

        # shortcut ops
        self.conv7 = partial(tf.layers.conv2d, kernel_size=7, activation=tf.nn.elu, padding='SAME')
        self.conv5 = partial(tf.layers.conv2d, kernel_size=5, activation=tf.nn.elu, padding='SAME')
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')

    def build_generator(self, x, mask, reuse=False, name='blind_inpaint_net', config=None):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]

        if config is not None:
            self.config = config
        # network with three branches
        cnum = self.config.g_cnum
        b_names = ['b1', 'b2', 'b3', 'merge']

        conv_7 = self.conv7
        conv_5 = self.conv5
        conv_3 = self.conv3

        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(x, cnum*2, 5, stride=2, name='mask_conv2')
            # x = resblock(x, cnum*2, 3, stride=1, name='mask_conv21')
            x = resblock(x, cnum*4, 3, stride=2, name='mask_conv3')
            # x = resblock(x, cnum * 4, 3, stride=1, name='mask_conv31')

            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            xb3 = tf.image.resize_bilinear(mx_feat, [xh, xw], align_corners=True)
            # x = resblock(mx_feat, cnum * 4, 3, stride=1, name='mask_conv8')
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

    def wgan_patch_discriminator(self, x, mask, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('discriminator_local', reuse=reuse):
            h, w = mask.get_shape().as_list()[1:3]
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum*2, name='conv2')
            x = self.conv5_ds(x, filters=cnum*4, name='conv3')
            x = self.conv5_ds(x, filters=cnum*8, name='conv4')
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

    def wgan_local_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_local', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')

            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_global_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_global', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv4')
            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_discriminator(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def wgan_mask_discriminator(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
        return dout_local, dout_global, mask_local

    def build_net(self, batch_data, batch_noise, config, summary=True, reuse=False):
        self.config = config
        batch_pos = batch_data / 127.5 - 1.
        batch_noise = batch_noise / 127.5 - 1
        # generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=8, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=20, maxLength=80, maxVertex=16)
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            batch_incomplete = batch_pos * (1. - mask_soft) + batch_noise * mask_soft
        else:
            batch_incomplete = batch_pos * (1. - mask) + batch_noise * mask
        mask_priority = priority_loss_mask(mask)
        batch_predicted, mask_pred, mask_logit = self.build_generator(batch_incomplete, mask, reuse=reuse, config=config)

        losses = {}
        # apply mask and complete image
        if config.use_blend is True:
            batch_complete = batch_predicted
        else:
            batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        if config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            local_patch_mask = local_patch(mask, bbox)
            local_patch_batch_pred = local_patch(batch_predicted, bbox)
            mask_priority = local_patch(mask_priority, bbox)
        else:
            local_patch_batch_pos = batch_pos
            local_patch_batch_complete = batch_complete
            local_patch_batch_pred = batch_predicted

        if config.pretrain_network:
            print('Pretrain the whole net with only reconstruction loss.')

        if not config.pretrain_network:
            config.feat_style_layers = {'conv3_2': 1.0, 'conv4_2': 1.0}
            config.feat_content_layers = {'conv4_2': 1.0}

            config.mrf_style_w = 1.0
            config.mrf_content_w = 1.0

            ID_MRF_loss = id_mrf_reg(local_patch_batch_pred, local_patch_batch_pos, config)
            # ID_MRF_loss = id_mrf_reg(batch_predicted, batch_pos, config)

            losses['ID_MRF_loss'] = ID_MRF_loss
            tf.summary.scalar('losses/ID_MRF_loss', losses['ID_MRF_loss'])

        pretrain_l1_alpha = config.pretrain_l1_alpha
        losses['l1_loss'] = \
            pretrain_l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_batch_pred) * mask_priority)
        if not config.pretrain_network:
            losses['l1_loss'] += tf.reduce_mean(ID_MRF_loss * config.mrf_alpha)
        losses['ae_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - batch_predicted) * (1. - mask))
        if not config.pretrain_network:
            losses['ae_loss'] += pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - batch_predicted) * (1. - mask))
        losses['ae_loss'] /= tf.reduce_mean(1. - mask)
        losses['mask_loss'] = bce_weighted(mask_logit, mask, mask)
        if summary:
            batch_mask_vis = tf.tile(mask_pred, [1, 1, 1, 3]) * 2 - 1
            mask_vis = tf.tile(mask, [config.batch_size, 1, 1, 3]) * 2 - 1

            viz_img = tf.concat([batch_pos, batch_incomplete, batch_complete, batch_mask_vis, mask_vis], axis=2)
            tf.summary.image('gt__degraded__predicted__mask-pred__mask', f2uint(viz_img))
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
            tf.summary.scalar('losses/mask_loss', losses['mask_loss'])

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        if config.mask_type == 'rect':
            # local deterministic patch
            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
            # wgan with gradient penalty
            pos_neg_local, pos_neg_global = self.wgan_discriminator(local_patch_batch_pos_neg,
                                                                    batch_pos_neg, config.d_cnum, reuse=reuse)
        else:
            pos_neg_local, pos_neg_global, mask_local = self.wgan_mask_discriminator(batch_pos_neg,
                                                                                     mask, config.d_cnum, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        global_wgan_loss_alpha = 1.0
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        if config.mask_type == 'rect':
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            dout_local, dout_global = self.wgan_discriminator(
                interpolates_local, interpolates_global, config.d_cnum, reuse=True)
        else:
            interpolates_local = interpolates_global
            dout_local, dout_global, _ = self.wgan_mask_discriminator(interpolates_global, mask, config.d_cnum, reuse=True)

        # apply penalty
        if config.mask_type == 'rect':
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
        else:
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('convergence/d_loss', losses['d_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global)
            tf.summary.scalar('gan_wgan_loss/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gan_wgan_loss/gp_penalty_local', penalty_local)
            tf.summary.scalar('gan_wgan_loss/gp_penalty_global', penalty_global)

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        losses['g_loss'] += config.mask_loss_alpha * losses['mask_loss']
        ##

        print('Set L1_LOSS_ALPHA to %f' % config.l1_loss_alpha)
        print('Set GAN_LOSS_ALPHA to %f' % config.gan_loss_alpha)

        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        print('Set AE_LOSS_ALPHA to %f' % config.ae_loss_alpha)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

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
        batch_predict, mask_pred, mask_logit = self.build_generator(im, mask, config=config, reuse=reuse)
        # apply mask and reconstruct
        batch_complete = batch_predict * mask_pred + im * (1 - mask_pred)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_logit))
        return batch_predict, batch_complete, mask_pred, bce, im


class PartialConvNet:
    def __init__(self):
        self.config = None

        # shortcut ops
        self.conv7 = partial(tf.layers.conv2d, kernel_size=7, activation=tf.nn.elu, padding='SAME')
        self.conv5 = partial(tf.layers.conv2d, kernel_size=5, activation=tf.nn.elu, padding='SAME')
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')

    def build_generator(self, x, mask=None, reuse=False, name='blind_inpaint_net', config=None):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        xin = x

        # network with three branches
        cnum = self.config.g_cnum

        conv_7 = self.conv7
        conv_5 = self.conv5
        conv_3 = self.conv3
        bc = ['mask', 'b1', 'b2']
        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(xin, cnum*2, 5, stride=2, name='mask_conv2')
            # x = resblock(x, cnum*2, 3, stride=1, name='mask_conv21')
            x = resblock(x, cnum*4, 3, stride=2, name='mask_conv3')
            # x = resblock(x, cnum * 4, 3, stride=1, name='mask_conv31')

            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            # xb3 = tf.image.resize_bilinear(mx_feat, [xh, xw], align_corners=True)
            x = resblock(mx_feat, cnum * 4, 3, stride=1, name='mask_conv8')
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

    def build_generator_soft(self, x, mask=None, reuse=False, name='blind_inpaint_net', config=None):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        xin = x

        # network with three branches
        cnum = self.config.g_cnum

        conv_7 = self.conv7
        conv_5 = self.conv5
        conv_3 = self.conv3
        bc = ['mask', 'b1', 'b2']
        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(xin, cnum*2, 5, stride=2, name='mask_conv2')
            # x = resblock(x, cnum*2, 3, stride=1, name='mask_conv21')
            x = resblock(x, cnum*4, 3, stride=2, name='mask_conv3')
            # x = resblock(x, cnum * 4, 3, stride=1, name='mask_conv31')

            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            # xb3 = tf.image.resize_bilinear(mx_feat, [xh, xw], align_corners=True)
            x = resblock(mx_feat, cnum * 4, 3, stride=1, name='mask_conv8')
            x = conv_3(inputs=x, filters=cnum * 4, strides=1, name='mask_conv8')

            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            x = resblock(x, cnum * 2, 3, stride=1, name='mask_deconv9')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            x = resblock(x, cnum, 3, stride=1, name='mask_deconv10')

            x = conv_3(inputs=x, filters=cnum // 2, strides=1, name='mask_compress_conv')
            mask_logit = tf.layers.conv2d(inputs=x, kernel_size=3, filters=1, strides=1, activation=None, padding='SAME',
                                 name='mask_output')
            mask_pred = tf.clip_by_value(mask_logit, 0., 1.)
            mask_soft = tf.sigmoid(mask_logit)

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

        return x, mask_pred, mask_logit, mask_soft

    def wgan_patch_discriminator(self, x, mask, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('discriminator_local', reuse=reuse):
            h, w = mask.get_shape().as_list()[1:3]
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum*2, name='conv2')
            x = self.conv5_ds(x, filters=cnum*4, name='conv3')
            x = self.conv5_ds(x, filters=cnum*8, name='conv4')
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

    def wgan_local_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_local', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')

            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_global_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_global', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv4')
            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_discriminator(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def wgan_mask_discriminator(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
        return dout_local, dout_global, mask_local

    def build_net(self, batch_data, batch_noise, config, summary=True, reuse=False):
        self.config = config
        batch_pos = batch_data / 127.5 - 1.
        batch_noise = batch_noise / 127.5 - 1
        # generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=8, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=20, maxLength=80, maxVertex=16)

        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            batch_incomplete = batch_pos * (1. - mask_soft) + batch_noise * mask_soft
        else:
            batch_incomplete = batch_pos * (1. - mask) + batch_noise * mask
        mask_priority = priority_loss_mask_cp(mask)
        mask_in = mask
        batch_predicted, batch_mask, batch_mask_logit = self.build_generator(batch_incomplete, mask=mask_in,
                                                                             config=config, reuse=reuse)

        losses = {}
        # apply mask and complete image
        soft_batch_mask = priority_loss_mask(1-batch_mask, hsize=15, iters=4) + batch_mask
        soft_batch_mask = tf.minimum(soft_batch_mask, 1)
        # batch_complete = batch_predicted * batch_mask + batch_incomplete * (1. - batch_mask)
        batch_complete = batch_predicted * soft_batch_mask + batch_incomplete * (1. - soft_batch_mask)
        if config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            local_patch_mask = local_patch(mask, bbox)
            local_patch_batch_pred = local_patch(batch_predicted, bbox)
            local_mask_priority = local_patch(mask_priority, bbox)
        else:
            local_patch_batch_pos = batch_pos
            local_patch_batch_complete = batch_complete
            local_patch_batch_pred = batch_predicted

        if config.pretrain_network:
            print('Pretrain the whole net with only reconstruction loss.')

        pretrain_l1_alpha = config.pretrain_l1_alpha
        # losses['l1_loss'] = pretrain_l1_alpha * tf.reduce_mean((batch_complete-batch_pos)**2)
        if config.mask_type == 'rect':
            losses['l1_loss'] = pretrain_l1_alpha * tf.reduce_mean(
                tf.abs(local_patch_batch_complete - local_patch_batch_pos))
        else:
            losses['l1_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_complete - batch_pos) * mask)
        # losses['l1_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_complete-batch_pos))
        losses['l1_loss'] /= tf.reduce_mean(mask)
        losses['ae_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_predicted - batch_pos) * (1. - mask))
        # losses['ae_loss'] /= (tf.reduce_mean(1. - batch_mask)+1e-5)
        # losses['ae_loss'] = 0

        losses['semantic_consistent_loss'] = 0
        # losses['semantic_consistent_loss'] = semantic_consistent_loss(batch_predicted, batch_pos, config,
        #                                                               reuse=reuse_vgg)
        # config.perceptual_layers = {'conv3_2': 1.0, 'conv4_2': 1.0}
        # losses['semantic_consistent_loss'] = perceptual_loss(batch_predicted, batch_pos, config)

        # losses['geometric_consistent_loss'] = geometric_consistent_loss(batch_predicted, batch_pos,
        #                                                                 mask_priority - 1 + mask)
        # losses['geometric_consistent_loss'] = edge_loss_coarse2fine(batch_predicted, batch_pos)
        # todo: maybe we should shut down the geometric loss, it seems harm generation performance.
        # losses['geometric_consistent_loss'] = relative_total_variation_loss(batch_predicted)
        losses['geometric_consistent_loss'] = 0
        losses['mask_loss'] = bce_weighted(batch_mask_logit, mask, mask)

        if summary:
            batch_mask_vis = tf.tile(batch_mask, [1, 1, 1, 3]) * 2 - 1
            soft_batch_mask_vis = tf.tile(soft_batch_mask, [1, 1, 1, 3]) * 2 - 1
            mask_vis = tf.tile(mask, [config.batch_size, 1, 1, 3]) * 2 - 1
            viz_img = tf.concat([batch_pos, batch_incomplete, batch_predicted,
                                 batch_complete, soft_batch_mask_vis, batch_mask_vis, mask_vis], axis=2)
            tf.summary.image('gt__degraded__predicted__completed__soft-mask__mask', f2uint(viz_img))
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
            tf.summary.scalar('losses/semantic_consistent_loss', losses['semantic_consistent_loss'])
            tf.summary.scalar('losses/geometric_consistent_loss', losses['geometric_consistent_loss'])
            tf.summary.scalar('losses/mask_loss', losses['mask_loss'])

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        if config.mask_type == 'rect':
            # local deterministic patch
            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
            # wgan with gradient penalty
            pos_neg_local, pos_neg_global = self.wgan_discriminator(local_patch_batch_pos_neg,
                                                                    batch_pos_neg, config.d_cnum, reuse=reuse)
        else:
            pos_neg_local, pos_neg_global, mask_local = self.wgan_mask_discriminator(batch_pos_neg,
                                                                                     mask, config.d_cnum, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        global_wgan_loss_alpha = 1.0
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        if config.mask_type == 'rect':
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            dout_local, dout_global = self.wgan_discriminator(
                interpolates_local, interpolates_global, config.d_cnum, reuse=True)
        else:
            interpolates_local = interpolates_global
            dout_local, dout_global, _ = self.wgan_mask_discriminator(interpolates_global, mask, config.d_cnum, reuse=True)

        # apply penalty
        if config.mask_type == 'rect':
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
        else:
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('convergence/d_loss', losses['d_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global)
            tf.summary.scalar('gan_wgan_loss/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gan_wgan_loss/gp_penalty_local', penalty_local)
            tf.summary.scalar('gan_wgan_loss/gp_penalty_global', penalty_global)

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']

        if config.use_mrf:
            losses['g_loss'] += config.mrf_alpha * losses['ID_MRF_loss']

        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        losses['g_loss'] += config.semantic_loss_alpha * losses['semantic_consistent_loss']
        losses['g_loss'] += config.geometric_loss_alpha * losses['geometric_consistent_loss']
        losses['g_loss'] += config.mask_loss_alpha * losses['mask_loss']
        ##
        if summary:
            tf.summary.scalar('G_loss', losses['g_loss'])

        print('Set L1_LOSS_ALPHA to %f' % config.l1_loss_alpha)
        print('Set GAN_LOSS_ALPHA to %f' % config.gan_loss_alpha)
        print('Set AE_LOSS_ALPHA to %f' % config.ae_loss_alpha)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

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
        batch_predict, mask_pred, mask_logit = self.build_generator(im, mask, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * mask + im * (1 - mask)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_logit))
        return batch_predict, batch_complete, mask_pred, bce, im

    def evaluate_soft(self, im, noise, mask, config, reuse=False):
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
        batch_predict, _, mask_logit, mask_pred = self.build_generator_soft(im, mask, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * mask_pred + im * (1 - mask_pred)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_logit))
        return batch_predict, batch_complete, mask_pred, bce, im


# todo: in this version, we use resblock to build generator for estimating lost pixels
# more compact structure
# todo: 2 branches: mask estimation, inpainting 3x3
# todo: the decoding part fuses discriminative encoding and generative encoding
# only two branches.
class VCNModel_lite:
    def __init__(self, config=None):
        self.config = config

        # shortcut ops
        self.conv7 = partial(tf.layers.conv2d, kernel_size=7, activation=tf.nn.elu, padding='SAME')
        self.conv5 = partial(tf.layers.conv2d, kernel_size=5, activation=tf.nn.elu, padding='SAME')
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')

    def build_generator(self, x, mask=None, reuse=False, name='blind_inpaint_net', config=None, rho=None):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        xin = x

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

    def wgan_patch_discriminator(self, x, mask, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('discriminator_local', reuse=reuse):
            h, w = mask.get_shape().as_list()[1:3]
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum*2, name='conv2')
            x = self.conv5_ds(x, filters=cnum*4, name='conv3')
            x = self.conv5_ds(x, filters=cnum*8, name='conv4')
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

    def wgan_local_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_local', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')

            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_global_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_global', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv4')
            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_discriminator(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def wgan_mask_discriminator(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
        return dout_local, dout_global, mask_local

    def build_net(self, batch_data, batch_noise, config, summary=True, reuse=False):
        self.config = config
        batch_pos = batch_data / 127.5 - 1.
        batch_noise = batch_noise / 127.5 - 1
        # generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=config.parts, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=config.brush_width, maxLength=config.brush_length,
                                     maxVertex=config.vertex)
        mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask

        coin = tf.random_uniform([1], minval=0, maxval=1.0)[0]
        mask_used = tf.cond(coin > 0.5, lambda: mask_soft, lambda: mask)

        batch_incomplete = batch_pos * (1. - mask_used) + batch_noise * mask_used
        mask_priority = priority_loss_mask_cp(mask)

        mask_in = None
        if config.phase == 'acc':
            mask_in = mask

        if config.paired is False:
            batch_predicted, batch_mask, batch_mask_logit = self.build_generator(batch_incomplete, mask=mask_in,
                                                                                 config=config, reuse=reuse)
        else:
            batch_predicted, batch_mask, batch_mask_logit = self.build_generator(batch_noise, mask=mask_in,
                                                                                 config=config, reuse=reuse)
        losses = {}
        # apply mask and complete image
        soft_batch_mask = priority_loss_mask(1-batch_mask, hsize=15, iters=4) + batch_mask
        soft_batch_mask = tf.minimum(soft_batch_mask, 1)
        # batch_complete = batch_predicted * batch_mask + batch_incomplete * (1. - batch_mask)
        batch_complete = batch_predicted * soft_batch_mask + batch_incomplete * (1. - soft_batch_mask)
        if config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            local_patch_mask = local_patch(mask, bbox)
        else:
            local_patch_batch_pos = batch_pos
            local_patch_batch_complete = batch_complete

        if config.pretrain_network:
            print('Pretrain the whole net with only reconstruction loss.')

        reuse_vgg = False
        if config.use_mrf:
            config.feat_style_layers = {'conv3_2': 0.5}
            config.feat_content_layers = {'conv3_2': 0.5}

            config.mrf_style_w = 1.0
            config.mrf_content_w = 1.0

            ID_MRF_loss = id_mrf_reg(batch_predicted, batch_pos, config)

            losses['ID_MRF_loss'] = ID_MRF_loss
            tf.summary.scalar('losses/ID_MRF_loss', losses['ID_MRF_loss'])

            reuse_vgg = True

        pretrain_l1_alpha = config.pretrain_l1_alpha

        losses['l1_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_predicted-batch_pos))
        losses['ae_loss'] = 0

        if config.paired is True:
            losses['l1_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_predicted - batch_pos))
            losses['ae_loss'] = 0

        losses['semantic_consistent_loss'] = semantic_consistent_loss(batch_predicted, batch_pos, config,
                                                                      reuse=reuse_vgg)
        losses['geometric_consistent_loss'] = 0
        losses['mask_loss'] = bce_weighted(batch_mask_logit, mask, mask)

        if summary:
            batch_mask_vis = tf.tile(batch_mask, [1, 1, 1, 3]) * 2 - 1
            soft_batch_mask_vis = tf.tile(soft_batch_mask, [1, 1, 1, 3]) * 2 - 1
            mask_vis = tf.tile(mask, [config.batch_size, 1, 1, 3]) * 2 - 1
            if config.paired is False:
                viz_img = tf.concat([batch_pos, batch_incomplete, batch_predicted,
                                     batch_complete, soft_batch_mask_vis, batch_mask_vis, mask_vis], axis=2)
            else:
                viz_img = tf.concat([batch_pos, batch_noise, batch_predicted,
                                     batch_complete, soft_batch_mask_vis, batch_mask_vis, mask_vis], axis=2)
            tf.summary.image('gt__degraded__predicted__completed__soft-mask__mask', f2uint(viz_img))
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
            tf.summary.scalar('losses/semantic_consistent_loss', losses['semantic_consistent_loss'])
            tf.summary.scalar('losses/geometric_consistent_loss', losses['geometric_consistent_loss'])
            tf.summary.scalar('losses/mask_loss', losses['mask_loss'])

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        if config.mask_type == 'rect':
            # local deterministic patch
            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
            # wgan with gradient penalty
            pos_neg_local, pos_neg_global = self.wgan_discriminator(local_patch_batch_pos_neg,
                                                                    batch_pos_neg, config.d_cnum, reuse=reuse)
        else:
            pos_neg_local, pos_neg_global, mask_local = self.wgan_mask_discriminator(batch_pos_neg,
                                                                                     mask, config.d_cnum, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        global_wgan_loss_alpha = 1.0
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        if config.mask_type == 'rect':
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            dout_local, dout_global = self.wgan_discriminator(
                interpolates_local, interpolates_global, config.d_cnum, reuse=True)
        else:
            interpolates_local = interpolates_global
            dout_local, dout_global, _ = self.wgan_mask_discriminator(interpolates_global, mask, config.d_cnum, reuse=True)

        # apply penalty
        if config.mask_type == 'rect':
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
        else:
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('convergence/d_loss', losses['d_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global)
            tf.summary.scalar('gan_wgan_loss/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gan_wgan_loss/gp_penalty_local', penalty_local)
            tf.summary.scalar('gan_wgan_loss/gp_penalty_global', penalty_global)

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']

        if config.use_mrf:
            losses['g_loss'] += config.mrf_alpha * losses['ID_MRF_loss']

        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        losses['g_loss'] += config.semantic_loss_alpha * losses['semantic_consistent_loss']
        losses['g_loss'] += config.geometric_loss_alpha * losses['geometric_consistent_loss']
        if config.paired is False:
            losses['g_loss'] += config.mask_loss_alpha * losses['mask_loss']
        ##
        if summary:
            tf.summary.scalar('G_loss', losses['g_loss'])

        print(f'l1 lambda: {config.l1_loss_alpha}')
        print(f'gan_loss lambda: {config.gan_loss_alpha}')
        print(f'ae_loss lambda: {config.ae_loss_alpha}')
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

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
        batch_predict, batch_mask, batch_mask_logit = self.build_generator(im, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + im * (1 - batch_mask)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=batch_mask_logit))

        return batch_predict, batch_complete, batch_mask, bce, batch_input

    def evaluate_soft(self, im, noise, mask, config, reuse=False):
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
        batch_predict, _, batch_mask_logit, batch_mask = self.build_generator_soft(im, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + im * (1 - batch_mask)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=batch_mask_logit))
        # batch_complete = batch_predict
        return batch_predict, batch_complete, batch_mask, bce, batch_input

    def de_fence(self, im, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        batch_input = im
        # inpaint
        self.config.phase = 'acc'
        batch_predict, batch_mask, batch_mask_logit = self.build_generator(im, mask=mask, reuse=reuse, config=self.config)
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


class InpaintGatedModel_MEN:
    def __init__(self, config=None):
        self.name = 'InpaintCAModel-MEN'
        self.config = config
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')
        return

    def build_inpaint_net(self, x, mask,
                          config=None,
                          reuse=False,
                          training=True, padding='SAME', name='blind_inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]

        if config is None:
            cnum = self.config.g_cnum
        else:
            cnum = config.g_cnum
        conv_3 = self.conv3
        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(xin, cnum * 2, 5, stride=2, name='mask_conv2')
            # x = resblock(x, cnum*2, 3, stride=1, name='mask_conv21')
            x = resblock(x, cnum * 4, 3, stride=2, name='mask_conv3')
            # x = resblock(x, cnum * 4, 3, stride=1, name='mask_conv31')

            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            xb3 = tf.image.resize_bilinear(mx_feat, [xh, xw], align_corners=True)
            # x = resblock(mx_feat, cnum * 4, 3, stride=1, name='mask_conv8')
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

            # branch 3
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
            # conv branch
            # xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
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
            x = gen_gatedconv(x, 4*cnum, 3, 1, name='pmconv6',
                                activation=tf.nn.relu)
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

    # todo: align FLAGS with config
    def build_net(
            self, batch_data, batch_noise, config, training=True, summary=True,
            reuse=False):
        # here FLAGS should be config in our code
        batch_pos = batch_data / 127.5 - 1.
        batch_noise = batch_noise / 127.5 - 1
        # generate mask, 1 represents masked point
        # bbox = random_bbox(FLAGS)
        FLAGS = config
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=8, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=20, maxLength=80, maxVertex=16)

        if config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            batch_incomplete = batch_pos * (1. - mask_soft) + batch_noise * mask_soft
        else:
            batch_incomplete = batch_pos * (1. - mask) + batch_noise * mask

        xin = batch_incomplete
        x1, x2, offset_flow, mask_pred, mask_logit = self.build_inpaint_net(batch_incomplete, mask, config=config, reuse=reuse, training=training)
        batch_predicted = x2
        losses = {}
        # apply mask and complete image
        # batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        if config.use_blend is True:
            batch_complete = batch_predicted
        else:
            batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
        # local patches
        l1_loss_alpha = 1
        losses['ae_loss'] = l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x1))
        losses['ae_loss'] += l1_loss_alpha * tf.reduce_mean(tf.abs(batch_pos - x2))

        losses['mask_loss'] = bce_weighted(mask_logit, mask, mask)
        if summary:
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
            tf.summary.scalar('losses/mask_loss', losses['mask_loss'])

            batch_mask_vis = tf.tile(mask_pred, [1, 1, 1, 3]) * 2 - 1
            mask_vis = tf.tile(mask, [config.batch_size, 1, 1, 3]) * 2 - 1

            viz_img = [batch_pos, batch_incomplete, batch_complete, batch_mask_vis, mask_vis]
            # if FLAGS.guided:
            #     viz_img = [
            #         batch_pos,
            #         batch_incomplete + edge,
            #         batch_complete]
            # else:
            #     viz_img = [batch_pos, batch_incomplete, batch_complete]
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_nearest_neighbor))
            tf.summary.image('raw__incomplete__predicted__flow__mask-pred__mask', f2uint(tf.concat(viz_img, axis=2)))

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        # if FLAGS.gan_with_mask:
        batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [FLAGS.batch_size*2, 1, 1, 1])], axis=3)
        # wgan with gradient penalty
        # if FLAGS.gan == 'sngan':
        pos_neg = self.build_gan_discriminator(batch_pos_neg, training=training, reuse=reuse)
        pos, neg = tf.split(pos_neg, 2)
        g_loss, d_loss = gan_hinge_loss(pos, neg)
        losses['g_loss'] = g_loss
        losses['d_loss'] = d_loss
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_loss', d_loss)
        # else:
        #     raise NotImplementedError('{} not implemented.'.format(FLAGS.gan))
        gan_loss_alpha = 1

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = gan_loss_alpha * losses['g_loss']
        # if FLAGS.ae_loss:
        losses['g_loss'] += losses['ae_loss']
        losses['g_loss'] += config.mask_loss_alpha * losses['mask_loss']
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def evaluate(self, batch_data, noise, mask, config=None, reuse=False, is_training=False):

        im = batch_data / 127.5 - 1.
        noise = noise / 127.5 - 1
        if config.use_blend is True:
            mask_soft = priority_loss_mask(1-mask, hsize=15, iters=4)+mask
            im = im * (1 - mask_soft) + noise * mask_soft
        else:
            im = im * (1 - mask) + noise * mask

        x1, x2, flow, mask_pred, mask_logit = self.build_inpaint_net(im, mask, config=config, reuse=reuse, training=False)
        batch_predict = x2

        batch_complete = batch_predict * mask_pred + im * (1 - mask_pred)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_logit))

        return batch_predict, batch_complete, mask_pred, bce, im


class blindinpaint_model:
    def __init__(self, config=None):
        self.config = config

    def build_generator(self, x, mask=None, reuse=False, name='blind_inpaint_net', config=None, rho=None):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        xin = x

        # network with three branches
        cnum = self.config.g_cnum

        cn_type = self.config.cn_type
        conv_3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')

        if rho is not None:
            config.rho = rho
        with tf.variable_scope(name, reuse=reuse):

            # branch mask
            x = resblock(xin, cnum * 2, 5, stride=2, name='mask_conv2')
            x = resblock(x, cnum * 4, 3, stride=2, name='mask_conv3')
            x = resblock(x, cnum * 4, 3, stride=1, rate=2, name='mask_conv4_atrous')
            mx_feat = resblock(x, cnum * 4, 3, stride=1, rate=4, name='mask_conv5_atrous')
            xb3 = tf.image.resize_bilinear(mx_feat, [xh, xw], align_corners=True)
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

            if config.use_cn is True:
                if config.phase == 'tune':
                    mask = mask_pred
            else:
                mask = None
            if config.embrace is True:
                xin = xin * (1 - mask)
            x = context_resblock(xin, mask, cnum, 5, stride=1, name='cmp_conv1', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 2, 3, stride=2, name='cmp_conv2', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 2, 3, stride=1, name='cmp_conv21', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=2, name='cmp_conv3', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=1, name='cmp_conv31', debug=cn_type, alpha=config.rho)

            x = context_resblock(x, mask, cnum * 4, 3, stride=1, rate=2, name='cmp_conv4_atrous', debug=cn_type,
                                 alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=1, rate=2, name='cmp_conv5_atrous', alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=1, rate=4, name='cmp_conv6_atrous', alpha=config.rho)
            x = context_resblock(x, mask, cnum * 4, 3, stride=1, rate=4, name='cmp_conv7_atrous', debug=cn_type,
                                 alpha=config.rho)

            x = context_resblock(x, mask, cnum * 4, 3, stride=1, name='cmp_conv8', debug=cn_type, alpha=config.rho)

            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            x = context_resblock(x, mask, cnum * 2, 3, stride=1, name='cmp_deconv9', debug=cn_type, alpha=config.rho)
            x = context_resblock(x, mask, cnum * 2, 3, stride=1, name='cmp_deconv91', debug=cn_type, alpha=config.rho)
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            x = context_resblock(x, mask, cnum, 3, stride=1, name='cmp_deconv10', debug=cn_type, alpha=config.rho)
            xb1 = context_resblock(x, mask, cnum, 3, stride=1, name='cmp_deconv101', debug=cn_type, alpha=config.rho)

            x = tf.concat([xb1, xb3], axis=-1)
            x = conv_3(inputs=x, filters=cnum, strides=1, name='cmp_compress_conv1')
            x = conv_3(inputs=x, filters=cnum // 2, strides=1, name='cmp_compress_conv2')
            x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=3, strides=1, activation=None, padding='SAME',
                                 name='cmp_output')
            x = tf.clip_by_value(x, -1., 1.)

        return x, mask_pred, mask_logit

    def wgan_patch_discriminator(self, x, mask, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('discriminator_local', reuse=reuse):
            h, w = mask.get_shape().as_list()[1:3]
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
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

    def wgan_local_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_local', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')

            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_global_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_global', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv4')
            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_discriminator(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def wgan_mask_discriminator(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
        return dout_local, dout_global, mask_local

    def build_net(self, batch_data, batch_noise, config, summary=True, reuse=False):
        self.config = config
        batch_pos = batch_data / 127.5 - 1.
        batch_noise = batch_noise / 127.5 - 1
        # generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=config.parts, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=config.brush_width, maxLength=config.brush_length,
                                     maxVertex=config.vertex)
        mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask

        coin = tf.random_uniform([1], minval=0, maxval=1.0)[0]
        mask_used = tf.cond(coin > 0.5, lambda: mask_soft, lambda: mask)

        batch_incomplete = batch_pos * (1. - mask_used) + batch_noise * mask_used
        mask_priority = priority_loss_mask_cp(mask)

        mask_in = None
        if config.phase == 'acc':
            mask_in = mask

        if config.paired is False:
            batch_predicted, batch_mask, batch_mask_logit = self.build_generator(batch_incomplete, mask=mask_in,
                                                                                 config=config, reuse=reuse)
        else:
            batch_predicted, batch_mask, batch_mask_logit = self.build_generator(batch_noise, mask=mask_in,
                                                                                 config=config, reuse=reuse)
        losses = {}
        # apply mask and complete image
        soft_batch_mask = priority_loss_mask(1 - batch_mask, hsize=15, iters=4) + batch_mask
        soft_batch_mask = tf.minimum(soft_batch_mask, 1)
        # batch_complete = batch_predicted * batch_mask + batch_incomplete * (1. - batch_mask)
        batch_complete = batch_predicted * soft_batch_mask + batch_incomplete * (1. - soft_batch_mask)
        if config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            local_patch_mask = local_patch(mask, bbox)
        else:
            local_patch_batch_pos = batch_pos
            local_patch_batch_complete = batch_complete

        if config.pretrain_network:
            print('Pretrain the whole net with only reconstruction loss.')

        reuse_vgg = False
        if config.use_mrf:
            config.feat_style_layers = {'conv3_2': 0.5}
            config.feat_content_layers = {'conv3_2': 0.5}

            config.mrf_style_w = 1.0
            config.mrf_content_w = 1.0

            ID_MRF_loss = id_mrf_reg(batch_predicted, batch_pos, config)

            losses['ID_MRF_loss'] = ID_MRF_loss
            tf.summary.scalar('losses/ID_MRF_loss', losses['ID_MRF_loss'])

            reuse_vgg = True

        pretrain_l1_alpha = config.pretrain_l1_alpha

        losses['l1_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_predicted - batch_pos))
        losses['ae_loss'] = 0

        if config.paired is True:
            losses['l1_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_predicted - batch_pos))
            losses['ae_loss'] = 0

        losses['semantic_consistent_loss'] = semantic_consistent_loss(batch_predicted, batch_pos, config,
                                                                      reuse=reuse_vgg)
        losses['geometric_consistent_loss'] = 0
        losses['mask_loss'] = bce_weighted(batch_mask_logit, mask, mask)

        if summary:
            batch_mask_vis = tf.tile(batch_mask, [1, 1, 1, 3]) * 2 - 1
            soft_batch_mask_vis = tf.tile(soft_batch_mask, [1, 1, 1, 3]) * 2 - 1
            mask_vis = tf.tile(mask, [config.batch_size, 1, 1, 3]) * 2 - 1
            if config.paired is False:
                viz_img = tf.concat([batch_pos, batch_incomplete, batch_predicted,
                                     batch_complete, soft_batch_mask_vis, batch_mask_vis, mask_vis], axis=2)
            else:
                viz_img = tf.concat([batch_pos, batch_noise, batch_predicted,
                                     batch_complete, soft_batch_mask_vis, batch_mask_vis, mask_vis], axis=2)
            tf.summary.image('gt__degraded__predicted__completed__soft-mask__mask', f2uint(viz_img))
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])
            tf.summary.scalar('losses/semantic_consistent_loss', losses['semantic_consistent_loss'])
            tf.summary.scalar('losses/geometric_consistent_loss', losses['geometric_consistent_loss'])
            tf.summary.scalar('losses/mask_loss', losses['mask_loss'])

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

        if config.mask_type == 'rect':
            # local deterministic patch
            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
            # wgan with gradient penalty
            pos_neg_local, pos_neg_global = self.wgan_discriminator(local_patch_batch_pos_neg,
                                                                    batch_pos_neg, config.d_cnum, reuse=reuse)
        else:
            pos_neg_local, pos_neg_global, mask_local = self.wgan_mask_discriminator(batch_pos_neg,
                                                                                     mask, config.d_cnum, reuse=reuse)
        pos_local, neg_local = tf.split(pos_neg_local, 2)
        pos_global, neg_global = tf.split(pos_neg_global, 2)
        # wgan loss
        global_wgan_loss_alpha = 1.0
        g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
        g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
        losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
        losses['d_loss'] = d_loss_global + d_loss_local
        # gp
        interpolates_global = random_interpolates(batch_pos, batch_complete)
        if config.mask_type == 'rect':
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            dout_local, dout_global = self.wgan_discriminator(
                interpolates_local, interpolates_global, config.d_cnum, reuse=True)
        else:
            interpolates_local = interpolates_global
            dout_local, dout_global, _ = self.wgan_mask_discriminator(interpolates_global, mask, config.d_cnum,
                                                                      reuse=True)

        # apply penalty
        if config.mask_type == 'rect':
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
        else:
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=mask)
        penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local + penalty_global)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        if summary and not config.pretrain_network:
            tf.summary.scalar('convergence/d_loss', losses['d_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global)
            tf.summary.scalar('gan_wgan_loss/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gan_wgan_loss/gp_penalty_local', penalty_local)
            tf.summary.scalar('gan_wgan_loss/gp_penalty_global', penalty_global)

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']

        if config.use_mrf:
            losses['g_loss'] += config.mrf_alpha * losses['ID_MRF_loss']

        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']
        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        losses['g_loss'] += config.semantic_loss_alpha * losses['semantic_consistent_loss']
        losses['g_loss'] += config.geometric_loss_alpha * losses['geometric_consistent_loss']
        if config.paired is False:
            losses['g_loss'] += config.mask_loss_alpha * losses['mask_loss']
        ##
        if summary:
            tf.summary.scalar('G_loss', losses['g_loss'])

        print(f'l1 lambda: {config.l1_loss_alpha}')
        print(f'gan_loss lambda: {config.gan_loss_alpha}')
        print(f'ae_loss lambda: {config.ae_loss_alpha}')
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

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
        batch_predict, batch_mask, batch_mask_logit = self.build_generator(im, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + im * (1 - batch_mask)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=batch_mask_logit))

        return batch_predict, batch_complete, batch_mask, bce, batch_input

    def evaluate_soft(self, im, noise, mask, config, reuse=False):
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
        batch_predict, _, batch_mask_logit, batch_mask = self.build_generator_soft(im, reuse=reuse, config=config)
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + im * (1 - batch_mask)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=batch_mask_logit))
        # batch_complete = batch_predict
        return batch_predict, batch_complete, batch_mask, bce, batch_input

    def de_fence(self, im, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        batch_input = im
        # inpaint
        self.config.phase = 'acc'
        batch_predict, batch_mask, batch_mask_logit = self.build_generator(im, mask=mask, reuse=reuse,
                                                                           config=self.config)
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

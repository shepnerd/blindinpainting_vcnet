import tensorflow as tf
from net.ops import *
from net.loss import *
from util.util import f2uint
from functools import partial, reduce

from tensorflow.contrib.framework.python.ops import arg_scope
from net.generator import VCNModel
from net.discriminator import wgan_discriminator, wgan_mask_discriminator, build_gan_discriminator

class Blindinpaint_model:
    def __init__(self, config=None):
        self.config = config

        self.netG = VCNModel(self.config).net

        if not hasattr(config, 'pretrain_network') or config.pretrain_network:
            self.netD = None
        else:
            self.netD = wgan_discriminator if config.mask_type == 'rect' else wgan_mask_discriminator

    def _preprocess(self, x):
        return x / 127.5 - 1

    def _get_mask(self):
        config = self.config
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask = bbox2mask(bbox, config, name='mask_c')
        else:
            mask = free_form_mask_tf(parts=config.parts, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=config.brush_width, maxLength=config.brush_length,
                                     maxVertex=config.vertex)
            bbox = None

        mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask

        return mask, mask_soft, bbox

    def _get_local(self, xin, xout, mask=None, bbox=None):
        if self.config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(xin, bbox)
            local_patch_batch_complete = local_patch(xout, bbox)
            local_patch_mask = local_patch(mask, bbox)
        else:
            local_patch_batch_pos = xin
            local_patch_batch_complete = xout
            local_patch_mask = mask

        return local_patch_batch_pos, local_patch_batch_complete, local_patch_mask

    def get_training_losses(self, batch_data, batch_noise, summary=True, reuse=False):
        config = self.config
        batch_pos = self._preprocess(batch_data)
        batch_noise = self._preprocess(batch_noise)
        # generate mask, 1 represents masked point

        mask, mask_soft, bbox = self._get_mask()
        coin = tf.random_uniform([1], minval=0, maxval=1.0)[0]
        mask_used = tf.cond(coin > 0.5, lambda: mask_soft, lambda: mask)

        batch_incomplete = batch_pos * (1. - mask_used) + batch_noise * mask_used

        mask_in = None
        if config.phase == 'acc':
            mask_in = mask

        data_input = batch_noise if config.paired else batch_incomplete
        batch_predicted, batch_mask, batch_mask_logit = self.netG(data_input, mask=mask_in, reuse=reuse)

        losses = {}
        # apply mask and complete image
        soft_batch_mask = priority_loss_mask(1 - batch_mask, hsize=15, iters=4) + batch_mask
        soft_batch_mask = tf.minimum(soft_batch_mask, 1)
        # batch_complete = batch_predicted * batch_mask + batch_incomplete * (1. - batch_mask)

        batch_complete = batch_predicted * soft_batch_mask + batch_incomplete * (1. - soft_batch_mask)

        local_patch_batch_pos, local_patch_batch_complete, local_patch_mask = self._get_local(batch_pos, batch_complete, mask=mask, bbox=bbox)

        # if config.pretrain_network:
        #     print('Pretrain the whole net with only reconstruction loss.')

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
        if config.pretrain_network is False:
            batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)

            local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0) if config.mask_type == 'rect' else None

            pos_neg_local, pos_neg_global = self.netD(batch_global=batch_pos_neg, batch_local=local_patch_batch_pos_neg, mask=mask, d_cnum=config.d_cnum, reuse=reuse)

            pos_local, neg_local = tf.split(pos_neg_local, 2)
            
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # wgan loss
            # global_wgan_loss_alpha = 1.0
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
            # losses['g_loss'] = global_wgan_loss_alpha * g_loss_global + g_loss_local
            losses['g_loss'] = g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete) if config.mask_type == 'rect' else interpolates_global

            dout_local, dout_global = self.netD(batch_global=interpolates_global, batch_local=interpolates_local, mask=mask, d_cnum=config.d_cnum, reuse=True)

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

            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        else:
            losses['g_loss'] = 0

        # if config.pretrain_network:
        #     losses['g_loss'] = 0
        # else:
        #     losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']

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

        # print(f'l1 lambda: {config.l1_loss_alpha}')
        # print(f'gan_loss lambda: {config.gan_loss_alpha}')
        # print(f'ae_loss lambda: {config.ae_loss_alpha}')
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'blind_inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def evaluate_compose(self, im, noise, mask, config, reuse=False):
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

    def evaluate(self, im, noise, mask, reuse=False):
        # generate mask, 1 represents masked point
        im = im / 127.5 - 1
        noise = noise / 127.5 - 1
        if self.config.use_blend is True:
            mask_soft = priority_loss_mask(1 - mask, hsize=15, iters=4) + mask
            im = im * (1 - mask_soft) + noise * mask_soft
        else:
            im = im * (1 - mask) + noise * mask
        batch_input = im
        # inpaint
        batch_predict, batch_mask, batch_mask_logit = self.netG(im, reuse=reuse)
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
        batch_predict, batch_mask, batch_mask_logit = self.netG(im, mask=mask, reuse=reuse)
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
        batch_predict, batch_mask, batch_mask_logit = self.netG(im, reuse=reuse)
        # apply mask and reconstruct
        batch_complete = batch_predict * batch_mask + im * (1 - batch_mask)
        # batch_complete = batch_predict
        return batch_predict, batch_complete, batch_mask

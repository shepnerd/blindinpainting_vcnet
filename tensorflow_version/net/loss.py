import tensorflow as tf
import numpy as np
from net.ops import random_bbox, bbox2mask, local_patch
from net.ops import priority_loss_mask
from net.ops import gan_wgan_loss, gradients_penalty, random_interpolates
from net.ops import free_form_mask_tf
from net.vgg import Vgg19
from util.util import f2uint
from functools import partial, reduce
import scipy.stats as st

from tensorflow.contrib.framework.python.ops import add_arg_scope


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


def gauss_kernel(size=21, sigma=3.0):
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


def relative_total_variation_map(im, win_size=9, sigma=3.0, eps=1e-5, padding='VALID'):
    g = tf_make_guass_var(win_size, sigma)
    fh = tf.constant([[[[1], [1], [1]]], [[[-1], [-1], [-1]]]], tf.float32)
    fw = tf.constant([[[[1], [1], [1]], [[-1], [-1], [-1]]]], tf.float32)
    dy = tf.nn.conv2d(im, fh, strides=[1, 1, 1, 1], padding='SAME')
    dx = tf.nn.conv2d(im, fw, strides=[1, 1, 1, 1], padding='SAME')

    Dy = tf.nn.conv2d(tf.abs(dy), g, strides=[1, 1, 1, 1], padding=padding)
    Dx = tf.nn.conv2d(tf.abs(dx), g, strides=[1, 1, 1, 1], padding=padding)

    Ly = tf.abs(tf.nn.conv2d(dy, g, strides=[1, 1, 1, 1], padding=padding))
    Lx = tf.abs(tf.nn.conv2d(dx, g, strides=[1, 1, 1, 1], padding=padding))

    rtv_map = Dy / (Ly + eps) + Dx / (Lx + eps)
    return rtv_map


# using relative total variation to compute the discrepancy between pred and gt
def edge_loss_coarse2fine(pred, gt):
    discrepancy = tf.abs(relative_total_variation_map(pred, padding='SAME') -
                         relative_total_variation_map(gt, padding='SAME')) ** 2
    return tf.reduce_mean(discrepancy)


def relative_total_variation_loss(im, win_size=9, sigma=3.0):
    rtv_map = relative_total_variation_map(im, win_size, sigma)
    return tf.reduce_mean(rtv_map)


def sobel_filter(oritention='y'):
    f = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    if oritention == 'x':
        f = np.transpose(f)
    f = np.reshape(f, [3, 3, 1, 1])
    f = np.tile(f, [1, 1, 3, 3])
    return f


def tf_sobel_filter(oritention='y'):
    return tf.Variable(tf.convert_to_tensor(sobel_filter(oritention)))


# using sobel filter to compute image gradient (like edges)
def tf_im_gradient(im):
    im_sobel_y = tf.nn.conv2d(im, tf_sobel_filter('y'), strides=[1, 1, 1, 1], padding='SAME')
    im_sobel_x = tf.nn.conv2d(im, tf_sobel_filter('x'), strides=[1, 1, 1, 1], padding='SAME')
    im_gradient = tf.sqrt(im_sobel_y**2+im_sobel_x**2)
    return im_gradient


def bce_weighted(pred, gt, mask, eps=1e-5):
    # unknown
    # loss_unknown = gt * tf.log(1 + tf.exp(-pred)) * mask
    # loss_known = (1 - gt) * (pred + tf.log(1 + tf.exp(-pred))) * (1 - mask)
    pred = tf.reduce_mean(pred, axis=[0], keep_dims=True)

    h, w, c = pred.get_shape().as_list()[1:]
    cnt = h * w * c
    unknown_w = tf.reduce_sum(mask, [1, 2, 3]) / cnt
    known_w = 1 - unknown_w

    coef = known_w / (unknown_w+eps)
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(gt, pred, coef))
    # loss = tf.reduce_mean(loss_unknown / (unknown_w+eps) + loss_known / (known_w+eps))
    return loss


def geometric_consistent_loss(pred, gt, spatial_variant_w=None, win_size=5, sigma=2.0):

    gt_rtv_map = relative_total_variation_map(gt, win_size, sigma, padding='SAME')

    gt_gradient = tf_im_gradient(gt)
    pred_gradient = tf_im_gradient(pred)

    g = tf_make_guass_var(11, 1.5)
    g = tf.tile(g, [1, 1, 3, 3])

    gt_gradient_smooth = tf.nn.conv2d(gt_gradient, g, strides=[1, 1, 1, 1], padding='SAME')
    pred_gradient_smooth = tf.nn.conv2d(pred_gradient, g, strides=[1, 1, 1, 1], padding='SAME')

    if spatial_variant_w is None:
        spatial_variant_w = tf.ones_like(gt_rtv_map)

    gc_loss = tf.reduce_mean(tf.abs(gt_rtv_map * (pred_gradient_smooth - gt_gradient_smooth) * spatial_variant_w))
    return gc_loss


def semantic_consistent_loss(pred, gt, config, reuse=False):
    vgg = Vgg19(filepath=config.vgg19_path)
    src_vgg = vgg.build_vgg19((pred + 1) * 127.5, reuse=reuse)
    dst_vgg = vgg.build_vgg19((gt + 1) * 127.5, reuse=True)

    # semantic_loss = tf.reduce_mean(tf.abs(src_vgg['conv5_2'] - dst_vgg['conv5_2']))
    semantic_loss = tf.reduce_mean((src_vgg['conv3_2'] - dst_vgg['conv3_2']) ** 2)
    return semantic_loss


def spatial_variant_loss(pred, gt, spatial_variant_w=None):
    h, w = pred.get_shape().as_list()[1:3]
    if spatial_variant_w is None:
        spatial_variant_w = tf.ones((1, h, w, 1))
    else:
        spatial_variant_w = tf.image.resize_nearest_neighbor(spatial_variant_w, [h, w])

    return tf.reduce_mean(tf.abs(pred - gt) * spatial_variant_w)


def perceptual_loss(pred, gt, config, reuse=False):
    vgg = Vgg19(filepath=config.vgg19_path)
    src_vgg = vgg.build_vgg19((pred + 1) * 127.5, reuse=reuse)
    dst_vgg = vgg.build_vgg19((gt + 1) * 127.5, reuse=True)

    layers = config.perceptual_layers

    ploss = [w * tf.reduce_mean(tf.abs(src_vgg[layer] - dst_vgg[layer])) for layer, w in layers.items()]
    return tf.reduce_mean(ploss)


def perceptual_loss_spatial_variant(pred, gt, config, spatial_variant_w=None, reuse=False):
    vgg = Vgg19(filepath=config.vgg19_path)
    src_vgg = vgg.build_vgg19((pred + 1) * 127.5, reuse=reuse)
    dst_vgg = vgg.build_vgg19((gt + 1) * 127.5, reuse=True)

    layers = config.perceptual_layers

    ploss = [w * spatial_variant_loss(src_vgg[layer], dst_vgg[layers], spatial_variant_w)
             for layer, w in layers.items()]
    return tf.reduce_mean(ploss)


"""
id-mrf
"""
from enum import Enum

class Distance(Enum):
    L2 = 0
    DotProduct = 1

class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=3):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = tf.exp((self.b - scaled_distances) / self.sigma, name='weights_before_normalization')
        self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)

    def reversed_direction_CS(self):
        cs_flow_opposite = CSFlow(self.sigma, self.b)
        cs_flow_opposite.raw_distances = self.raw_distances
        work_axis = [1, 2]
        relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
        cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
        return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.1), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        with tf.name_scope('CS'):
            sT = T_features.shape.as_list()
            sI = I_features.shape.as_list()

            Ivecs = tf.reshape(I_features, (sI[0], -1, sI[3]))
            Tvecs = tf.reshape(T_features, (sI[0], -1, sT[3]))
            r_Ts = tf.reduce_sum(Tvecs * Tvecs, 2)
            r_Is = tf.reduce_sum(Ivecs * Ivecs, 2)
            raw_distances_list = []
            for i in range(sT[TensorAxis.N]):
                Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
                A = tf.matmul(Tvec,tf.transpose(Ivec))
                cs_flow.A = A
                # A = tf.matmul(Tvec, tf.transpose(Ivec))
                r_T = tf.reshape(r_T, [-1, 1])  # turn to column vector
                dist = r_T - 2 * A + r_I
                cs_shape = sI[:3] + [dist.shape[0].value]
                cs_shape[0] = 1
                dist = tf.reshape(tf.transpose(dist), cs_shape)
                # protecting against numerical problems, dist should be positive
                dist = tf.maximum(float(0.0), dist)
                # dist = tf.sqrt(dist)
                raw_distances_list += [dist]

            cs_flow.raw_distances = tf.convert_to_tensor([tf.squeeze(raw_dist, axis=0) for raw_dist in raw_distances_list])

            relative_dist = cs_flow.calc_relative_distances()
            cs_flow.__calculate_CS(relative_dist)
            return cs_flow

    #--
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(1.0), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        with tf.name_scope('CS'):
            # prepare feature before calculating cosine distance
            T_features, I_features = cs_flow.center_by_T(T_features, I_features)
            with tf.name_scope('TFeatures'):
                T_features = CSFlow.l2_normalize_channelwise(T_features)
            with tf.name_scope('IFeatures'):
                I_features = CSFlow.l2_normalize_channelwise(I_features)
                # work seperatly for each example in dim 1
                cosine_dist_l = []
                N, _, _, _ = T_features.shape.as_list()
                for i in range(N):
                    T_features_i = tf.expand_dims(T_features[i, :, :, :], 0)
                    I_features_i = tf.expand_dims(I_features[i, :, :, :], 0)
                    patches_i = cs_flow.patch_decomposition(T_features_i)
                    cosine_dist_i = tf.nn.conv2d(I_features_i, patches_i, strides=[1, 1, 1, 1],
                                                        padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
                    cosine_dist_l.append(cosine_dist_i)

                cs_flow.cosine_dist = tf.concat(cosine_dist_l, axis = 0)

                cosine_dist_zero_to_one = -(cs_flow.cosine_dist - 1) / 2
                cs_flow.raw_distances = cosine_dist_zero_to_one

                relative_dist = cs_flow.calc_relative_distances()
                cs_flow.__calculate_CS(relative_dist)
                return cs_flow

    def calc_relative_distances(self, axis=3):
        epsilon = 1e-5
        div = tf.reduce_min(self.raw_distances, axis=axis, keep_dims=True)
        # div = tf.reduce_mean(self.raw_distances, axis=axis, keep_dims=True)
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    def weighted_average_dist(self, axis=3):
        if not hasattr(self, 'raw_distances'):
            raise exception('raw_distances property does not exists. cant calculate weighted average l2')

        multiply = self.raw_distances * self.cs_NHWC
        return tf.reduce_sum(multiply, axis=axis, name='weightedDistPerPatch')

    # --
    @staticmethod
    def create(I_features, T_features, distance : Distance, nnsigma=float(1.0), b=float(1.0)):
        if distance.value == Distance.DotProduct.value:
            cs_flow = CSFlow.create_using_dotP(I_features, T_features, nnsigma, b)
        elif distance.value == Distance.L2.value:
            cs_flow = CSFlow.create_using_L2(I_features, T_features, nnsigma, b)
        else:
            raise "not supported distance " + distance.__str__()
        return cs_flow

    @staticmethod
    def sum_normalize(cs, axis=3):
        reduce_sum = tf.reduce_sum(cs, axis, keep_dims=True, name='sum')
        return tf.divide(cs, reduce_sum, name='sumNormalized')

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size

        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT, self.varT = tf.nn.moments(
            T_features, axes, name='TFeatures/moments')
        # we do not divide by std since its causing the histogram
        # for the final cs to be very thin, so the NN weights
        # are not distinctive, giving similar values for all patches.
        # stdT = tf.sqrt(varT, "stdT")
        # correct places with std zero
        # stdT[tf.less(stdT, tf.constant(0.001))] = tf.constant(1)
        with tf.name_scope('TFeatures/centering'):
            self.T_features_centered = T_features - self.meanT
        with tf.name_scope('IFeatures/centering'):
            self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = tf.norm(features, ord='euclidean', axis=3, name='norm')
        # expanding the norms tensor to support broadcast division
        norms_expanded = tf.expand_dims(norms, 3)
        features = tf.divide(features, norms_expanded, name='normalized')
        return features

    def patch_decomposition(self, T_features):
        # patch decomposition
        patch_size = 1
        patches_as_depth_vectors = tf.extract_image_patches(
            images=T_features, ksizes=[1, patch_size, patch_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID',
            name='patches_as_depth_vectors')

        self.patches_NHWC = tf.reshape(
            patches_as_depth_vectors,
            shape=[-1, patch_size, patch_size, patches_as_depth_vectors.shape[3].value],
            name='patches_PHWC')

        self.patches_HWCN = tf.transpose(
            self.patches_NHWC,
            perm=[1, 2, 3, 0],
            name='patches_HWCP')  # tf.conv2 ready format

        return self.patches_HWCN


def mrf_loss(T_features, I_features, distance=Distance.DotProduct, nnsigma=float(1.0)):
    T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
    I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)

    with tf.name_scope('cx'):
        cs_flow = CSFlow.create(I_features, T_features, distance, nnsigma)
        # sum_normalize:
        height_width_axis = [1, 2]
        # To:
        cs = cs_flow.cs_NHWC
        k_max_NC = tf.reduce_max(cs, axis=height_width_axis)
        CS = tf.reduce_mean(k_max_NC, axis=[1])
        CS_as_loss = 1 - CS
        CS_loss = -tf.log(1 - CS_as_loss)
        CS_loss = tf.reduce_mean(CS_loss)
        return CS_loss


def random_sampling(tensor_in, n, indices=None):
    N, H, W, C = tf.convert_to_tensor(tensor_in).shape.as_list()
    S = H * W
    tensor_NSC = tf.reshape(tensor_in, [N, S, C])
    all_indices = list(range(S))
    shuffled_indices = tf.random_shuffle(all_indices)
    indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices
    res = tf.gather(tensor_NSC, indices, axis=1)
    return res, indices


def random_pooling(feats, output_1d_size=100):
    is_input_tensor = type(feats) is tf.Tensor

    if is_input_tensor:
        feats = [feats]

    # convert all inputs to tensors
    feats = [tf.convert_to_tensor(feats_i) for feats_i in feats]

    N, H, W, C = feats[0].shape.as_list()
    feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2)
    res = [feats_sampled_0]
    for i in range(1, len(feats)):
        feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
        res.append(feats_sampled_i)

    res = [tf.reshape(feats_sampled_i, [N, output_1d_size, output_1d_size, C]) for feats_sampled_i in res]
    if is_input_tensor:
        return res[0]
    return res


def crop_quarters(feature_tensor):
    N, fH, fW, fC = feature_tensor.shape.as_list()
    quarters_list = []
    quarter_size = [N, round(fH / 2), round(fW / 2), fC]
    quarters_list.append(tf.slice(feature_tensor, [0, 0, 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, 0, round(fW / 2), 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), round(fW / 2), 0], quarter_size))
    feature_tensor = tf.concat(quarters_list, axis=0)
    return feature_tensor


def id_mrf_reg_feat(feat_A, feat_B, config):
    if config.crop_quarters is True:
        feat_A = crop_quarters(feat_A)
        feat_B = crop_quarters(feat_B)

    N, fH, fW, fC = feat_A.shape.as_list()
    if fH * fW <= config.max_sampling_1d_size ** 2:
        print(' #### Skipping pooling ....')
    else:
        print(' #### pooling %d**2 out of %dx%d' % (config.max_sampling_1d_size, fH, fW))
        feat_A, feat_B = random_pooling([feat_A, feat_B], output_1d_size=config.max_sampling_1d_size)

    return mrf_loss(feat_A, feat_B, distance=config.Dist, nnsigma=config.nn_stretch_sigma)


#from easydict import EasyDict as edict
# scale of im_src and im_dst: [-1, 1]
def id_mrf_reg(im_src, im_dst, config):
    vgg = Vgg19(filepath=config.vgg19_path)

    src_vgg = vgg.build_vgg19((im_src + 1) * 127.5)
    dst_vgg = vgg.build_vgg19((im_dst + 1) * 127.5, reuse=True)

    feat_style_layers = config.feat_style_layers
    feat_content_layers = config.feat_content_layers

    mrf_style_w = config.mrf_style_w
    mrf_content_w = config.mrf_content_w

    mrf_config = edict()
    mrf_config.crop_quarters = False
    mrf_config.max_sampling_1d_size = 65
    mrf_config.Dist = Distance.DotProduct
    mrf_config.nn_stretch_sigma = 0.5  # 0.1

    mrf_style_loss = [w * id_mrf_reg_feat(src_vgg[layer], dst_vgg[layer], mrf_config)
                      for layer, w in feat_style_layers.items()]
    mrf_style_loss = tf.reduce_sum(mrf_style_loss)

    mrf_content_loss = [w * id_mrf_reg_feat(src_vgg[layer], dst_vgg[layer], mrf_config)
                        for layer, w in feat_content_layers.items()]
    mrf_content_loss = tf.reduce_sum(mrf_content_loss)

    id_mrf_loss = mrf_style_loss * mrf_style_w + mrf_content_loss * mrf_content_w
    return id_mrf_loss
import argparse
import os
import time

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.dataset_training_paths = {
            'celebahq': '/data/home/litingwang/download/celeba_hq_images/train.txt',
            'celeba': '/data/yiwang/download/img_align_celeba/keys.txt',
            'places2': '/data/yiwang/proj/place2/train.txt',
            'imagenet': '/data/xyshen/Datasets/ImageNet/train/keys.txt',
            'paris_streetview': '/data/yiwang/download/paris_streetview/paris_train_original/keys.txt',
            'places2full': '/data/yiwang/download/train_large.txt',
            'adobe_5k': '/data/yiwang/download/adobe_5k_train.txt',
            'celeba_wild': '/data/yiwang/download/train_celeba_wild.txt'
        }

    def initialize(self):
        self.parser.add_argument('--dataset', type=str, default='paris_streetview', help='The dataset of the experiment.')
        self.parser.add_argument('--data_file', type=str, default='', help='the file storing training file paths')
        self.parser.add_argument('--data_noise', type=str, default='', help='the file storing noisy file paths')
        self.parser.add_argument('--data_noise_aux', type=str, default='', help='the file storing additional noisy file paths')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=str, default='', help='pretrained models are given here')
        self.parser.add_argument('--model_prefix', type=str, default='snap', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='vcn', help='which model does we use.')
        self.parser.add_argument('--phase', type=str, default='acc', help='the training phase we are in (acc | tune | full)')
        
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--random_mask', type=int, default=1)
        self.parser.add_argument('--mask_type', type=str, default='rect')
        self.parser.add_argument('--random_crop', type=int, default=0)
        self.parser.add_argument('--pretrain_network', action='store_true')
        self.parser.add_argument('--gan_loss_alpha', type=float, default=1e-3)
        self.parser.add_argument('--wgan_gp_lambda', type=float, default=10)
        self.parser.add_argument('--pretrain_l1_alpha', type=float, default=1.2)
        self.parser.add_argument('--l1_loss_alpha', type=float, default=1.4)
        self.parser.add_argument('--ae_loss_alpha', type=float, default=1.2)
        self.parser.add_argument('--mask_loss_alpha', type=float, default=2.0)
        self.parser.add_argument('--rho', type=float, default=0.5, help='control ratio for context normalization')
        self.parser.add_argument('--semantic_loss_alpha', type=float, default=1e-4)
        self.parser.add_argument('--geometric_loss_alpha', type=float, default=1e-3)
        self.parser.add_argument('--mrf_alpha', type=float, default=0.1)
        self.parser.add_argument('--gc_cfd', type=float, default=0.8)
        self.parser.add_argument('--gc_smooth', type=float, default=2e-4)
        self.parser.add_argument('--random_seed', type=bool, default=False)
        self.parser.add_argument('--random_alpha', type=int, default=0)
        self.parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')
        self.parser.add_argument('--cfd_alpha', type=float, default=1.0, help='multiplier of confidence processing')
        self.parser.add_argument('--cfd_magic', type=float, default=0.25, help='the interactive number in the formula')
        self.parser.add_argument('--cfd_type', type=str, default='Poly')
        self.parser.add_argument('--use_cn', type=int, default=0)
        self.parser.add_argument('--cn_type', type=str, default='v1', choices=['v1', 'se', 'old'], help='[v1|se|old]')
        self.parser.add_argument('--use_mrf', type=int, default=0)
        self.parser.add_argument('--use_blend', type=int, default=1)
        self.parser.add_argument('--embrace', type=int, default=0, help='use the all data or not, default is not')
        self.parser.add_argument('--critic', type=str, default='single')

        self.parser.add_argument('--train_spe', type=int, default=1000)
        self.parser.add_argument('--max_iters', type=int, default=40000)
        self.parser.add_argument('--viz_steps', type=int, default=5)

        self.parser.add_argument('--img_shapes', type=str, default='256,256,3',
                                 help='given shape parameters: h,w,c or h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='128,128',
                                 help='given mask parameters: h,w')
        self.parser.add_argument('--max_delta_shapes', type=str, default='32,32')
        self.parser.add_argument('--margins', type=str, default='0,0')
        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=64,
                                 help='# of discriminator filters in first conv layer')
        self.parser.add_argument('--d_iters', type=int, default=5)

        self.parser.add_argument('--vgg19_path', type=str, default='vgg19_weights/imagenet-vgg-verydeep-19.mat')

        self.parser.add_argument('--parts', type=int, default=8)
        self.parser.add_argument('--brush_width', type=int, default=20)
        self.parser.add_argument('--brush_length', type=int, default=80)
        self.parser.add_argument('--vertex', type=int, default=16)

        # data configuration
        self.parser.add_argument('--paired', type=int, default=0)
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.dataset_path = \
            self.dataset_training_paths[self.opt.dataset] if self.opt.data_file == '' else self.opt.data_file

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(str(id))

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.random_crop in [0, 1]
        self.opt.random_crop = True if self.opt.random_crop == 1 else False

        assert self.opt.random_alpha in [0, 1]
        self.opt.random_alpha = True if self.opt.random_alpha == 1 else False

        assert self.opt.use_cn in [0, 1]
        self.opt.use_cn = True if self.opt.use_cn == 1 else False

        assert self.opt.cn_type in ['v1', 'se', 'old']

        assert self.opt.use_mrf in [0, 1]
        self.opt.use_mrf = True if self.opt.use_mrf == 1 else False

        assert self.opt.use_blend in [0, 1]
        self.opt.use_blend = True if self.opt.use_blend == 1 else False

        assert self.opt.embrace in [0, 1]
        self.opt.embrace = True if self.opt.embrace == 1 else False

        assert self.opt.paired in [0, 1]
        self.opt.paired = True if self.opt.paired == 1 else False

        assert self.opt.critic in ['single', 'double']

        self.opt.data_noise_aux = None if self.opt.data_noise_aux == '' else self.opt.data_noise_aux

        assert self.opt.mask_type in ['rect', 'stroke']

        assert self.opt.cfd_type in ['Shepard', 'CrossEntropy', 'Poly', 'Exp']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        str_max_delta_shapes = self.opt.max_delta_shapes.split(',')
        self.opt.max_delta_shapes = [int(x) for x in str_max_delta_shapes]

        str_margins = self.opt.margins.split(',')
        self.opt.margins = [int(x) for x in str_margins]

        # model name and date
        self.opt.date_str = time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_name = self.opt.model
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.model_name
        self.opt.model_folder += '_' + self.opt.dataset
        self.opt.model_folder += '_b' + str(self.opt.batch_size)
        self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        self.opt.model_folder += '_dc' + str(self.opt.d_cnum)
        self.opt.model_folder += '_r' + str(self.opt.rho*10)

        self.opt.model_folder += '_randmask-' + self.opt.mask_type if self.opt.random_mask else ''
        self.opt.model_folder += '_RM' if self.opt.embrace else ''
        self.opt.model_folder += '_pretrain' if self.opt.pretrain_network else ''

        if os.path.isdir(self.opt.checkpoints_dir) is False:
            os.mkdir(self.opt.checkpoints_dir)

        self.opt.model_folder = os.path.join(self.opt.checkpoints_dir, self.opt.model_folder)
        if os.path.isdir(self.opt.model_folder) is False:
            os.mkdir(self.opt.model_folder)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.opt.gpu_ids)

        return self.opt

    def __string__(self):
        args = vars(self.opt)
        doc = '------------ Options -------------\n'
        for k, v in sorted(args.items()):
            doc += f'{str(k)}: {str(v)}\n'
        doc += '-------------- End ----------------'

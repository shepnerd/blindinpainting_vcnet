import argparse
import os
import time

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataset', type=str, default='paris_streetview',
                                 help='The dataset of the experiment.')
        self.parser.add_argument('--data_file', type=str, default='./imgs/paris-streetview_256x256', help='the file storing testing file paths')
        self.parser.add_argument('--test_dir', type=str, default='./test_results', help='models are saved here')
        self.parser.add_argument('--load_model_dir', type=str, default='./checkpoints', help='pretrained models are given here')
        self.parser.add_argument('--model_prefix', type=str, default='snap')
        self.parser.add_argument('--seed', type=int, default=1, help='random seed')

        self.parser.add_argument('--use_noise', type=int, default=0)
        self.parser.add_argument('--data_noise', type=str, default='')
        self.parser.add_argument('--data_noise_aux', type=str, default='')
        self.parser.add_argument('--use_blend', type=int, default=1)
        self.parser.add_argument('--embrace', type=int, default=0, help='use the all data or not, default is not')
        self.parser.add_argument('--rho', type=float, default=0.5, help='control ratio in context normalization')

        self.parser.add_argument('--model', type=str, default='vcn')
        self.parser.add_argument('--random_mask', type=int, default=0,
                                 help='using random mask')

        self.parser.add_argument('--use_cn', type=int, default=1)
        self.parser.add_argument('--cn_type', type=str, default='v1', choices=['v1', 'se', 'old'], help='[v1|se|old]')
        self.parser.add_argument('--use_mrf', type=int, default=0)
        self.parser.add_argument('--phase', type=str, default='tune')
        self.parser.add_argument('--img_shapes', type=str, default='256,256,3',
                                 help='given shape parameters: h,w,c or h,w')
        self.parser.add_argument('--mask_shapes', type=str, default='128,128',
                                 help='given mask parameters: h,w')
        self.parser.add_argument('--mask_type', type=str, default='stroke')
        self.parser.add_argument('--test_num', type=int, default=-1)
        self.parser.add_argument('--mode', type=str, default='save')
        self.parser.add_argument('--save_intermediate', type=int, default=0)

        # for generator
        self.parser.add_argument('--g_cnum', type=int, default=32,
                                 help='# of generator filters in first conv layer')
        self.parser.add_argument('--d_cnum', type=int, default=64,
                                 help='# of discriminator filters in first conv layer')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.data_file != '':
            self.opt.dataset_path = self.opt.data_file

        if os.path.exists(self.opt.test_dir) is False:
            os.mkdir(self.opt.test_dir)

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.use_cn in [0, 1]
        self.opt.use_cn = True if self.opt.use_cn == 1 else False

        assert self.opt.use_mrf in [0, 1]
        self.opt.use_mrf = True if self.opt.use_mrf == 1 else False

        assert self.opt.use_noise in [0, 1]
        self.opt.use_noise = True if self.opt.use_noise == 1 else False

        assert self.opt.use_blend in [0, 1]
        self.opt.use_blend = True if self.opt.use_blend == 1 else False

        assert self.opt.save_intermediate in [0, 1]
        self.opt.save_intermediate = True if self.opt.save_intermediate == 1 else False

        assert self.opt.embrace in [0, 1]
        self.opt.embrace = True if self.opt.embrace == 1 else False

        assert self.opt.mask_type in ['rect', 'stroke']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        # model name and date
        self.opt.date_str = 'test_'+time.strftime('%Y%m%d-%H%M%S')
        self.opt.model_folder = self.opt.date_str + '_' + self.opt.dataset + '_' + self.opt.model
        self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        self.opt.model_folder += '_r' + str(10 * self.opt.rho)
        self.opt.model_folder += '_randmask-' + self.opt.mask_type if self.opt.random_mask else ''
        self.opt.model_folder += '_RM' if self.opt.embrace else ''
        if self.opt.random_mask:
            self.opt.model_folder += '_seed-' + str(self.opt.seed)
        self.opt.saving_path = os.path.join(self.opt.test_dir, self.opt.model_folder)

        if os.path.exists(self.opt.saving_path) is False and self.opt.mode == 'save':
            os.mkdir(self.opt.saving_path)

        return self.opt

    def __string__(self):
        args = vars(self.opt)
        doc = '------------ Options -------------\n'
        for k, v in sorted(args.items()):
            doc += f'{str(k)}: {str(v)}\n'
        doc += '-------------- End ----------------'
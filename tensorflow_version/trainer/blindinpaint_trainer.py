import os
import tensorflow as tf
from net.model import Blindinpaint_model
from data.data import DataLoader
from options.train_options import TrainOptions


class BlindInpaint_Trainer:
    def __init__(self, config):
        self.config = config

        self.dataLoader = DataLoader(file_in=config.dataset_path, file_out=config.data_noise,
                                     batch_size=config.batch_size, im_size=config.img_shapes,
                                     file_aux=config.data_noise_aux, paired=config.paired)

        self.model = Blindinpaint_model(config)

        self.g_optimizer, self.d_optimizer = self.build_optimizers(config)

        self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    def build_optimizers(self, config):
        lr = tf.get_variable('lr', shape=[], trainable=False, initializer=tf.constant_initializer(config.lr))

        g_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5,
                                             beta2=0.9) if config.model != 'gatedconv' \
            else tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
        d_optimizer = g_optimizer

        config.d_iters = 1 if config.model == 'gatedconv' else config.d_iters
        return g_optimizer, d_optimizer

    def run(self):
        images = self.dataLoader.next()

        gt, noise = tf.split(images, 2, axis=3)

        g_vars, d_vars, losses = self.model.get_training_losses(gt, noise)

        g_train_op = self.g_optimizer.minimize(losses['g_loss'], var_list=g_vars)

        if self.config.pretrain_network is False:
            d_train_op = self.d_optimizer.minimize(losses['d_loss'], var_list=d_vars)

        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if self.config.load_model_dir != '':
                print('[-] Loading the pretrained model from: {}'.format(self.config.load_model_dir))
                ckpt = tf.train.get_checkpoint_state(self.config.load_model_dir)
                if ckpt:
                    # saver.restore(sess, tf.train.latest_checkpoint(config.load_model_dir))
                    assign_ops = list(
                        map(lambda x: tf.assign(x,
                                                tf.contrib.framework.load_variable(self.config.load_model_dir, x.name)),
                            g_vars))
                    sess.run(assign_ops)
                    print('-- load G')

                    if self.config.pretrain_network is False:
                        try:
                            assign_ops = list(
                            map(lambda x: tf.assign(x,
                                                    tf.contrib.framework.load_variable(self.config.load_model_dir, x.name)),
                                d_vars))
                            sess.run(assign_ops)
                            print('-- load D')
                        except RuntimeError:
                            pass

                    print("[*] Loading SUCCESS.")
                else:
                    print("[x] Loading ERROR.")

            summary_writer = tf.summary.FileWriter(self.config.model_folder, sess.graph, flush_secs=30)

            coord = tf.train.Coordinator()
            thread = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(1, self.config.max_iters + 1):

                if self.config.pretrain_network is False:
                    for _ in range(self.config.d_iters):
                        _, d_loss = sess.run([d_train_op, losses['d_loss']])

                _, g_loss = sess.run([g_train_op, losses['g_loss']])

                if step % self.config.viz_steps == 0:
                    print('[{:04d}, {:04d}] G_loss > {}'.format(step // self.config.train_spe,
                                                                step % self.config.train_spe,
                                                                g_loss))

                    if self.config.pretrain_network is False:
                        print(' '*12 + f' D_loss > {d_loss}')
                    
                    summary_writer.add_summary(sess.run(summary_op), global_step=step)

                if step % self.config.train_spe == 0:
                    self.saver.save(sess, os.path.join(self.config.model_folder, self.config.model_prefix), step)
                    print(f'-- save model in {os.path.join(self.config.model_folder, self.config.model_prefix)}')

            coord.request_stop()
            coord.join(thread)

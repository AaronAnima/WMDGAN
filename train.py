import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_dataset_train
from models import get_G, get_E, get_z_D, get_trans_func, get_img_D
import random
import argparse
import math
import scipy.stats as stats
import tensorflow_probability as tfp

import ipdb


G = get_G([None, 4, 4, 512])
D = get_img_D([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
E = get_E([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
D_z = get_z_D([None, flags.z_dim])
C = get_z_D([None, flags.z_dim])
f_ab = get_trans_func([None, flags.z_dim])
f_ba = get_trans_func([None, flags.z_dim])
D_zA = get_z_D([None, flags.z_dim])
D_zB = get_z_D([None, flags.z_dim])


def KStest(real_z, fake_z):
    p_list = []
    for i in range(flags.batch_size_train):
        _, tmp_p = stats.ks_2samp(fake_z[i], real_z[i])
        p_list.append(tmp_p)
    return np.min(p_list), np.mean(p_list)


def train_GE(con=False):
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset
    print(con)
    if con:
        G.load_weights('./checkpoint/G.npz')
        D.load_weights('./checkpoint/D.npz')
        E.load_weights('./checkpoint/E.npz')
        D_z.load_weights('./checkpoint/Dz.npz')

    G.train()
    D.train()
    E.train()
    D_z.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    lr_G = flags.lr_G
    lr_E = flags.lr_E
    lr_D = flags.lr_D
    lr_Dz = flags.lr_Dz

    d_optimizer = tf.optimizers.Adam(lr_D, beta_1=flags.beta1, beta_2=flags.beta2)
    g_optimizer = tf.optimizers.Adam(lr_G, beta_1=flags.beta1, beta_2=flags.beta2)
    e_optimizer = tf.optimizers.Adam(lr_E, beta_1=flags.beta1, beta_2=flags.beta2)
    dz_optimizer = tf.optimizers.Adam(lr_Dz, beta_1=flags.beta1, beta_2=flags.beta2)
    # tfd = tfp.distributions
    # dist_normal = tfd.Normal(loc=0., scale=1.)
    # dist_Bernoulli = tfd.Bernoulli(probs=0.5)
    # dist_beta = tfd.Beta(0.5, 0.5)
    eval_batch = None
    for step, image_labels in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        if step == 0:
            eval_batch  = image_labels[0]
            tl.visualize.save_images(eval_batch.numpy(), [8, 8],
                                     '{}/eval_samples.png'.format(flags.sample_dir,
                                                                       step // n_step_epoch, step))
        batch_imgs = image_labels[0]

        # ipdb.set_trace()

        epoch_num = step // n_step_epoch
        with tf.GradientTape(persistent=True) as tape:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            # tl.visualize.save_images(batch_imgs.numpy(), [8, 8],
            #                          '{}/raw_samples.png'.format(flags.sample_dir))
            # fake_z = E(batch_imgs)
            # fake_imgs = G(fake_z)
            # fake_logits = D(fake_imgs)
            # real_logits = D(batch_imgs)
            # fake_logits_z = D(G(z))
            # real_z_logits = D_z(z)
            # fake_z_logits = D_z(fake_z)
            #
            # e_loss_z = - tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.zeros_like(fake_z_logits)) + \
            #            tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.ones_like(fake_z_logits))
            #
            # recon_loss = flags.lamba_recon * tl.cost.absolute_difference_error(batch_imgs, fake_imgs, is_mean=True)
            # g_loss_x = - tl.cost.sigmoid_cross_entropy(fake_logits, tf.zeros_like(fake_logits)) + \
            #            tl.cost.sigmoid_cross_entropy(fake_logits, tf.ones_like(fake_logits))
            # g_loss_z = - tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.zeros_like(fake_logits_z)) + \
            #            tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.ones_like(fake_logits_z))
            #
            # e_loss = recon_loss + e_loss_z
            # g_loss = recon_loss + g_loss_x + g_loss_z
            #
            # d_loss = tl.cost.sigmoid_cross_entropy(real_logits, tf.ones_like(real_logits)) + \
            #          tl.cost.sigmoid_cross_entropy(fake_logits, tf.zeros_like(fake_logits)) + \
            #          tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.zeros_like(fake_logits_z))
            #
            # dz_loss = tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.zeros_like(fake_z_logits)) + \
            #           tl.cost.sigmoid_cross_entropy(real_z_logits, tf.ones_like(real_z_logits))
            fake_z = E(batch_imgs)
            recon_x = G(fake_z)
            recon_loss = tl.cost.absolute_difference_error(batch_imgs, recon_x, is_mean=True)
            latent_loss = tl.cost.absolute_difference_error
            e_loss = recon_loss
            g_loss = recon_loss
        # Updating Encoder
        grad = tape.gradient(e_loss, E.trainable_weights)
        e_optimizer.apply_gradients(zip(grad, E.trainable_weights))

        # Updating Generator
        grad = tape.gradient(g_loss, G.trainable_weights)
        g_optimizer.apply_gradients(zip(grad, G.trainable_weights))

        # # Updating Discriminator
        # grad = tape.gradient(d_loss, D.trainable_weights)
        # d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
        #
        # # Updating D_z & D_h
        # grad = tape.gradient(dz_loss, D_z.trainable_weights)
        # dz_optimizer.apply_gradients(zip(grad, D_z.trainable_weights))

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] e_loss: {:.5f}, g_loss: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, e_loss, g_loss))
            # Kstest
            # p_min, p_avg = KStest(z, fake_z)
            # print("kstest: min:{}, avg:{}", p_min, p_avg)

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
            D.save_weights('{}/D.npz'.format(flags.checkpoint_dir), format='npz')
            E.save_weights('{}/E.npz'.format(flags.checkpoint_dir), format='npz')
            D_z.save_weights('{}/Dz.npz'.format(flags.checkpoint_dir), format='npz')
            # G.train()
        if np.mod(step, flags.eval_step) == 0 and step != 0:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            G.eval()
            recon_imgs = G(E(eval_batch))
            G.train()
            # tl.visualize.save_images(result.numpy(), [8, 8],
            #                          '{}/sample_{:02d}_{:04d}.png'.format(flags.sample_dir,
            #                                                                 step // n_step_epoch, step))
            tl.visualize.save_images(recon_imgs.numpy(), [8, 8],
                                     '{}/recon_{:02d}_{:04d}.png'.format(flags.sample_dir,
                                                                            step // n_step_epoch, step))
        del tape


def train_C():
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset
    # G.load_weights('./checkpoint/{}/G.npz'.format(flags.param_dir))
    # E.load_weights('./checkpoint/{}/E.npz'.format(flags.param_dir))
    G.eval()
    E.eval()
    C.train()

    c_optimizer = tf.optimizers.Adam(flags.lr_C, beta_1=flags.beta1, beta_2=flags.beta2)

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)
    acc_sum = 0
    acc_step = 0
    for step, image_labels in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        batch_imgs = image_labels[0]
        batch_labels = image_labels[1]
        batch_labels = (batch_labels + 1) / 2
        batch_labels = tf.reshape(batch_labels, [flags.batch_size_train, 1])
        batch_labels = tf.cast(batch_labels, tf.float32)
        epoch_num = step // n_step_epoch

        with tf.GradientTape(persistent=True) as tape:
            real_z = E(batch_imgs)
            z_logits = C(real_z)
            z_logits = tf.cast(z_logits, tf.float32)
            loss_c = tl.cost.sigmoid_cross_entropy(z_logits, batch_labels)
            logits = ((tf.sign(z_logits * 2 - 1) + 1)/2)
            labels = batch_labels
            acc_num = flags.batch_size_train + tf.reduce_sum(logits * labels) - tf.reduce_sum(logits) \
                      - tf.reduce_sum(labels)
            batch_acc = acc_num / flags.batch_size_train
            acc_sum += batch_acc
            acc_step += 1
        # Updating Encoder
        grad = tape.gradient(loss_c, C.trainable_weights)
        c_optimizer.apply_gradients(zip(grad, C.trainable_weights))
        del tape

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] batch_acc is {}, loss_c: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, batch_acc, loss_c))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            C.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')

        if np.mod(step, flags.acc_step) == 0 and step != 0:
            acc = acc_sum / acc_step
            print("The avg step_acc is {:.3f} in {} step".format(acc, acc_step))
            acc_sum = 0
            acc_step = 0


def train_F_D():
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset

    G.load_weights('./checkpoint/G.npz')
    E.load_weights('./checkpoint/E.npz')
    f_ab.train()
    f_ba.train()
    D_zA.train()
    D_zB.train()
    G.eval()
    E.eval()

    f_optimizer = tf.optimizers.Adam(flags.lr_F, beta_1=flags.beta1, beta_2=flags.beta2)
    d_optimizer = tf.optimizers.Adam(flags.lr_D, beta_1=flags.beta1, beta_2=flags.beta2)

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    for step, batch_imgs in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        image_a = batch_imgs[0]
        image_b = batch_imgs[1]
        epoch_num = step // n_step_epoch

        with tf.GradientTape(persistent=True) as tape:
            za_real = E()
        # Updating Encoder
        grad = tape.gradient(loss_c, C.trainable_weights)
        c_optimizer.apply_gradients(zip(grad, C.trainable_weights))
        del tape

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] batch_acc is {}, loss_c: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, batch_acc, loss_c))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            C.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')

        if np.mod(step, flags.acc_step) == 0 and step != 0:
            acc = acc_sum / acc_step
            print("The avg step_acc is {:.3f} in {} step".format(acc, acc_step))
            acc_sum = 0
            acc_step = 0


#### Main part, opt z to translate raw img to another latent manifold ####
def opt_z():
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset

    G.eval()
    E.eval()
    C.eval()

    z_optimizer = tf.optimizers.Adam(lr_C, beta_1=flags.beta1, beta_2=flags.beta2)


    for step, image_labels in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        if step >= flags.sample_cnt:
            break
        print('Now start {} img'.format(str(step)))
        batch_imgs = image_labels[0]

        batch_labels = image_labels[1]
        batch_labels = (batch_labels + 1) / 2
        tran_label = tf.ones_like(batch_labels) - batch_labels

        for step_z in range(flags.step_z):
            with tf.GradientTape(persistent=True) as tape:

                real_z = E(batch_imgs)
                z = tf.Variable(real_z)
                z_logits = C(z)
                loss_z = tl.cost.sigmoid_cross_entropy(z_logits, tran_label)

            # Updating Encoder
            grad = tape.gradient(loss_z, z.trainable_weights)
            z_optimizer.apply_gradients(zip(grad, z.trainable_weights))
            opt_img = G(z)
            del tape
            if np.mod(step_z, 10) == 0:
                tl.visualize.save_images(opt_img.numpy(), [8, 8],
                                         '{}/opt_img{:02d}_step{:02d}.png'.format(flags.opt_sample_dir, step, step_z))


        tl.visualize.save_images(batch_imgs.numpy(), [8, 8],
                                 '{}/raw_img{:02d}.png'.format(flags.opt_sample_dir, step))
        tl.visualize.save_images(opt_img.numpy(), [8, 8],
                                 '{}/opt_img{:02d}.png'.format(flags.opt_sample_dir, step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DWGAN', help='train or eval')
    parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
    args = parser.parse_args()
    train_GE(con=args.is_continue)
    train_C()
    opt_z()

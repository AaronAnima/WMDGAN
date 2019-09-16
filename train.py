import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_dataset_train
from models import get_G, get_E, get_z_D, get_img_D, get_z_G
import random
import argparse
import math
import scipy.stats as stats
import tensorflow_probability as tfp

import ipdb
# import sys
# f = open('a.log', 'a')
# sys.stdout = f
# sys.stderr = f # redirect std err, if necessary



G = get_G([None, 4, 4, 512])
E = get_E([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
D_z = get_z_D([None, 4, 4, 512])
G_z = get_z_G([None, flags.z_dim])


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
        E.load_weights('./checkpoint/E.npz')

    G.train()
    E.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    lr_G = flags.lr_G
    lr_E = flags.lr_E

    g_optimizer = tf.optimizers.Adam(lr_G, beta_1=flags.beta1, beta_2=flags.beta2)
    e_optimizer = tf.optimizers.Adam(lr_E, beta_1=flags.beta1, beta_2=flags.beta2)
    eval_batch = None
    for step, image_labels in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        if step == 0:
            eval_batch  = image_labels[0]
            tl.visualize.save_images(eval_batch.numpy(), [8, 8],
                                     '{}/eval_samples.png'.format(flags.sample_dir, step // n_step_epoch, step))
        batch_imgs = image_labels[0]

        # ipdb.set_trace()

        epoch_num = step // n_step_epoch
        with tf.GradientTape(persistent=True) as tape:
            tl.visualize.save_images(batch_imgs.numpy(), [8, 8],
                                     '{}/raw_samples.png'.format(flags.sample_dir))
            fake_z = E(batch_imgs)
            recon_x = G(fake_z)
            recon_loss = tl.cost.absolute_difference_error(batch_imgs, recon_x, is_mean=True)
            reg_loss = tf.math.maximum(tl.cost.mean_squared_error(fake_z, tf.zeros_like(fake_z)), 1)
            # reg_loss = tl.cost.mean_squared_error(tl.cost.mean_squared_error(fake_z, tf.zeros_like(fake_z)), 1)
            e_loss = flags.lamba_recon * recon_loss + reg_loss
            g_loss = flags.lamba_recon * recon_loss

        # Updating Encoder
        grad = tape.gradient(e_loss, E.trainable_weights)
        e_optimizer.apply_gradients(zip(grad, E.trainable_weights))

        # Updating Generator
        grad = tape.gradient(g_loss, G.trainable_weights)
        g_optimizer.apply_gradients(zip(grad, G.trainable_weights))

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] e_loss: {:.5f}, g_loss: {:.5f}, recon_loss: {:.5f}, reg_loss: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, e_loss, g_loss, recon_loss, reg_loss))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
            E.save_weights('{}/E.npz'.format(flags.checkpoint_dir), format='npz')
            # G.train()
        if np.mod(step, flags.eval_step) == 0 and step != 0:
            # z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            G.eval()
            E.eval()
            recon_imgs = G(E(eval_batch))
            G.train()
            E.train()
            tl.visualize.save_images(recon_imgs.numpy(), [8, 8],
                                     '{}/recon_{:02d}_{:04d}.png'.format(flags.sample_dir, step // n_step_epoch, step))
        del tape


def train_Gz():
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset
    G.load_weights('./checkpoint/G.npz')
    E.load_weights('./checkpoint/E.npz')
    G.eval()
    E.eval()
    G_z.train()
    D_z.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)

    lr_Dz = flags.lr_Dz
    lr_Gz = flags.lr_Gz

    dt_optimizer = tf.optimizers.Adam(lr_Dz, beta_1=flags.beta1, beta_2=flags.beta2)
    gt_optimizer = tf.optimizers.Adam(lr_Gz, beta_1=flags.beta1, beta_2=flags.beta2)
    for step, image_labels in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        batch_imgs = image_labels[0]

        epoch_num = step // n_step_epoch
        with tf.GradientTape(persistent=True) as tape:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            fake_tensor = G_z(z)
            real_tensor = E(batch_imgs)

            fake_tensor_logits = D_z(fake_tensor)
            real_tensor_logits = D_z(real_tensor)

            gt_loss = tl.cost.sigmoid_cross_entropy(fake_tensor_logits, tf.ones_like(fake_tensor_logits))
            dt_loss = tl.cost.sigmoid_cross_entropy(real_tensor_logits, tf.ones_like(real_tensor_logits)) + \
                      tl.cost.sigmoid_cross_entropy(fake_tensor_logits, tf.zeros_like(fake_tensor_logits))
        # Updating Generator
        grad = tape.gradient(gt_loss, G_z.trainable_weights)
        gt_optimizer.apply_gradients(zip(grad, G_z.trainable_weights))
        #
        # Updating D_z & D_h
        grad = tape.gradient(dt_loss, D_z.trainable_weights)
        dt_optimizer.apply_gradients(zip(grad, D_z.trainable_weights))

        # basic
        if np.mod(step, flags.show_freq) == 0 and step != 0:
            print("Epoch: [{}/{}] [{}/{}] dt_loss: {:.5f}, gt_loss: {:.5f}".format
                  (epoch_num, n_epoch, step, n_step_epoch, dt_loss, gt_loss))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            G_z.save_weights('{}/G_z.npz'.format(flags.checkpoint_dir), format='npz')
            D_z.save_weights('{}/D_z.npz'.format(flags.checkpoint_dir), format='npz')
            # G.train()
        if np.mod(step, flags.eval_step) == 0 and step != 0:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            G.eval()
            sample_tensor = G_z(z)
            sample_img = G(sample_tensor)
            G.train()
            tl.visualize.save_images(sample_img.numpy(), [8, 8],
                                     '{}/sample_{:02d}_{:04d}.png'.format(flags.sample_dir,
                                                                         step // n_step_epoch, step))
        del tape


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DWGAN', help='train or eval')
    parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
    args = parser.parse_args()
    # train_GE(con=args.is_continue)
    train_Gz()

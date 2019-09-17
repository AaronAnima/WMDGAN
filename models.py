import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, BatchNorm, Concat, GaussianNoise
from config import flags
from utils import SpectralNormConv2d


def get_G(shape_z):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    ni = Input(shape_z)
    # input size is [None, 8, 8, 128]
    nn = DeConv2d(128, (1, 1), (1, 1), W_init=w_init, b_init=None, act=None)(ni)

    nn = DeConv2d(64, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = DeConv2d(32, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = DeConv2d(32, (3, 3), (1, 1), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = DeConv2d(32, (3, 3), (1, 1), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = DeConv2d(3, (4, 4), (2, 2), act=tf.nn.tanh, W_init=w_init, b_init=None)(nn)

    return tl.models.Model(inputs=ni, outputs=nn)


# E is reverse of G without activation in the output layer
def get_E(shape_z):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    ni = Input(shape_z)
    # output size is [None, 8, 8, 128]

    nn = Conv2d(32, (3, 3), (1, 1), W_init=w_init, b_init=None)(ni)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = Conv2d(32, (3, 3), (1, 1), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = Conv2d(32, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = Conv2d(64, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = Conv2d(128, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)

    nn = Conv2d(128, (1, 1), (1, 1), W_init=w_init, b_init=None, act=None)(nn)

    return tl.models.Model(inputs=ni, outputs=nn)


def get_z_D(shape_z):
    w_init = tf.random_normal_initializer(stddev=0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    nz = Input(shape_z)
    # 8 8 128
    n = SpectralNormConv2d(128, (3, 3), (1, 1), act=lrelu, W_init=w_init, padding='VALID')(nz)
    # 6 6 128
    n = SpectralNormConv2d(128, (3, 3), (1, 1), act=lrelu, W_init=w_init)(n)
    # 6 6 128
    n = SpectralNormConv2d(256, (3, 3), (1, 1), act=lrelu, W_init=w_init, padding='VALID')(n)
    # 4 4 256
    n = SpectralNormConv2d(512, (4, 4), (1, 1), act=lrelu, W_init=w_init, padding='VALID')(n)
    # 1 1 512
    n = Reshape(shape=[-1, 512])(n)
    n = Dense(n_units=1, act=tf.identity, W_init=w_init, b_init=None)(n)
    return tl.models.Model(inputs=nz, outputs=n)


# 4 * 4 * 512 = 128 * 4 * 4 * 4 G is the transpose of D
def get_z_G(shape_z):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    # lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    nz = Input(shape_z)
    n = Dense(n_units=4 * 4 * 256, W_init=w_init, b_init=None, act=None)(nz)
    n = Reshape(shape=[-1, 4, 4, 256])(n)
    n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(n)

    n = DeConv2d(128, (3, 3), (1, 1), W_init=w_init, padding='VALID', b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(n)

    n = DeConv2d(128, (3, 3), (1, 1), W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(n)

    n = Conv2d(128, (3, 3), (1, 1), W_init=w_init, padding='VALID')(n)

    return tl.models.Model(inputs=nz, outputs=n)


# def get_trans_func(shape=[None, flags.z_dim]):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     act = 'relu'  # lambda x : tf.nn.leaky_relu(x, 0.2)
#
#     ni = Input(shape)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(ni)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(nn)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(nn)
#     nn = Dense(n_units=flags.z_dim, W_init=w_init)(nn)
#
#     return tl.models.Model(inputs=ni, outputs=nn)


# def get_img_D(shape):
#     df_dim = 8
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
#     ni = Input(shape)
#     n = Conv2d(df_dim, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(df_dim * 2, (5, 5), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(df_dim * 4, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(df_dim * 8, (5, 5), (1, 1), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(df_dim * 8, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     nf = Flatten(name='flatten')(n)
#     n = Dense(n_units=1, act=None, W_init=w_init)(nf)
#     return tl.models.Model(inputs=ni, outputs=n, name='img_Discriminator')



#
# def get_classifier(shape=[None, flags.z_dim], df_dim=64):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     act = 'relu'  # lambda x : tf.nn.leaky_relu(x, 0.2)
#
#     ni = Input(shape)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(ni)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(nn)
#     nn = Dense(n_units=flags.z_dim, act=act, W_init=w_init)(nn)
#     nn = Dense(n_units=1, W_init=w_init)(nn)
#
#     return tl.models.Model(inputs=ni, outputs=nn)


# def get_G(shape_z, gf_dim=64):    # Dimension of gen filters in first conv layer. [64]
#
#     image_size = 32
#     s16 = image_size // 16
#     # w_init = tf.glorot_normal_initializer()
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#
#     ni = Input(shape_z)
#     nn = Dense(n_units=(gf_dim * 16 * s16 * s16), W_init=w_init, b_init=None)(ni)
#     nn = Reshape(shape=[-1, s16, s16, gf_dim * 16])(nn) # [-1, 2, 2, gf_dim * 8]
#     nn = BatchNorm(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
#     nn = DeConv2d(gf_dim * 8, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 4, 4, gf_dim * 4]
#     nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
#     nn = DeConv2d(gf_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 8, 8, gf_dim * 2]
#     nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
#     nn = DeConv2d(gf_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=None)(nn) # [-1, 16, 16, gf_dim *]
#     nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
#     nn = DeConv2d(gf_dim, (5, 5), (2, 2), b_init=None, W_init=w_init)(nn) # [-1, 32, 32, 3]
#     nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
#     nn = DeConv2d(3, (5, 5), (2, 2), act=tf.nn.tanh, W_init=w_init)(nn)  # [-1, 64, 64, 3]
#
#     return tl.models.Model(inputs=ni, outputs=nn, name='generator')

# def get_E(shape):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     gamma_init = tf.random_normal_initializer(1., 0.02)
#     lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
#     ni = Input(shape)   # (1, 64, 64, 3)
#     n = Conv2d(3, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)  # (1, 16, 16, 3)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(32, (5, 5), (1, 1), padding="VALID", act=None, W_init=w_init, b_init=None)(n)  # (1, 12, 12, 32)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Conv2d(64, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)  # (1, 6, 6, 64)
#     n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
#     n = Flatten(name='flatten')(n)
#     nz = Dense(n_units=flags.z_dim, act=None, W_init=w_init)(n)
#     return tl.models.Model(inputs=ni, outputs=nz, name='encoder')

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:57:57 2018

@author: yamane
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda
from chainer import optimizers

import utils

# Original implementation: https://github.com/chainer/chainer/tree/master/examples/vae
class AAE(chainer.Chain):
    """Variational AutoEncoder"""
    def __init__(self, n_in, n_latent, n_h):
        super(AAE, self).__init__(
            # encoder
            le1 = L.Linear(n_in, n_h),
            le2 = L.Linear(n_h, n_h),
            le3 = L.Linear(n_h,  n_latent),
            # decoder
            ld1 = L.Linear(n_latent, n_h),
            ld2 = L.Linear(n_h, n_h),
            ld3 = L.Linear(n_h, n_in))
        self.dim_z = n_in

    def __call__(self, x, sigmoid=True):
        """ AutoEncoder """
        return self.decoder(self.encoder(x))

    def encoder(self, x):
        h1 = F.relu(self.le1(x))
        h2 = F.relu(self.le2(h1))
        h3 = self.le3(h2)
        return h3

    def decoder(self, z):
        h1 = F.relu(self.ld1(z))
        h2 = F.relu(self.ld2(h1))
        h3 = self.ld3(h2)
        return h3

    def draw_sample(self, size=100):
        z = self.xp.random.uniform(-1, 1, size=(size, self.dim_z)).astype('f')
        return self.encoder(z)


class Discriminator(chainer.Chain):
    def __init__(self, c_in=2):
        c_1 = 200
        c_2 = 200
        super(Discriminator, self).__init__(
            fc1=L.Linear(c_in, c_1),
            fc2=L.Linear(c_1, c_2),
            fc3=L.Linear(c_2, 1))

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def swiss_roll(batchsize, ndim, num_labels):
	def sample(label, num_labels):
		uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
		r = math.sqrt(uni) * 3.0
		rad = np.pi * 4.0 * np.sqrt(uni)
		x = r * math.cos(rad)
		y = r * math.sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, ndim), dtype=np.float32)
	for batch in range(batchsize):
		for zi in range(ndim // 2):
			z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, num_labels - 1), num_labels)
	return z


if __name__ == '__main__':
    # settings
    gpu = 0
    n_epoch = 1000
    n_in = 784
    n_latent = 2
    n_h = 500
    batchsize = 200
    real_var = 2
    real_mean = 0
    update_interval_g = 1
    update_interval_d = 1

    test = False
    graph_gen = True

    # Prepare dataset
    x_train, x_test, y_train, y_test, N, N_test, N_train = utils.load_mnist_dataset(test)

#    x_real = np.random.normal(real_mean, real_var, (batchsize, 50)).astype(np.float32)
    x_real = swiss_roll(batchsize, 2, 10)
    ds_real = chainer.datasets.TupleDataset(x_real)
    it_real = chainer.iterators.SerialIterator(ds_real, batchsize)

    # Prepare AAE model
    model = AAE(n_in, n_latent, n_h)
    discriminator = Discriminator()

    if gpu >= 0:
        cuda.get_device_from_id(gpu).use()
        model.to_gpu()
        discriminator.to_gpu()
    xp = np if gpu < 0 else cuda.cupy

    # Setup optimizer
    opt_encoder = optimizers.Adam()
    opt_encoder.setup(model)
    opt_discriminator = optimizers.Adam()
    opt_discriminator.setup(discriminator)

    epoch_loss_g = []
    epoch_loss_d = []
    for epoch in range(1, n_epoch + 1):
        print('epoch', epoch)
        loss_g_list = []
        loss_d_list = []
        # training
        perm = np.random.permutation(N)
        for i in range(0, N, batchsize):
            x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))

            t_fake = cuda.to_gpu(np.zeros((len(x), 1), dtype=np.int32))
            t_real = cuda.to_gpu(np.ones((len(x), 1), dtype=np.int32))

            z_fake = model.encoder(x)
            z_real = cuda.to_gpu(chainer.dataset.concat_examples(next(it_real))[0])
            z_real = chainer.Variable(z_real)

            y_fake = discriminator(z_fake)
            y_real = discriminator(z_real)

            with chainer.using_config('train', True):
                # generation loss
                loss_g = F.sigmoid_cross_entropy(y_fake, t_real)
                # discrimination loss
                loss_d = F.sigmoid_cross_entropy(y_real, t_real) + F.sigmoid_cross_entropy(y_fake, t_fake)

            # generator  phase
            if epoch % update_interval_g == 0:
                # 勾配を初期化
                model.cleargrads()
                with chainer.using_config('train', True):
                    loss_g.backward()
                opt_encoder.update()

            # adversarial phase
            if epoch % update_interval_d == 0:
                # 勾配を初期化
                discriminator.cleargrads()
                with chainer.using_config('train', True):
                    # 逆伝搬を計算
                    loss_d.backward()
                opt_discriminator.update()

            loss_g_list.append(cuda.to_cpu(loss_g.data))
            loss_d_list.append(cuda.to_cpu(loss_d.data))
        epoch_loss_g.append(np.mean(loss_g_list))
        epoch_loss_d.append(np.mean(loss_d_list))

        plt.plot(epoch_loss_g)
        plt.title("generation loss")
        plt.legend(["loss_g"], loc="upper right")
        plt.grid()
        plt.show()
        print('generation loss:', epoch_loss_g[epoch-1])

        plt.plot(epoch_loss_d)
        plt.title("discrimination loss")
        plt.legend(["loss_d"], loc="upper right")
        plt.grid()
        plt.show()
        print('discrimination loss:', epoch_loss_d[epoch-1])

        n_sample = 5000
        x_sample = x_test[:n_sample]
#        real_sample = real_std * np.random.randn(n_sample, 2).astype(np.float32) + real_mean
        real_sample = swiss_roll(batchsize, 2, 10)

        z = model.encoder(chainer.cuda.to_gpu(x_sample))
        z.to_cpu()
        z = z.data

        plt.figure(figsize=(10, 8))
        plt.title('real sample')
        plt.scatter(real_sample[:, 0], real_sample[:, 1], alpha=0.6)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.title('generated sample')
        plt.scatter(z[:, 0], z[:, 1], alpha=0.6)
        plt.grid()
        plt.show()
        print('update_interval', update_interval_g)


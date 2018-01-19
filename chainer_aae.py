# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 20:38:47 2018

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

from dataset import Dataset
import sampler
import utils


class AdvarsarialAutoEncoder(chainer.Chain):
    """Variational AutoEncoder"""
    def __init__(self, ndim_x=28*28, ndim_y=20, ndim_z=2, ndim_h=1000):
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.ndim_z = ndim_z
        self.ndim_h = ndim_h
        super(AdvarsarialAutoEncoder, self).__init__(
            # encoder
            le1 = L.Linear(ndim_x, ndim_h),
            le2 = L.Linear(ndim_h, ndim_h),
            head_y = L.Linear(ndim_h, ndim_y),
            head_z = L.Linear(ndim_h, ndim_z),
            cluster_head = L.Linear(ndim_y, ndim_z, nobias=True),
            # decoder
            ld1 = L.Linear(ndim_z, ndim_h),
            ld2 = L.Linear(ndim_h, ndim_h),
            ld3 = L.Linear(ndim_h, ndim_x),
            # discriminator_y
            ly1=L.Linear(ndim_y, ndim_h),
            ly2=L.Linear(ndim_h, ndim_h),
            ly3=L.Linear(ndim_h, 2),
            # discriminator_z
            lz1=L.Linear(ndim_z, ndim_h),
            lz2=L.Linear(ndim_h, ndim_h),
            lz3=L.Linear(ndim_h, 2))

    def __call__(self, x, sigmoid=True):
        """ AutoEncoder """
        y, z = self.encode_x_yz(x)
        representation = self.encode_yz_representation(y, z)
        x_reconstruction = self.decode_representation_x(representation)
        return x_reconstruction

    def encoder(self, x):
        h = F.relu(self.le1(x))
        return F.relu(self.le2(h))

    def decoder(self, representation):
        h = F.relu(self.ld1(representation))
        h = F.relu(self.ld2(h))
        return F.tanh(self.ld3(h))

    def encode_x_yz(self, x, apply_softmax_y=True):
        internal = self.encoder(x)
        y = self.head_y(internal)
        z = self.head_z(internal)
        if apply_softmax_y:
            y = F.softmax(y)
        return y, z

    def encode_yz_representation(self, y, z):
        cluster_head = self.cluster_head(y)
        return cluster_head + z

    def decode_representation_x(self, representation):
        return self.decoder(representation)

    def discriminator_y(self, y):
        h = F.relu(self.ly1(y))
        h = F.relu(self.ly2(h))
        return self.ly3(h)

    def discriminator_z(self, z):
        h = F.relu(self.lz1(z))
        h = F.relu(self.lz2(h))
        return self.lz3(h)

    def discriminate_y(self, y, apply_softmax=False):
        logit = self.discriminator_y(y)
        if apply_softmax:
            return F.softmax(logit)
        return logit

    def discriminate_z(self, z, apply_softmax=False):
        logit = self.discriminator_z(z)
        if apply_softmax:
            return F.softmax(logit)
        return logit


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
    n_epoch = 5000
    batchsize = 100
    ndim_x = 28 * 28
    ndim_y = 20
    ndim_z = 2
    ndim_h = 1000

    alpha = 0.001

    test = False
    graph_gen = True

    # Prepare VAE model
    model = AdvarsarialAutoEncoder(ndim_x, ndim_y, ndim_z, ndim_h)

    mnist_train, mnist_test = chainer.datasets.get_mnist()
    images_train, labels_train = mnist_train._datasets
    images_test, labels_test = mnist_test._datasets

    # normalize
    images_train = (images_train - 0.5) * 2
    images_test = (images_test - 0.5) * 2

    dataset = Dataset(train=(images_train, labels_train), test=(images_test, labels_test))

    total_iterations_train = len(images_train) // batchsize

    # Setup optimizer
    optimizer = optimizers.Adam(alpha)
    optimizer.setup(model)

    using_gpu = False
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
        using_gpu = True
    xp = model.xp

    # 0 -> true sample
    # 1 -> generated sample
    class_true = np.zeros(batchsize, dtype=np.int32)
    class_fake = np.ones(batchsize, dtype=np.int32)

    if using_gpu:
        class_true = cuda.to_gpu(class_true)
        class_fake = cuda.to_gpu(class_fake)

    epoch_loss_autoencoder = []
    epoch_loss_generator = []
    epoch_loss_discriminator = []

    for epoch in range(1, n_epoch + 1):
        print(epoch)
        loss_ae_list = []
        loss_g_list = []
        loss_d_list = []
        # training
        for itr in range(total_iterations_train):
            with chainer.using_config('train', True):
                x_u, _, _ = dataset.sample_minibatch(batchsize, gpu=using_gpu)

                ### reconstruction phase ###
                if True:
                    y_onehot_u, z_u = model.encode_x_yz(x_u, apply_softmax_y=True)
                    repr_u = model.encode_yz_representation(y_onehot_u, z_u)
                    x_reconstruction_u = model.decode_representation_x(repr_u)
                    loss_reconstruction = F.mean_squared_error(x_u, x_reconstruction_u)

                    model.cleargrads()
                    loss_reconstruction.backward()
                    optimizer.update()

                ### adversarial phase ###
                if True:
                    y_onehot_fake_u, z_fake_u = model.encode_x_yz(x_u, apply_softmax_y=True)

#                    z_true = sampler.gaussian(batchsize, model.ndim_z, mean=0, var=1)
                    z_true = sampler.swiss_roll(batchsize, model.ndim_z, 10)
                    y_onehot_true = sampler.onehot_categorical(batchsize, model.ndim_y)
                    if using_gpu:
                        z_true = cuda.to_gpu(z_true)
                        y_onehot_true = cuda.to_gpu(y_onehot_true)

                    dz_true = model.discriminate_z(z_true, apply_softmax=False)
                    dz_fake = model.discriminate_z(z_fake_u, apply_softmax=False)
                    dy_true = model.discriminate_y(y_onehot_true, apply_softmax=False)
                    dy_fake = model.discriminate_y(y_onehot_fake_u, apply_softmax=False)

                    discriminator_z_confidence_true = float(xp.mean(F.softmax(dz_true).data[:, 0]))
                    discriminator_z_confidence_fake = float(xp.mean(F.softmax(dz_fake).data[:, 1]))
                    discriminator_y_confidence_true = float(xp.mean(F.softmax(dy_true).data[:, 0]))
                    discriminator_y_confidence_fake = float(xp.mean(F.softmax(dy_fake).data[:, 1]))

                    loss_discriminator_z = F.softmax_cross_entropy(dz_true, class_true) + F.softmax_cross_entropy(dz_fake, class_fake)
                    loss_discriminator_y = F.softmax_cross_entropy(dy_true, class_true) + F.softmax_cross_entropy(dy_fake, class_fake)
                    loss_discriminator = loss_discriminator_z + loss_discriminator_y

                    model.cleargrads()
                    loss_discriminator.backward()
                    optimizer.update()

                ### generator phase ###
                if True:
                    y_onehot_fake_u, z_fake_u = model.encode_x_yz(x_u, apply_softmax_y=True)

                    dz_fake = model.discriminate_z(z_fake_u, apply_softmax=False)
                    dy_fake = model.discriminate_y(y_onehot_fake_u, apply_softmax=False)

                    loss_generator = F.softmax_cross_entropy(dz_fake, class_true) + F.softmax_cross_entropy(dy_fake, class_true)

                    model.cleargrads()
                    loss_generator.backward()
                    optimizer.update()

                ### additional cost ###
#                if True:
#                    distance = model.compute_distance_of_cluster_heads()
#                    loss_cluster_head = -F.sum(distance)
#
#                    model.cleargrads()
#                    loss_cluster_head.backward()
#                    optimizer_cluster_head.update()

            loss_ae_list.append(cuda.to_cpu(loss_reconstruction.data))
            loss_g_list.append(cuda.to_cpu(loss_generator.data))
            loss_d_list.append(cuda.to_cpu(loss_discriminator.data))
        epoch_loss_autoencoder.append(np.mean(loss_ae_list))
        epoch_loss_generator.append(np.mean(loss_g_list))
        epoch_loss_discriminator.append(np.mean(loss_d_list))

        if epoch % 1 == 0:
            plt.plot(epoch_loss_autoencoder)
            plt.title("reconstruction loss")
            plt.legend(["loss_rec"], loc="upper right")
            plt.grid()
            plt.show()
            print('reconstraction loss:', epoch_loss_autoencoder[epoch-1])

            plt.plot(epoch_loss_generator)
            plt.title("generation loss")
            plt.legend(["loss_g"], loc="upper right")
            plt.grid()
            plt.show()
            print('g loss:', epoch_loss_generator[epoch-1])

            plt.plot(epoch_loss_discriminator)
            plt.title("discrimination loss")
            plt.legend(["loss_d"], loc="upper right")
            plt.grid()
            plt.show()
            print('d loss:', epoch_loss_discriminator[epoch-1])

        if epoch % 1 == 0:
            # OriginalとReconstrauctionの比較
            x_test, _, _ = dataset.test(batchsize, gpu=using_gpu)
            utils.draw_result(model, x_test)

        n_sample = 1000
        x_test, y_test, _ = dataset.test(n_sample, gpu=using_gpu)
        y_test = cuda.to_cpu(y_test)
#        z_true = sampler.gaussian(n_sample, model.ndim_z, mean=0, var=1)
        z_true = sampler.swiss_roll(n_sample, model.ndim_z, 10)

        _, z_fake = model.encode_x_yz(x_test)
        z_fake.to_cpu()
        z_fake = z_fake.data

        plt.figure(figsize=(10, 8))
        plt.title('real sample')
        plt.scatter(z_true[:, 0], z_true[:, 1], alpha=0.6)
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.title('fake sample')
        plt.scatter(z_fake[:, 0], z_fake[:, 1], c=y_test, cmap="rainbow", alpha=0.6)
        for i in range(10):
            m = utils.get_mean(z_fake, y_test, i)
            plt.text(m[0], m[1], "{}".format(i), fontsize=20)
        plt.grid()
        plt.show()

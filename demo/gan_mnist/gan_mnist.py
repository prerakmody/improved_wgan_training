import os, sys
import time
import matplotlib
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

import tensorflow as tf
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

# lib.print_model_settings(locals().copy())

class GAN_MNIST():
    def __init__(self, mode = 'wgan-gp', dim = 64, output_dim = 28 * 28, batch_size = 50, lambdaa = 10):  
        """
        mode : str
            - dcgan, wgan, or wgan-gp
        """
        self.mode = mode
        self.dim = dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lambdaa = lambdaa

        self.loss()

        self.train_gen, self.dev_gen, self.test_gen = lib.mnist.load(self.batch_size, self.batch_size)
    
    def LeakyReLU(self, x, alpha=0.2):
        return tf.maximum(alpha*x, x)

    def ReLULayer(self, name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(
            name+'.Linear', 
            n_in, 
            n_out, 
            inputs,
            initialization='he'
        )
        return tf.nn.relu(output)

    def LeakyReLULayer(self, name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(
            name+'.Linear', 
            n_in, 
            n_out, 
            inputs,
            initialization='he'
        )
        return self.LeakyReLU(output)

    def Generator(self, n_samples, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 4 * self.dim, noise)
        if self.mode == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4 * self.dim, 4, 4])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * self.dim, 2 * self.dim, 5, output)
        if self.mode == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
        output = tf.nn.relu(output)

        output = output[:,:,:7,:7]

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * self.dim, self.dim, 5, output)
        if self.mode == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', self.dim, 1, 5, output)
        output = tf.nn.sigmoid(output)

        return tf.reshape(output, [-1, self.output_dim])

    def Discriminator(self, inputs):
        output = tf.reshape(inputs, [-1, 1, 28, 28])

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 1, self.dim, 5, output,stride=2)
        output = self.LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', self.dim, 2 * self.dim, 5, output, stride=2)
        if self.dim == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
        output = self.LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*self.dim, 4*self.dim, 5, output, stride=2)
        if self.mode == 'wgan':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
        output = self.LeakyReLU(output)

        output = tf.reshape(output, [-1, 4*4*4*self.dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*self.dim, 1, output)

        return tf.reshape(output, [-1])

    def loss(self):
        self.real_data = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dim])
        self.fake_data = self.Generator(self.batch_size)

        self.disc_real = self.Discriminator(self.real_data)
        self.disc_fake = self.Discriminator(self.fake_data)

        gen_params = lib.params_with_name('Generator')
        disc_params = lib.params_with_name('Discriminator')

        if self.mode == 'wgan':
            self.gen_cost = -tf.reduce_mean(self.disc_fake)
            self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

            self.gen_train_op = tf.train.RMSPropOptimizer(
                learning_rate=5e-5
            ).minimize(self.gen_cost, var_list=gen_params)
            self.disc_train_op = tf.train.RMSPropOptimizer(
                learning_rate=5e-5
            ).minimize(self.disc_cost, var_list=disc_params)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(
                    tf.assign(
                        var, 
                        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                    )
                )
            self.clip_disc_weights = tf.group(*clip_ops)

        elif self.mode == 'wgan-gp':
            self.gen_cost = -tf.reduce_mean(self.disc_fake)
            self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)

            alpha = tf.random_uniform(
                shape=[self.batch_size,1], 
                minval=0.,
                maxval=1.
            )
            differences = self.fake_data - self.real_data
            interpolates = self.real_data + (alpha*differences)
            gradients = tf.gradients(self.Discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1.)**2)
            self.disc_cost += self.lambdaa * gradient_penalty

            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, 
                beta1=0.5,
                beta2=0.9
            ).minimize(self.gen_cost, var_list=gen_params)
            self.disc_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, 
                beta1=0.5, 
                beta2=0.9
            ).minimize(self.disc_cost, var_list=disc_params)

            self.clip_disc_weights = None

        elif MODE == 'dcgan':
            pass
            # gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     disc_fake, 
            #     tf.ones_like(disc_fake)
            # ))

            # disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     disc_fake, 
            #     tf.zeros_like(disc_fake)
            # ))
            # disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     disc_real, 
            #     tf.ones_like(disc_real)
            # ))
            # disc_cost /= 2.

            # gen_train_op = tf.train.AdamOptimizer(
            #     learning_rate=2e-4, 
            #     beta1=0.5
            # ).minimize(gen_cost, var_list=gen_params)
            # disc_train_op = tf.train.AdamOptimizer(
            #     learning_rate=2e-4, 
            #     beta1=0.5
            # ).minimize(disc_cost, var_list=disc_params)

            # clip_disc_weights = None

        # For saving samples
        fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
        self.fixed_noise_samples = self.Generator(128, noise=fixed_noise)

    def generate_image(self, session, iteration, true_dist):
        samples = session.run(self.fixed_noise_samples)

        save_dir = os.path.join(os.getcwd(), 'gan_mnist_output')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_name = os.path.join(save_dir, 'frame' + str(iteration) + '.png')

        lib.save_images.save_images(
            samples.reshape((128, 28, 28)), 
            file_name
        )

    # Dataset iterator
    def inf_train_gen(self):
        while True:
            for images,targets in self.train_gen():
                yield images

    def train(self, critic_iters = 5, iters = 10000):
        # Train loop
        with tf.Session() as session:
            print ('\n ======= TRAIN =======')
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
            print (' ---> Initialized all variables')

            gen = self.inf_train_gen()

            for iteration in range(iters):
                if iteration == 0:
                    print (' ---> Network and data initialized. Let the training begin! And may the best weights win!')
                start_time = time.time()

                # TRAIN GENERATOR
                if iteration > 0:
                    _ = session.run(self.gen_train_op)

                # TRAIN DISCRIMINATOR
                if self.mode == 'dcgan':
                    disc_iters = 1
                else:
                    disc_iters = critic_iters

                for i in range(disc_iters):
                    _data = gen.__next__()
                    _disc_cost, _ = session.run(
                        [self.disc_cost, self.disc_train_op],
                        feed_dict={self.real_data: _data}
                    )
                    if self.clip_disc_weights is not None:
                        _ = session.run(self.clip_disc_weights)

                # LOGGING
                # lib.plot.plot('train disc cost', _disc_cost)
                # lib.plot.plot('time', time.time() - start_time)
                # Calculate dev loss and generate samples every 100 iters
                if iteration % 10 == 0:
                    dev_disc_costs = []
                    for images,_ in self.dev_gen():
                        _dev_disc_cost = session.run(
                            self.disc_cost, 
                            feed_dict={self.real_data: images}
                        )
                        dev_disc_costs.append(_dev_disc_cost)
                    print ('Iteration:', iteration, np.mean(dev_disc_costs))
                    # lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                    self.generate_image(session, iteration, _data)

                # Write logs every 100 iters
                # if (iteration < 5) or (iteration % 100 == 99):
                #     lib.plot.flush()

                # lib.plot.tick()

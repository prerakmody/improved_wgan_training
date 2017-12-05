import os, sys
import time
import random
import matplotlib
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

import tensorflow as tf
import tflib.plot
import tflib as lib
import tflib.ops.linear


# lib.print_model_settings(locals().copy())


class GAN_Toy():
    def __init__(self, mode = 'wgan-gp', dim = 512, lambdaa = 0.1, fixed_generator = False , batch_size = 256):
        self.mode = mode
        self.dim = dim
        self.lambdaa = lambdaa
        self.fixed_generator = fixed_generator
        self.batch_size = batch_size

        self.define()

    def ReLULayer(self, name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(
            name+'.Linear',
            n_in,
            n_out,
            inputs,
            initialization='he'
        )
        output = tf.nn.relu(output)
        return output

    def Generator(self, n_samples, real_data):
        if self.fixed_generator:
            return real_data + (1.*tf.random_normal(tf.shape(real_data)))
        else:
            noise = tf.random_normal([n_samples, 2])
            output = self.ReLULayer('Generator.1', 2, self.dim, noise)
            output = self.ReLULayer('Generator.2', self.dim, self.dim, output)
            output = self.ReLULayer('Generator.3', self.dim, self.dim, output)
            output = lib.ops.linear.Linear('Generator.4', self.dim, 2, output) #o/p has same dims as i/p
            return output

    def Discriminator(self, inputs):
        output = self.ReLULayer('Discriminator.1', 2, self.dim, inputs)
        output = self.ReLULayer('Discriminator.2', self.dim, self.dim, output)
        output = self.ReLULayer('Discriminator.3', self.dim, self.dim, output)
        output = lib.ops.linear.Linear('Discriminator.4', self.dim, 1, output) # o/p is [self.dim, 1]
        return tf.reshape(output, [-1]) # flattens the previous step

    def define(self):
        self.real_data = tf.placeholder(tf.float32, shape=[None, 2]) #2-dimensional data
        self.fake_data = self.Generator(self.batch_size, self.real_data)

        self.disc_real = self.Discriminator(self.real_data)
        self.disc_fake = self.Discriminator(self.fake_data)

        # WGAN loss
        print (' ---> Defining Disc and Gen Loss')
        self.disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
        self.gen_cost = -tf.reduce_mean(self.disc_fake)

        # WGAN gradient penalty
        if self.mode == 'wgan-gp':
            alpha = tf.random_uniform(
                shape=[self.batch_size,1], 
                minval=0.,
                maxval=1.
            )
            interpolates = alpha * self.real_data + ((1-alpha) * self.fake_data)  #Eh??
            disc_interpolates = self.Discriminator(interpolates)
            gradients = tf.gradients(disc_interpolates, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes-1)**2)
        
            self.disc_cost += self.lambdaa * gradient_penalty

        print (' ---> Aggregating all variables for Disc and Gen in two distinct variables')
        disc_params = lib.params_with_name('Discriminator')
        gen_params = lib.params_with_name('Generator')

        if self.mode == 'wgan-gp':
            print (' ---> Defining Optimizers for training')
            self.disc_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, 
                beta1=0.5, 
                beta2=0.9
            ).minimize(
                self.disc_cost, 
                var_list = disc_params
            )
            if len(gen_params) > 0:
                self.gen_train_op = tf.train.AdamOptimizer(
                    learning_rate=1e-4, 
                    beta1=0.5, 
                    beta2=0.9
                ).minimize(
                    self.gen_cost, 
                    var_list = gen_params
                )
            else:
                self.gen_train_op = tf.no_op()

        else:
            pass
            # disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
            #     disc_cost, 
            #     var_list=disc_params
            # )
            # if len(gen_params) > 0:
            #     gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
            #         gen_cost, 
            #         var_list=gen_params
            #     )
            # else:
            #     gen_train_op = tf.no_op()


            # # Build an op to do the weight clipping
            # clip_ops = []
            # for var in disc_params:
            #     clip_bounds = [-.01, .01]
            #     clip_ops.append(
            #         tf.assign(
            #             var, 
            #             tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            #         )
            #     )
            # clip_disc_weights = tf.group(*clip_ops)

            # print "Generator params:"
            # for var in lib.params_with_name('Generator'):
            #     print "\t{}\t{}".format(var.name, var.get_shape())
            # print "Discriminator params:"
            # for var in lib.params_with_name('Discriminator'):
            #     print "\t{}\t{}".format(var.name, var.get_shape())

    def generate_image(self, session, iteration, true_dist, dataset_type):
        """
        Generates and saves a plot of the true distribution, the generator, and the
        critic.
        """
        N_POINTS = 128
        RANGE = 3

        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
        points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
        points = points.reshape((-1,2))

        samples, disc_map = session.run(
            [self.fake_data, self.disc_real], 
            feed_dict={self.real_data : points}
        )
        disc_map = session.run(self.disc_real, feed_dict={self.real_data : points}) #this is repeated!!??

        plt.clf()

        x = y = np.linspace(-RANGE, RANGE, N_POINTS)
        plt.contour(x,y,disc_map.reshape((len(x), len(y))).transpose())

        plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+') # generated dataset
        plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+') # predicted dataset

        if iteration == 0:
            print ('\nTrue Dist (orange +) : ', true_dist.shape)
            print ('Samples (green +) : ', samples.shape)
            print ('Discrimnator Map : ', disc_map.shape)

        save_dir = os.path.join(os.getcwd(), 'gan_toy_output')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # file_name = os.path.join(save_dir, 'frame' + str(self.frame_index[0]) + '.jpg')
        file_name = os.path.join(save_dir, 'frame' + str(iteration) + '.png')
        plt.savefig(file_name)

    # Dataset iterator
    def inf_train_gen(self, dataset_type):
        if dataset_type == '25gaussians':
            dataset = []
            for i in range(100000/25):
                for x in range(-2, 3):
                    for y in range(-2, 3):
                        point = np.random.randn(2)*0.05
                        point[0] += 2*x
                        point[1] += 2*y
                        dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            np.random.shuffle(dataset)
            dataset /= 2.828 # stdev
            while True:
                for i in range(len(dataset) / self.batch_size):
                    yield dataset[i * self.batch_size : (i+1) * self.batch_size]

        elif dataset_type == 'swissroll':
            while True:
                data = sklearn.datasets.make_swiss_roll(
                    n_samples = self.batch_size, 
                    noise = 0.25
                )[0]
                data = data.astype('float32')[:, [0, 2]]
                data /= 7.5 # stdev plus a little
                yield data

        elif dataset_type == '8gaussians':
            scale = 2.
            centers = [
                (1,0),
                (-1,0),
                (0,1),
                (0,-1),
                (1./np.sqrt(2), 1./np.sqrt(2)),
                (1./np.sqrt(2), -1./np.sqrt(2)),
                (-1./np.sqrt(2), 1./np.sqrt(2)),
                (-1./np.sqrt(2), -1./np.sqrt(2))
            ]
            centers = [(scale*x,scale*y) for x,y in centers]
            while True:
                dataset = []
                for i in range(self.batch_size):
                    point = np.random.randn(2)*.02
                    center = random.choice(centers)
                    point[0] += center[0]
                    point[1] += center[1]
                    dataset.append(point)
                dataset = np.array(dataset, dtype='float32')
                dataset /= 1.414 # stdev (sqrt of 2) 
                yield dataset
        
        else:
            print ('Cannot generate this dataset type as of yet')

    def train(self, iters = 100000, critic_iters = 5, dataset_type = '8gaussians'):
        # Train loop!
        with tf.Session() as session:
            print ('\n ======= TRAINING ======= ')
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
            
            gen = self.inf_train_gen(dataset_type)
            
            for iteration in range(iters):
                # TRAIN GENERATOR
                if iteration > 0:
                    _ = session.run(self.gen_train_op)
                
                # TRAIN CRITIC
                for i in range(critic_iters):
                    _data = gen.__next__()  #[batch_size, 2]
                    _disc_cost, _ = session.run(
                        [self.disc_cost, self.disc_train_op],
                        feed_dict={self.real_data: _data}
                    )
                    # if self.mode == 'wgan':
                    #     _ = session.run([clip_disc_weights])
                
                # LOGGING
                # lib.plot.plot('disc cost', _disc_cost)
                if iteration % 10 == 0:
                    # lib.plot.flush()
                    print ('Iter:', iteration,' ---> Disc Cost:', _disc_cost)
                    self.generate_image(session, iteration, _data, dataset_type)
                # slib.plot.tick()

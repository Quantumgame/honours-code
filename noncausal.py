import tensorflow as tf

import numpy as np
from datetime import datetime
from PIL import Image

import time

def get_weights(shape):
    return tf.Variable(tf.contrib.layers.xavier_initializer()(shape))

def get_bias_weights(shape):
    return tf.Variable(tf.zeros_initializer()(shape))

class NonCausal:
    def gate(self, p):
        p1, p2 = tf.split(p, 2, -1)
        return tf.multiply(tf.tanh(p1), tf.sigmoid(p2))
        
    def conv2d(self, input, shape):
        assert len(shape) == 4
        W = get_weights(shape)
        b = get_bias_weights(shape[-1])
        return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') + b
        
    def fully_connected(self, input, in_features, out_features, bias=True):
        W = get_weights([1, 1, in_features, out_features])
        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID')
        if bias:
            out += get_bias_weights([out_features])
        return out    
        
    def layer(self, input, first_layer=False, last_layer=False):
        assert not (first_layer and last_layer)
        out = self.gate(self.conv2d(input, [self.filter_size, self.filter_size, self.channels if first_layer else self.features, 2*self.features]))
        residual = self.fully_connected(out, self.features, self.end_features if last_layer else self.features)
        skip = self.fully_connected(out, self.features, self.end_features)
        if first_layer:
            input = self.fully_connected(input, self.channels, self.features, bias=False)
        elif last_layer:
            input = self.fully_connected(input, self.features, self.end_features, bias=False)
        return input + residual, skip
        
    def end(self, outs):
        out = tf.nn.relu(tf.reduce_sum(outs, 0))
        out = tf.nn.relu(self.fully_connected(out, self.end_features, self.end_features))
        out = self.fully_connected(out, self.end_features, self.channels * self.values)
        return tf.reshape(out, shape=[self.batch_size, self.height, self.width, self.channels, self.values])
        
    def noncausal(self):
        X = self.X_in
        outs = []
        for i in range(self.layers):
            X, skip = self.layer(X, first_layer=(i==0), last_layer=(i==self.layers-1))
            outs.append(skip)
        outs.append(X)
        logits = self.end(outs)
        predictions = self.sample(logits)
        
        return logits, predictions
        
    def sample(self, logits):
        probabilities = logits / self.temperature
        return tf.reshape(tf.cast(tf.multinomial(tf.reshape(probabilities, shape=[self.batch_size*self.height*self.width*self.channels, self.values]), 1), tf.float32), shape=[self.batch_size,self.height,self.width,self.channels])
        
    def logitise(self, images):
        return tf.stop_gradient(tf.one_hot(tf.to_int32(images * (self.values-1)), self.values))
        
    def generate_ten_thousand_samples_graph(self, sess):
        print('Generating graph for noncausal')
        outer_samples = []
        test_data = self.data.get_noise_values()
        for b in range(self.data.total_test_batches):
            print('batch', b)
            inner_samples = []
            X_corrupted = sess.run(test_data)
            inner_samples.append(X_corrupted)
            for _ in range(40):#self.test_iterations):
                X_corrupted = sess.run(self.predictions, feed_dict={self.X_in:X_corrupted})
                inner_samples.append(X_corrupted)
            outer_samples.append(inner_samples)
        samples = np.concatenate(outer_samples, 1)
        print(samples.shape)
        return samples
        
    def generate_ten_thousand_samples(self, sess):
        from data.dataset import noise
        print('Generating 10000 samples for noncausal')
        start = time.perf_counter()
        samples = []
        test_data = self.data.get_noise_values()
        for j in range(self.data.total_test_batches):
            print('batch', j)
            X_corrupted = sess.run(test_data)
            for i in range(self.test_iterations):
                X_corrupted = sess.run(self.predictions, feed_dict={self.X_in:X_corrupted})
                if self.markov_type == 'renoise_per_two':
                    X_corrupted = sess.run(self.predictions, feed_dict={self.X_in:X_corrupted})
                if self.markov_type != 'only_denoise':
                    noise_prop = (self.test_iterations - i - 1) / self.test_iterations
                    X_corrupted = np.array([noise(arr, noise_prop, noise_prop)[0] for arr in X_corrupted])
            samples.append(X_corrupted)
        samples = np.concatenate(samples)
        print(time.perf_counter() - start)
        return samples
        
    def update_clamp(self, data, newdata, filename):
        if filename == 'topgap':
            data = data.copy()
            data[:,:data.shape[1]//2,:,:] = newdata[:,:data.shape[1]//2,:,:]
            return data
        elif filename == 'bottomgap':
            data = data.copy()
            data[:,data.shape[1]//2:,:,:] = newdata[:,data.shape[1]//2:,:,:]
            return data
        else:
            return newdata
        
    def generate_diversity_samples(self, sess, filename, X_corrupted, X_true, horz_samples=10, vert_samples=10):
        from data.dataset import noise
        print('Generating samples for', filename)
        predictions = [X_true[:vert_samples,:,:,:].reshape(1, vert_samples, self.height, self.width), X_corrupted[:vert_samples,:,:,:].reshape(1, vert_samples, self.height, self.width)]
        X_corrupted = np.repeat(X_corrupted[:vert_samples,:,:,:], horz_samples, axis=0)
        X_true = np.repeat(X_true[:vert_samples,:,:,:], horz_samples, axis=0)
        
        start = time.perf_counter()
        for i in range(self.test_iterations // 2 if filename == 'denoise' else self.test_iterations):
            X_corrupted = self.update_clamp(X_corrupted, sess.run(self.predictions, feed_dict={self.X_in:X_corrupted}), filename)
            if self.markov_type == 'renoise_per_two':
                X_corrupted = self.update_clamp(X_corrupted, sess.run(self.predictions, feed_dict={self.X_in:X_corrupted}), filename)
            if self.markov_type != 'only_denoise':
                if filename == 'denoise':
                    noise_prop = ((self.test_iterations//2 - i - 1) / self.test_iterations)
                else:
                    noise_prop = (self.test_iterations - i - 1) / self.test_iterations
                X_corrupted = self.update_clamp(X_corrupted, np.array([noise(arr, noise_prop, noise_prop)[0] for arr in X_corrupted]), filename)
        print(time.perf_counter() - start)
        
        
        predictions.append(X_corrupted.reshape(vert_samples, horz_samples, self.height, self.width).transpose(1,0,2,3))
        images = np.concatenate(tuple(predictions), axis=0)
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((self.height * vert_samples, self.width * (horz_samples + 2)))
        
        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M_noncausal_diversity_' + filename)+".png"
        Image.fromarray((images*255).astype(np.int32)).convert('RGB').save(filename)
        
    def generate_one_group_of_samples(self, sess, filename, X_corrupted, X_true=None, horz_samples=10):
        from data.dataset import noise
        print('Generating samples for', filename)
        X_corrupted = X_corrupted[:horz_samples,:,:,:]
        predictions = [X_true[:horz_samples,:,:,:]] if X_true is not None else []
        predictions.append(X_corrupted)
        
        start = time.perf_counter()
        for i in range(self.test_iterations // 2 if filename == 'denoise' else self.test_iterations):
            X_corrupted = self.update_clamp(X_corrupted, sess.run(self.predictions, feed_dict={self.X_in:X_corrupted}), filename)
            if self.markov_type == 'renoise_per_two':
                X_corrupted = self.update_clamp(X_corrupted, sess.run(self.predictions, feed_dict={self.X_in:X_corrupted}), filename)
            predictions.append(X_corrupted)
            if self.markov_type != 'only_denoise':
                if filename == 'denoise':
                    noise_prop = ((self.test_iterations//2 - i - 1) / self.test_iterations)
                else:
                    noise_prop = (self.test_iterations - i - 1) / self.test_iterations
                X_corrupted = self.update_clamp(X_corrupted, np.array([noise(arr, noise_prop, noise_prop)[0] for arr in X_corrupted]), filename)
        print(time.perf_counter() - start)
        
        predictions = tuple(p.reshape((horz_samples, 1, self.height, self.width)) for p in predictions)
        images = np.concatenate(predictions, axis=1)
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((self.height * ((self.test_iterations//2 if filename == 'denoise' else self.test_iterations) + 1 + (1 if X_true is not None else 0)), self.width * horz_samples))
        
        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M_noncausal_' + filename)+".png"
        Image.fromarray((images*255).astype(np.int32)).convert('RGB').save(filename)
        
    def generate_grid_of_samples(self, sess, X_corrupted):
        from data.dataset import noise
        print('Generating samples for grid')
        horz_samples = 10
        vert_samples = 10
        X_corrupted = X_corrupted[:horz_samples*vert_samples,:,:,:]

        start = time.perf_counter()
        for i in range(self.test_iterations):
            X_corrupted = sess.run(self.predictions, feed_dict={self.X_in:X_corrupted})
            if self.markov_type == 'renoise_per_two':
                X_corrupted = sess.run(self.predictions, feed_dict={self.X_in:X_corrupted})
            if self.markov_type != 'only_denoise':
                noise_prop = (self.test_iterations - i - 1) / self.test_iterations
                X_corrupted = np.array([noise(arr, noise_prop, noise_prop)[0] for arr in X_corrupted])
        print(time.perf_counter() - start)

        images = X_corrupted.reshape((vert_samples, horz_samples, self.height, self.width))
        images = images.transpose(0, 2, 1, 3)
        images = images.reshape((self.height * vert_samples, self.width * horz_samples))

        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M_noncausal_grid')+".png"
        Image.fromarray((images*255).astype(np.int32)).convert('RGB').save(filename)
        
    def generate_samples(self, sess):
        denoising_values = self.data.get_denoising_values()
        X_corrupted, X_true, _ = sess.run(denoising_values)
        self.generate_one_group_of_samples(sess, 'denoise', X_corrupted, X_true)
        X_corrupted, X_true, _ = sess.run(denoising_values)
        self.generate_diversity_samples(sess, 'denoise', X_corrupted, X_true)

        topgap_values = self.data.get_topgap_values()
        X_corrupted, X_true, _ = sess.run(topgap_values)
        self.generate_one_group_of_samples(sess, 'topgap', X_corrupted, X_true)
        X_corrupted, X_true, _ = sess.run(topgap_values)
        self.generate_diversity_samples(sess, 'topgap', X_corrupted, X_true)
        
        bottomgap_values = self.data.get_bottomgap_values()
        X_corrupted, X_true, _ = sess.run(bottomgap_values)
        self.generate_one_group_of_samples(sess, 'bottomgap', X_corrupted, X_true)
        X_corrupted, X_true, _ = sess.run(bottomgap_values)
        self.generate_diversity_samples(sess, 'bottomgap', X_corrupted, X_true)
        
        noise_values = self.data.get_noise_values()
        X_corrupted = sess.run(noise_values)
        self.generate_one_group_of_samples(sess, 'purenoise', X_corrupted)
        X_corrupted = sess.run(noise_values)
        self.generate_grid_of_samples(sess, X_corrupted)
        
        
    def samples(self):
        assert self.restore
        saver = tf.train.Saver()
        with tf.Session() as sess: 
            ckpt = tf.train.get_checkpoint_state('ckpts', latest_filename='noncausal')
            saver.restore(sess, ckpt.model_checkpoint_path)
            self.generate_samples(sess)
            
    def get_test_samples(self):
        assert self.restore
        saver = tf.train.Saver()

        with tf.Session() as sess: 
            ckpt = tf.train.get_checkpoint_state('ckpts', latest_filename='noncausal')
            saver.restore(sess, ckpt.model_checkpoint_path)
            return self.generate_ten_thousand_samples(sess)
            
    def get_test_samples_graph(self):
        assert self.restore
        saver = tf.train.Saver()

        with tf.Session() as sess: 
            ckpt = tf.train.get_checkpoint_state('ckpts', latest_filename='noncausal')
            saver.restore(sess, ckpt.model_checkpoint_path)
            return self.generate_ten_thousand_samples_graph(sess)

    def run(self):
        saver = tf.train.Saver()
        train_data = self.data.get_corrupted_values()
        test_data = self.data.get_corrupted_test_values()
        
        summary_writer = tf.summary.FileWriter('logs/noncausal')

        with tf.Session() as sess: 
            if self.restore:
                ckpt = tf.train.get_checkpoint_state('ckpts', latest_filename='noncausal')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            print('Started training at time', time.perf_counter())
            starttime = time.perf_counter()
            global_step = sess.run(self.global_step)
            print("Started Model Training...")
            while global_step < self.iterations:
                #print('running', global_step, self.iterations)
                X_corrupted, X_true, _, proportion = sess.run(train_data)
                predictions, train_summary, _, _ = sess.run([self.predictions, self.train_summary, self.loss, self.train_step], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.proportion:proportion})
                summary_writer.add_summary(train_summary, global_step)
                global_step = sess.run(self.global_step)
                
                train_summary, _, _ = sess.run([self.train_summary, self.loss, self.train_step], feed_dict={self.X_in:predictions, self.X_true:X_true, self.proportion:proportion})
                summary_writer.add_summary(train_summary, global_step)
                global_step = sess.run(self.global_step)

                if global_step%1000 == 0 or global_step >= self.iterations:
                    saver.save(sess, 'ckpts/noncausal.ckpt', global_step=global_step, latest_filename='noncausal')
                    sess.run([self.reset_loss_mean])
                  
                    for i in range(self.data.total_test_batches):
                        X_corrupted, X_true, _, proportion = sess.run(test_data)
                        predictions, _ = sess.run([self.predictions, self.update_loss_mean], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.proportion:proportion})
                        sess.run([self.update_loss_mean], feed_dict={self.X_in:predictions, self.X_true:X_true, self.proportion:proportion})
                      
                    test_summary, test_loss = sess.run([self.test_summary, self.loss_mean])
                    summary_writer.add_summary(test_summary, global_step)
                  
                    print("iteration %d, test loss %g"%(global_step, test_loss))
            print('Finished training at time', time.perf_counter())
            print('Time elapsed', time.perf_counter() - starttime)
            
    def run_tests(self):
        print('Noncausal basic test:')
        train_data = self.data.get_corrupted_values()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            X_corrupted, X_true, _, proportion = sess.run(train_data)
            loss, predictions, _ = sess.run([self.loss, self.predictions, self.train_step], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.proportion:proportion})
            
        print('Loss', loss)
        print('Predictions', [p.shape for p in predictions])
        print('Test completed')

    def __init__(self, conf, data):
        self.channels = data.channels
        assert data.channels == 1
        self.height = data.height
        self.width = data.width
        self.values = data.values
        self.labels = data.labels
        self.data = data
        self.filter_size = conf.filter_size
        self.features = conf.features
        self.end_features = conf.end_features
        self.layers = conf.layers
        self.temperature = conf.temperature
        self.iterations = conf.iterations
        self.learning_rate = conf.learning_rate
        self.restore = conf.restore
        self.markov_type = conf.markov_type

        self.test_iterations = conf.test_iterations

        self.global_step = tf.Variable(0, trainable=False)
        
        self.X_in = tf.placeholder(tf.float32, [None,self.height,self.width,self.channels])
        self.batch_size = tf.shape(self.X_in)[0]
        self.X_true = tf.placeholder(tf.float32, [None,self.height,self.width,self.channels])
        self.proportion = tf.placeholder(tf.float32, [None])
        self.expanded_proportion = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.reciprocal(self.proportion), -1), -1), -1), [1,self.height,self.width,self.channels])
        logits, self.predictions = self.noncausal()
        self.loss = tf.reduce_mean(tf.multiply(self.expanded_proportion, tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.logitise(self.X_true))))
        self.train_summary = tf.summary.scalar('train_loss', self.loss)
        with tf.name_scope('loss_mean_calc'):
            self.loss_mean, self.update_loss_mean = tf.metrics.mean(self.loss)
        self.test_summary = tf.summary.scalar('test_loss', self.loss_mean)
        
        self.reset_loss_mean = tf.initialize_variables([i for i in tf.local_variables() if 'loss_mean_calc' in i.name])
         
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        
        print('trainable variables:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


import tensorflow as tf

import numpy as np
from datetime import datetime
from PIL import Image

def get_weights(name, shape):
    return tf.get_variable(name, trainable=True, initializer=tf.contrib.layers.xavier_initializer()(shape))

def get_bias_weights(name, shape):
    return tf.get_variable(name, trainable=True, initializer=tf.zeros_initializer()(shape))

class NonCausal:
    def gate(self, p):
        p1, p2 = tf.split(p, 2, -1)
        return tf.multiply(tf.tanh(p1), tf.sigmoid(p2))
        
    def conv2d(self, input, shape):
        assert len(shape) == 4
        W = get_weights('W', shape)
        b = get_bias_weights('b', shape[-1])
        return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') + b
        
    def fully_connected(self, input, in_features, out_features, bias=True):
        W = get_weights('W', [1, 1, in_features, out_features])
        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID')
        if bias:
            out += get_bias_weights('b', [out_features])
        return out
        
    def apply_conditioning(self, out_features):
        if self.conditioning == 'none':
            return tf.zeros([self.batch_size, self.height, self.width, out_features])
        elif self.conditioning == 'generic':
            return tf.tile(tf.reshape(self.fully_connected(self.y, self.labels, out_features), shape=[self.batch_size, 1, 1, out_features]), [1, self.height, self.width, 1])
        elif self.conditioning == 'localised':
            return tf.reshape(self.fully_connected(self.y, self.labels, self.height * self.width * out_features), shape=[self.batch_size, self.height, self.width, out_features])
        else:
            assert False     
        
    def layer(self, input, first_layer=False, last_layer=False):
        assert not (first_layer and last_layer)
        with tf.variable_scope('cond'):
            condition = self.apply_conditioning(2*self.features)
        with tf.variable_scope('main'):
            out = self.gate(self.conv2d(input, [self.filter_size, self.filter_size, self.channels if first_layer else self.features, 2*self.features]) + condition)
        with tf.variable_scope('res'):
            residual = self.fully_connected(out, self.features, self.end_features if last_layer else self.features)
        with tf.variable_scope('skip'):
            skip = self.fully_connected(out, self.features, self.end_features)
        with tf.variable_scope('proj'):
            if first_layer:
                input = self.fully_connected(input, self.channels, self.features, bias=False)
            elif last_layer:
                input = self.fully_connected(input, self.features, self.end_features, bias=False)
        return input + residual, skip
        
    def end(self, outs):
        with tf.variable_scope('cond'):
            condition = self.apply_conditioning(self.end_features)
        outs.append(condition)
        out = tf.nn.relu(tf.reduce_sum(outs, 0))
        with tf.variable_scope('full1'):
            out = tf.nn.relu(self.fully_connected(out, self.end_features, self.end_features))
        with tf.variable_scope('full2'):
            out = self.fully_connected(out, self.end_features, self.channels * self.values)
        return tf.reshape(out, shape=[self.batch_size, self.height, self.width, self.channels, self.values])
        
    def noncausal(self):
        X = self.X_in
        train_logits = []
        test_logits = []
        test_predictions = []
        
        with tf.variable_scope('noncausal', reuse=tf.AUTO_REUSE):
            for iteration in range(max(self.train_iterations, self.test_iterations)):
                outs = []
                for i in range(self.layers):
                    with tf.variable_scope('layer' + str(i)):
                        X, skip = self.layer(X, first_layer=(i==0), last_layer=(i==self.layers-1))
                    outs.append(skip)
                outs.append(X)

                with tf.variable_scope('end'):
                    logits = self.end(outs)
                predictions = tf.stop_gradient(self.sample(logits))
                X = predictions
                if iteration < self.train_iterations:
                    train_logits.append(logits)
                if iteration < self.test_iterations:
                    test_logits.append(logits)
                    test_predictions.append(predictions)
        
        return train_logits, test_logits, test_predictions
        
    def sample(self, logits):
        probabilities = logits / self.temperature
        return tf.reshape(tf.cast(tf.multinomial(tf.reshape(probabilities, shape=[self.batch_size*self.height*self.width*self.channels, self.values]), 1), tf.float32), shape=[self.batch_size,self.height,self.width,self.channels])
        
    def logitise(self, images):
        return tf.stop_gradient(tf.one_hot(tf.to_int32(images * (self.values-1)), self.values))
        
    def generate_samples(self, sess):
        print("Generating Sample Images...")
        X_corrupted, X_true, y = sess.run(self.data.get_corrupted_test_values())

        predictions = sess.run([p[:100,:,:,:] for p in self.predictions], feed_dict={self.X_in:X_corrupted, self.y_raw:y})
        
        X_in = X_corrupted.reshape((100, 1, self.height, self.width))
        predictions = tuple(p.reshape((100, 1, self.height, self.width)) for p in predictions)
        assert len(predictions) == self.test_iterations
        X_true = X_true.reshape((100, 1, self.height, self.width))
        images = np.concatenate((X_in,) + predictions + (X_true,), axis=1)
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((self.height * (self.test_iterations + 2), self.width * 100)) #TODO: support more than one channel

        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M')+".png"
        Image.fromarray((images*255).astype(np.int32)).convert('RGB').save(filename)

    def run(self):
        saver = tf.train.Saver()
        train_data = self.data.get_corrupted_values()
        noise_train_data = self.data.get_noise_values()
        test_data = self.data.get_corrupted_test_values()
        noise_test_data = self.data.get_noise_test_values()
        
        summary_writer = tf.summary.FileWriter('logs/noncausal')

        with tf.Session() as sess: 
            if self.restore:
                ckpt = tf.train.get_checkpoint_state('ckpts')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            global_step = sess.run(self.global_step)
            print("Started Model Training...")
            while global_step < self.epochs:
              #print('running', global_step, self.epochs)
              if global_step % 2 == 0:
                X_corrupted, X_true, y = sess.run(train_data)
                train_summary, _, _ = sess.run([self.train_summary, self.train_loss, self.train_step], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
                summary_writer.add_summary(train_summary, global_step)
                global_step = sess.run(self.global_step)
              else:
                X_corrupted, X_true, y = sess.run(noise_train_data)
                first_iteration_train_summary, _, _ = sess.run([self.first_iteration_train_summary, self.first_iteration_train_loss, self.first_iteration_train_step], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
                summary_writer.add_summary(first_iteration_train_summary, global_step)
                global_step = sess.run(self.global_step)

              if global_step%1000 == 0 or global_step == self.epochs:
                saver.save(sess, 'ckpts/noncausal.ckpt', global_step=global_step)
                sess.run([self.reset_loss_mean, self.reset_first_iteration_loss_mean])
                
                for i in range(200*5):
                    X_corrupted, X_true, y = sess.run(test_data)
                    sess.run([self.update_loss_mean], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
                    
                test_summary, test_loss = sess.run([self.test_summary, self.loss_mean])
                summary_writer.add_summary(test_summary, global_step)

                    
                for i in range(200):
                    X_corrupted, X_true, y = sess.run(noise_test_data)
                    sess.run([self.update_first_iteration_loss_mean], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
                    
                first_iteration_test_summary, first_iteration_test_loss = sess.run([self.first_iteration_test_summary, self.first_iteration_loss_mean])
                summary_writer.add_summary(first_iteration_test_summary, global_step)
                
                print("epoch %d, test loss %g, noise test loss %g"%(global_step, test_loss, first_iteration_test_loss))
              
            
    def run_tests(self):
        print('Noncausal basic test:')
        train_data = self.data.get_corrupted_values()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            X_corrupted, X_true, y = sess.run(train_data)
            train_loss, _ = sess.run([self.train_loss, self.train_step], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
            first_iteration_train_loss, _ = sess.run([self.first_iteration_train_loss, self.first_iteration_train_step], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
            test_loss, predictions = sess.run([self.test_loss, self.predictions], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
            first_iteration_test_loss, predictions = sess.run([self.first_iteration_test_loss, self.predictions], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
        print('Train loss', train_loss)
        print('First Iter Train loss', first_iteration_train_loss)
        print('Test loss', test_loss)
        print('First Iter Test loss', first_iteration_test_loss)
        print('Predictions', [p.shape for p in predictions])
        print('Test completed')
        
    def cross_entropy_loss(self, logits_list, ground_truth):
        return tf.reduce_mean([tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.logitise(ground_truth)) for logits in logits_list])
        
    def compute_train_loss(self, logits_list, ground_truth):
        return self.cross_entropy_loss(logits_list, ground_truth)
        
    def compute_test_loss(self, logits_list, ground_truth):
        return self.cross_entropy_loss(logits_list, ground_truth)

    def __init__(self, conf, data):
        self.channels = data.channels
        self.height = data.height
        self.width = data.width
        self.values = data.values
        self.labels = data.labels
        self.data = data
        self.filter_size = conf.filter_size
        self.features = conf.features
        self.end_features = conf.end_features
        self.layers = conf.layers
        self.conditioning = conf.conditioning
        self.temperature = conf.temperature
        self.epochs = conf.epochs
        self.learning_rate = conf.learning_rate
        self.restore = conf.restore
        
        self.train_iterations = conf.train_iterations
        self.test_iterations = conf.test_iterations

        self.global_step = tf.Variable(0, trainable=False)
        
        self.X_in = tf.placeholder(tf.float32, [None,self.height,self.width,self.channels])
        self.batch_size = tf.shape(self.X_in)[0]
        self.X_true = tf.placeholder(tf.float32, [None,self.height,self.width,self.channels])
        self.y_raw = tf.placeholder(tf.int32, [None])
        self.y = tf.reshape(self.y_raw, shape=[self.batch_size, 1, 1])
        self.y = tf.stop_gradient(tf.one_hot(self.y, self.labels))
        train_logits, test_logits, test_predictions = self.noncausal()
        self.train_loss = self.compute_train_loss(train_logits, self.X_true)
        self.first_iteration_train_loss = self.compute_train_loss(train_logits[:1], self.X_true)
        
        self.test_loss = self.compute_test_loss(test_logits, self.X_true)
        with tf.name_scope('loss_mean_calc'):
            self.loss_mean, self.update_loss_mean = tf.metrics.mean(self.test_loss)
        self.test_summary = tf.summary.scalar('test_loss', self.loss_mean)
        self.reset_loss_mean = tf.initialize_variables([i for i in tf.local_variables() if 'loss_mean_calc' in i.name])
        
        self.first_iteration_test_loss = self.compute_test_loss(test_logits[:1], self.X_true)
        with tf.name_scope('loss_mean_first_iter_calc'):
            self.first_iteration_loss_mean, self.update_first_iteration_loss_mean = tf.metrics.mean(self.first_iteration_test_loss)
        self.first_iteration_test_summary = tf.summary.scalar('first_iteration_test_loss', self.first_iteration_loss_mean)
        self.reset_first_iteration_loss_mean = tf.initialize_variables([i for i in tf.local_variables() if 'loss_mean_first_iter_calc' in i.name])
        
        self.train_summary = tf.summary.scalar('train_loss', self.train_loss)
        self.first_iteration_train_summary = tf.summary.scalar('first_iteration_train_loss', self.first_iteration_train_loss)
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.train_loss, global_step=self.global_step)
        self.first_iteration_train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.first_iteration_train_loss, global_step=self.global_step)
        self.predictions = test_predictions
        
        print('trainable variables:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


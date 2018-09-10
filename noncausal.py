import tensorflow as tf

import numpy as np
from datetime import datetime
from PIL import Image

def get_weights(shape):
    return tf.Variable(tf.contrib.layers.xavier_initializer()(shape))

def get_bias_weights(shape):
    return tf.Variable(tf.zeros_initializer()(shape))

class NonCausal:
    def gate(self, p1, p2):
        return tf.multiply(tf.tanh(p1), tf.sigmoid(p2))
        
    def apply_conditioning(self, weights=None):
        if self.conditioning == 'none':
            return tf.zeros([self.batch_size, self.height, self.width, 2*self.features]), None
            
        if weights is None:
            if self.conditioning == 'generic':
                W = get_weights([1, 1, self.labels, 2*self.features])
                b = get_bias_weights([2*self.features])
            elif self.conditioning == 'localised':
                W = get_weights([1, 1, self.labels, self.height * self.width * 2*self.features])
                b = get_bias_weights([self.height * self.width * 2*self.features])
        else:
            W, b = weights
            
        if self.conditioning == 'generic':
            result = tf.tile(tf.reshape(tf.nn.conv2d(self.y, W, strides=[1, 1, 1, 1], padding='VALID') + b, shape=[self.batch_size, 1, 1, 2*self.features]), [1, self.height, self.width, 1])
        elif self.conditioning == 'localised':
            result = tf.reshape(tf.nn.conv2d(self.y, W, strides=[1, 1, 1, 1], padding='VALID') + b, shape=[self.batch_size, self.height, self.width, 2*self.features])
        else:
            assert False
            
        return result, (W, b)        
        
    def start(self, input, weights=None):
        if weights is None:
            W = get_weights([1, 1, self.channels, 2*self.features])
            b = get_bias_weights([2*self.features])
            condition, cond_weights = self.apply_conditioning()
        else:
            W, b, cond_weights = weights
            condition, _ = self.apply_conditioning(cond_weights)

        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b + condition
        out = self.gate(out[:,:,:,:self.features], out[:,:,:,self.features:])
        
        return out, (W, b, cond_weights)
        
    def layer(self, input, weights=None):
        if weights is None:
            W = get_weights([self.filter_size, self.filter_size, self.features, 2*self.features])
            b = get_bias_weights([2*self.features])
            W2 = get_weights([1, 1, self.features, self.features])
            b2 = get_bias_weights([self.features])
            condition, cond_weights = self.apply_conditioning()
        else:
            W, b, W2, b2, cond_weights = weights
            condition, _ = self.apply_conditioning(cond_weights)

        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') + b + condition
        out = self.gate(conv[:,:,:,:self.features], conv[:,:,:,self.features:])
        residual = tf.nn.conv2d(out, W2, strides=[1, 1, 1, 1], padding='VALID') + b2
        
        return input + residual, (W, b, W2, b2, cond_weights)
        
    def end(self, input, weights=None):
        if weights is None:
            W = get_weights([1, 1, self.features, 2*self.channels])
            b = get_bias_weights([2*self.channels])
        else:
            W, b = weights

        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b
        out = self.gate(out[:,:,:,:self.channels], out[:,:,:,self.channels:])
        
        return out, (W, b)
        
    def noncausal(self):
        X = self.X_in
        weights = []
        print(X.shape)
        X, W = self.start(X)
        weights.append(W)
        for i in range(self.layers):
            print(X.shape)
            X, W = self.layer(X)
            weights.append(W)
            
        print(X.shape)
        X, W = self.end(X)
        weights.append(W)
        print(X.shape)
        X_single = X
        
        for iteration in range(self.iterations - 1): ## TODO remove 'iterations'
            print(X.shape)
            X, _ = self.start(X, weights[0])
            for i in range(self.layers):
                print(X.shape)
                X, _ = self.layer(X, weights[i+1])
                
            print(X.shape)
            X, _ = self.end(X, weights[-1])
            print(X.shape)
        return X, X_single
        
    def generate_samples(self, sess):
        print("Generating Sample Images...")
        X_in, X_true, y = sess.run(self.data.get_test_values())

        predictions = sess.run(self.predictions, feed_dict={self.X_in:X_in, self.y_raw:y})
        
        X_in = X_in.reshape((self.batch_size, 1, self.height, self.width))
        predictions = predictions.reshape((self.batch_size, 1, self.height, self.width))
        X_true = X_true.reshape((self.batch_size, 1, self.height, self.width))
        images = np.concatenate((X_in, predictions, X_true), axis=1)
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((self.height * 3, self.width * self.batch_size))

        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M')+".png"
        Image.fromarray((images*255).astype(np.int32)).convert('RGB').save(filename)

    def run(self):
        #saver = tf.train.Saver(tf.trainable_variables())
        X_in_tf, X_tf, y_tf = self.data.get_values()       

        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            #if os.path.exists(conf.ckpt_file):
            #    saver.restore(sess, conf.ckpt_file)
            #    print("Model Restored")
            print("Started Model Training...")
            for i in range(self.epochs):
              X_in, X_true, y = sess.run([X_in_tf, X_tf, y_tf])
              _, loss = sess.run([self.train_step,self.loss], feed_dict={self.X_in:X_in, self.X_true:X_true, self.y_raw:y})
              if i%10 == 0:
                #saver.save(sess, conf.ckpt_file)
                print("batch %d, loss %g"%(i, loss))
            self.generate_samples(sess)
            
    def run_tests(self):
        print('No tests available')

    def __init__(self, conf, data):
        self.batch_size = conf.batch_size
        self.channels = data.channels
        self.height = data.height
        self.width = data.width
        self.values = data.values
        self.labels = data.labels
        self.data = data
        self.filter_size = conf.filter_size
        self.features = conf.features
        self.layers = conf.layers
        self.train_iterations = conf.train_iterations
        self.test_iterations = conf.test_iterations
        self.conditioning = conf.conditioning
        self.temperature = conf.temperature
        self.epochs = conf.epochs
        self.learning_rate = conf.learning_rate

        
        self.X_in = tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channels])
        self.X_true = tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channels])
        self.y_raw = tf.placeholder(tf.int32, [self.batch_size])
        self.y = tf.reshape(self.y_raw, shape=[self.batch_size, 1, 1])
        self.y = tf.one_hot(self.y, self.labels)
        X_out, X_single = self.noncausal()
        self.loss = tf.nn.l2_loss((X_out if self.train_type == 'full' else X_single) - self.X_true) ## TODO: train_type is gone
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        print(X_out.shape)
        self.predictions = tf.to_float(tf.less(tf.random_uniform([self.batch_size,self.height,self.width,self.channels]), X_out))


import tensorflow as tf

import numpy as np
from datetime import datetime
from PIL import Image

class Simultaneous:
    def gate(self, p1, p2):
        return tf.multiply(tf.tanh(p1), tf.sigmoid(p2))
        
    def apply_conditioning(self, weights=None):
        if self.conditioning == 'none':
            return tf.zeros([self.batch_size, self.height, self.width, 2*self.features]), None
            
        if weights is None:
            if self.conditioning == 'local':
                W = tf.Variable(tf.truncated_normal([1, 1, self.labels, 2*self.features], stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[2*self.features]))
            elif self.conditioning == 'global':
                W = tf.Variable(tf.truncated_normal([1, 1, self.labels, self.height * self.width * 2*self.features], stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[self.height * self.width * 2*self.features]))
        else:
            W, b = weights
            
        if self.conditioning == 'local':
            result = tf.tile(tf.reshape(tf.nn.conv2d(self.y, W, strides=[1, 1, 1, 1], padding='VALID') + b, shape=[self.batch_size, 1, 1, 2*self.features]), [1, self.height, self.width, 1])
        elif self.conditioning == 'global':
            result = tf.reshape(tf.nn.conv2d(self.y, W, strides=[1, 1, 1, 1], padding='VALID') + b, shape=[self.batch_size, self.height, self.width, 2*self.features])
        else:
            assert False
            
        return result, (W, b)
        
    def apply_recurrence(self, weights=None, recurrences=None):
        if self.recurrence == 'end':
            return tf.zeros([self.batch_size, self.height, self.width, self.features]), None
            
        if weights is None:
            W = tf.Variable(tf.truncated_normal([self.filter_size, self.filter_size, self.features, 2*self.features], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[2*self.features]))
            W2 = tf.Variable(tf.truncated_normal([1, 1, self.features, self.features], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[self.features]))
        else:
            W, b, W2, b2 = weights
            
        if recurrences is None:
            recurrences = tf.zeros([self.batch_size, self.height, self.width, self.features])

        result = tf.nn.conv2d(recurrences, W, strides=[1, 1, 1, 1], padding='SAME') + b
        result = self.gate(result[:,:,:,:self.features], result[:,:,:,self.features:])
        result = tf.nn.conv2d(result, W2, strides=[1, 1, 1, 1], padding='VALID') + b2
        
        return result, (W, b, W2, b2)
        
        
    def start(self, input, weights=None, recurrences=None):
        if weights is None:
            W = tf.Variable(tf.truncated_normal([1, 1, self.channels, 2*self.features], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[2*self.features]))
            condition, cond_weights = self.apply_conditioning()
        else:
            W, b, cond_weights = weights
            condition, _ = self.apply_conditioning(cond_weights)

        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b + condition
        out = self.gate(out[:,:,:,:self.features], out[:,:,:,self.features:])
        
        return out, (W, b, cond_weights), None
        
    def layer(self, input, weights=None, recurrences=None):
        if weights is None:
            W = tf.Variable(tf.truncated_normal([self.filter_size, self.filter_size, self.features, 2*self.features], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[2*self.features]))
            W2 = tf.Variable(tf.truncated_normal([1, 1, self.features, self.features], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[self.features]))
            condition, cond_weights = self.apply_conditioning()
            recurrence, rec_weights = self.apply_recurrence()
        else:
            W, b, W2, b2, cond_weights, rec_weights = weights
            condition, _ = self.apply_conditioning(cond_weights)
            recurrence, _ = self.apply_recurrence(rec_weights, recurrences)

        conv = tf.nn.conv2d(input + recurrence, W, strides=[1, 1, 1, 1], padding='SAME') + b + condition
        out = self.gate(conv[:,:,:,:self.features], conv[:,:,:,self.features:])
        residual = tf.nn.conv2d(out, W2, strides=[1, 1, 1, 1], padding='VALID') + b2
        
        return input + residual, (W, b, W2, b2, cond_weights, rec_weights), out
        
    def end(self, input, weights=None, recurrences=None):
        if weights is None:
            W = tf.Variable(tf.truncated_normal([1, 1, self.features, 2*self.channels], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[2*self.channels]))
        else:
            W, b = weights

        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b
        out = self.gate(out[:,:,:,:self.channels], out[:,:,:,self.channels:])
        
        return out, (W, b), None
        
    def simultaneous(self):
        X = self.X
        weights = []
        recurrences = []
        print(X.shape)
        X, W, R = self.start(X)
        weights.append(W)
        recurrences.append(R)
        for i in range(self.layers):
            print(X.shape)
            X, W, R = self.layer(X)
            weights.append(W)
            recurrences.append(R)
            
        print(X.shape)
        X, W, R = self.end(X)
        weights.append(W)
        recurrences.append(R)
        print(X.shape)
        X_single = X
        
        for iteration in range(self.iterations - 1):
            print(X.shape)
            X, _, recurrences[0] = self.start(X, weights[0], recurrences[0])
            for i in range(self.layers):
                print(X.shape)
                X, _, recurrences[i+1] = self.layer(X, weights[i+1], recurrences[i+1])
                
            print(X.shape)
            X, _, recurrences[-1] = self.end(X, weights[-1], recurrences[-1])
            print(X.shape)
        return X, X_single
        
    def generate_samples(self, sess):
        print("Generating Sample Images...")
        samples = np.zeros((self.batch_size, self.height, self.width, self.channels), dtype=np.float32)
        conditions = [0]*self.batch_size

        predictions = sess.run(self.predictions, feed_dict={self.X:samples, self.y_raw:conditions})

        images = predictions.reshape((self.batch_size, 1, self.height, self.width))
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((self.height * 1, self.width * self.batch_size))

        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M')+".jpg"
        Image.fromarray(images.astype(np.int8)*255, mode='L').convert(mode='RGB').save(filename)

    def run(self):
        #saver = tf.train.Saver(tf.trainable_variables())
        iterator = self.data.train.make_one_shot_iterator()
        X_tf, y_tf = iterator.get_next()        

        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            #if os.path.exists(conf.ckpt_file):
            #    saver.restore(sess, conf.ckpt_file)
            #    print("Model Restored")
            print("Started Model Training...")
            for i in range(self.batches):
              X, y = sess.run([X_tf, y_tf])
              _, loss = sess.run([self.train_step,self.loss], feed_dict={self.X:X, self.y_raw:y})
              if i%10 == 0:
                #saver.save(sess, conf.ckpt_file)
                print("batch %d, loss %g"%(i, loss))
            self.generate_samples(sess)

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
        self.recurrence = conf.recurrence
        self.iterations = conf.iterations
        self.train_type = conf.train_type
        self.conditioning = conf.conditioning
        self.temperature = conf.temperature
        self.batches = conf.batches

        
        self.X = tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channels])
        self.y_raw = tf.placeholder(tf.int32, [self.batch_size])
        self.y = tf.reshape(self.y_raw, shape=[self.batch_size, 1, 1])
        self.y = tf.one_hot(self.y, self.labels)
        X_out, X_single = self.simultaneous()
        self.loss = tf.nn.l2_loss((X_out if self.train_type == 'full' else X_single) - self.X)
        self.train_step = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss)
        print(X_out.shape)
        self.predictions = tf.to_float(tf.less(tf.random_uniform([self.batch_size,self.height,self.width,self.channels]), X_out))


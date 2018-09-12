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
        
    def logitise(self, images):
        return tf.one_hot(tf.to_int32(images * (self.values-1)), self.values)
        
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
        test_data = self.data.get_corrupted_test_values()
        # TODO: new loss function
        
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
              X_corrupted, X_true, y = sess.run(train_data)
              train_summary, _, _ = sess.run([self.train_summary, self.train_loss, self.train_step], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
              summary_writer.add_summary(train_summary, global_step)
              
              if global_step%1000 == 0 or global_step == (self.epochs - 1):
                saver.save(sess, 'ckpts/noncausal.ckpt', global_step=global_step)
                X_corrupted, X_true, y = sess.run(test_data)
                test_summary, test_loss = sess.run([self.test_summary, self.test_loss], feed_dict={self.X_in:X_corrupted, self.X_true:X_true, self.y_raw:y})
                summary_writer.add_summary(test_summary, global_step)
                print("batch %d, test loss %g"%(global_step, test_loss))
              global_step = sess.run(self.global_step)
            
    def run_tests(self):
        print('No tests available')
        
    def sample(self, logits):
        probabilities = logits / self.temperature
        return tf.reshape(tf.multinomial(tf.reshape(probabilities, shape=[self.batch_size*self.height*self.width*self.channels, self.values]), 1), shape=[self.batch_size,self.height,self.width,self.channels])

    def __init__(self, conf, data):
        self.channels = data.channels
        self.height = data.height
        self.width = data.width
        self.values = data.values
        self.labels = data.labels
        self.data = data
        self.filter_size = conf.filter_size
        self.features = conf.features
        self.layers = conf.layers
        self.conditioning = conf.conditioning
        self.temperature = conf.temperature
        self.epochs = conf.epochs
        self.learning_rate = conf.learning_rate
        self.restore = conf.restore
        
        self.train_iterations = conf.train_iterations
        self.test_iterations = conf.test_iterations
        # TODO: new loss function

        self.global_step = tf.Variable(0, trainable=False)
        
        self.X_in = tf.placeholder(tf.float32, [None,self.height,self.width,self.channels])
        self.batch_size = tf.shape(self.X_in)[0]
        self.X_true = tf.placeholder(tf.float32, [None,self.height,self.width,self.channels])
        self.y_raw = tf.placeholder(tf.int32, [None])
        self.y = tf.reshape(self.y_raw, shape=[self.batch_size, 1, 1]) # If this throws an error at runtime, y_raw was fed with the wrong batch size. Can't seem to find a way to constrain the size at feed time.
        self.y = tf.one_hot(self.y, self.labels)
        train_predictions, test_predictions, test_sampled = self.noncausal()
        self.train_loss = tf.reduce_mean([tf.nn.softmax_cross_entropy_with_logits_v2(logits=p, labels=self.logitise(self.X_true)) for p in train_predictions])
        self.test_loss = tf.reduce_mean([tf.nn.softmax_cross_entropy_with_logits_v2(logits=p, labels=self.logitise(self.X_true)) for p in test_predictions])
        self.train_summary = tf.summary.scalar('train_loss', self.train_loss)
        self.test_summary = tf.summary.scalar('test_loss', self.test_loss)
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.train_loss, global_step=self.global_step)
        print(X_out.shape)
        self.predictions = test_sampled
        #print(self.predictions.shape)
        
        print('trainable variables:', len(tf.trainable_variables()))


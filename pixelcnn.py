import tensorflow as tf

import numpy as np
from datetime import datetime
from PIL import Image

def get_weights(shape):
    return tf.Variable(tf.contrib.layers.xavier_initializer()(shape))

def get_bias_weights(shape):
    return tf.Variable(tf.zeros_initializer()(shape))

class PixelCNN:
    def gate(self, p1, p2):
        return tf.multiply(tf.tanh(p1), tf.sigmoid(p2))

    def vertical_stack(self, v_in, first=False):
        # Could replace this with the complicated Nx1 -> 1xN thing, but probably not worth it.
        assert self.filter_size % 2 == 1
        n = self.filter_size
        v_in = tf.pad(v_in, [[0,0],[n-1,0],[n//2, n//2],[0,0]])
        
        W = get_weights([n, n, self.channels if first else self.features, 2*self.features])
        b = get_bias_weights([2*self.features])
        v_conv = tf.nn.conv2d(v_in, W, strides=[1, 1, 1, 1], padding='VALID') + b
        return v_conv
        
    def horizontal_stack(self, h_in, v_conv, first=False):
        n = self.filter_size
        h_in = tf.pad(h_in, [[0,0],[0,0],[n-1, 0],[0,0]])
        
        v_conv = tf.pad(v_conv, [[0,0],[1,0],[0, 0],[0,0]])
        v_conv = v_conv[:,:-1,:,:]
        
        if first:
            # first layer: don't include current pixel in convolution
            h_in = h_in[:,:,:-1,:]
            n = n-1
            
        W = get_weights([1, n, self.channels if first else self.features, 2*self.features])
        b = get_bias_weights([2*self.features])
        h_conv = tf.nn.conv2d(h_in, W, strides=[1, 1, 1, 1], padding='VALID') + b
        
        W2 = get_weights([1, 1, 2*self.features, 2*self.features])
        b2 = get_bias_weights([2*self.features])
        v_conv = tf.nn.conv2d(v_conv, W2, strides=[1, 1, 1, 1], padding='VALID') + b2
        
        h_conv += v_conv
        h_conv += self.apply_conditioning(True)
        
        h_out = self.gate(h_conv[:,:,:,:self.features], h_conv[:,:,:,self.features:])
        
        return h_out

    def layer(self, v_in, h_in, first=False):
        v_conv = self.vertical_stack(v_in, first=first)
        v_out = v_conv + self.apply_conditioning(True)
        v_out = self.gate(v_out[:,:,:,:self.features], v_out[:,:,:,self.features:])
        h_out = self.horizontal_stack(h_in, v_conv, first=first)
        
        skip = self.fully_connected(h_out, False)
        if not first:
            h_out = self.fully_connected(h_out, False) + h_in
        
        return v_out, h_out, skip

    def fully_connected(self, input, final):
        W = get_weights([1, 1, self.features, self.channels * self.values if final else self.features])
        b = get_bias_weights([self.channels * self.values if final else self.features])
        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b
        if final:
            out = tf.reshape(out, shape=[self.batch_size, self.height, self.width, self.channels, self.values])
        return out
        
    def apply_conditioning(self, doubled):
        doubler = 2 if doubled else 1
        if self.conditioning == 'none':
            return tf.zeros([self.batch_size, self.height, self.width, self.features * doubler])
        elif self.conditioning == 'local':
            W = get_weights([1, 1, self.labels, self.features * doubler])
            b = get_bias_weights([self.features * doubler])
            return tf.tile(tf.reshape(tf.nn.conv2d(self.y, W, strides=[1, 1, 1, 1], padding='VALID') + b, shape=[self.batch_size, 1, 1, self.features * doubler]), [1, self.height, self.width, 1])
        elif self.conditioning == 'global':
            W = get_weights([1, 1, self.labels, self.height * self.width * self.features * doubler])
            b = get_bias_weights([self.height * self.width * self.features * doubler])
            return tf.reshape(tf.nn.conv2d(self.y, W, strides=[1, 1, 1, 1], padding='VALID') + b, shape=[self.batch_size, self.height, self.width, self.features * doubler])
        else:
            assert False

    def pixelcnn(self):
        print(self.X.shape, self.y.shape)
        v_in, h_in = self.X, self.X # 'pass through a "causal convolution"???'
        outs = []
        for i in range(self.layers):
            v_in, h_in, skip = self.layer(v_in, h_in, first=(i == 0))
            outs.append(skip)
        v_in = tf.pad(v_in, [[0,0],[1,0],[0, 0],[0,0]])
        v_in = v_in[:,:-1,:,:]
        outs.append(self.fully_connected(v_in, False))
        outs.append(self.fully_connected(h_in, False))
        outs.append(self.apply_conditioning(False))
        out = tf.reduce_sum(outs, 0)
        print(out.shape)
        out = tf.nn.relu(out)
        out = self.fully_connected(out, False)
        out = tf.nn.relu(out)
        print(out.shape)
        out = self.fully_connected(out, True)
        print(out.shape)
        return out
        
    def logitise(self, images):
        return tf.one_hot(tf.to_int32(images * (self.values-1)), self.values)
        
    def generate_samples(self, sess):
        print("Generating Sample Images...")
        samples = np.zeros((100, self.height, self.width, self.channels), dtype=np.float32)
        conditions = [i // 10 for i in range(100)]

        for i in range(self.height):
            for j in range(self.width):
                predictions = sess.run(self.predictions, feed_dict={self.X:samples, self.y_raw:conditions})
                samples[:, i, j, :] = predictions[:, i, j, :]

        images = samples.reshape((100, 1, self.height, self.width))
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((self.height * 10, self.width * 10))

        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M')+".png"
        Image.fromarray(images.astype(np.int8)*255, mode='L').convert(mode='RGB').save(filename)
        
    def run_tests(self):
        self.causality_test()
        
    def causality_test(self):
        import matplotlib.pyplot as plt
        print('PixelCNN causality test:')
        batch_size = 1
        if (self.filter_size // 2) * self.layers < max(self.height, self.width):
            print('Filter size and/or number of layers too low to capture whole image in receptive field. Not running test.')
            return
        data_length = self.height * self.width * self.channels
        data_length = int(np.ceil(data_length / batch_size) * batch_size)
        data = np.zeros((data_length, self.height, self.width, self.channels))
        answer = np.zeros((data_length, self.height, self.width, self.channels))
        conditions = [0]*(batch_size)
        for y in range(self.height):
            for x in range(self.width):
                for c in range(self.channels):
                    data[y*self.width * self.channels + x*self.channels + c,y,x,c] = np.nan
                    for yy in range(y, self.height):
                        for xx in range(self.width):
                            for cc in range(self.channels):
                                if yy > y or xx > x:
                                    answer[y*self.width * self.channels + x*self.channels + c,yy,xx,cc] = np.nan
        
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            for batch in range(data_length//batch_size):
                X = data[batch:batch+batch_size,:,:,:]
                y = answer[batch:batch+batch_size,:,:,:]
                predictions = sess.run(self.predictions, feed_dict={self.X:X, self.y_raw:conditions})
                # weirdly tf.multinomial outputs n+1 if one of the inputs is nan:
                if np.all((predictions==self.values) == np.isnan(y)):
                    print('Success')
                else:
                    print('Fail')
                    plt.close()
                    plt.subplot(121)
                    plt.imshow((predictions == self.values)[0,:,:,0])
                    plt.subplot(122)
                    plt.imshow(np.isnan(y)[0,:,:,0])
                    plt.show()
        
        
    def run(self):
        saver = tf.train.Saver()
        X_in_tf, X_tf, y_tf = self.data.get_values()
        X_test_in_tf, X_test_tf, y_test_tf = self.data.get_test_values()
        
        summary_writer = tf.summary.FileWriter('logs/pixelcnn')

        with tf.Session() as sess: 
            if self.restore:
                ckpt = tf.train.get_checkpoint_state('ckpts')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            global_step = sess.run(self.global_step)
            print("Started Model Training...")
            while global_step < self.batches:
              print('running', global_step, self.batches)
              X, y = sess.run([X_tf, y_tf])
              summary, _ = sess.run([self.summaries, self.train_step], feed_dict={self.X:X, self.y_raw:y})
              summary_writer.add_summary(summary, global_step)
              
              if global_step%1000 == 0 or global_step == (self.batches - 1):
                saver.save(sess, 'ckpts/pixelcnn.ckpt', global_step=global_step)
                X, y = sess.run([X_test_tf, y_test_tf])
                summary, test_loss = sess.run([self.summaries, self.loss], feed_dict={self.X:X, self.y_raw:y})
                summary_writer.add_summary(summary, global_step)
                print("batch %d, test loss %g"%(global_step, test_loss))
              global_step = sess.run(self.global_step)
                
            saver.save(sess, 'ckpts/pixelcnn.ckpt')
            
            #self.generate_samples(sess)
            
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
        self.batches = conf.batches
        self.learning_rate = conf.learning_rate
        self.restore = conf.restore
        
        self.global_step = tf.Variable(0, trainable=False)
        
        self.X = tf.placeholder(tf.float32, [None,self.height,self.width,self.channels])
        self.batch_size = tf.shape(self.X)[0]
        self.y_raw = tf.placeholder(tf.int32, [None])
        self.y = tf.reshape(self.y_raw, shape=[self.batch_size, 1, 1]) # If this throws an error at runtime, y_raw was fed with the wrong batch size. Can't seem to find a way to constrain the size at feed time.
        self.y = tf.one_hot(self.y, self.labels)
        X_out = self.pixelcnn() # softmax has not been applied here, shape is [batch, height, width, channels, values]
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=X_out, labels=self.logitise(self.X)))
        tf.summary.scalar('loss', self.loss)
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        print(X_out.shape)
        probabilities = X_out / self.temperature
        self.predictions = tf.reshape(tf.multinomial(tf.reshape(probabilities, shape=[self.batch_size*self.height*self.width*self.channels, self.values]), 1), shape=[self.batch_size,self.height,self.width,self.channels])
        print(self.predictions.shape)
        
        self.summaries = tf.summary.merge_all()


import tensorflow as tf

import numpy as np
from datetime import datetime
from PIL import Image

class PixelCNN:
    def gate(self, p1, p2):
        return tf.multiply(tf.tanh(p1), tf.sigmoid(p2))

    def vertical_stack(self, v_in, first=False):
        # Could replace this with the complicated Nx1 -> 1xN thing, but probably not worth it.
        assert self.filter_size % 2 == 1
        n = self.filter_size
        v_in = tf.pad(v_in, [[0,0],[n-1,0],[n//2, n//2],[0,0]])
        
        W = tf.Variable(tf.truncated_normal([n, n, self.channels if first else self.features, 2*self.features], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[2*self.features]))
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
            
        W = tf.Variable(tf.truncated_normal([1, n, self.channels if first else self.features, 2*self.features], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[2*self.features]))
        h_conv = tf.nn.conv2d(h_in, W, strides=[1, 1, 1, 1], padding='VALID') + b
        
        W2 = tf.Variable(tf.truncated_normal([1, 1, 2*self.features, 2*self.features], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[2*self.features]))
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
        W = tf.Variable(tf.truncated_normal([1, 1, self.features, self.channels * self.values if final else self.features], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[self.channels * self.values if final else self.features]))
        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b
        if final:
            out = tf.reshape(out, shape=[self.batch_size, self.height, self.width, self.channels, self.values])
        return out
        
    def apply_conditioning(self, doubled):
        doubler = 2 if doubled else 1
        if self.conditioning == 'none':
            return tf.zeros([self.batch_size, self.height, self.width, self.features * doubler])
        elif self.conditioning == 'local':
            W = tf.Variable(tf.truncated_normal([1, 1, self.labels, self.features * doubler], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.features * doubler]))
            return tf.tile(tf.reshape(tf.nn.conv2d(self.y, W, strides=[1, 1, 1, 1], padding='VALID') + b, shape=[self.batch_size, 1, 1, self.features * doubler]), [1, self.height, self.width, 1])
        elif self.conditioning == 'global':
            W = tf.Variable(tf.truncated_normal([1, 1, self.labels, self.height * self.width * self.features * doubler], stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[self.height * self.width * self.features * doubler]))
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
        samples = np.zeros((self.batch_size, self.height, self.width, self.channels), dtype=np.float32)
        conditions = [i // self.batch_size for i in range(self.batch_size)]

        for i in range(self.height):
            for j in range(self.width):
                predictions = sess.run(self.predictions, feed_dict={self.X:samples, self.y_raw:conditions})
                samples[:, i, j, :] = predictions[:, i, j, :]

        images = samples.reshape((self.batch_size, 1, self.height, self.width))
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((self.height * 1, self.width * self.batch_size))

        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M')+".png"
        Image.fromarray(images.astype(np.int8)*255, mode='L').convert(mode='RGB').save(filename)
        
    def run_tests(self):
        self.causality_test()
        
    def causality_test(self):
        import matplotlib.pyplot as plt
        print('PixelCNN causality test:')
        if self.batch_size > 1:
            print('Recommend using --batch_size 1 for interpretable results')
        if (self.filter_size // 2) * self.layers < max(self.height, self.width):
            print('Filter size and/or number of layers too low to capture whole image in receptive field. Not running test.')
            return
        data_length = self.height * self.width * self.channels
        data_length = int(np.ceil(data_length / self.batch_size) * self.batch_size)
        data = np.zeros((data_length, self.height, self.width, self.channels))
        answer = np.zeros((data_length, self.height, self.width, self.channels))
        conditions = [0]*self.batch_size
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
            for batch in range(data_length//self.batch_size):
                X = data[batch:batch+self.batch_size,:,:,:]
                y = answer[batch:batch+self.batch_size,:,:,:]
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
        #saver = tf.train.Saver(tf.trainable_variables())
        X_in_tf, X_tf, y_tf = self.data.get_values()    

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
        self.conditioning = conf.conditioning
        self.temperature = conf.temperature
        self.batches = conf.batches
        
        self.X = tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channels])
        self.y_raw = tf.placeholder(tf.int32, [self.batch_size])
        self.y = tf.reshape(self.y_raw, shape=[self.batch_size, 1, 1])
        self.y = tf.one_hot(self.y, self.labels)
        X_out = self.pixelcnn() # softmax has not been applied here, shape is [batch, height, width, channels, values]
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=X_out, labels=self.logitise(self.X)))
        self.train_step = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss) # "RMSprop with a learning rate schedule starting at 1e-4 and decaying to 1e-5, trained for 200k steps with batch size of 128"
        print(X_out.shape)
        probabilities = X_out / self.temperature
        self.predictions = tf.reshape(tf.multinomial(tf.reshape(probabilities, shape=[self.batch_size*self.height*self.width*self.channels, self.values]), 1), shape=[self.batch_size,self.height,self.width,self.channels])
        print(self.predictions.shape)


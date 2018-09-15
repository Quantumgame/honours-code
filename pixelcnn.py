import tensorflow as tf

import numpy as np
from datetime import datetime
from PIL import Image

def get_weights(shape):
    return tf.Variable(tf.contrib.layers.xavier_initializer()(shape))

def get_bias_weights(shape):
    return tf.Variable(tf.zeros_initializer()(shape))

class PixelCNN:
    def gate(self, p):
        p1, p2 = tf.split(p, 2, -1)
        return tf.multiply(tf.tanh(p1), tf.sigmoid(p2))
        
    def padded_conv2d(self, input, shape, pad_shape):
        assert len(shape) == 4
        W = get_weights(shape)
        b = get_bias_weights(shape[-1])
        input = tf.pad(input, pad_shape)
        return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b
        
    def fully_connected(self, input, in_features, out_features, bias=True):
        W = get_weights([1, 1, in_features, out_features])
        out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID')
        if bias:
            out += get_bias_weights([out_features])
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
            
    def shift_up(self, v):
        return tf.pad(v, [[0,0],[1,0],[0, 0],[0,0]])[:,:-1,:,:]
        
    def shift_left(self, h):
        return tf.pad(h, [[0,0],[0,0],[1, 0],[0,0]])[:,:,:-1,:]

    def layer(self, v_in, h_in, first_layer=False, last_layer=False):
        assert not (first_layer and last_layer)
        assert self.filter_size % 2 == 1

        v_cond = self.apply_conditioning(2*self.features)
        v_conv = self.padded_conv2d(v_in,
            [self.filter_size-1, self.filter_size, self.channels if first_layer else self.features, 2*self.features],
            [[0,0],[self.filter_size-2,0],[self.filter_size//2, self.filter_size//2],[0,0]]
        )
        v_out = self.gate(v_conv + v_cond)
        
        if first_layer:
            h_in = self.shift_left(h_in) # TODO: encode dependencies among channels for the same pixel

        h_cond = self.apply_conditioning(2*self.features)
        v_cross = self.fully_connected(self.shift_up(v_conv), 2*self.features, 2*self.features)
        h_conv = self.padded_conv2d(h_in,
            [1, self.filter_size, self.channels if first_layer else self.features, 2*self.features],
            [[0,0],[0,0],[self.filter_size-1, 0],[0,0]]
        )
        h_out = self.gate(h_conv + v_cross + h_cond)
        
        residual = self.fully_connected(h_out, self.features, self.end_features if last_layer else self.features)
        skip = self.fully_connected(h_out, self.features, self.end_features)
            
        if first_layer:
            h_in = self.fully_connected(h_in, self.channels, self.features, bias=False)
        elif last_layer:
            h_in = self.fully_connected(h_in, self.features, self.end_features, bias=False)
        
        return v_out, h_in + residual, skip
        
    def end(self, outs):
        condition = self.apply_conditioning(self.end_features)
        outs.append(condition)
        out = tf.nn.relu(tf.reduce_sum(outs, 0))
        out = tf.nn.relu(self.fully_connected(out, self.end_features, self.end_features))
        out = self.fully_connected(out, self.end_features, self.channels * self.values)
        return tf.reshape(out, shape=[self.batch_size, self.height, self.width, self.channels, self.values])

    def pixelcnn(self):
        v, h = self.X, self.X
        outs = []
        for i in range(self.layers):
            v, h, skip = self.layer(v, h, first_layer=(i == 0), last_layer=(i==self.layers-1))
            outs.append(skip)
        outs.append(h)
        logits = self.end(outs)
        predictions = self.sample(logits)

        return logits, predictions
        
    def sample(self, logits):
        probabilities = logits / self.temperature
        return tf.reshape(tf.multinomial(tf.reshape(probabilities, shape=[self.batch_size*self.height*self.width*self.channels, self.values]), 1), shape=[self.batch_size,self.height,self.width,self.channels])
        
    def logitise(self, images):
        return tf.stop_gradient(tf.one_hot(tf.to_int32(images * (self.values-1)), self.values))
        
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
        images = images.reshape((self.height * 10, self.width * 10)) #TODO: support more than one channel

        filename = datetime.now().strftime('samples/%Y_%m_%d_%H_%M')+".png"
        Image.fromarray((images*255).astype(np.int32)).convert('RGB').save(filename)
        
    def run(self):
        saver = tf.train.Saver()
        train_data = self.data.get_plain_values()
        test_data = self.data.get_plain_test_values()
        
        summary_writer = tf.summary.FileWriter('logs/pixelcnn')

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
              X, y = sess.run(train_data)
              train_summary, _, _ = sess.run([self.train_summary, self.loss, self.train_step], feed_dict={self.X:X, self.y_raw:y})
              summary_writer.add_summary(train_summary, global_step)
              global_step = sess.run(self.global_step)

              if global_step%1000 == 0 or global_step == self.epochs:
                saver.save(sess, 'ckpts/pixelcnn.ckpt', global_step=global_step)
                X, y = sess.run(test_data)
                test_summary, test_loss = sess.run([self.test_summary, self.loss], feed_dict={self.X:X, self.y_raw:y})
                summary_writer.add_summary(test_summary, global_step)
                print("epoch %d, test loss %g"%(global_step, test_loss))
              
              
    def run_tests(self):
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
        
        self.global_step = tf.Variable(0, trainable=False)
        
        self.X = tf.placeholder(tf.float32, [None,self.height,self.width,self.channels])
        self.batch_size = tf.shape(self.X)[0]
        self.y_raw = tf.placeholder(tf.int32, [None])
        self.y = tf.reshape(self.y_raw, shape=[self.batch_size, 1, 1])
        self.y = tf.stop_gradient(tf.one_hot(self.y, self.labels))
        logits, predictions = self.pixelcnn()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.logitise(self.X)))
        self.train_summary = tf.summary.scalar('train_loss', self.loss)
        self.test_summary = tf.summary.scalar('test_loss', self.loss)
        self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        self.predictions = predictions
        
        print('trainable variables:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


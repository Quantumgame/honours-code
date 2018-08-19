import tensorflow as tf

class Simultaneous:
    def gate(self, p1, p2):
        return tf.multiply(tf.tanh(p1), tf.sigmoid(p2))
        
    def layer(self, X):
        W = tf.Variable(tf.truncated_normal([self.filter_size, self.filter_size, X.shape[3].value, 2*self.features], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[2*self.features]))
        conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME') + b
        out = self.gate(conv[:,:,:,:self.features], conv[:,:,:,self.features:])
        return out
        
    def fully_connected(self, X):
        
        W = tf.Variable(tf.truncated_normal([1, 1, self.features, 2], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[2]))
        out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='VALID') + b
        
        return self.gate(out[:,:,:,:1], out[:,:,:,1:])
        
    def simultaneous(self, X):
        for i in range(self.layers):
            print(X.shape)
            X = self.layer(X)
            
        print(X.shape)
        X = self.fully_connected(X)
        print(X.shape)
        return X
        
    def __init__(self, conf, data):
        self.batch_size = conf.batch_size
        self.channels = data.channels
        self.height = data.height
        self.width = data.width
        self.data = data
        self.filter_size = conf.filter_size
        self.features = conf.features
        self.layers = conf.layers
        self.recurrence = conf.recurrence
        self.iterations = conf.iterations
        self.train_type = conf.train_type
        self.conditioning = conf.conditioning
        
        X = tf.placeholder(tf.float32, [self.batch_size,self.height,self.width,self.channels])
        X_out = self.simultaneous(X)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=X_out, labels=X))
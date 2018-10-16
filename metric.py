import tensorflow as tf
from collections import namedtuple

from data.dataset import Dataset

def get_metric(real, fakes):
    print('computing MMD metric')
    real, fakes = get_feature_vectors(real, fakes)
    print('computing scores')
    scores = [compute_score(real, fake) for fake in fakes]
    print('your scores are', scores)
    return scores
    
def compute_score(real, fake):
    Mxx = squared_dist(real, real)
    Mxy = squared_dist(real, fake)
    Myy = squared_dist(fake, fake)
    return mmd(Mxx, Mxy, Myy)

def squared_dist(A, B):
  """ https://stackoverflow.com/a/51186660 """
  assert A.shape.as_list() == B.shape.as_list()

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A + row_norms_B - 2 * tf.matmul(A, tf.transpose(B))

def mmd(Mxx, Mxy, Myy, sigma=1):
    scale = tf.reduce_mean(Mxx)
    Mxx = tf.math.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = tf.math.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = tf.math.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = tf.sqrt(tf.reduce_mean(Mxx) + tf.reduce_mean(Myy) - 2 * tf.reduce_mean(Mxy))
    return mmd

    
def get_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def get_bias_weights(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
  
def conv2d(input, shape):
    assert len(shape) == 4
    W = get_weights(shape)
    b = get_bias_weights([shape[-1]])
    return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') + b

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def classifier():
    input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    
    x = max_pool_2x2(tf.nn.relu(conv2d(input, [5, 5, 1, 32])))
    x = max_pool_2x2(tf.nn.relu(conv2d(x, [5, 5, 32, 64])))
    x = tf.reshape(x, [-1, 7 * 7 * 64])
    feature_vectors = tf.nn.relu(tf.matmul(x, get_weights([7 * 7 * 64, 1024])) + get_bias_weights([1024]))
    x = tf.nn.dropout(feature_vectors, keep_prob)
    logits = tf.matmul(x, get_weights([1024, 10])) + get_bias_weights([10])
    return input, keep_prob, feature_vectors, logits

def get_feature_vectors(real, fakes):
  train_data = Dataset(namedtuple('Conf', 'batch_size')(50), only_plain=True).get_plain_values()
  
  labels = tf.placeholder(tf.int64, [None])
  input, keep_prob, feature_vectors, logits = classifier()

  cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('training classifier')
    for i in range(20):#000):
      X, y = sess.run(train_data)
      train_step.run(feed_dict={input: X, labels: y, keep_prob: 0.5})
      if i % 100 == 0:
        print(' iteration', i, 'of', 20000)
    print('evaluating feature vectors')
    real = feature_vectors.eval(feed_dict={input: real})
    fakes = [feature_vectors.eval(feed_dict={input: fake}) for fake in fakes]
  return real, fakes
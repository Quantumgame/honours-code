import tensorflow as tf
import numpy as np
import scipy.ndimage.filters

import data.mnist as mnist
    
def noise(arr, min_prop, max_prop):
    arr = arr.copy()
    proportion = min_prop + np.random.random() * (max_prop - min_prop)
    mask = np.random.random(size=arr.shape[:-1]) < proportion
    arr[mask] = mnist_noise(arr.shape)[mask,:]
    return arr, proportion
    
def noise_top(arr):
    arr = arr.copy()
    half = arr[:arr.shape[0]//2,:,:]
    arr[:arr.shape[0]//2,:,:] = mnist_noise(half.shape)
    return arr
    
def noise_bottom(arr):
    arr = arr.copy()
    half = arr[arr.shape[0]//2:,:,:]
    arr[arr.shape[0]//2:,:,:] = mnist_noise(half.shape)
    return arr
    
def mnist_noise(shape):
    return np.random.choice([0.0, 1.0], size=shape, p=[0.87, 0.13])

class Dataset:
    # TODO: make the datasets savable
    def __init__(self, conf, only_plain=False):
        images, labels = mnist.train()
        test_images, test_labels = mnist.test()
        self.values = 2
        self.labels = 10
        self.test_size = 10000
        self.test_batch = 100
        self.total_test_batches = self.test_size // self.test_batch
            
        input_shape = images.output_shapes
        assert len(input_shape) == 3
        self.height, self.width, self.channels = input_shape.as_list()
        assert self.channels == 1
        
        self.plain_data = tf.data.Dataset.zip((images, labels)).repeat().batch(conf.batch_size)
        self.plain_test_data = tf.data.Dataset.zip((test_images, test_labels)).repeat().batch(self.test_batch)
        self.plain_test_data_full = tf.data.Dataset.zip((test_images, test_labels)).batch(self.test_size)
        
        if only_plain:
            return
        
        self.corrupted_data = tf.data.Dataset.zip((images, labels)).repeat().map(lambda image, label: self.make_corrupted_data(image, label, conf)).batch(conf.batch_size)
        self.corrupted_test_data = tf.data.Dataset.zip((test_images, test_labels)).repeat().map(lambda image, label: self.make_corrupted_data(image, label, conf)).batch(self.test_batch)
        
        self.noise_test_data = tf.data.Dataset.zip((test_images, test_labels)).repeat().map(lambda image, label: self.make_noise_data(image, label, conf)).batch(self.test_batch)
        self.denoising_test_data = tf.data.Dataset.zip((test_images, test_labels)).repeat().map(lambda image, label: self.make_denoising_data(image, label, conf)).batch(self.test_batch)
        self.topgap_test_data = tf.data.Dataset.zip((test_images, test_labels)).repeat().map(lambda image, label: self.make_topgap_data(image, label, conf)).batch(self.test_batch)
        self.bottomgap_test_data = tf.data.Dataset.zip((test_images, test_labels)).repeat().map(lambda image, label: self.make_bottomgap_data(image, label, conf)).batch(self.test_batch)
        
        self.plain_proportions = Dataset.range(1, 2).repeat().batch(conf.batch_size)
        self.plain_proportions_test = Dataset.range(1, 2).repeat().batch(self.test_size)
        
    def make_corrupted_data(self, image, label, conf):    
        corrupted, proportion = tf.py_func(lambda arr: noise(arr, conf.min_noise_prop, conf.max_noise_prop), [image], [tf.float32, tf.float32])
        return (corrupted, image, label, proportion)
        
    def make_denoising_data(self, image, label, conf):    
        corrupted, proportion = tf.py_func(lambda arr: noise(arr, 0.5, 0.5), [image], [tf.float32, tf.float32])
        return (corrupted, image, label)
        
    def make_topgap_data(self, image, label, conf):    
        corrupted = tf.py_func(lambda arr: noise_top(arr), [image], [tf.float32])
        return (corrupted, image, label)
        
    def make_bottomgap_data(self, image, label, conf):    
        corrupted = tf.py_func(lambda arr: noise_bottom(arr), [image], [tf.float32])
        return (corrupted, image, label)
        
    def make_noise_data(self, image, label, conf):
        pure_noise, proportion = tf.py_func(lambda arr: noise(arr, 1.0, 1.0), [image], [tf.float32, tf.float32])
        return pure_noise
        
    def get_plain_values(self):
        # Returns (input, label) tuples
        return self.plain_data.make_one_shot_iterator().get_next()
        
    def get_corrupted_values(self):
        # Returns (corrupted, true, label, proportion) tuples
        return self.corrupted_data.make_one_shot_iterator().get_next()
        
    def get_plain_test_values(self):
        # Returns (input, label) tuples
        return self.plain_test_data.make_one_shot_iterator().get_next()
        
    def get_plain_test_values_full(self):
        # Returns (input, label) tuples
        return self.plain_test_data_full.make_one_shot_iterator().get_next()
        
    def get_corrupted_test_values(self):
        # Returns (corrupted, true, label, proportion) tuples
        return self.corrupted_test_data.make_one_shot_iterator().get_next()
        
    def get_plain_proportions(self):
        return self.plain_proportions.make_one_shot_iterator().get_next()
        
    def get_plain_proportions_test(self):
        return self.plain_proportions_test.make_one_shot_iterator().get_next()
        
    def get_noise_values(self):
        # Returns noise values
        return self.noise_test_data.make_one_shot_iterator().get_next()
        
    def get_denoising_values(self):
        # Returns (corrupted, true, label) tuples
        return self.denoising_test_data.make_one_shot_iterator().get_next()
        
    def get_topgap_values(self):
        # Returns (corrupted, true, label) tuples
        return self.topgap_test_data.make_one_shot_iterator().get_next()
        
    def get_bottomgap_values(self):
        # Returns (corrupted, true, label) tuples
        return self.bottomgap_test_data.make_one_shot_iterator().get_next()
import tensorflow as tf
import numpy as np
import scipy.ndimage.filters

import data.mnist as mnist
import data.imagenet as imagenet    
    
def noise(arr, proportion, generator):
    arr = arr.copy()
    mask = np.random.random(size=arr.shape[:-1]) < proportion
    arr[mask] = generator(arr.shape)[mask,:]
    return arr
    
def noise_top(arr, generator):
    arr = arr.copy()
    half = arr[:arr.shape[0]//2,:,:]
    arr[:arr.shape[0]//2,:,:] = generator(half.shape)
    return arr
    
def noise_bottom(arr, generator):
    arr = arr.copy()
    half = arr[arr.shape[0]//2:,:,:]
    arr[arr.shape[0]//2:,:,:] = generator(half.shape)
    return arr
    
def mnist_noise(shape):
    return np.random.choice([0.0, 1.0], size=shape, p=[0.87, 0.13])
    
def blur(arr, sigma):
    return scipy.ndimage.filters.gaussian_filter(arr, [sigma,sigma,0])
    
def downsample(image, factor, height, width):
    assert height % factor == 0
    assert width % factor == 0
    return tf.image.resize_images(tf.image.resize_images(image, [height//factor, width//factor], method=tf.image.ResizeMethod.BICUBIC), [height, width], method=tf.image.ResizeMethod.BILINEAR)
    
def concat_datasets(ds):
    d = tf.data.Dataset.from_tensors(ds[0])
    for dd in ds[1:]:
        d = d.concatenate(tf.data.Dataset.from_tensors(dd))
    return d
    

num_corruptions = 5

class Dataset:
    def __init__(self, conf):        
        if conf.dataset == 'mnist':
            images, labels = mnist.train()
            test_images, test_labels = mnist.test()
            self.values = 2
            self.labels = 10
            self.test_size = 10000
            self.noise_generator = mnist_noise
        else:
            images, labels = imagenet.train()
            test_images, test_labels = imagenet.test()
            self.values = 256
            self.labels = 1000
            self.test_size = None # TODO
            self.noise_generator = None # TODO
            
        input_shape = images.output_shapes
        assert len(input_shape) == 3
        self.height, self.width, self.channels = input_shape.as_list()
        
        self.plain_data = tf.data.Dataset.zip((images, labels))
        self.plain_data = self.plain_data.repeat().batch(conf.batch_size)
        
        self.plain_test_data = tf.data.Dataset.zip((test_images, test_labels))
        self.plain_test_data = self.plain_test_data.repeat().batch(50)
        
        self.corrupted_data = tf.data.Dataset.zip((images, labels)).flat_map(lambda image, label: self.make_corrupted_data(image, label, conf))
        self.corrupted_data = self.corrupted_data.repeat().batch(conf.batch_size)
        
        self.corrupted_test_data = tf.data.Dataset.zip((test_images, test_labels)).flat_map(lambda image, label: self.make_corrupted_data(image, label, conf))
        self.corrupted_test_data = self.corrupted_test_data.repeat().batch(50)
        
        self.noise_data = tf.data.Dataset.zip((images, labels)).map(lambda image, label: self.make_noise_data(image, label, conf))
        self.noise_data = self.noise_data.repeat().batch(conf.batch_size)
        
        self.noise_test_data = tf.data.Dataset.zip((test_images, test_labels)).map(lambda image, label: self.make_noise_data(image, label, conf))
        self.noise_test_data = self.noise_test_data.repeat().batch(50)
        
    def make_corrupted_data(self, image, label, conf):    
        #pure_noise = tf.py_func(lambda arr: noise(arr, 1.0, self.noise_generator), [image], tf.float32)
        gap_top = tf.py_func(lambda arr: noise_top(arr, self.noise_generator), [image], tf.float32)
        gap_bottom = tf.py_func(lambda arr: noise_bottom(arr, self.noise_generator), [image], tf.float32)
        noisy = tf.py_func(lambda arr: noise(arr, conf.noise_prop, self.noise_generator), [image], tf.float32)
        blurry = tf.py_func(lambda arr: blur(arr, conf.blur_sigma), [image], tf.float32)
        downsampled = downsample(image, conf.upsample_factor, self.height, self.width)
        
        datasets = [gap_top, gap_bottom, noisy, blurry, downsampled]
        assert len(datasets) == num_corruptions
        return concat_datasets([(data, image, label) for data in datasets])
        
    def make_noise_data(self, image, label, conf):
        pure_noise = tf.py_func(lambda arr: noise(arr, 1.0, self.noise_generator), [image], tf.float32)
        return (pure_noise, image, label)
        
    def get_plain_values(self):
        # Returns (input, label) tuples
        return self.plain_data.make_one_shot_iterator().get_next()
        
    def get_corrupted_values(self):
        # Returns (corrupted, true, label) tuples
        return self.corrupted_data.make_one_shot_iterator().get_next()
        
    def get_noise_values(self):
        # Returns (noise, true, label) tuples
        return self.noise_data.make_one_shot_iterator().get_next()
        
    def get_plain_test_values(self):
        # Returns (input, label) tuples
        return self.plain_test_data.make_one_shot_iterator().get_next()
        
    def get_corrupted_test_values(self):
        # Returns (corrupted, true, label) tuples
        return self.corrupted_test_data.make_one_shot_iterator().get_next()
        
    def get_noise_test_values(self):
        # Returns (noise, true, label) tuples
        return self.noise_test_data.make_one_shot_iterator().get_next()
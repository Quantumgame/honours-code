import tensorflow as tf
import numpy as np
import scipy.ndimage.filters

import data.mnist as mnist
import data.imagenet as imagenet    
    
def noise(arr, proportion): #TODO: do not treat channels separately
    arr = arr.copy()
    mask = np.random.random(size=arr.shape) < proportion
    arr[mask] = np.random.random(size=arr.shape)[mask]
    return arr
    
def noise_top(arr):
    arr = arr.copy()
    half = arr[:arr.shape[0]//2,:,:]
    arr[:arr.shape[0]//2,:,:] = np.random.random(size=half.shape)
    return arr
    
def noise_bottom(arr):
    arr = arr.copy()
    half = arr[arr.shape[0]//2:,:,:]
    arr[arr.shape[0]//2:,:,:] = np.random.random(size=half.shape)
    return arr
    
def get_noisy(data, proportion):
    return data.map(lambda d: tf.py_func(lambda arr: noise(arr, proportion), [d], tf.float32))
    
def blur(arr, sigma):
    return scipy.ndimage.filters.gaussian_filter(arr, [sigma,sigma,0])
    
def get_blurry(data, sigma):
    return data.map(lambda d: tf.py_func(lambda arr: blur(arr, sigma), [d], tf.float32))
    
def downsample(image, factor, height, width):
    assert height % factor == 0
    assert width % factor == 0
    return tf.image.resize_images(tf.image.resize_images(image, [height//factor, width//factor], method=tf.image.ResizeMethod.BICUBIC), [height, width], method=tf.image.ResizeMethod.BILINEAR)
    
def get_downsampled(data, factor, height, width):
    assert height % factor == 0
    assert width % factor == 0
    return data.map(lambda d: downsample(d, factor, height, width))
    
def concat_datasets(ds):
    d = tf.data.Dataset.from_tensors(ds[0])
    for dd in ds[1:]:
        d = d.concatenate(tf.data.Dataset.from_tensors(dd))
    return d
    
def make_test_data(image, label, conf, height, width):
    data = tf.data.Dataset.from_tensors(())
    
    pure_noise = tf.py_func(lambda arr: noise(arr, 1.0), [image], tf.float32)
    gap_top = tf.py_func(lambda arr: noise_top(arr), [image], tf.float32)
    gap_bottom = tf.py_func(lambda arr: noise_bottom(arr), [image], tf.float32)
    denoise = tf.py_func(lambda arr: noise(arr, conf.noise_prop), [image], tf.float32)
    deblur = tf.py_func(lambda arr: blur(arr, conf.blur_sigma), [image], tf.float32)
    upsample = downsample(image, conf.upsample_factor, height, width)
    
    datasets = [pure_noise, gap_top, gap_bottom, denoise, deblur, upsample]
    datasets = [(data, image, label) for data in datasets]
    return concat_datasets(datasets)
    

class Dataset:
    def __init__(self, conf):
        assert conf.plain_images or conf.denoising or conf.deblurring or conf.upsampling
        
        if conf.dataset == 'mnist':
            images, labels = mnist.train()
            test_images, test_labels = mnist.test()
            self.values = 2
            self.labels = 10
        else:
            images, labels = imagenet.train()
            test_images, test_labels = imagenet.test()
            self.values = 256
            self.labels = 1000
            
        input_shape = images.output_shapes
        assert len(input_shape) == 3
        self.height, self.width, self.channels = input_shape.as_list()
            
        plain = tf.data.Dataset.zip((images, images, labels))
        noisy = tf.data.Dataset.zip((get_noisy(images, conf.noise_prop), images, labels))
        blurry = tf.data.Dataset.zip((get_blurry(images, conf.blur_sigma), images, labels))
        downsampled = tf.data.Dataset.zip((get_downsampled(images, conf.upsample_factor, self.height, self.width), images, labels))
        
        datas = []
        if conf.plain_images:
            datas.append(plain)
        if conf.denoising:
            datas.append(noisy)
        if conf.deblurring:
            datas.append(blurry)
        if conf.upsampling:
            datas.append(downsampled)
        
        # Interleave datasets, see: https://stackoverflow.com/questions/47343228/interleaving-tf-data-datasets
        data = tf.data.Dataset.zip(tuple(datas)).flat_map(lambda *ds: concat_datasets(ds))
        self.data = data.repeat().batch(conf.batch_size)
        
        self.test_data = tf.data.Dataset.zip((test_images, test_labels)).flat_map(lambda image, label: make_test_data(image, label, conf, self.height, self.width))
        self.test_data = self.test_data.batch(conf.batch_size)
        
        
    def get_values(self):
        # Returns (input, output, label) tuples
        return self.data.make_one_shot_iterator().get_next()
        
    def get_test_values(self):
        return self.test_data.make_one_shot_iterator().get_next()
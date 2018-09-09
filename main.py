import argparse

import tensorflow as tf

from data.dataset import Dataset
from pixelcnn import PixelCNN
from simultaneous import Simultaneous
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--model', type=str, choices=['pixelcnn', 'simultaneous'], default='simultaneous')
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--features', type=int, default=None) 
    parser.add_argument('--filter_size', type=int, default=None)
    parser.add_argument('--batches', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'imagenet'], default='mnist')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--train_type', type=str, choices=['full', 'one_iteration'], default='full')
    parser.add_argument('--conditioning', type=str, choices=['none', 'global', 'local'], default='none')
    parser.add_argument('--temperature', type=float, default=1/1.05)
    parser.add_argument('--plain_images', action='store_false') #todo fix this
    parser.add_argument('--denoising', action='store_true')
    parser.add_argument('--noise_prop', type=float, default=0.5)
    parser.add_argument('--deblurring', action='store_true')
    parser.add_argument('--blur_sigma', type=float, default=1.0)
    parser.add_argument('--upsampling', action='store_true')
    parser.add_argument('--upsample_factor', type=int, default=2)
    conf = parser.parse_args()
    
    if conf.dataset == 'mnist':
        conf.layers = conf.layers or 15
        conf.features = conf.features or 16
        conf.filter_size = conf.filter_size or 5
        conf.batches = conf.batches or 200000
        conf.batch_size = conf.batch_size or 32
        conf.learning_rate = conf.learning_rate or 1e-4
    elif conf.dataset == 'imagenet':
        conf.layers = conf.layers or 15
        conf.features = conf.features or 128
        conf.filter_size = conf.filter_size or 5 # "We trained the model on 32 × 32 images. The PixelCNN module used 10 layers with 128 feature maps"
        conf.batches = conf.batches or 200000
        conf.batch_size = conf.batch_size or 128 # "RMSprop with a learning rate schedule starting at 1e-4 and decaying to 1e-5, trained for 200k steps with batch size of 128"
        conf.learning_rate = conf.learning_rate or 1e-4
    else:
        assert False
    
    # for quick test use python .\main.py --layers 3 --features 5 --batches 101 --batch_size 6 --conditioning none
    
    data = Dataset(conf)
    model = PixelCNN(conf, data) if conf.model == 'pixelcnn' else Simultaneous(conf, data)
    
    if conf.test:
        model.run_tests()
    else:
        model.run()

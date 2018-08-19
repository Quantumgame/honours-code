import argparse

import tensorflow as tf

import data.mnist as mnist
import data.imagenet as imagenet
from data.dataset import Dataset
from pixelcnn import PixelCNN
from simultaneous import Simultaneous

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'imagenet'], default='mnist')
    parser.add_argument('--recurrence', type=str, choices=['double', 'end', 'within'], default='end')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--train_type', type=str, choices=['full', 'one_iteration'], default='full')
    parser.add_argument('--conditioning', type=str, choices=['none', 'global', 'local'], default='global')
    parser.add_argument('--temperature', type=float, default=1/1.05)
    #parser.add_argument('--grad_clip', type=int, default=1)
    #parser.add_argument('--ckpt_path', type=str, default='ckpts')
    #parser.add_argument('--summary_path', type=str, default='logs')
    conf = parser.parse_args()
    # for quick test use python .\main.py --layers 3 --features 5 --batches 101 --batch_size 6 --conditioning none
    
    data = Dataset(mnist.train(), mnist.test(), 2, 10, conf.batch_size) if conf.dataset == 'mnist' else Dataset(imagenet.train(), imagenet.test(), 256, 1000, conf.batch_size)
    #pixelcnn = PixelCNN(conf, data)
    #pixelcnn.run()
    simultaneous = Simultaneous(conf, data)
    simultaneous.run()

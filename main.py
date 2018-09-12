import argparse

import tensorflow as tf

from data.dataset import Dataset
from pixelcnn import PixelCNN
from noncausal import NonCausal
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='run unit tests instead of training model')
    parser.add_argument('--restore', action='store_true', help='restore from checkpoint (must exist)')
    parser.add_argument('--model', type=str, choices=['pixelcnn', 'noncausal'], required=True)
    parser.add_argument('--layers', type=int, default=None, help='number of convolutional layers')
    parser.add_argument('--features', type=int, default=None, help='number of convolutional filters per layer')
    parser.add_argument('--end_features', type=int, default=None, help='number of features in the final fully-connected layer') 
    parser.add_argument('--filter_size', type=int, default=None, help='size of convolutional filters')
    parser.add_argument('--epochs', type=int, default=None, help='number of training epochs to run')
    parser.add_argument('--batch_size', type=int, default=None, help='number of samples per minibatch (training epoch)')
    parser.add_argument('--learning_rate', type=float, default=None, help='RMSProp learning rate')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'imagenet'], default='mnist')
    parser.add_argument('--conditioning', type=str, choices=['none', 'generic', 'localised'], default='none', help='how to condition on class labels: none = not at all, generic = overall condition for whole image, localised = condition can vary across the image')
    parser.add_argument('--temperature', type=float, default=1/1.05, help='Temperature for random sampling at the softmax layer')
    ## These options only apply to noncausal
    parser.add_argument('--noise_prop', type=float, default=0.5, help='Proportion of pixels to replace with noise for denoising trials')
    parser.add_argument('--blur_sigma', type=float, default=1.0, help='Sigma for gaussian blur for deblurring trials')
    parser.add_argument('--upsample_factor', type=int, default=2, help='Factor by which to downsample the data for upsampling trials')
    ## These options apply only to noncausal
    parser.add_argument('--train_iterations', type=int, default=2, help='How many times to apply the noncausal model to the inputs, backpropogating errors each time')
    parser.add_argument('--test_iterations', type=int, default=4, help='At test time, how many times to apply the noncausal model to the inputs, measuring the error each time')
    conf = parser.parse_args()
    
    if conf.dataset == 'mnist':
        conf.layers = conf.layers or 15
        conf.features = conf.features or 16
        conf.end_features = conf.end_features or 64
        conf.filter_size = conf.filter_size or 5
        conf.epochs = conf.epochs or 200000
        conf.batch_size = conf.batch_size or 32
        conf.learning_rate = conf.learning_rate or 1e-4
    elif conf.dataset == 'imagenet':
        conf.layers = conf.layers or 15
        conf.features = conf.features or 128
        conf.end_features = conf.end_features or 1024
        conf.filter_size = conf.filter_size or 5
        conf.epochs = conf.epochs or 200000
        conf.batch_size = conf.batch_size or 128
        conf.learning_rate = conf.learning_rate or 1e-4
    else:
        assert False
    
    # for quick test use python .\main.py --layers 3 --features 5 --end_features 10 --epochs 101 --batch_size 6 --model pixelcnn
    
    data = Dataset(conf)
    model = PixelCNN(conf, data) if conf.model == 'pixelcnn' else NonCausal(conf, data)
    
    if conf.test:
        model.run_tests()
    else:
        model.run()

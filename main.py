import argparse

import tensorflow as tf

from data.dataset import Dataset
from pixelcnn import PixelCNN
from noncausal import NonCausal
from metric import get_metric
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='run unit tests instead of training model')
    parser.add_argument('--restore', action='store_true', help='restore from checkpoint (must exist)')
    parser.add_argument('--samples', action='store_true', help='generate samples from trained model')
    parser.add_argument('--model', type=str, choices=['pixelcnn', 'noncausal', 'evaluate'], required=True)
    parser.add_argument('--layers', type=int, default=15, help='number of convolutional layers')
    parser.add_argument('--features', type=int, default=16, help='number of convolutional filters per layer')
    parser.add_argument('--end_features', type=int, default=64, help='number of features in the final fully-connected layer') 
    parser.add_argument('--filter_size', type=int, default=5, help='size of convolutional filters')
    parser.add_argument('--iterations', type=int, default=200000, help='number of training iterations to run')
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples per minibatch (training epoch)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='RMSProp learning rate')
    parser.add_argument('--temperature', type=float, default=1/1.05, help='Temperature for random sampling at the softmax layer')
    ## These options only apply to noncausal
    parser.add_argument('--noise_prop', type=float, default=0.5, help='Proportion of pixels to replace with noise for denoising trials')
    parser.add_argument('--blur_sigma', type=float, default=1.0, help='Sigma for gaussian blur for deblurring trials')
    parser.add_argument('--upsample_factor', type=int, default=2, help='Factor by which to downsample the data for upsampling trials')
    ## These options apply only to noncausal
    parser.add_argument('--train_iterations', type=int, default=2, help='How many times to apply the noncausal model to the inputs, backpropogating errors each time')
    parser.add_argument('--test_iterations', type=int, default=4, help='At test time, how many times to apply the noncausal model to the inputs, measuring the error at the end')
    conf = parser.parse_args()
    
    # for quick test use python .\main.py --layers 3 --features 5 --end_features 10 --iterations 101 --batch_size 6 --model pixelcnn
    
    if conf.model == 'evaluate':
        test_data = Dataset(conf).get_plain_test_values_full()
        with tf.Session() as sess: 
            X, _ = sess.run(test_data)
        get_metric(X, [X])
    else:
        data = Dataset(conf)
        model = PixelCNN(conf, data) if conf.model == 'pixelcnn' else NonCausal(conf, data)
        
        if conf.test:
            model.run_tests()
        elif conf.samples:
            model.samples()
        else:
            model.run()

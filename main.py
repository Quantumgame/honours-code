import argparse

import tensorflow as tf
import numpy as np

from data.dataset import Dataset
from pixelcnn import PixelCNN
from noncausal import NonCausal
from metric import get_metric
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='run unit tests instead of training model')
    parser.add_argument('--restore', action='store_true', help='restore from checkpoint (must exist)')
    parser.add_argument('--samples', action='store_true', help='generate samples from trained model')
    parser.add_argument('--model', type=str, choices=['pixelcnn', 'denoising', 'noncausal', 'evaluate'], required=True)
    parser.add_argument('--layers', type=int, default=15, help='number of convolutional layers')
    parser.add_argument('--features', type=int, default=16, help='number of convolutional filters per layer')
    parser.add_argument('--end_features', type=int, default=64, help='number of features in the final fully-connected layer') 
    parser.add_argument('--filter_size', type=int, default=5, help='size of convolutional filters')
    parser.add_argument('--iterations', type=int, default=200000, help='number of training iterations to run')
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples per minibatch (training epoch)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='RMSProp learning rate')
    parser.add_argument('--temperature', type=float, default=1/1.05, help='Temperature for random sampling at the softmax layer')
    ## These options only apply to noncausal and denoising
    parser.add_argument('--min_noise_prop', type=float, default=0.05, help='Minimum proportion of pixels to replace with noise for training')
    parser.add_argument('--max_noise_prop', type=float, default=0.95, help='Maximum proportion of pixels to replace with noise for training')
    #parser.add_argument('--train_iterations', type=int, default=2, help='How many times to apply the noncausal model to the inputs, backpropagating errors each time')
    parser.add_argument('--test_iterations', type=int, default=20, help='When generating samples, how many times to apply the noncausal and denoising models to the inputs')
    conf = parser.parse_args()
    
    # for quick test use python .\main.py --layers 3 --features 5 --end_features 10 --iterations 101 --batch_size 6 --model pixelcnn
    
    if conf.model == 'evaluate':
        data = Dataset(conf)
        test_data = data.get_plain_test_values()
        with tf.Session() as sess:
            samples = []
            for _ in range(data.total_test_batches):
                X, _ = sess.run(test_data)
                samples.append(X)
            X = np.concatenate(samples)
            print(X.shape)
        X_pixelcnn = PixelCNN(conf, data, False).get_test_samples()
        X_denoising = PixelCNN(conf, data, True).get_test_samples()
        X_noncausal = NonCausal(conf, data).get_test_samples()
        get_metric(X, [X_pixelcnn,X_denoising,X_noncausal])
    else:
        data = Dataset(conf)
        model = NonCausal(conf, data) if conf.model == 'noncausal' else PixelCNN(conf, data, conf.model == 'denoising')
        
        if conf.test:
            model.run_tests()
        elif conf.samples:
            model.samples()
        else:
            model.run()

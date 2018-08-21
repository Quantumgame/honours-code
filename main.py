import argparse

import tensorflow as tf

from data.dataset import Dataset
from pixelcnn import PixelCNN
from simultaneous import Simultaneous
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=str, choices=['pixelcnn', 'simultaneous'], default='simultaneous')
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--features', type=int, default=128) # "We trained the model on 32 × 32 images. The PixelCNN module used 10 layers with 128 feature maps"
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'imagenet'], default='mnist')
    parser.add_argument('--recurrence', type=str, choices=['double', 'end', 'within'], default='end')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--train_type', type=str, choices=['full', 'one_iteration'], default='full')
    parser.add_argument('--conditioning', type=str, choices=['none', 'global', 'local'], default='global')
    parser.add_argument('--temperature', type=float, default=1/1.05)
    parser.add_argument('--plain_images', action='store_true')
    parser.add_argument('--denoising', action='store_true')
    parser.add_argument('--noise_prop', type=float, default=0.5)
    parser.add_argument('--deblurring', action='store_true')
    parser.add_argument('--blur_sigma', type=float, default=1.0)
    parser.add_argument('--upsampling', action='store_true')
    parser.add_argument('--upsample_factor', type=int, default=2)
    #parser.add_argument('--grad_clip', type=int, default=1)
    #parser.add_argument('--ckpt_path', type=str, default='ckpts')
    #parser.add_argument('--summary_path', type=str, default='logs')
    conf = parser.parse_args()
    
    # for quick test use python .\main.py --layers 3 --features 5 --batches 101 --batch_size 6 --conditioning none
    
    data = Dataset(conf)
    model = PixelCNN(conf, data) if conf.model == 'pixelcnn' else Simultaneous(conf, data)
    
    if conf.test:
        model.run_tests()
    else:
        model.run()

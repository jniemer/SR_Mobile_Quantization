import os
import argparse
import cv2
import numpy as np
from options import parse
from data import DIV2K
from data.converter import DataConverter
import tensorflow as tf
#import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Image to process')
    parser.add_argument('-m', '--model', default='best_model.h5', help='model file')
    args = parser.parse_args()
    input_file = args.input
    model_file = args.model
    print(input_file)
    print(model_file)

    if (os.path.exists(input_file)):
        if (os.path.exists(model_file)):
            img = cv2.imread(input_file)      
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            model = tf.keras.models.load_model(model_file, custom_objects={'tf': tf})

            model.predict(img)
        else:
            print('Model file does not exist, exiting')
    else:
        print('Image file does not exist, exiting')'''
    
    
    if __name__ == '__main__':

        parser = argparse.ArgumentParser(description='FSRCNN Demo')
        parser.add_argument('--opt', required=True)
        parser.add_argument('--name', required=True)
        parser.add_argument('--scale', default=3, type=int)
        parser.add_argument('--ps', default=48, type=int, help='patch_size')
        parser.add_argument('--bs', default=16, type=int, help='batch_size')
        parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
        parser.add_argument('--gpu_ids', default=None)
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--resume_path', default=None)
        parser.add_argument('--qat', action='store_true', default=False)
        parser.add_argument('--qat_path', default=None)

        args = parser.parse_args()
        args, lg = parse(args)

        # create dataset
        train_data = DIV2K(args['datasets']['train'])
        lg.info('Create train dataset successfully!')
        lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))
        print(train_data.get_patch().shape())
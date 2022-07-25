import os
import argparse
import cv2
import numpy as np
from data.converter import DataConverter
import tensorflow as tf
#import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
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
        print('Image file does not exist, exiting')
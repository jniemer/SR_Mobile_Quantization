import os
import argparse
import cv2
import numpy as np
from data.converter import DataConverter
import tensorflow as tf
import pickle
import solvers.networks.base7
#import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Image to process')
    parser.add_argument('-i', '--output', required=True, help='Output file')
    parser.add_argument('-m', '--model', default='models/best_status', help='model file')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    model_file = args.model
    print(input_file)
    print(model_file)

    if (os.path.exists(input_file)):
        if (os.path.exists(model_file)):
            #img = cv2.imread(input_file)      
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with open(input_file, "rb") as file:
                img = pickle.load(file)
                img_new = img.reshape(1, 256, 256, 3)

                model = tf.keras.models.load_model(model_file, custom_objects={'tf': tf})

                output = model.predict(img_new)
                output = output.reshape(1024, 1024, 3)
                print(output.shape)
                outputimg = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_file, outputimg)
                print('done!')
        else:
            print('Model file does not exist, exiting')
    else:
        print('Image file does not exist, exiting')

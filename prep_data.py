import os
import argparse
import cv2
import numpy as np
from data.converter import DataConverter
from options import parse
from solvers import Solver
from data import DIV2K
import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
import os
import os.path as osp
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args, lg = parse(args)

    # create dataset
    lg.info('Converting training data')
    train_data_converter = DataConverter(args['datasets']['train'])
    lg.info('Training data converted!')

    
    lg.info('Converting validation data')
    train_data_converter = DataConverter(args['datasets']['val'])
    lg.info('Validation data converted!')
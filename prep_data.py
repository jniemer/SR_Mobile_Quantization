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
    lg.info('Converting training data')
    train_data_converter = DataConverter(args['datasets']['train'])
    train_data_converter.convert_images()
    lg.info('Training data converted!')

    
    lg.info('Converting validation data')
    val_data_converter = DataConverter(args['datasets']['val'])
    val_data_converter.convert_images()
    lg.info('Validation data converted!')
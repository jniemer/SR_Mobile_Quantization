import h5py
import numpy as np
import tensorflow as tf
import cv2
import random
import os
import os.path as osp
import pickle

class DataConverter():
    def __init__(self, dataset):
        self.dataset = dataset

    def convert_images(self):
        self.convert_img_to_pt(key='dataroot_HR')
        self.convert_img_to_pt(key='dataroot_LR')

    def convert_img_to_pt(self, key):
        if self.dataset[key][-1] == '/':
            self.dataset[key] = self.dataset[key][:-1]
        img_list = os.listdir(self.dataset[key])
        
        need_convert = False
        for i in range(len(img_list)):
            _, ext = osp.splitext(img_list[i])
            if ext != '.pt':
                need_convert = True
                break
        if need_convert == False:
            return
        
        new_dir_path = self.dataset[key] + '_pt'
        if osp.exists(new_dir_path) and len(os.listdir(new_dir_path))==len(img_list):
            self.dataset[key] = new_dir_path
            return

        os.makedirs(new_dir_path, exist_ok=True)
        for i in range(len(img_list)):
            base, ext = osp.splitext(img_list[i])
            src_path = osp.join(self.dataset[key], img_list[i])
            dst_path = osp.join(new_dir_path, base+'.pt')             
            with open(dst_path, 'wb') as _f:
                img = cv2.imread(src_path)      
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pickle.dump(img, _f)
        self.dataset[key] = new_dir_path
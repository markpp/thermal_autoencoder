import os
import numpy as np
import torch
import torch.utils.data
import argparse

import cv2
import random
import math
from glob import glob

from views import views

parser = argparse.ArgumentParser(description='modify script parameters when necessary.')
parser.add_argument('--root_dir', type=str, help='where is the data located?', default='/home/markpp/datasets/harbour_frames/2/')
parser.add_argument('--set', type=str, help='what type of dataset should this be?', default='train') #or 'train' or 'val'
parser.add_argument('--view', type=str, help='which view folder', default='view1_normal')
parser.add_argument('--output', type=str, help='where to put the output?', default='data/')
parser.add_argument('--use_flow', type=bool, help='use flow vectors?', default=False)
args = parser.parse_args()

class HarborDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, crop_size=64, crop_idx=0, USE_FLOW=False):
        self.crop_size = crop_size
        self.crop_idx = crop_idx
        self.USE_FLOW = USE_FLOW
        self.image_list = sorted([y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], 'img_*.jpg'))])
        if not len(self.image_list)>0:
            print("did not find any files")

    def load_sample(self, image_path):
        thermal = cv2.imread(image_path)
        self.image_h, self.image_w, _ = thermal.shape
        if self.USE_FLOW:
            thermal[:,:,1] = cv2.imread(image_path.replace('img_','flow_x_'), -1)
            thermal[:,:,2] = cv2.imread(image_path.replace('img_','flow_y_'), -1)
        else:
            thermal = thermal[:,:,0]
        return thermal, image_path.split('/')[-3]

    def crop(self, img, view):
        x, y = views[view][self.crop_idx]['x'], views[view][self.crop_idx]['y']
        crop = img[y:y+self.crop_size, x:x+self.crop_size]
        return crop

    def __getitem__(self, idx):
        img, view = self.load_sample(self.image_list[idx])
        img = self.crop(img, view)
        return img, view

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    view_dir = os.path.join(args.output,args.view)
    if not os.path.exists(view_dir):
        os.mkdir(view_dir)

    for crop_idx in range(len(views[args.view])):
        crop_dir = os.path.join(view_dir,"crop{}".format(crop_idx))
        if not os.path.exists(crop_dir):
            os.mkdir(crop_dir)

        set_dir = os.path.join(crop_dir,args.set)
        if not os.path.exists(set_dir):
            os.mkdir(set_dir)

        data_dir = os.path.join(args.root_dir,args.view)
        dataset = HarborDataset(data_dir,
                                crop_size = 64,
                                crop_idx = crop_idx,
                                USE_FLOW=args.use_flow)

        print(len(dataset))
        for i, data in enumerate(dataset):
            input, view = data
            cv2.imwrite(os.path.join(set_dir,"{}.png".format(str(i).zfill(5))),input)

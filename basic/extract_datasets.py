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
parser.add_argument('--root_dir', type=str, help='where is the data located?', default='data/')
#parser.add_argument('--root_dir', type=str, help='where is the data located?', default='/home/markpp/datasets/harbour_frames/2/')
parser.add_argument('--set', type=str, help='what type of dataset should this be?', default='train') #or 'train' or 'val'
parser.add_argument('--view', type=str, help='which view folder', default='normal')
parser.add_argument('--output', type=str, help='where to put the output?', default='output/')
parser.add_argument('--use_flow', type=bool, help='use flow vectors?', default=False)
args = parser.parse_args()


def four_point_transform(rect):
    # make sure the points follow a consistent order
    # unpack them
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    return M, maxWidth, maxHeight

class HarborDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, crop_size=64):
        self.crop_size = crop_size
        self.image_list = sorted([y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], 'img_*.jpg'))])
        self.folder = self.image_list[0].split('/')[-2].replace(' ','')
        print(self.folder)
        #self.image_list = sorted(glob(os.path.join(root_dir, '*.jpg')))
        if not len(self.image_list)>0:
            print("did not find any files")

        # points from roi.txt 
        pts = np.array([[52,108],[67,108],[170,265],[112,270]])
        rect = np.zeros((4, 2), dtype = "float32")
        rect[0], rect[3] = pts[0], pts[3]
        rect[1], rect[2] = pts[1], pts[2]
        self.M, self.maxWidth, self.maxHeight = four_point_transform(rect)

    def load_sample(self, image_path):
        thermal = cv2.imread(image_path)
        self.image_h, self.image_w, _ = thermal.shape
        thermal = thermal[:,:,0]
        return thermal

    def crop(self, img):
        '''
        # simple square crops
        x, y = views[view][self.crop_idx]['x'], views[view][self.crop_idx]['y']
        crop = img[y:y+self.crop_size, x:x+self.crop_size]
        '''
        # warped roi crop
        crop = cv2.warpPerspective(img, self.M, (self.maxWidth, self.maxHeight))
        crop = cv2.resize(crop, (64,192), interpolation=cv2.INTER_LINEAR)
        return crop

    def __getitem__(self, idx):
        img = self.load_sample(self.image_list[idx])
        img = self.crop(img)
        return img, self.folder

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    set_dir = os.path.join(args.output,args.set)
    if not os.path.exists(set_dir):
        os.mkdir(set_dir)

    view_dir = os.path.join(set_dir,args.view)
    if not os.path.exists(view_dir):
        os.mkdir(view_dir)

    input_dir = os.path.join(args.root_dir,args.set,args.view)
    dataset = HarborDataset(input_dir)

    print(len(dataset))
    for i, data in enumerate(dataset):
        crop, folder_name = data
        cv2.imwrite(os.path.join(view_dir,"{}_{}.jpg".format(folder_name,str(i).zfill(5))),crop)

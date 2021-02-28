# coding=utf-8
from __future__ import division
import sys
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import PIL.Image as pil
import matplotlib.cm as cm
from depth_evaluation_utils_show import *

parser = argparse.ArgumentParser()
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--GT_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument("--test_file_list", type=str, default='/media/deep/Ubuntu2/ubuntu/unlearner_mask_0/data/kitti/test_files_dashiyan.txt', help="Path to the list of test files")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
args = parser.parse_args()

def main():
    pred_depths = np.load(args.pred_file)
    test_files = read_text_lines(args.test_file_list)
    rgb_file, gt_files, gt_calib, im_sizes, im_files, cams= \
        read_file_data_dashiyan(test_files, args.kitti_dir)
    num_test = len(im_files)
    gt_depths = []
    pred_depths_resized = []
    gt_path = os.path.join(args.GT_dir,"gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
    for t_id in range(num_test):
    #for t_id in range(1):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        pred_depths_resized_one=cv2.resize(pred_depths[t_id],
                       (im_sizes[t_id][1], im_sizes[t_id][0]),
                       interpolation=cv2.INTER_LINEAR)
        Ground_truth_depths_resized_one = cv2.resize(gt_depths[t_id],
                                         (im_sizes[t_id][1], im_sizes[t_id][0]),
                                         interpolation=cv2.INTER_LINEAR)
        pred_depthss=normalize_depth_for_display(pred_depths_resized_one)
        pred_depths_resized.append(pred_depths_resized_one )
        gt_depth = normalize_depth_for_display(Ground_truth_depths_resized_one)
        #depth, depth_interp = generate_depth_map(gt_calib[t_id],
        #                           gt_files[t_id],
        #                           im_sizes[t_id],
        #                           camera_id,
        #                           True,
        #                           True)
        #depth = gray2rgb(depth, cmap=cmap)
        #gt_depths.append(depth.astype(np.float32))
        #if int(t_id) in range(1, num_test, 100):
        fh = open(rgb_file[t_id], 'rb')
        raw_im = pil.open(fh)
        scaled_im = raw_im.resize((640,360), pil.ANTIALIAS)
        plt.imshow(scaled_im)
        plt.show()
        plt.imshow(gt_depth,cmap='inferno')
        plt.show()
        plt.imshow(pred_depthss)
        plt.show()
#光流计算看图展示


main()
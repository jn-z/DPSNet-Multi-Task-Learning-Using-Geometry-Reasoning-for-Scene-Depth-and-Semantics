# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
import imageio


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.70, 0, 0.25, 0],
                           [0, 1.18, 0.32, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1280, 760)
        self.resize_shape = (512, 128)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.K_real = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape_real = (1242, 375)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def get_color_real(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def get_label(self, folder, frame_index, do_flip):
        color = self.pil_loader_label(self.get_label_path(folder, frame_index))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, do_flip):
        depth_gt = cv2.imread(self.get_depth_path(folder, frame_index), cv2.IMREAD_UNCHANGED) # uint16
        depth_gt = depth_gt.astype(np.int32)  # Convert to int32 for torch compatibility
        #depth_gt = cv2.imread(self.get_depth_path(folder, frame_index), cv2.IMREAD_UNCHANGED)
        #depth_gt = cv2.resize(depth_gt, self.full_res_shape, interpolation=cv2.INTER_NEAREST)
        #depth_gt = depth_gt
        depth_gt_trans = (depth_gt[:, :, 0] + depth_gt[:, :, 1] * 256 + depth_gt[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1) * 100
        depth_gt_trans = cv2.resize(depth_gt_trans, self.resize_shape, interpolation=cv2.INTER_NEAREST)
        #depth_gt = np.array(depth_gt).astype(np.uint16) #(760, 1280, 3)
        # 5000 * (R + G*256 + B*256*256) / (256*256*256 - 1)
        if do_flip:
            depth_gt_trans = np.fliplr(depth_gt_trans)

        return depth_gt_trans



class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "{:d}".format(int(folder)),
            "RGB",
            f_str)

        return image_path

    def get_label_path(self, folder, frame_index):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        label_path = os.path.join(
            self.data_path,
            "{:d}".format(int(folder)),
            "GT",
            f_str)

        return label_path

    def get_depth_path(self, folder, frame_index):
        f_str = "{:06d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            "{:d}".format(int(folder)),
            "Depth",
            f_str)

        return depth_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path



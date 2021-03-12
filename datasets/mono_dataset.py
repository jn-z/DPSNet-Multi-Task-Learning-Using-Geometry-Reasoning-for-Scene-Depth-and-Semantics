# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_loader_label(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    return Image.open(path)


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 ids,
                 height,
                 width,
                 min_samples,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 trainsets=True):
        super(MonoDataset, self).__init__()

        self.height = height
        self.width = width
        self.min_samples = min_samples
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.frame_idxs = frame_idxs
        self.data_path = data_path
        self.is_train = is_train
        self.img_ext = img_ext
        self.trainsets = trainsets
        self.load_depth = True
        self.loader = pil_loader
        self.pil_loader_label = pil_loader_label
        self.to_tensor = transforms.ToTensor()
        self.ids = ids
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        self.resize_label ={}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
            self.resize_label[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=Image.NEAREST)

        #self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):

            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

            if "color_real" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

            if "label" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_label[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                #inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            if "color_real" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                    # inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            if "label" in k:
                n, im, i = k
                f = self.convert_labels(f)
                f = self.to_tensor(f)
                inputs[(n, im, i)] = self.to_longTensor_gt(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder, num_frames = self.ids[index]
        num_frames_rand = random.randint(1, self.min_samples)
        folder_rand = random.randint(1, 90)
        folder_rand_0 = random.randint(91, 99)
        folder_rand_1 = random.randint(100, 101)
        for i in self.frame_idxs:
            if folder <= 90:
                inputs[("color", i, -1)] = self.get_color(folder, (num_frames + i) + 101, do_flip)
                inputs[("label", i, -1)] = self.get_label(folder, (num_frames + i) + 101, do_flip)
                if self.trainsets:
                    inputs[("color_real", i, -1)] = self.get_color_real(folder_rand_0, (num_frames_rand + i), do_flip)
                else:
                    inputs[("color_real", i, -1)] = self.get_color_real(folder_rand_1, (num_frames_rand + i),do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder_rand, (num_frames_rand + i) + 101, do_flip)
                inputs[("label", i, -1)] = self.get_label(folder_rand, (num_frames_rand + i) + 101, do_flip)
                inputs[("color_real", i, -1)] = self.get_color_real(folder, (num_frames + i), do_flip)
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            # for kitti
            K_real = self.K_real.copy()
            K_real[0, :] *= self.width // (2 ** scale)
            K_real[1, :] *= self.height // (2 ** scale)
            inv_K_real = np.linalg.pinv(K_real)
            inputs[("K_real", scale)] = torch.from_numpy(K_real)
            inputs[("inv_K_real", scale)] = torch.from_numpy(inv_K_real)
            #for synthia
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        if self.load_depth:
            if folder <= 90:
                depth_gt = self.get_depth(folder, (num_frames) + 101, do_flip)
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_real",i, -1)]
            #del inputs[("color_aug", i, -1)]

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, do_flip):
        raise NotImplementedError

    def get_color_real(self, folder, frame_index, do_flip):
        raise NotImplementedError

    def get_label(self, folder, frame_index, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, do_flip):
        raise NotImplementedError

    # this function converts the Synthia labels so they are simpler
    def convert_labels(self, labels):
        labels = np.array(labels)

        return Image.fromarray(labels)

    # this function takes a tensor and turns it into a long tensor if it is an int tensor. This one is used for the segmentation ground truth, so no divisions.
    def to_longTensor_gt(self, img):
        # convert a IntTensor to a LongTensor
        if isinstance(img, torch.IntTensor):
            return img.type(torch.LongTensor)
        elif isinstance(img, torch.LongTensor):
            return img
        else:
            return img
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from networks.networks_ import AdvLoss
import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
import os
from skimage.measure import label
from segmentation import CrossEntropyLoss2d


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        # label zero is to be ignored. synthia data
        weight = torch.ones(self.opt.num_classes)
        weight[0] = 0
        self.criterion_CE = CrossEntropyLoss2d(weight.to(self.device))
        self.criterion_adv = AdvLoss(self.opt.adv_loss).to(self.device)
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
        # segmentation model
        self.models["encoder_rgb"]= networks.Encoder(self.opt.input_nc).to(self.device)
        self.models["discriminator"] = networks.Discriminator(1).to(self.device)
        self.models["decoder_1"] = networks.Decoder_part_one().to(self.device)
        self.models["decoder_2_segmentation"] = networks.Decoder_part_two(num_classes=self.opt.num_classes).to(self.device)

        if self.opt.predictive_mask:
            #assert self.opt.disable_automasking, \
            #    "When using predictive_mask, please disable automasking with --disable_automasking"

            self.models["decoder_3_instance_mask"] = networks.Decoder_part_two_instance().to(self.device)
            self.parameters_to_train += list(self.models["decoder_3_instance_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # Segmentation parameters for train
        self.optimizer_G_shared = optim.Adam(
            list(self.models["encoder_rgb"].parameters()) + list(self.models["decoder_1"].parameters()),
            lr=self.opt.lr_seg_mask)
        self.optimizer_D = optim.Adam(self.models["discriminator"].parameters(), lr=self.opt.lr_seg_mask)
        self.optimizer_G_segmentation = optim.Adam(self.models["decoder_2_segmentation"].parameters(),
                                                   lr=self.opt.lr_seg_mask)
        # self.optimizer_instance_mask = torch.optim.Adam(self.models["decoder_3_instance_mask"].parameters(), lr=self.opt.lr_seg_mask)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_real": datasets.KITTIRAWDataset_real,
                         "kitti_odom_real": datasets.KITTIOdomDataset_real
                         }
        self.dataset = datasets_dict[self.opt.dataset]
        #for SYNTHIA
        num_train_samples = 0
        num_train_samples_val = 0
        img_ext = '.png' if self.opt.png else '.jpg'
        self.data_path = os.path.join(self.opt.data_path, self.opt.phase)
        self.data_path_val = os.path.join(self.opt.data_path, 'val/')

        self.ids = []
        self.ids_val = []

        for seq_id in range(1, len(os.listdir(self.data_path))):
            num_samples = len(os.listdir(os.path.join(self.data_path, str(seq_id), 'RGB')))
            if seq_id ==1:
                min_samples =num_samples-2
            else:
                if min_samples > num_samples-2:
                   min_samples = num_samples-2
            for sample_id in range(1,num_samples-2):
                self.ids.append((seq_id, sample_id))
            num_train_samples += num_samples
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        train_dataset = self.dataset(
            self.data_path, self.ids, self.opt.height, self.opt.width,min_samples,
            self.opt.frame_ids, 4, is_train=True,  img_ext=img_ext, trainsets = True)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        if self.opt.phase =="train":
            for seq_id_ in range(1, len(os.listdir(self.data_path_val))-2):
               num_samples_ = len(os.listdir(os.path.join(self.data_path_val, str(seq_id_), 'RGB')))
               if seq_id_ == 1:
                   min_samples = num_samples_-2
               else:
                   if min_samples > num_samples_-2:
                       min_samples = num_samples_-2
               for sample_id in range(1, num_samples_ - 2):
                   self.ids_val.append((seq_id_, sample_id))
               num_train_samples_val += num_samples_
            val_dataset = self.dataset(
                self.data_path_val, self.ids_val, self.opt.height, self.opt.width,min_samples,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext,trainsets = False)
            self.val_loader = DataLoader(
                 val_dataset, self.opt.batch_size, True,
                 num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        self.model_lr_scheduler.step()
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs = self.process_batch_D(inputs)

            outputs, losses = self.process_batch(inputs,outputs)
            #if "depth_gt" in inputs:
            #    losses = self.compute_depth_losses(inputs, outputs, losses)

            self.model_optimizer.zero_grad()
            self.optimizer_G_shared.zero_grad()
            self.optimizer_G_segmentation.zero_grad()
            self.optimizer_D.zero_grad()
            losses["loss"].backward(retain_graph=True)
            losses["loss_tatol_seg"].backward(retain_graph=True)
            self.backward_D(inputs, outputs)
            self.model_optimizer.step()
            self.optimizer_G_shared.step()
            self.optimizer_G_segmentation.step()
            self.optimizer_D.step()


            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
    def process_batch_D(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        outputs = {}
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        all_color_aug = torch.cat([inputs[("color", i, 0)] for i in self.opt.frame_ids])  # [12,640,480]
        # outputs mask:
        encoded_rgb_all = self.models["encoder_rgb"](all_color_aug)

        # re-arranging the feature lists so they can be passed into the decoder parts.
        # this is very messy. I don't like it, but can't be arsed to fix it now.
        # [prev[6], curr[6],later[6]]
        b = self.opt.batch_size
        ix7s = [encoded_rgb_all[6][b:2*b,:,:,:], encoded_rgb_all[6][:b,:,:,:],encoded_rgb_all[6][2*b:3*b,:,:,:]]  # [[1,256,8,2], [1,256,8,2]]
        ix6s = [encoded_rgb_all[5][b:2*b,:,:,:], encoded_rgb_all[5][:b,:,:,:],encoded_rgb_all[5][2*b:3*b,:,:,:]]  # [1,256,8,2] [1,256,8,2]
        ix5s = [encoded_rgb_all[4][b:2*b,:,:,:], encoded_rgb_all[4][:b,:,:,:],encoded_rgb_all[4][2*b:3*b,:,:,:]]  # [1,256,16,4] [1,256,16,4]
        ix4s = [encoded_rgb_all[3][b:2*b,:,:,:], encoded_rgb_all[3][:b,:,:,:],encoded_rgb_all[3][2*b:3*b,:,:,:]]  # [1,256,32,8] [1,256,32,8]
        ix3s = [encoded_rgb_all[2][b:2*b,:,:,:], encoded_rgb_all[2][:b,:,:,:],encoded_rgb_all[2][2*b:3*b,:,:,:]]  # [1,128,64,16] [1,128,64,16]
        ix2s = [encoded_rgb_all[1][b:2*b,:,:,:], encoded_rgb_all[1][:b,:,:,:],encoded_rgb_all[1][2*b:3*b,:,:,:]]  # [1,64,128,32] [1,64,128,32]
        ix1s = [encoded_rgb_all[0][b:2*b,:,:,:], encoded_rgb_all[0][:b,:,:,:],encoded_rgb_all[0][2*b:3*b,:,:,:]]  # [1,32,256,64] [1,32,256,64]

        all_color_real = torch.cat([inputs[("color_real", i, 0)] for i in self.opt.frame_ids])  # [12,640,480]
        encoded_rgb_all_real = self.models["encoder_rgb"](all_color_real)
        ix7s_real = [encoded_rgb_all_real[6][b:2 * b, :, :, :], encoded_rgb_all_real[6][:b, :, :, :],
                     encoded_rgb_all_real[6][2 * b:3 * b, :, :, :]]  # [[1,256,8,2], [1,256,8,2]]
        ix6s_real = [encoded_rgb_all_real[5][b:2 * b, :, :, :], encoded_rgb_all_real[5][:b, :, :, :],
                     encoded_rgb_all_real[5][2 * b:3 * b, :, :, :]]  # [1,256,8,2] [1,256,8,2]
        ix5s_real = [encoded_rgb_all_real[4][b:2 * b, :, :, :], encoded_rgb_all_real[4][:b, :, :, :],
                     encoded_rgb_all_real[4][2 * b:3 * b, :, :, :]]  # [1,256,16,4] [1,256,16,4]
        ix4s_real = [encoded_rgb_all_real[3][b:2 * b, :, :, :], encoded_rgb_all_real[3][:b, :, :, :],
                     encoded_rgb_all_real[3][2 * b:3 * b, :, :, :]]  # [1,256,32,8] [1,256,32,8]
        ix3s_real = [encoded_rgb_all_real[2][b:2 * b, :, :, :], encoded_rgb_all_real[2][:b, :, :, :],
                     encoded_rgb_all_real[2][2 * b:3 * b, :, :, :]]  # [1,128,64,16] [1,128,64,16]
        ix2s_real = [encoded_rgb_all_real[1][b:2 * b, :, :, :], encoded_rgb_all_real[1][:b, :, :, :],
                     encoded_rgb_all_real[1][2 * b:3 * b, :, :, :]]  # [1,64,128,32] [1,64,128,32]
        ix1s_real = [encoded_rgb_all_real[0][b:2 * b, :, :, :], encoded_rgb_all_real[0][:b, :, :, :],
                     encoded_rgb_all_real[0][2 * b:3 * b, :, :, :]]  # [1,32,256,64] [1,32,256,64]
        for i, frame_id in enumerate(self.opt.frame_ids):
            decoder_1_output = self.models["decoder_1"](ix7s[i], ix6s[i], ix5s[i], ix4s[i])
            decoder_1_output_real = self.models["decoder_1"](ix7s_real[i], ix6s_real[i], ix5s_real[i], ix4s_real[i])
            # build predict_mask
            if i == 0 and self.opt.predictive_mask:
                instance_mask_pyramid = self.models["decoder_3_instance_mask"](decoder_1_output, ix3s[0],ix2s[0], ix1s[0])
                instance_mask_pyramid_real = self.models["decoder_3_instance_mask"](decoder_1_output_real, ix3s_real[0], ix2s_real[0],ix1s_real[0])
                for scale in self.opt.scales:
                    outputs[("instance_mask_pyramid", 0, scale)] = instance_mask_pyramid[scale]
                    outputs[("instance_mask_pyramid_real", 0, scale)] = instance_mask_pyramid_real[scale]

            # build segmentation
            label_fake_pyramid = self.models["decoder_2_segmentation"](decoder_1_output,ix3s[i], ix2s[i], ix1s[i])
            label_fake_pyramid_real = self.models["decoder_2_segmentation"](decoder_1_output_real, ix3s_real[i], ix2s_real[i], ix1s_real[i])
            for scale in self.opt.scales:
                outputs[("label_fake_pyramid", i, scale)] = label_fake_pyramid[scale]
                outputs[("label_fake_pyramid_real", i, scale)] = label_fake_pyramid_real[scale]

        return outputs
    # backward pass to train the discriminator with the loss
    def backward_D(self,inputs, outputs):

        # to train the discriminator on real RGB-D sample from the current frame:
        loss_D_real = 0
        loss_D_fake = 0
        for i, frame_id in enumerate(self.opt.frame_ids):
            if i ==0:
                disp_vir = outputs[("disp", 0)].detach()
                label_fake_trans = outputs[("label_fake_pyramid", i, 0)].detach().argmax(dim=1, keepdim=True).float()
                dis_out_real = self.models["discriminator"](disp_vir * label_fake_trans)
                loss_D_real += self.criterion_adv(dis_out_real, True)
                disp_real = outputs[("disp_real", 0)].detach()
                label_fake_trans_real = outputs[("label_fake_pyramid_real", i, 0)].detach().argmax(dim=1, keepdim=True).float()
                # now to train the discriminator on the fake RGB-D sample from the current frame:
                dis_out_curr = self.models["discriminator"](disp_real * label_fake_trans_real)
                loss_D_fake += self.criterion_adv(dis_out_curr, False)

        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()

    def process_batch(self, inputs, outputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.

            all_color_aug = torch.cat([inputs[("color", i, 0)] for i in self.opt.frame_ids]) #[12,640,480]
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            all_color_real = torch.cat([inputs[("color_real", i, 0)] for i in self.opt.frame_ids])  # [12,640,480]
            all_features_real = self.models["encoder"](all_color_real)
            all_features_real = [torch.split(f, self.opt.batch_size) for f in all_features_real]

            features = {}
            features_real = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
                features_real[k] = [f[i] for f in all_features_real]
            outputs.update(self.models["depth"](features[0]))
            outputs.update(self.models["depth"](features_real[0]), estimate_real = True)
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color", 0, 0])
            outputs.update(self.models["depth"](features))
            features_real = self.models["encoder"](inputs["color_real", 0, 0])
            outputs.update(self.models["depth"](features_real, estimate_real = True))

        # current segmentation frame generate by the model:
        # for instance_mask_segment

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, features_real))

        outputs.update(self.colorize_segmentaions(inputs, outputs))
        outputs.update(self.colorize_segmentaions_instance(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)


        return outputs, losses

    def predict_poses(self, inputs, features, features_real):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
                pose_feats_real = {f_i: features_real[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color", f_i, 0] for f_i in self.opt.frame_ids}
                pose_feats_real = {f_i: inputs["color_real", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                        pose_inputs_real = [pose_feats_real[f_i], pose_feats_real[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]
                        pose_inputs_real = [pose_feats_real[0], pose_feats_real[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        pose_inputs_real = [self.models["pose_encoder"](torch.cat(pose_inputs_real, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)
                        pose_inputs_real = torch.cat(pose_inputs_real, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    axisangle_real, translation_real = self.models["pose"](pose_inputs_real)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    outputs[("cam_T_cam_real", 0, f_i)] = transformation_from_parameters(
                        axisangle_real[:, 0], translation_real[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
                pose_inputs_real = torch.cat(
                    [inputs[("color_real", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]
                    pose_inputs_real = [self.models["pose_encoder"](pose_inputs_real)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]
                pose_inputs_real = [features_real[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)
            axisangle_real, translation_real = self.models["pose"](pose_inputs_real)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])
                    outputs[("axisangle_real", 0, f_i)] = axisangle_real
                    outputs[("translation_real", 0, f_i)] = translation_real
                    outputs[("cam_T_cam_real", 0, f_i)] = transformation_from_parameters(
                        axisangle_real[:, i], translation_real[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs = self.process_batch_D(inputs)
            outputs, losses = self.process_batch(inputs, outputs)

            #if "depth_gt" in inputs:
            #    self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    # -----------------------------------------
    def colorize_segmentaions(self, inputs, outputs, test=None):
        for scale in self.opt.scales:
          for i, frame_id in enumerate(self.opt.frame_ids):
            fake = outputs[("label_fake_pyramid", i, scale)].detach().cpu().numpy()[:, :, :, :]
            #fake_real = outputs[("label_fake_pyramid_real", i, scale)].detach().cpu().numpy()[:, :, :, :]
            real_0 = (inputs[("label", frame_id, scale)].cpu().numpy()[:, :, :, :] * 255).astype(int)
            real = np.squeeze(real_0,axis=1)
            fake_0 = np.argmax(fake, axis=1)
            #fake_real_0 = np.argmax(fake_real, axis=1)
            fake_0 = np.expand_dims(fake_0, axis=1)
            #fake_real_0 = np.expand_dims(fake_real_0, axis=1)
            ind = np.squeeze(fake_0,axis=1)  # #[batch_size,128,512]
            outputs[("label_fake_numpy", i, scale)] = torch.from_numpy(fake_0).type(torch.cuda.FloatTensor)
            #outputs[("label_fake_numpy_real", i, scale)] = torch.from_numpy(fake_real_0).type(torch.cuda.FloatTensor)
            outputs[("label_real_numpy", i, scale)] = torch.from_numpy(real).type(torch.cuda.FloatTensor)
            if test is None:
                ind[real == 0] = 0
            else:
                real = real

            r = ind.copy()
            g = ind.copy()
            b = ind.copy()

            r_gt = real.copy()
            g_gt = real.copy()
            b_gt = real.copy()

            Void = [0, 0, 0]
            Road = [128, 64, 128]
            Sidewalk = [0, 0, 192]
            Building = [128, 0, 0]
            Wall = [102, 102, 156]
            Fence = [64, 64, 128]
            Pole = [192, 192, 128]
            TrafficLight = [0, 128, 128]
            TrafficSign = [192, 128, 128]
            Vegetation = [128, 128, 0]
            Terrain = [152, 251, 152]
            Sky = [128, 128, 128]
            Person = [64, 64, 0]
            Rider = [0, 128, 192]
            Car = [64, 0, 128]
            Truck = [0, 0, 70]
            Bus = [0, 60, 100]
            Train = [0, 80, 100]
            Motorcycle = [0, 0, 230]
            Bicycle = [119, 11, 32]
            RoadLines = [0, 172, 0]
            Other = [72, 0, 98]
            RoadWorks = [167, 106, 29]

            label_colours = np.array(
                [Void, Road, Sidewalk, Building, Wall, Fence, Pole, TrafficLight, TrafficSign, Vegetation, Terrain, Sky,
                 Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, RoadLines, Other, RoadWorks])

            for l in range(0, len(label_colours)):
                r[ind == l] = label_colours[l, 0]
                g[ind == l] = label_colours[l, 1]
                b[ind == l] = label_colours[l, 2]

                r_gt[real == l] = label_colours[l, 0]
                g_gt[real == l] = label_colours[l, 1]
                b_gt[real == l] = label_colours[l, 2]

            rgb = np.zeros((self.opt.batch_size, 3, ind.shape[1], ind.shape[2]))

            rgb[:, 0, :, :] = r
            rgb[:, 1, :, :] = g
            rgb[:, 2, :, :] = b

            rgb_gt = np.zeros((self.opt.batch_size, 3, ind.shape[1], ind.shape[2]))

            rgb_gt[:, 0, :, :] = r_gt
            rgb_gt[:, 1, :, :] = g_gt
            rgb_gt[:, 2, :, :] = b_gt
            outputs[("label_fake_vis", i, scale)] = torch.from_numpy(rgb).type(torch.cuda.FloatTensor)
            outputs[("label_real_vis", i, scale)] = torch.from_numpy(rgb_gt).type(torch.cuda.FloatTensor)
        return outputs

    def colorize_segmentaions_instance(self, inputs, outputs):
        for scale in self.opt.scales:
            real_0 = (inputs[("label", 0, scale)].cpu().numpy()[:, :, :, :] * 255).astype(int)

            real = np.squeeze(real_0,axis=1)

            instance_object = np.zeros((self.opt.batch_size,  real.shape[1], real.shape[2]))
            instance_object_inverse = np.ones((self.opt.batch_size,  real.shape[1], real.shape[2]))
            #r1 = np.zeros((self.opt.batch_size, real.shape[1], real.shape[2]))
            #g1 = np.zeros((self.opt.batch_size, real.shape[1], real.shape[2]))
            #b1 = np.zeros((self.opt.batch_size, real.shape[1], real.shape[2]))
            for j in range(12, 20):
                instance_object[real == j] = 1
                instance_object_inverse[real == j] = 0
                #r1[real == j] = 220
                #g1[real == j] = 20
                #b1[real == j] = 60
            #rgb1 = np.zeros((self.opt.batch_size, 3, real.shape[1], real.shape[2]))

            #rgb1[:, 0, :, :] = r1
            #rgb1[:, 1, :, :] = g1
            #rgb1[:, 2, :, :] = b1
            instance_object = np.expand_dims(instance_object, axis=1)
            instance_object_inverse = np.expand_dims(instance_object_inverse, axis=1)
            outputs[("instance_real", 0, scale)] = torch.from_numpy(instance_object).type(torch.cuda.FloatTensor)
            outputs[("instance_real_inverse", 0, scale)] = torch.from_numpy(instance_object_inverse).type(torch.cuda.FloatTensor)
            #outputs[("instance_real_vis", 0, scale)] = torch.from_numpy(rgb1).type(torch.cuda.FloatTensor)

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp_real = outputs[("disp_real", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale

            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                disp_real = F.interpolate(
                    disp_real, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            _, depth_real = disp_to_depth(disp_real, self.opt.min_depth_real, self.opt.max_depth_real)

            outputs[("depth", 0, scale)] = depth
            outputs[("depth_real", 0, scale)] = depth_real

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                    T_real = outputs[("cam_T_cam_real", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    axisangle_real = outputs[("axisangle_real", 0, frame_id)]
                    translation_real = outputs[("translation_real", 0, frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    inv_depth_real = 1 / depth_real
                    mean_inv_depth_real = inv_depth_real.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                    T_real = transformation_from_parameters(
                        axisangle_real[:, 0], translation_real[:, 0] * mean_inv_depth_real[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                cam_points_real = self.backproject_depth[source_scale](
                    depth_real, inputs[("inv_K_real", source_scale)])
                pix_coords_real = self.project_3d[source_scale](
                    cam_points_real, inputs[("K_real", source_scale)], T_real)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("sample_real", frame_id, scale)] = pix_coords_real

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="zeros")

                outputs[("color_real", frame_id, scale)] = F.grid_sample(
                    inputs[("color_real", frame_id, source_scale)],
                    outputs[("sample_real", frame_id, scale)],
                    padding_mode="zeros")

                outputs[("label_vis_sys", i+1, scale)] = F.grid_sample(
                    outputs[("label_real_vis", i+1, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="zeros")#border

                if not self.opt.disable_automasking_real:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

                    outputs[("color_identity_real", frame_id, scale)] = \
                        inputs[("color_real", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_loss_label_sys = 0
        loss_tatol_seg = 0

        for scale in self.opt.scales:
            loss_mask = 0
            loss_segmentation_L1 = 0
            loss_segmentation_adv = 0
            loss_segmentation = 0

            reprojection_losses = []
            real_reprojection_losses = []
            label_real_reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale

            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            target_label_real_vis = outputs[("label_real_vis", 0, source_scale)]

            disp_real = outputs[("disp_real", scale)]
            color_real = inputs[("color_real", 0, scale)]
            target_real = inputs[("color_real", 0, source_scale)]

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                pred = outputs[("color", frame_id, scale)]
                reprojection_loss = self.compute_reprojection_loss(pred, target)
                label_real_sys_pred = outputs[("label_vis_sys", i+1, scale)]
                label_real_reprojection_loss = self.compute_reprojection_loss(label_real_sys_pred, target_label_real_vis)

                pred_real = outputs[("color_real", frame_id, scale)]
                reprojection_losse_real = self.compute_reprojection_loss(pred_real, target_real)
                # for label sys smooth
                reprojection_losses.append(reprojection_loss)
                real_reprojection_losses.append(reprojection_losse_real)
                label_real_reprojection_losses.append(label_real_reprojection_loss)
            reprojection_losses = torch.cat(reprojection_losses, 1)
            real_reprojection_losses = torch.cat(real_reprojection_losses, 1)
            label_real_reprojection_losses = torch.cat(label_real_reprojection_losses, 1)
            if self.opt.predictive_mask:
                mask = outputs[("instance_mask_pyramid", 0, scale)]
                instance_mask = outputs[("instance_real", 0, scale)]
                instance_mask_inverse = outputs[("instance_real_inverse", 0, scale)]
                # instance_real_mask = outputs[("instance_real_mask", i+1, source_scale)]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                    instance_mask = F.interpolate(
                        instance_mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)
                    instance_mask_inverse = F.interpolate(
                        instance_mask_inverse, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)
                instance_mask_prev = torch.ones(mask.shape).cuda() - mask

                # for compute instance mask expect loss
                instance_mask_detach = instance_mask.cpu().numpy()[:, :, :, :]  # .detach()
                instance_mask_detach = np.squeeze(instance_mask_detach, axis=1)
                instance_mask_inverse_detach = instance_mask_inverse.cpu().numpy()[:, :, :, :]  # .detach()
                instance_mask_inverse_detach = torch.from_numpy(instance_mask_inverse_detach).type(
                    torch.cuda.FloatTensor)

                mask_inverse_loss = 0.2 * nn.BCELoss()(instance_mask_inverse_detach * mask, instance_mask_inverse_detach)
                loss_mask = self.opt.ones_mask_weight * mask_inverse_loss.mean()
                labeled_img, num = label(instance_mask_detach, background=0, return_num=True, connectivity=2)
                for i in range(0, num):
                    label_index = i + 1
                    mcr = (labeled_img == label_index)
                    if np.sum(mcr, axis=(0, 1, 2)) >= 40:
                        mcr = np.expand_dims(mcr, axis=1)
                        mcr = torch.from_numpy(mcr).type(torch.cuda.FloatTensor)
                        # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                        weighting_loss = 0.2 * nn.BCELoss()(mcr * mask, mcr)
                        # pan duan shi fou shi an zhao suo xiang de1 lai
                        loss_mask += self.opt.ones_moving_mask_weight * weighting_loss.mean()
                reprojection_losses *= mask
                label_real_reprojection_losses *= mask
                losses["loss_mask_loss/{}".format(scale)] = loss_mask
            if not self.opt.disable_automasking_real:
                identity_reprojection_losses = []
                for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                    pred_real = inputs[("color_real", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred_real, target_real))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
                # use the predicted mask
                #for i, frame_id in enumerate(self.opt.frame_ids):
                #outputs[("instance_real_mask", 0, source_scale)] = label(outputs[("instance_real", 0, source_scale)], connectivity=2)
            if self.opt.avg_reprojection:
                reprojection_loss_total = reprojection_losses.mean(1, keepdim=True)
                reprojection_loss_total_real = real_reprojection_losses.mean(1, keepdim=True)
                label_real_reprojection_loss_total = label_real_reprojection_losses.mean(1, keepdim=True)

            else:
                reprojection_loss_total = reprojection_losses
                reprojection_loss_total_real = real_reprojection_losses
                label_real_reprojection_loss_total = label_real_reprojection_losses


            if not self.opt.disable_automasking_real:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001 * 2

                combined_real = torch.cat((identity_reprojection_loss, reprojection_loss_total_real), dim=1)
            else:
                combined_real = reprojection_loss_total_real

            combined = reprojection_loss_total
            combined_label = label_real_reprojection_loss_total

            if combined_real.shape[1] == 1:
                to_optimise = combined
                to_optimise_real = combined_real
                to_optimise_label = combined_label

            else:
                to_optimise, idxs = torch.min(combined, dim=1)
                to_optimise_real, idxs_real = torch.min(combined_real, dim=1)
                to_optimise_label, idxs_label = torch.min(combined_label, dim=1)

            if not self.opt.disable_automasking_real:
                outputs["identity_selection/{}".format(scale)] = (idxs_real > identity_reprojection_loss.shape[1] - 1).float()

            image_sys_loss = self.opt.image_sys_weight * to_optimise.mean()
            image_real_sys_loss = self.opt.image_real_sys_weight * to_optimise_real.mean()
            losses["image_sys_loss/{}".format(scale)] = image_sys_loss
            losses["image_real_sys_loss/{}".format(scale)] = image_real_sys_loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            mean_disp_real = disp_real.mean(2, True).mean(3, True)
            norm_disp_real = disp_real / (mean_disp_real + 1e-7)
            smooth_loss = self.opt.disparity_smoothness * get_smooth_loss(norm_disp, color) / (2 ** scale)
            smooth_loss += self.opt.disparity_smoothness * get_smooth_loss(norm_disp_real, color_real) / (2 ** scale)
            total_loss += smooth_loss
            total_loss += image_sys_loss
            total_loss += image_real_sys_loss
            total_loss += loss_mask
            losses["total_loss/{}".format(scale)] = total_loss

            label_sys_loss =  self.opt.label_sys_weight * to_optimise_label.mean()
            losses["label_sys_loss/{}".format(scale)] = label_sys_loss
            total_loss_label_sys += label_sys_loss

            for i, frame_id in enumerate(self.opt.frame_ids):
                real_label = inputs[("label", frame_id, source_scale)]
                output_target_label = outputs[("label_fake_numpy", i, source_scale)]
                loss_segmentation_L1 += torch.nn.L1Loss()(output_target_label, real_label)
                real_label_numpy = outputs[("label_real_numpy", i, scale)]
                loss_segmentation += self.criterion_CE(outputs[("label_fake_pyramid", i, scale)],real_label_numpy.long())
                if source_scale ==0:
                    dis_out = self.models["discriminator"](output_target_label)
                    loss_segmentation_adv += ((1 - dis_out) ** 2).mean()

            losses_segmentation = self.opt.segmentation_weight * loss_segmentation
            losses_segmentation_L1 = self.opt.l1_weight * loss_segmentation_L1
            losses_segmentation_adv = self.opt.adv_label_weight * loss_segmentation_adv
            losses["losses_segmentation/{}".format(scale)] = losses_segmentation
            losses["losses_segmentation_L1/{}".format(scale)] = losses_segmentation_L1
            losses["losses_segmentation_adv/{}".format(scale)] = losses_segmentation_adv
            loss_tatol_seg += losses_segmentation_L1
            loss_tatol_seg += losses_segmentation
            loss_tatol_seg += losses_segmentation_adv

        total_loss_ = total_loss + total_loss_label_sys
        total_loss_ /= self.num_scales
        losses["loss"] = total_loss_

        loss_tatol_seg /= self.num_scales
        losses["loss_tatol_seg"] = loss_tatol_seg
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [128, 512], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()
        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        #crop_mask = torch.zeros_like(mask)
        #crop_mask[:, :, 153:371, 44:1197] = 1
        #mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())
        losses["loss"] += losses["de/abs_rel"] + 0.5*losses["de/rms"]
        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            #for s in self.opt.scales:
                for i, frame_id in enumerate(self.opt.frame_ids):
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, 0, j),
                        inputs[("color", frame_id, 0)][j].data, self.step)
                    writer.add_image(
                        "color_real_{}_{}/{}".format(frame_id, 0, j),
                        inputs[("color_real", frame_id, 0)][j].data, self.step)
                    if frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, 0, j),
                            outputs[("color", frame_id, 0)][j].data, self.step)
                        writer.add_image(
                            "color_real_pred_{}_{}/{}".format(frame_id, 0, j),
                            outputs[("color_real", frame_id, 0)][j].data, self.step)
                        writer.add_image(
                            "label_sys_pred_{}_{}/{}".format(frame_id, 0, j),
                            outputs[("label_vis_sys", i, 0)][j].data, self.step)
                    if i ==0:
                        if self.opt.predictive_mask:
                            writer.add_image(
                                "predictive_mask_{}_{}/{}".format(0, 0, j),
                                outputs[("instance_mask_pyramid", 0, 0)][j].data, self.step)
                            writer.add_image(
                                "real_predictive_mask_{}_{}/{}".format(0, 0, j),
                                outputs[("instance_mask_pyramid_real", 0, 0)][j].data, self.step)
                            writer.add_image(
                                "instance_real_{}_{}/{}".format(0, 0, j),
                                outputs[("instance_real", 0, 0)][j].data, self.step)
                            writer.add_image(
                                "label_fake_vis_{}_{}/{}".format(0, 0, j),
                                outputs[("label_fake_vis", 0, 0)][j].data, self.step)


                        elif not self.opt.disable_automasking:
                            writer.add_image(
                                "automask_{}/{}".format(0, j),
                                outputs["identity_selection/{}".format(0)][j][None, ...], self.step)

                writer.add_image(
                    "disp_{}/{}".format(0, j),
                    normalize_image(outputs[("disp", 0)][j]), self.step)
                writer.add_image("disp_real_{}/{}".format(0, j),
                    normalize_image(outputs[("disp_real", 0)][j]), self.step)



    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

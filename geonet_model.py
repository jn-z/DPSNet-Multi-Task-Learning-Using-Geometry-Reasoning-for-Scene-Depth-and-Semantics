# coding=utf-8
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from geonet_nets import *
from utils import *

class GeoNetModel(object):

    def __init__(self, opt, tgt_image, src_image_stack, intrinsics):
        self.opt = opt
        self.tgt_image = self.preprocess_image(tgt_image)
        self.src_image_stack = self.preprocess_image(src_image_stack)
        self.intrinsics = intrinsics

        self.build_model()
        if not opt.mode in ['train_rigid', 'train_move']:
            return

        self.build_losses()
        self.collect_summaries()

    def build_model(self):
        opt = self.opt
        self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)
        self.tgt_image_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) \
                                      for img in self.tgt_image_pyramid]
        # src images concated along batch dimension
        if self.src_image_stack != None:
            self.src_image_concat = tf.concat([self.src_image_stack[:,:,:,3*i:3*(i+1)] \
                                    for i in range(opt.num_source)], axis=0)
            self.src_image_concat_pyramid = self.scale_pyramid(self.src_image_concat, opt.num_scales)

        if opt.add_dispnet:
            self.build_dispnet()

        if opt.add_posenet:
            self.build_posenet()

        if opt.add_dispnet and opt.add_posenet:
            self.build_rigid_flow_warping()


    def build_dispnet(self):
        opt = self.opt

        # build dispnet_inputs
        if opt.mode == 'test_depth':
            # for test_depth mode we only predict the depth of the target image
            self.dispnet_inputs = self.tgt_image
        else:
            # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            self.dispnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.dispnet_inputs = tf.concat([self.dispnet_inputs, self.src_image_stack[:,:,:,3*i:3*(i+1)]], axis=0)
        
        # build dispnet
        with tf.variable_scope('depth_1', reuse=False):
            self.pred_disp = disp_net(opt, self.dispnet_inputs)

        if opt.scale_normalize:
            # As proposed in https://arxiv.org/abs/1712.00175, this can 
            # bring improvement in depth estimation, but not included in our paper.
            self.pred_disp = [self.spatial_normalize(disp) for disp in self.pred_disp]

        self.pred_depth = [1./d for d in self.pred_disp]

    def build_posenet(self):
        opt = self.opt

        # build posenet_inputs
        self.posenet_inputs = tf.concat([self.tgt_image, self.src_image_stack], axis=3)
        self.posenet_inputs_inverse_1 = tf.concat([self.src_image_stack[:,:,:,:3],self.tgt_image, self.src_image_stack[:,:,:,3:6]], axis=3)
        self.posenet_inputs_inverse_2 = tf.concat([self.src_image_stack[:,:,:,3:6], self.src_image_stack[:,:,:,:3],self.tgt_image], axis=3)
        # build posenet
        with tf.variable_scope('pose', reuse=False):
            self.pred_poses,self.move_inputs = pose_net(opt, self.posenet_inputs)
        if opt.add_movenet:
            with tf.variable_scope('move_net', reuse=False):
               self.mask_cnn = move_net(opt, self.move_inputs)
        with tf.variable_scope('pose', reuse=True):
            _, self.move_inputs_0_1 = pose_net(opt, self.posenet_inputs_inverse_1)
            _, self.move_inputs_2_1 = pose_net(opt, self.posenet_inputs_inverse_2)
        if opt.add_movenet:
            with tf.variable_scope('move_net', reuse=True):
               self.mask_cnn_0_1 = move_net(opt, self.move_inputs_0_1)
               self.mask_cnn_2_1 = move_net(opt, self.move_inputs_2_1)

    def build_rigid_flow_warping(self):
        opt = self.opt
        bs = opt.batch_size

        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []
        self.fwd_rigid_warp_pyramid = []
        self.bwd_rigid_warp_pyramid = []
        self.fwd_mask_image_pyramid = []
        self.bwd_mask_image_pyramid = []
        self.fwd_rigid_flow_no_moving_concat_pyramid = []
        self.bwd_rigid_flow_no_moving_concat_pyramid = []
        self.bwd_rigid_warp_no_moving_pyramid = []
        self.fwd_rigid_warp_no_moving_pyramid = []
        self.bwd_mask_no_moving_pyramid = []
        self.fwd_mask_no_moving_pyramid = []
        self.ref_exp_mask_pyramid = []
        self.fwd_exp_concat_pyramid = []
        self.bwd_exp_concat_pyramid = []
        self.fwd_rigid_warp_with_depth_pyramid = []
        self.bwd_rigid_warp_with_depth_pyramid = []
        self.fwd_exp_logits_concat_pyramid = []
        self.bwd_exp_logits_concat_pyramid = []
        for s in range(opt.num_scales):
            for i in range(opt.num_source):
                fwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][:bs], axis=3),
                                 self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], False)
                bwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                                 self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], True)
                if not i:
                    fwd_rigid_flow_concat = fwd_rigid_flow
                    bwd_rigid_flow_concat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
                    bwd_rigid_flow_concat = tf.concat([bwd_rigid_flow_concat, bwd_rigid_flow], axis=0)
            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_concat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_concat)

        # warping by rigid flow

        for s in range(opt.num_scales):
            fwd_rigid_warp, fwd_mask_image=flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_pyramid[s])   #注意蒙版在{0：8,：，：}面积与image是一样的，后面的{8：16,：，：}面积比image小
            fwd_mask_image = tf.stop_gradient(fwd_mask_image)

            bwd_rigid_warp, bwd_mask_image= flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_rigid_flow_pyramid[s])#注意蒙版在{0：8,：，：}面积比image小，后面的{8：16,：，：}面积与image是一样的
            bwd_mask_image = tf.stop_gradient(bwd_mask_image)

            self.fwd_rigid_warp_pyramid.append(fwd_rigid_warp)
            self.bwd_rigid_warp_pyramid.append(bwd_rigid_warp)
            self.fwd_mask_image_pyramid.append(fwd_mask_image)
            self.bwd_mask_image_pyramid.append(bwd_mask_image)
        if opt.add_movenet:        
            self.depthnet_inputs_tgt = self.tgt_image
            self.depthnet_inputs_src = self.tgt_image
            for i in range(opt.num_source):
                self.depthnet_inputs_tgt = tf.concat([self.depthnet_inputs_tgt, self.fwd_rigid_warp_pyramid[0][bs * i:bs * (i + 1), :, :, :]], axis=0)  # shape=[4*batch,128,416,3]
                self.depthnet_inputs_src = tf.concat([self.depthnet_inputs_src, self.bwd_rigid_warp_pyramid[0][bs * i:bs * (i + 1), :, :, :]], axis=0)  # shape=[4*batch,128,416,3]
            with tf.variable_scope('depth_1', reuse=True):
                self.pred_disp_no_moving_depth_tgt = disp_net(opt, self.depthnet_inputs_tgt) #all is depth at frame 0
                self.pred_disp_no_moving_depth_src = disp_net(opt, self.depthnet_inputs_src)  # all is depth at frame -1,1
            if opt.scale_normalize:
                self.pred_disp_no_moving_tgt = [self.spatial_normalize(disp) for disp in self.pred_disp_no_moving_depth_tgt]
                self.pred_disp_no_moving_src = [self.spatial_normalize(disp) for disp in self.pred_disp_no_moving_depth_src]
        #需要停止梯度
            self.pred_depth_no_moving_tgt = [tf.stop_gradient(1./d) for d in self.pred_disp_no_moving_tgt]
            self.pred_depth_no_moving_src = [tf.stop_gradient(1. / d) for d in self.pred_disp_no_moving_src]
            for s in range(opt.num_scales):
                for i in range(opt.num_source):
                    fwd_rigid_flow_no_moving= compute_rigid_flow(tf.squeeze(self.pred_depth_no_moving_tgt[s][bs*i+bs:bs*(i+1)+bs,:,:,:], axis=3),self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], reverse_pose=False)
                    bwd_rigid_flow_no_moving = compute_rigid_flow(tf.squeeze(self.pred_depth_no_moving_src[s][bs * i + bs:bs * (i + 1) + bs, :, :, :], axis=3),
                                                              self.pred_poses[:, i, :], self.intrinsics[:, s, :, :], reverse_pose=True)

                    if not i:
                        fwd_rigid_flow_no_moving_concat = fwd_rigid_flow_no_moving
                        bwd_rigid_flow_no_moving_concat = bwd_rigid_flow_no_moving
                    else:
                        fwd_rigid_flow_no_moving_concat = tf.concat([fwd_rigid_flow_no_moving_concat, fwd_rigid_flow_no_moving], axis=0)
                        bwd_rigid_flow_no_moving_concat = tf.concat([bwd_rigid_flow_no_moving_concat, bwd_rigid_flow_no_moving], axis=0)
                self.bwd_rigid_flow_no_moving_concat_pyramid.append(bwd_rigid_flow_no_moving_concat)
                self.fwd_rigid_flow_no_moving_concat_pyramid.append(fwd_rigid_flow_no_moving_concat)
            for s in range(opt.num_scales):
                fwd_rigid_warp_no_moving_, fwd_mask_image_no_moving_ = flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_no_moving_concat_pyramid[s])  #  指的就是3帧到1,2,4,5帧合成的图以及对应的模板
                bwd_rigid_warp_no_moving, bwd_mask_image_no_moving = flow_warp(fwd_rigid_warp_no_moving_, self.bwd_rigid_flow_pyramid[s]) #  指的就是1,2,4,5帧到3帧的合成图以及对应的模板

                bwd_rigid_warp_no_moving_, bwd_mask_image_no_moving_ = flow_warp(self.tgt_image_tile_pyramid[s],
                                                                             self.bwd_rigid_flow_no_moving_concat_pyramid[s])  # 指的就是3帧到1,2,4,5帧合成的图以及对应的模板
                fwd_rigid_warp_no_moving, fwd_mask_image_no_moving = flow_warp(bwd_rigid_warp_no_moving_,
                                                                           self.fwd_rigid_flow_pyramid[s])  # 指的就是1,2,4,5帧到3帧的合成图以及对应的模板
            # compute_with_depth_no_with_redepth
                bwd_rigid_warp_with_depth, bwd_mask_image_no_moving_ = flow_warp(self.fwd_rigid_warp_pyramid[s],
                                                                             self.bwd_rigid_flow_pyramid[s])  # 指的就是3帧到1,2,4,5帧合成的图以及对应的模板
                fwd_rigid_warp_with_depth, fwd_mask_image_no_moving_ = flow_warp(self.bwd_rigid_warp_pyramid[s],
                                                                           self.fwd_rigid_flow_pyramid[s])  # 指的就是1,2,4,5帧到3帧的合成图以及对应的模板
                self.bwd_rigid_warp_no_moving_pyramid.append(bwd_rigid_warp_no_moving)
                self.fwd_rigid_warp_no_moving_pyramid.append(fwd_rigid_warp_no_moving)
                self.fwd_rigid_warp_with_depth_pyramid.append(fwd_rigid_warp_with_depth)
                self.bwd_rigid_warp_with_depth_pyramid.append(bwd_rigid_warp_with_depth)
        # mask_image-compute
        if opt.add_movenet:
            for s in range(opt.num_scales):
            	ref_exp_mask = self.get_reference_explain_mask(s)
            	for i in range(opt.num_source):
                	fwd_exp_logits = tf.slice(self.mask_cnn[s],
                                               [0, 0, 0, i * 2],
                                               [-1, -1, -1, 2])
                	fwd_exp = tf.nn.softmax(fwd_exp_logits)

                	if not i:
                    		bwd_exp_logits_0 = tf.slice(self.mask_cnn_0_1[s], [0, 0, 0, i * 2], [-1, -1, -1, 2])
                    		bwd_exp_0 = tf.nn.softmax(bwd_exp_logits_0)
                    		fwd_exp_concat = fwd_exp
                    		fwd_exp_logits_ = fwd_exp_logits

                	else:
                    		bwd_exp_logits_1 = tf.slice(self.mask_cnn_2_1[s], [0, 0, 0, i * 2], [-1, -1, -1, 2])
                    		bwd_exp_1 = tf.nn.softmax(bwd_exp_logits_1)
                    		fwd_exp_concat = tf.concat([fwd_exp_concat, fwd_exp], axis=0)
                    		bwd_exp_concat = tf.concat([bwd_exp_0, bwd_exp_1], axis=0)
                    		fwd_exp_logits_concat = tf.concat([fwd_exp_logits_, fwd_exp_logits], axis=0)
                    		bwd_exp_logits_concat = tf.concat([bwd_exp_logits_0, bwd_exp_logits_1], axis=0)
            	self.fwd_exp_concat_pyramid.append(fwd_exp_concat)
            	self.bwd_exp_concat_pyramid.append(bwd_exp_concat)
            	self.fwd_exp_logits_concat_pyramid.append(fwd_exp_logits_concat)
            	self.bwd_exp_logits_concat_pyramid.append(bwd_exp_logits_concat)
            	self.ref_exp_mask_pyramid.append(ref_exp_mask)

        # compute reconstruction error
        self.fwd_rigid_error_pyramid = [self.image_similarity_with_mask(self.fwd_rigid_warp_pyramid[s],self.fwd_mask_image_pyramid[s],
                                        self.tgt_image_tile_pyramid[s]) for s in range(opt.num_scales)]
        self.bwd_rigid_error_pyramid = [self.image_similarity_with_mask(self.bwd_rigid_warp_pyramid[s],self.bwd_mask_image_pyramid[s],
                                        self.src_image_concat_pyramid[s]) for s in range(opt.num_scales)]

        # self-systhzsis
        if opt.add_movenet:
            self.fwd_rigid_error_pyramid_self_loop = [self.image_similarity_with_mask(self.fwd_rigid_warp_no_moving_pyramid[s],self.fwd_mask_image_pyramid[s],
                                                                        self.tgt_image_tile_pyramid[s]) for s in range(opt.num_scales)]

            self.bwd_rigid_error_pyramid_self_loop = [self.image_similarity_with_mask(self.bwd_rigid_warp_no_moving_pyramid[s],self.bwd_mask_image_pyramid[s],
                                                                        self.src_image_concat_pyramid[s]) for s in range(opt.num_scales)]

    def build_losses(self):
        opt = self.opt
        bs = opt.batch_size
        self.rigid_warp_loss = 0
        self.disp_smooth_loss = 0
        self.exp_loss = 0
        self.rigid_warp_photo_loop_loss = 0
        self.exp_loss_cc = 0
        self.total_loss = 0  # regularization_loss
        self.mask_indicate_pyramid = []
        self.mask_indicate_all_pyramid = []
        self.fwd_object_mask_all_pyramid = []
        for s in range(opt.num_scales):

            # compute_mask_loss
            if opt.add_movenet:
                  self.exp_loss += opt.explain_reg_weight * opt.num_source / 2 * \
                             (self.compute_exp_reg_loss(self.fwd_exp_logits_concat_pyramid[s], self.ref_exp_mask_pyramid[s])+ \
                              self.compute_exp_reg_loss(self.bwd_exp_logits_concat_pyramid[s], self.ref_exp_mask_pyramid[s]))

            # rigid_warp_loss
            if opt.add_movenet:
            	if opt.mode == 'train_move' and opt.rigid_warp_weight > 0:
                	self.rigid_warp_loss += opt.rigid_warp_weight * opt.num_source / 2 * \
                                        (tf.reduce_mean(self.fwd_rigid_error_pyramid[s] * tf.expand_dims(
                                            self.fwd_exp_concat_pyramid[s][:, :, :, 1], -1)) + \
                                         tf.reduce_mean(self.bwd_rigid_error_pyramid[s] * tf.expand_dims(
                                             self.bwd_exp_concat_pyramid[s][:, :, :, 1], -1)))
            else:
                if opt.mode == 'train_rigid' and opt.rigid_warp_weight > 0:
                	self.rigid_warp_loss += opt.rigid_warp_weight * opt.num_source / 2 * \
                                        (tf.reduce_mean(self.fwd_rigid_error_pyramid[s]) + \
                                         tf.reduce_mean(self.bwd_rigid_error_pyramid[s]))

            # moving_object_loss
            if opt.add_movenet:
                fwd_compare_mask = tf.expand_dims(self.fwd_exp_concat_pyramid[s][:, :, :, 1], -1)
                bwd_compare_mask = tf.expand_dims(self.bwd_exp_concat_pyramid[s][:, :, :, 1], -1)
                mask_image_inverse = tf.ones_like(fwd_compare_mask, dtype=tf.float32)
                zeros_mask_image = tf.zeros_like(mask_image_inverse)
                fwd_object_mask = tf.abs(mask_image_inverse-fwd_compare_mask)
                bwd_object_mask = tf.abs(mask_image_inverse-bwd_compare_mask)
                self.rigid_warp_photo_loop_loss += opt.rigid_warp_photo_loop_weight * opt.num_source / 2 * (
                	tf.reduce_mean(self.fwd_rigid_error_pyramid_self_loop[s] * fwd_object_mask) +
                	tf.reduce_mean(self.bwd_rigid_error_pyramid_self_loop[s] * bwd_object_mask))

            	#compute_loss_cc_mask
                depth_difference = tf.abs(self.pred_depth_no_moving_tgt[s][bs:bs * 3, :, :, :]-tf.concat([self.pred_depth_no_moving_tgt[s][:bs, :, :, :],
                                                                                                      self.pred_depth_no_moving_tgt[s][:bs, :, :, :]],axis=0))
                mask_indicate = tf.where(tf.less(depth_difference,opt.depth_consistency_alpha), mask_image_inverse,zeros_mask_image)
                photo_loop_difference = tf.abs(self.fwd_rigid_warp_with_depth_pyramid[s] - self.fwd_rigid_warp_no_moving_pyramid[s])
                photo_loop_difference_ = tf.reduce_sum(photo_loop_difference,3,keep_dims=True)
                mask_indicate_all = tf.where(tf.less(photo_loop_difference_, opt.photo_loop_consistency_alpha),mask_image_inverse, mask_indicate)
                self.exp_loss_cc += opt.cc_reg_weight * opt.num_source / 2 * \
                             	(self.compute_exp_reg_loss(fwd_compare_mask, mask_indicate_all))  #only compute half

            # disp_smooth_loss
            if opt.disp_smooth_weight > 0:
                self.disp_smooth_loss += opt.disp_smooth_weight/(2**s) * self.compute_smooth_loss(self.pred_disp[s],
                                tf.concat([self.tgt_image_pyramid[s], self.src_image_concat_pyramid[s]], axis=0))
            if opt.add_movenet:
            	self.mask_indicate_pyramid.append(mask_indicate)
            	self.mask_indicate_all_pyramid.append(mask_indicate_all)
            	self.fwd_object_mask_all_pyramid.append(fwd_object_mask)
       
        self.total_loss += self.rigid_warp_loss + self.disp_smooth_loss + self.exp_loss + self.exp_loss_cc + self.rigid_warp_photo_loop_loss
    def collect_summaries(self):
        opt = self.opt
        bs = opt.batch_size
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("rigid_warp_loss", self.rigid_warp_loss)
        tf.summary.scalar("disp_smooth_loss", self.disp_smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        tf.summary.scalar("rigid_warp_photo_loop_loss", self.rigid_warp_photo_loop_loss)


        for s in [0]:  # 两个pose和两个图形大
            # tf.summary.histogram("scale%d_depth" % s, self.pred_depth_all[s])

            tf.summary.image('target_image_%d' % s,  self.deprocess_image(self.tgt_image_tile_pyramid[s]))  # 将tgt_image_all缩小并转换为unit8的格式，在记录节点
            tf.summary.image('scale%d_pred_depth' % s, 1. / self.pred_depth[s])
            if opt.add_movenet:
                tf.summary.image('scale%d_fwd_object_mask' % s, self.fwd_object_mask_all_pyramid[s])
            for i in [0,1]:
                tf.summary.image('scale%d_source_image_%d' % (s, i), self.deprocess_image(self.src_image_concat_pyramid[s][i * bs:(i + 1) * bs, :, :, :]))
                tf.summary.image('scale%d_fwd_rigid_flow_pyramid_%d' % (s, i), tf.expand_dims(self.deprocess_image(
                    self.fwd_rigid_flow_pyramid[s][bs * i:bs * (i + 1), :, :, 0]), -1))
                tf.summary.image('scale%d_fwd_warp_%d' % (s, i), self.deprocess_image(self.fwd_rigid_warp_pyramid[s][i * bs:(i + 1) * bs, :, :, :]))
                tf.summary.image('scale%d_bwd_warp_%d' % (s, i), self.deprocess_image(self.bwd_rigid_warp_pyramid[s][i * bs:(i + 1) * bs, :, :, :]))
                tf.summary.image('scale%d_fwd_rigid_error_%d' % (s, i), self.deprocess_image(self.fwd_rigid_error_pyramid[s][i * bs:(i + 1) * bs, :, :, :]))
                if opt.add_movenet:
                        tf.summary.image('scale%d_mask_indicate_difference_pyramid_%d' % (s, i), self.deprocess_image(self.mask_indicate_pyramid[s][i * bs:(i + 1) * bs, :, :, :]))
                        tf.summary.image('scale%d_mask_indicate_all_difference_pyramid_%d' % (s, i), self.deprocess_image(self.mask_indicate_all_pyramid[s][i * bs:(i + 1) * bs, :, :, :]))
                        tf.summary.image('scale%d_exp_mask_%d' % (s, i),tf.expand_dims(self.fwd_exp_concat_pyramid[s][i*bs:i*bs+bs, :, :, 1], -1))
                        tf.summary.image('scale%d_fwd_rigid_self_loop_error_%d' % (s, i), self.deprocess_image(self.fwd_rigid_error_pyramid_self_loop[s][i * bs:(i + 1) * bs, :, :, :]))                

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp,
                               (opt.batch_size,
                                int(opt.img_height/(2**downscaling)),
                                int(opt.img_width/(2**downscaling)),
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        ref_exp_mask_ = tf.concat([ref_exp_mask,ref_exp_mask],axis=0)
        return ref_exp_mask_

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity(self, x, y):
        return self.opt.alpha_recon_image * self.SSIM(x, y) + (1-self.opt.alpha_recon_image) * tf.abs(x-y)

    def image_similarity_with_mask(self, x, mask,y):
        return self.opt.alpha_recon_image * self.SSIM(x, y)*mask + (1-self.opt.alpha_recon_image) * tf.abs(x-y)*mask

    def L2_norm(self, x, axis=3, keep_dims=True):
        curr_offset = 1e-10
        l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keep_dims=keep_dims)
        return l2_norm

    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean

    def scale_pyramid(self, img, num_scales):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = int(h / ratio)
                nw = int(w / ratio)
                scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            return scaled_imgs

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def compute_smooth_loss(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(tf.expand_dims(flow[:,:,:,i], -1), img)
        return smoothness/2

    def preprocess_image(self, image):
        # Assuming input image is uint8
        if image == None:
            return None
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

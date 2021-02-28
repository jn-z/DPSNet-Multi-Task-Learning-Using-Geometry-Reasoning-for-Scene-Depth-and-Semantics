# coding=utf-8
# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
from __future__ import division
import numpy as np
import tensorflow as tf

def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = tf.shape(z)[0]
  N = 1
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size, _ = vec.get_shape().as_list()
  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(vec, [0, 3], [-1, 1])
  ry = tf.slice(vec, [0, 4], [-1, 1])
  rz = tf.slice(vec, [0, 5], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)
  return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords

def flow_warp(src_img, flow):
  """ inverse warp a source image to the target image plane based on flow field
  Args:
    src_img: the source  image [batch, height_s, width_s, 3]
    flow: target image to source image flow [batch, height_t, width_t, 2]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
  """
  batch, height, width, _ = src_img.get_shape().as_list()
  tgt_pixel_coords = tf.transpose(meshgrid(batch, height, width, False),
                     [0, 2, 3, 1])  # shape=[batch ,h,w, channel=(x,y)]
  src_pixel_coords = tgt_pixel_coords + flow  # so wo can predict flow is normlize flow ,because tgt_pixel_coords is not 1 pixel
  output_img,mask_image = bilinear_sampler(src_img, src_pixel_coords)
  return output_img, mask_image

def compute_rigid_flow(depth, pose, intrinsics, reverse_pose=False,pose_matrix=False):
  """Compute the rigid flow from target image plane to source image

  Args:
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source (or source to target if reverse_pose=True)
          camera transformation matrix [batch, 6], in the order of
          tx, ty, tz, rx, ry, rz;
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Rigid flow from target image to source image [batch, height_t, width_t, 2]
  """
  batch, height, width = depth.get_shape().as_list()    #  [4, 128, 416]
  # Convert pose vector to matrix
  if pose_matrix == False:
    pose = pose_vec2mat(pose)  #shape=[B, 4, 4]
  if reverse_pose:
    pose = tf.matrix_inverse(pose)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width) #[batch,  (3 if homogeneous), height, width]
  tgt_pixel_coords = tf.transpose(pixel_coords[:,:2,:,:], [0, 2, 3, 1]) #[batch,height,witdh,2]
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)  #tgt [4,4,128,416]  [batch,  (4 if homogeneous), height, width]
  # Construct a 4x4 intrinsic matrix
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose) # [batch,4,4]*[batch,4,4]=[]batch,4,4]
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  rigid_flow = src_pixel_coords - tgt_pixel_coords

  return rigid_flow

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear 双线性 sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  坐标：源像素坐标从[batch, height_s, width_s, channels ]。height_t / width_t对应的输出图像的尺寸（不需要height_s / width_s相同）。这两个通道分别对应于x和y坐标。
  ##%……&**新解：coords就像是在第一相机坐标系下，由target视图转化为source 视图的索引坐标，，source本身的坐标范围就是长宽，当把target中的像素转换到source时，那些能看见的像素就知道了
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
    mask_image A mask image for sampler
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([n_repeats,])), 1), [1, 0])       #shapes={1,n_repeats}=shape=(1, 53248)

    #rep = tf.cast(rep, 'float32')

    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)  # tf.reshape(x, (-1, 1)=shape=(4, 1),value=【0,dim,2*dim,3*dim】=【[0]，[53248]，[106496]，[159744]】,
    #  x=[4,hight*width]=[4,53248]  其实就是将每一个面积展开成1维向量，最后根据batch 组成=【batch，height*width】

    return tf.reshape(x, [-1])   # 【4*height*width】=(212992,)，即是全部展开成一位向量，（体会tensorflow的全部计算都是展开成向量进行计算）

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)  #分队别的应的是xy方向上的像素坐标，shape=[batch,h,w]
    inp_size = imgs.get_shape()    #[4, height_s, width_s, 3]
    coord_size = coords.get_shape()    #[4, height, width, 2]
    out_size = coords.get_shape().as_list()   #【4, height, width, 2】
    out_size[3] = imgs.get_shape().as_list()[3]    #转换成[4, height, width, 3]只是shape变哈

    coords_x = tf.cast(coords_x, 'float32')    #值全部转换成float值#coords_x(4, 128, 416, 1)
    coords_y = tf.cast(coords_y, 'float32')    #值全部转换成float值
    x0 = tf.floor(coords_x)   #向下取整，可以理解成在在整个标注坐标的图中全部向左移动成整数位置
    x1 = x0 + 1 #x1(4, 128, 416, 1)   #理解成在在整个标注坐标的图中全部向右移动1个单位
    y0 = tf.floor(coords_y)   #同理 值全部转换成float值#coords_y(4, 128, 416, 1)
    y1 = y0 + 1
    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')  #height-1，换成float
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')  #width-1，换成float
    zero = tf.zeros([1], dtype='float32')  #【0】
    ones=tf.ones([1],dtype='float32')
    #判断x，y是否在img的图像范围内不满足的转换，小于边界的转化成0，大于边界的转换成x_max
    #这个clip——byalues就是人工设置边界范围，根据设置的范围，建立confindence image
    x0_safe = tf.clip_by_value(x0, zero, x_max)#x0_safe(4, 128, 416, 1)，由于target图像是默认为第一相机坐标系，所以可以直接比较
    #tmp_zeros = tf.zeros_like(x0_safe, dtype=tf.float32)
    #tmp_ones = tf.ones_like(x0_safe, dtype=tf.float32)  # {32,2,48,64}
    #tm_ones = tf.where(x0_safe == zero, tmp_ones, tmp_zeros)  # 标注x0-safe中位移较多的像素坐标
    #x0_max = tf.where(x0_safe == x_max, tm_ones, tmp_zeros)  # 标注x0-safe中位移较多的像素坐标
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    wt_x0_mask = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    wt_x1_mask = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    wt_y0_mask= (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    wt_y1_mask = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    # 权重就是距离每一个像素坐标的距离，超出边界的权重仍然设为权重，是否有影响
    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from   从平面图像中提取的索引
    dim2 = tf.cast(inp_size[2], 'float32')       #指的是宽度
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')    #指的是面积
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,  #是一个list（里面存的是面积大小值）=【0,dim,2*dim,3*dim】=【0，53248，106496，159744，212992】,shape=[4,]
            coord_size[1] * coord_size[2]),      #coord_size[1] * coord_size[2]指的是像素的面积，最后将batch个面积的coord_size[1] * coord_size[2]全部展开成1维
        [out_size[0], out_size[1], out_size[2], 1])  #[4,height,width,1]   base代代表组成的合成图
    #新解：——repeat最后得到的是shape=【batch，height*witdh】=【4，53248】，具体数值为【0，：】=0，【1，：】=53248，【2，：】=106496，【3，：】=159744
    ######所以base的shape=【4，128，416，1】，其中每一个层为：【0，：，：，：】=0，【1，：，：，：】=53248，【2，：，：，：】=106496，【3，：，；，：】=159744
    #y,y0,y1, ory0_safe,y1_safe的shape都是=【batch，h，w】



    #base_y0(4, 128, 416, 1)=base(4,128,416,1)+y0_safe*dim2(4,126,416,1)
    base_y0 = base + y0_safe * dim2    #这里*宽度值，代表的是把y方向的位移全部乘以宽度，才能站是在实际中的运动位移（依运动的面积去表示的）
    #print("y0_safe * dim2s", (y0_safe * dim2).get_shape)  shape=(4, 128, 416, 1) ,就是说在每一个移动y坐标上乘以宽度

    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])     ####新解：因为将y方向的位移都转化到1维向量上去展示，而且都是一面的长度去表示的呈高倍数，因此x方向的位移就可以直接相加，且是数据值较小，不为正倍数，就可以分辨出是x方向   #idx00(212992, )
    idx01 = x0_safe + base_y1       #idx01(4, 128, 416, 1)等都是直接加起来，只是还未转换成列向量
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1
    ##4%……（*&（￥需要注意这就是位移向量（从target到source的位移向量），因为都是用第一相机坐标系下的像素坐标去表示的
    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))    #矩阵分解成【batch*height*width，3】  #imgs_flat(212992, 3)，注意他这里的排序，还将batch也排进来了，所以是与base相暗合（base这样的排序就是对的）
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)#imgs00(212992, 3)
    #$%^#构建一个mask的图，直接从理论中计算出来：
    mask_image_intial=tf.ones_like(imgs, dtype=tf.float32)
    mask_image=tf.reshape(mask_image_intial, tf.stack([-1, inp_size[3]]))
    mask_image = tf.cast(mask_image, 'float32')
    mask_im00 = tf.reshape(tf.gather(mask_image, tf.cast(idx00, 'int32')), out_size)  # imgs00(212992, 3)
    #将越界的范围用0标出来

    #print("imgs00 ", tf.gather(imgs_flat, tf.cast(idx00, 'int32')).shape)
    ###因此通过tf.gather去直接提取source的图像是完全成立的，用的是一同一个坐标系
    #还存在一个问题，就是边界的值是如何设置的呢？就会造成图里面的结果
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)#imgs01(4, 128, 416, 1, 3)
    mask_im01 = tf.reshape(tf.gather(mask_image, tf.cast(idx01, 'int32')), out_size)  # imgs00(212992, 3)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    mask_im10 = tf.reshape(tf.gather(mask_image, tf.cast(idx10, 'int32')), out_size)  # imgs00(212992, 3)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)
    mask_im11 = tf.reshape(tf.gather(mask_image, tf.cast(idx11, 'int32')), out_size)  # imgs00(212992, 3)

    w00 = wt_x0 * wt_y0  #w00(4, 128, 416, 1)
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    w00_mask = wt_x0_mask * wt_y0_mask  # w00(4, 128, 416, 1)
    w01_mask = wt_x0_mask * wt_y1_mask
    w10_mask = wt_x1_mask * wt_y0_mask
    w11_mask = wt_x1_mask * wt_y1_mask

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    mask_image = tf.add_n([
      w00_mask * mask_im00, w01_mask * mask_im01,
      w10_mask * mask_im10, w11_mask * mask_im11
    ])

    return output,mask_image

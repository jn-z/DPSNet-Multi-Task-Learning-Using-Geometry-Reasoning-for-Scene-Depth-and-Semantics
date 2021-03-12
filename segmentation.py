# This file contains util stuff needed for segmentation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight, ignore_index=0)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)

#-----------------------------------------
def colorize_segmentaions(batch_size, fake, real, test=None):

    fake = fake.numpy()[:, :, :, :]
    # transform to the image pixel:0~255
    real_0 = (real.cpu().numpy()[:, :, :, :]*255).astype(int)
    real = np.squeeze(real_0)
    fake_0 = np.argmax(fake, axis=1)
    fake_0 = np.expand_dims(fake_0,axis=1)
    ind = np.squeeze(fake_0)# #[batch_size,128,512]
    fake_0 = torch.from_numpy(fake_0)
    real_0 = torch.from_numpy(real)
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


    #Void = [0, 0, 0]
    #Sky = [128, 128, 128]
    #Building = [128, 0, 0]
    #Road = [128, 64, 128]
    #Sidewalk = [0, 0, 192]
    #Fence = [64, 64, 128]
    #Vegetation = [128, 128, 0]
    #Pole = [192, 192, 128]
    #Car = [64, 0, 128]
    #TrafficSign = [192, 128, 128]
    #Pedestrian = [64, 64, 0]
    #Bicycle = [0, 128, 192]
    #LaneMarking = [0, 172, 0]
    #TrafficLight = [0, 128, 128]
    #weizhi = [255,255,255]
    Void = [0, 0, 0]
    Road = [128, 64, 128]
    Sidewalk = [244, 35, 232]
    Building = [70, 70, 70]
    Wall = [102, 102, 156]
    Fence = [190, 153, 153]
    Pole = [153, 153, 153]
    TrafficLight = [250, 170, 30]
    TrafficSign = [220, 220, 0]
    Vegetation = [107, 142, 35]
    Terrain = [152,251, 152]
    Sky = [70, 130, 180]
    Person = [220, 20, 60]
    Rider = [255, 0 ,0]
    Car = [0, 0, 142]
    Truck = [0, 0, 70]
    Bus = [0, 60, 100]
    Train = [0, 80, 100]
    Motorcycle = [0, 0, 230]
    Bicycle = [119, 11, 32]
    RoadLines = [157,234, 50]
    Other = [72, 0, 98]
    RoadWorks = [167, 106, 29]


    label_colours = np.array([Void, Road,Sidewalk,Building,Wall,Fence,Pole,TrafficLight,TrafficSign,Vegetation,Terrain,Sky,Person,Rider,Car,Truck,Bus,Train,Motorcycle,Bicycle,RoadLines,Other,RoadWorks])

    for l in range(0, len(label_colours)):
        r[ind == l] = label_colours[l, 0]
        g[ind == l] = label_colours[l, 1]
        b[ind == l] = label_colours[l, 2]

        r_gt[real == l] = label_colours[l, 0]
        g_gt[real == l] = label_colours[l, 1]
        b_gt[real == l] = label_colours[l, 2]

    rgb = np.zeros((batch_size,3, ind.shape[1], ind.shape[2]))

    rgb[:, 0, :, :] = r
    rgb[:, 1, :, :] = g
    rgb[:, 2, :, :] = b

    rgb_gt = np.zeros((batch_size, 3, ind.shape[1], ind.shape[2]))

    rgb_gt[:, 0, :, :] = r_gt
    rgb_gt[:, 1, :, :] = g_gt
    rgb_gt[:, 2, :, :] = b_gt

    return torch.from_numpy(rgb), torch.from_numpy(rgb_gt), fake_0, real_0

def colorize_segmentaions_instance(real):

    real_0 = (real.cpu().numpy()[:, :, :, :]*255).astype(int)
    real = np.squeeze(real_0)

    instance_object = real.copy()

    for j in range(12,20):
        instance_object[real == j] == 1
        instance_object[real != j] == 0

    return  torch.from_numpy(instance_object)
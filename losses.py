# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Hao Wang
# based on https://github.com/LvXudong-HIT/LCCNet/losses.py

import torch
from torch import nn as nn
import numpy as np
from quaternion_distances import quaternion_distance
from utils import quat2mat, rotate_back, rotate_forward, tvector2mat, quaternion_from_matrix
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err,cam_calib):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.
        if self.rescale_trans != 0.:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.
        if self.rescale_rot != 0.:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_rot*loss_rot+self.rescale_trans*loss_transl

         
        optical_distance_loss = torch.tensor([0.0]).to(transl_err.device)
        h=0

        for i in range(len(point_clouds)):
            X= cam_calib[i].numpy()

            X1=np.insert(X, 3, np.array([0, 0, 0]), axis=0)
            X1=np.insert(X1, 3, np.array([0, 0, 0,1]), axis=1)
            X1=torch.Tensor(X1)
            X1=X1.cuda()
            point_cloud_gt = point_clouds[i].to(transl_err.device)
            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)
            RTCA = torch.mm(X1, RT_target)
            RTCB = torch.mm(RTCA, point_cloud_gt)
            temptensor=RTCB[2:3]
            RTCC=RTCB/temptensor
            RTCC.F=RTCC[0]
            RTCC.S=RTCC[1]
            Quanzhong1=RTCC.F-cam_calib[i][0][2]
            Quanzhong2=RTCC.S-cam_calib[i][1][2]
            Quanzhong=torch.cat(([Quanzhong1,Quanzhong2]),dim=0)
            Quanzhong=torch.reshape(Quanzhong,(2,-1))
            Quanzhong = Quanzhong.norm(dim=0)
            Quanzhong=1/Quanzhong
            Quanzhong= F.normalize(Quanzhong.float(),p=1,dim=0,eps=5)
            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)
            RTCA1 = torch.mm(X1, RT_predicted)
            RTCB1 = torch.mm(RTCA1, point_cloud_gt)
            temptensor1=RTCB1[2:3]
            RTCC1=RTCB1/temptensor1
            RTCC1.F=RTCC1[0]
            RTCC1.S=RTCC1[1]
            RTCC.F=torch.where(RTCC.F<1280,RTCC.F, RTCC1.F)
            RTCC.F=torch.where(RTCC.F>0,RTCC.F, RTCC1.F)
            RTCC1.F=torch.where(RTCC1.F>0,RTCC1.F, RTCC.F)
            RTCC1.F=torch.where(RTCC1.F<1280,RTCC1.F, RTCC.F)
            RTCC.S=torch.where(RTCC.S<384,RTCC.S, RTCC1.S)
            RTCC.S=torch.where(RTCC.S>0,RTCC.S, RTCC1.S)
            RTCC1.S=torch.where(RTCC1.S>0,RTCC1.S, RTCC.S)
            RTCC1.S=torch.where(RTCC1.S<384,RTCC1.S, RTCC.S)
            RTCCF=torch.cat((RTCC.F,RTCC.S),dim=0)
            RTCC1F=torch.cat((RTCC1.F,RTCC1.S),dim=0)
            RTCCF=torch.reshape(RTCCF,(2,-1))
            RTCC1F=torch.reshape(RTCC1F,(2,-1))
            error = (RTCCF - RTCC1F).norm(dim=0)
            error=torch.mul(error,Quanzhong)
            optical_distance_loss += error.mean()
            h=h+1
        #end = time.time()
        #print("3D Distance Time: ", end-start)
        if(h!=0):
            total_loss =  (1 - self.weight_point_cloud) * pose_loss+self.weight_point_cloud * (optical_distance_loss/h)
        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        return self.loss

from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from collections import namedtuple
import torch.nn.functional as F

from pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetFPModule, PointnetProposalModule



class Pointnet_Backbone(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet_Backbone, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.5,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[256, 256, 256, 256],
                use_xyz=use_xyz,
            )
        )
        self.cov_final = nn.Conv1d(256, 256, kernel_size=1)


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)


        return l_xyz[-1], self.cov_final(l_features[-1])




class Pointnet_Tracking(nn.Module):
    r"""
        xorr the search and the template
    """
    def __init__(self, input_channels=3, use_xyz=True, objective = False):
        super(Pointnet_Tracking, self).__init__()

        self.backbone_net = Pointnet_Backbone(input_channels, use_xyz)

        self.cosine = nn.CosineSimilarity(dim=1)

        self.mlp = pt_utils.SharedMLP([4+256,256,256,256], bn=True)

        self.FC_layer_cla = (
                pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(1, activation=None))
        self.fea_layer = (pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, activation=None))
        self.vote_layer = (
                pt_utils.Seq(3+256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3+256, activation=None))
        self.vote_aggregation = PointnetSAModule(
                radius=0.3,
                nsample=16,
                mlp=[1+256, 256, 256, 256],
                use_xyz=use_xyz)
        self.num_proposal = 64
        self.FC_proposal = (
                pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(3+1+1, activation=None))

    def xcorr(self, x_label, x_object, template_xyz):       

        B = x_object.size(0)
        f = x_object.size(1)
        n1 = x_object.size(2)
        n2 = x_label.size(2)
        final_out_cla = self.cosine(x_object.unsqueeze(-1).expand(B,f,n1,n2), x_label.unsqueeze(2).expand(B,f,n1,n2))

        fusion_feature = torch.cat((final_out_cla.unsqueeze(1),template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B,3,n1,n2)),dim = 1)

        fusion_feature = torch.cat((fusion_feature,x_object.unsqueeze(-1).expand(B,f,n1,n2)),dim = 1)

        fusion_feature = self.mlp(fusion_feature)

        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])
        fusion_feature = fusion_feature.squeeze(2)
        fusion_feature = self.fea_layer(fusion_feature)

        return fusion_feature

    def forward(self, template, search):
        r"""
            template: B*512*3 or B*512*6
            search: B*1024*3 or B*1024*6
        """
        template_xyz, template_feature = self.backbone_net(template, [256, 128, 64])

        search_xyz, search_feature = self.backbone_net(search, [512, 256, 128])

        fusion_feature = self.xcorr(search_feature, template_feature, template_xyz)

        estimation_cla = self.FC_layer_cla(fusion_feature).squeeze(1)

        score = estimation_cla.sigmoid()

        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(),fusion_feature),dim = 1)

        offset = self.vote_layer(fusion_xyz_feature)
        vote = fusion_xyz_feature + offset
        vote_xyz = vote[:,0:3,:].transpose(1, 2).contiguous()
        vote_feature = vote[:,3:,:]

        vote_feature = torch.cat((score.unsqueeze(1),vote_feature),dim = 1)

        center_xyzs, proposal_features = self.vote_aggregation(vote_xyz, vote_feature, self.num_proposal)

        proposal_offsets = self.FC_proposal(proposal_features)

        estimation_boxs = torch.cat((proposal_offsets[:,0:3,:]+center_xyzs.transpose(1, 2).contiguous(),proposal_offsets[:,3:5,:]),dim=1)

        return estimation_cla, vote_xyz, estimation_boxs.transpose(1, 2).contiguous(), center_xyzs
import torch
import torch.nn as nn
from nets.utils.identity import identity


class st_gcn_layer(nn.Module):

    def __init__(self, in_channels, out_channels, node_num, neigh_node_kinds, opt, bias=True, perNodeFF2=True, residual=True):

        super().__init__()

        self.inplace = True
        self.momentum = 0.1
        self.out_channels = out_channels
        self.neigh_node_kinds = neigh_node_kinds
        self.node_num = node_num
        self.opt = opt

        if opt.bn2d == True:
            self.batchnorm2d5 = nn.BatchNorm2d(out_channels, momentum=self.momentum)
        else:
            self.batchnorm2d5 = identity()

        if opt.activation_func == 'relu':
            self.activ_func = nn.ReLU(inplace=self.inplace)
        elif opt.activation_func == 'elu_1':
            self.activ_func = nn.ELU(alpha=1.0, inplace=self.inplace)
        elif opt.activation_func == 'elu_0.1':
            self.activ_func = nn.ELU(alpha=0.1, inplace=self.inplace)
        elif opt.activation_func == 'selu':
            self.activ_func = nn.SELU(inplace=self.inplace)
        elif opt.activation_func == 'tanh':
            self.activ_func = nn.Tanh()

        self.per_node_FF = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

        if opt.corre_weight_diversity == 'completely_unshared':
            if opt.channels_share_corre_weights:
                self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, node_num))  #4,1,17
            else:
                self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, node_num)) #randn(0,1),rand(0，1)
        elif opt.corre_weight_diversity == 'SeveralKinds_1node': #several categories of neigh nodes but nodes shared
            if opt.channels_share_corre_weights:
                self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
            else:
                self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'shared': #several categories of neigh nodes but nodes shared
            if opt.channels_share_corre_weights:
                self.corre_weight = nn.Parameter(torch.randn(1, 1, 1)) #
            else:
                self.corre_weight = nn.Parameter(torch.randn(out_channels, 1, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+6sym_limbs':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 7)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 7))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+4limbs':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 5)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 5))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+4limbs_1_23':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 9)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 9))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+2sym_limbs+2_5_3_6':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 11)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 11))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+2sym_limbs+2_5_3_6_pool_trunk+limb':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 11)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 11))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 2)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 2))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 2)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 2))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+2sym_limbs+2_5_3_6_pool_trunklimb':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 11)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 11))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+4sym_limbs+3_6_pool_trunklimb':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 9)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 9))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == '078_910_1114_14_2_5_3_6':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 12)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 12))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+141114+2_5_3_6':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 10)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 10))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+4sym_limbs+3_6':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 9)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 9))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_1trunk+4limbs_1_2_3':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 13)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 13))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_7center_0_1_2_3_4':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 5)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 5))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
        elif opt.corre_weight_diversity == 'SeveralKinds_7center_0_1_2_3_4_updown':
            if self.node_num == (2*opt.pad+1)*opt.n_joints:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 9)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 9))
            elif self.node_num == (2*opt.pad+1)*5:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))
            elif self.node_num == (2*opt.pad+1)*6:
                if opt.channels_share_corre_weights:
                    self.corre_weight = nn.Parameter(torch.randn(self.neigh_node_kinds, 1, 1)) #
                else:
                    self.corre_weight = nn.Parameter(torch.randn(out_channels, self.neigh_node_kinds, 1, 1))

        if perNodeFF2 == True:
            self.bn_relu_drop = nn.Sequential(
                self.batchnorm2d5,
                self.activ_func,
                nn.Dropout(opt.dropout5),)
        else:
            if residual:
                self.bn_relu_drop = nn.Sequential(
                    self.batchnorm2d5,
                    nn.Dropout(opt.dropout5),)
            else:
                self.bn_relu_drop = nn.Sequential(
                    self.batchnorm2d5,
                    self.activ_func,
                    nn.Dropout(opt.dropout5),)

    def forward(self, x, A):
        if self.opt.perNodeFF:
            x = self.per_node_FF(x) #  N, C, S, M   C:out_channels, S=1, M:node_num

        if self.opt.channels_share_corre_weights:
            corre_weight_kinds = torch.zeros([self.neigh_node_kinds, self.node_num, self.node_num]).cuda()
        else:
            corre_weight_kinds = torch.zeros([self.out_channels, self.neigh_node_kinds, self.node_num, self.node_num]).cuda()  #cuda,require_grad

        if self.opt.corre_weight_diversity in ['completely_unshared', 'SeveralKinds_1node','shared']:
            corre_weight2 = torch.sum(self.corre_weight.mul(A), dim=-3)  # A: (neigh_node_kinds, node_num, node_num)   obtain: C/noC,M,M

        elif self.opt.corre_weight_diversity == 'SeveralKinds_1trunk+6sym_limbs':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8,9,10], [1,4],[5,2],[3,6], [11,14], [12,15], [13,16]]
                corre_weight_parts = [[0,0,0,0,0], [1,1],[2,2], [3,3],[4,4], [5,5], [6,6]]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 2, 0, 3, 1]
                corre_weight_parts = [0, 0, 0, 0, 0]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [4, 2, 0, 3, 1, 5]
                corre_weight_parts = [0, 0, 0, 0, 0, 0]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_1trunk+4limbs':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8,9,10], [4,5,6],[1,2,3], [11,12,13],[14,15,16]]
                corre_weight_parts = [[0,0,0,0,0],[1,1,1],[2,2,2], [3,3,3],[4,4,4]]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 1, 0, 3, 2]
                corre_weight_parts = [0, 0, 0,0, 0]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [[4,5], 1, 0, 3, 2]
                corre_weight_parts = [[0,0], 0, 0, 0, 0]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_1trunk+2sym_limbs+2_5_3_6':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8,9,10],[1,4],[11,14], [2],[5],[12],[15],[3],[6], [13],[16]]
                corre_weight_parts = [[0,0,0,0,0],[1,1],[2,2],[3],[4],[5],[6],[7],[8],[9],[10] ]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 1, 0, 3, 2]
                corre_weight_parts = [0, 0, 0,0, 0]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [[4,5], 1, 0, 3, 2]
                corre_weight_parts = [[0,0], 0, 0, 0, 0]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_1trunk+2sym_limbs+2_5_3_6_pool_trunk+limb':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8,9,10],[1,4],[11,14], [2],[5],[12],[15],[3],[6], [13],[16]]
                corre_weight_parts = [[0,0,0,0,0],[1,1],[2,2],[3],[4],[5],[6],[7],[8],[9],[10] ]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 1, 0, 3, 2]
                corre_weight_parts = [0, 1, 1,1, 1]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [[4,5], 1, 0, 3, 2]
                corre_weight_parts = [[0,0], 1, 1, 1, 1]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_1trunk+2sym_limbs+2_5_3_6_pool_trunklimb':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8,9,10],[1,4],[11,14], [2],[5],[12],[15],[3],[6], [13],[16]]
                corre_weight_parts = [[0,0,0,0,0],[1,1],[2,2],[3],[4],[5],[6],[7],[8],[9],[10] ]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 1, 0, 3, 2]
                corre_weight_parts = [0, 0, 0,0, 0]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [[4,5], 1, 0, 3, 2]
                corre_weight_parts = [[0,0], 0, 0, 0, 0]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == '078_910_1114_14_2_5_3_6':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8],[9,10],[1,4],[11,14], [2],[5],[12],[15],[3],[6], [13],[16]]
                corre_weight_parts = [[0,0,0],[1,1],[2,2],[3,3],[4],[5],[6],[7],[8],[9],[10],[11] ]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 1, 0, 3, 2]
                corre_weight_parts = [0, 0, 0,0, 0]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [[4,5], 1, 0, 3, 2]
                corre_weight_parts = [[0,0], 0, 0, 0, 0]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_1trunk+141114+2_5_3_6':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8,9,10],[1,4,11,14], [2],[5],[12],[15],[3],[6], [13],[16]]
                corre_weight_parts = [[0,0,0,0,0],[1,1,1,1],[2],[3],[4],[5],[6],[7],[8],[9] ]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 1, 0, 3, 2]
                corre_weight_parts = [0, 0, 0,0, 0]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [[4,5], 1, 0, 3, 2]
                corre_weight_parts = [[0,0], 0, 0, 0, 0]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_1trunk+4sym_limbs+3_6':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8,9,10],[1,4],[11,14], [2,5],[12,15],[3],[6], [13],[16]]
                corre_weight_parts = [[0,0,0,0,0],[1,1],[2,2],[3,3],[4,4],[5],[6],[7],[8] ]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 1, 0, 3, 2]
                corre_weight_parts = [0, 0, 0,0, 0]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [[4,5], 1, 0, 3, 2]
                corre_weight_parts = [[0,0], 0, 0, 0, 0]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_1trunk+4limbs_1_2_3':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[0,7,8,9,10], [4],[5],[6],[1],[2],[3], [11],[12],[13],[14],[15],[16]]
                corre_weight_parts = [[0,0,0,0,0],[1],[2],[3],[4],[5],[6], [7],[8],[9],[10],[11],[12]]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [4, 1, 0, 3, 2]
                corre_weight_parts = [0, 0, 0,0, 0]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [[4,5], 1, 0, 3, 2]
                corre_weight_parts = [[0,0], 0, 0, 0, 0]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_7center_0_1_2_3_4':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[7],[0,8], [1,4,9,11,14],[2,5,10,12,15], [3,6,13,16]]
                corre_weight_parts = [[0], [1,1],[2,2,2,2,2], [3,3,3,3,3],[4,4,4,4]]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [[4], [1, 0, 3, 2]]
                corre_weight_parts = [0, [0, 0,0, 0]]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [4,[5, 1, 0, 3, 2]]
                corre_weight_parts = [0, [0, 0, 0, 0, 0]]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)
        elif self.opt.corre_weight_diversity == 'SeveralKinds_7center_0_1_2_3_4_updown':
            if self.node_num == (2*self.opt.pad+1)*self.opt.n_joints:
                skeleton_parts = [[7],[0],[8], [1,4],[9,11,14],[2,5],[10,12,15], [3,6],[13,16]]
                corre_weight_parts = [[0], [1],[2], [3,3],[4,4,4], [5,5],[6,6,6],[7,7],[8,8]]
            elif self.node_num == (2*self.opt.pad+1)*5:
                skeleton_parts = [[4], [1, 0], [3, 2]]
                corre_weight_parts = [0, [0, 0],[0, 0]]
            elif self.node_num == (2*self.opt.pad+1)*6:
                skeleton_parts = [4,[5, 1, 0], [3, 2]]
                corre_weight_parts = [0, [0, 0, 0], [0, 0]]

            for i in range (len(skeleton_parts)):
                corre_weight_kinds[...,skeleton_parts[i]] = self.corre_weight[...,corre_weight_parts[i]].mul(A[:,:,skeleton_parts[i]])
            corre_weight2 = torch.sum(corre_weight_kinds, dim=-3)


        if self.opt.channels_share_corre_weights:
            x = torch.einsum('ncsm,mw->ncsw', (x, corre_weight2))
        else:
            x = torch.einsum('ncsm,cmw->ncsw', (x, corre_weight2))

        x = self.bn_relu_drop(x)

        return x.contiguous(), A #
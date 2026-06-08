import torch
import torch.nn as nn

from nets.utils.st_gcn_layer import st_gcn_layer
from nets.utils.identity import identity
from nets.utils.graph_frames import Graph
from nets.utils.graph_frames_withpool_2 import Graph_pool


class Model(nn.Module):

    def __init__(self, opt):
        super().__init__()

        if opt.two_scale_channels=='128-384':
            inter_channels=[128, 384, 256] #128-384-256
        elif opt.two_scale_channels=='160-320':
            inter_channels = [160, 320, 640]
        else:
            inter_channels = [128, 256, 512]
        self.momentum = 0.1
        self.in_channels = opt.in_channels
        self.out_channels = opt.out_channels
        self.layout = opt.layout
        self.inplace = True
        self.pad = opt.pad
        self.framework = opt.framework
        self.opt = opt

        # original graph
        self.graph = Graph(self.layout, opt=opt, pad=opt.pad)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda()
        # pooled graph
        self.graph_pool = Graph_pool(self.layout, opt=opt, pad=opt.pad)
        self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cuda()

        neigh_node_kinds = self.A.size(0)
        neigh_node_kinds_pool = self.A_pool.size(0)

        self.input_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, self.momentum)

        if opt.bn2d == True:
            self.batchnorm2d0 = nn.BatchNorm2d(inter_channels[0], momentum=self.momentum)
            self.batchnorm2d1 = nn.BatchNorm2d(inter_channels[1], momentum=self.momentum)
            self.batchnorm2d2 = nn.BatchNorm2d(inter_channels[2], momentum=self.momentum)
        else:
            self.batchnorm2d0 = identity()
            self.batchnorm2d1 = identity()
            self.batchnorm2d2 = identity()

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

        self.conv_fro_aft_gcn = nn.ModuleList((
            conv_before_after_gcn(self.in_channels,  inter_channels[0], opt, residual=False),
            conv_before_after_gcn(inter_channels[0], inter_channels[0], opt, opt.conv_residual),
            conv_before_after_gcn(inter_channels[0], inter_channels[1], opt, opt.conv_residual),
            conv_before_after_gcn(inter_channels[1], inter_channels[1], opt, opt.conv_residual),
            conv_before_after_gcn(inter_channels[2], inter_channels[0], opt, opt.conv_residual),
            conv_before_after_gcn(inter_channels[2], inter_channels[2], opt, opt.conv_residual),
            conv_before_after_gcn(inter_channels[0], inter_channels[2], opt, opt.conv_residual),
            conv_before_after_gcn(inter_channels[1], inter_channels[2], opt, opt.conv_residual),
            conv_before_after_gcn(inter_channels[2], inter_channels[1], opt, opt.conv_residual),
            conv_before_after_gcn(inter_channels[1], inter_channels[0], opt, opt.conv_residual),
            conv_before_after_gcn(2*inter_channels[2], inter_channels[2], opt, opt.conv_residual),
            conv_before_after_gcn(2*inter_channels[2], inter_channels[1], opt, opt.conv_residual),))

        self.st_gcn_modules = nn.ModuleList((
            st_gcn(self.in_channels,  inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, residual=False),
            st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),))

        self.st_gcn_scale1 = nn.ModuleList((
                st_gcn(self.in_channels,  inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, residual=False),
                st_gcn(inter_channels[0], inter_channels[1], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
                st_gcn(inter_channels[1], inter_channels[1], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
                st_gcn(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, opt.perNodeFF2, opt.gcn_residual),
                st_gcn(self.in_channels, inter_channels[0], self.graph.num_node, neigh_node_kinds, opt, perNodeFF2=True, residual=False),))

        self.st_2gcns_scale1 = nn.ModuleList((
            two_st_gcns(inter_channels[0], inter_channels[0], self.graph.num_node, neigh_node_kinds, opt),))

        self.st_gcn_scale2 = nn.ModuleList((
            st_gcn(inter_channels[1], inter_channels[2], self.graph_pool.num_node, neigh_node_kinds_pool, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[2], inter_channels[2], self.graph_pool.num_node, neigh_node_kinds_pool, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[1], inter_channels[1], self.graph_pool.num_node, neigh_node_kinds_pool, opt, opt.perNodeFF2, opt.gcn_residual),
            st_gcn(inter_channels[0], inter_channels[1], self.graph_pool.num_node, neigh_node_kinds_pool, opt, opt.perNodeFF2, opt.gcn_residual), ))

        self.st_2gcns_scale2 = nn.ModuleList((
            two_st_gcns(inter_channels[1], inter_channels[1], self.graph_pool.num_node, neigh_node_kinds_pool, opt),))

        self.conv_scale2 = nn.Sequential(
            nn.Conv2d(inter_channels[2]*2, inter_channels[1], kernel_size=(1, 1), padding=(0, 0)),
            self.batchnorm2d1,
            self.activ_func,
            nn.Dropout(opt.dropout2)) #0.1

        self.conv_scale2_2 = nn.Sequential(
            nn.Conv2d(inter_channels[2], inter_channels[1], kernel_size=(1, 1), padding=(0, 0)),
            self.batchnorm2d1,
            self.activ_func,
            nn.Dropout(opt.dropout6)) #0.25

        self.conv_scale2_3 = nn.Sequential(
            nn.Conv2d(inter_channels[2], inter_channels[1], kernel_size=(1, 1), padding=(0, 0)),
            self.batchnorm2d1,
            self.activ_func,
            nn.Dropout(opt.dropout11)) #0.25

        self.conv_scale3 = nn.Sequential(
            nn.Conv2d(inter_channels[2], inter_channels[2], kernel_size=(1, 1), padding=(0, 0)),
            self.batchnorm2d2,
            self.activ_func,
            nn.Dropout(opt.dropout1)) #0.25

        if opt.two_scale_channels=='128-256-512':
            self.conv13_pool = nn.Sequential(
                nn.Conv2d(inter_channels[1], inter_channels[2], kernel_size=(1, 3), stride=(1,3),padding=(0, 0)),
                self.batchnorm2d1,
                self.activ_func,
                nn.Dropout(opt.dropout9)) #0.25
            self.conv12_pool = nn.Sequential(
                nn.Conv2d(inter_channels[1], inter_channels[2], kernel_size=(1, 2), stride=(1,2),padding=(0, 0)),
                self.batchnorm2d1,
                self.activ_func,
                nn.Dropout(opt.dropout13)) #0.25
        else:
            self.conv13_pool = nn.Sequential(
                nn.Conv2d(inter_channels[0], inter_channels[1], kernel_size=(1, 3), stride=(1,3),padding=(0, 0)),
                self.batchnorm2d1,
                self.activ_func,
                nn.Dropout(opt.dropout9))
            self.conv12_pool = nn.Sequential(
                nn.Conv2d(inter_channels[0], inter_channels[1], kernel_size=(1, 2), stride=(1,2),padding=(0, 0)),
                self.batchnorm2d1,
                self.activ_func,
                nn.Dropout(opt.dropout13)) #0.25

        if opt.two_scale_channels=='128-256-512':
            self.deconv_up_13 = nn.Sequential(
                nn.ConvTranspose2d(inter_channels[2], inter_channels[1],kernel_size=(1,3),stride=(1,3)),
                self.batchnorm2d0,
                self.activ_func,
                nn.Dropout(opt.dropout12))
            self.deconv_up_12 = nn.Sequential(
                nn.ConvTranspose2d(inter_channels[2], inter_channels[1],kernel_size=(1,2),stride=(1,2)),
                self.batchnorm2d0,
                self.activ_func,
                nn.Dropout(opt.dropout14))

            self.up_pre_conv = nn.Sequential(
                nn.Conv2d(inter_channels[2], inter_channels[1],kernel_size=(1,1),stride=(1,1)),
                self.batchnorm2d0,
                self.activ_func,
                nn.Dropout(opt.dropout10))
        else:
            self.deconv_up_13 = nn.Sequential(
                nn.ConvTranspose2d(inter_channels[1], inter_channels[0],kernel_size=(1,3),stride=(1,3)),
                self.batchnorm2d0,
                self.activ_func,
                nn.Dropout(opt.dropout12))
            self.deconv_up_12 = nn.Sequential(
                nn.ConvTranspose2d(inter_channels[1], inter_channels[0],kernel_size=(1,2),stride=(1,2)),
                self.batchnorm2d0,
                self.activ_func,
                nn.Dropout(opt.dropout14))

            self.up_pre_conv = nn.Sequential(
                nn.Conv2d(inter_channels[1], inter_channels[0],kernel_size=(1,1),stride=(1,1)),
                self.batchnorm2d0,
                self.activ_func,
                nn.Dropout(opt.dropout10))

        self.final_conv = nn.Conv2d(inter_channels[0], self.out_channels, kernel_size=1)
        self.final_conv1 = nn.Conv2d(inter_channels[1], self.out_channels, kernel_size=1)
        self.final_conv2 = nn.Conv2d(inter_channels[2], self.out_channels, kernel_size=1)

    def graph_max_pool(self, x, pool_size):
        if max(pool_size) > 1:
            x = nn.MaxPool2d(pool_size)(x)
            return x
        else:
            return x

    def graph_avg_pool(self, x, pool_size):
        if max(pool_size) > 1:
            x = nn.AvgPool2d(pool_size)(x)
            return x
        else:
            return x

    def graph_conv_pool_5(self, x):
        if not self.opt.graph_pool6_8unshared:
            x = self.conv13_pool(x)  #
            x = torch.cat((x_1,x_2),-1)
        return x

    def graph_conv_pool(self, x):
        if not self.opt.graph_pool6_8unshared:
            x = x[:,:,:,[11, 12, 13, 14, 15, 16, 4, 5, 6, 1, 2, 3, 0, 7, 8, 8, 9, 10]]
            x = self.conv13_pool(x)
        else:
            x_1 = self.conv13_pool(x[:,:,:,[11, 12, 13, 14, 15, 16, 4, 5, 6, 1, 2, 3, 0, 7, 8]])
            x_2 = self.conv12_pool(x[:,:,:,[9,10]])
            x = torch.cat((x_1,x_2),-1)
        return x

    def de_conv_upsample(self, x):
        if not self.opt.graph_pool6_8unshared:
            x = self.deconv_up_13(x)
            #    x[:,:,:,14] = (x[:,:,:,14]+x[:,:,:,15])/2
            x = x[:,:,:,[12,9,10,11,6,7,8,13,14,16,17,0,1,2,3,4,5]]
        else:
            x_1 = self.deconv_up_13(x[:,:,:,[0,1,2,3,4]])
            x_2 = self.deconv_up_12(x[:,:,:,[5]])
            x = torch.cat((x_1,x_2),-1)
            x = x[:,:,:,[12,9,10,11,6,7,8,13,14,15,16,0,1,2,3,4,5]]
      #11, 12, 13, 14, 15, 16, 4, 5, 6, 1, 2, 3, 0, 7, 8,9,10]
        return x


    def forward(self, x, out_all_frame=False):
        # input BN1d
        N, C, T, J, M= x.size() # C=2 T:frames J=17  M=1

        if self.opt.input_bn1d==True:
            x = x.permute(0, 4, 3, 1, 2).contiguous() # N, M, J, C, T
            x = x.view(N * M, J * C, T)

            x = self.input_bn(x)

            x = x.view(N, M, J, C, T)
            x = x.permute(0, 1, 3, 4, 2).contiguous() #N M C T J
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous() #N M C T J
        x = x.view(N * M, C, 1, -1)  # (N * M), C, 1, (T*J)

        if self.opt.framework == 'conv_gcn': #conv_res=false
            conv_list = list(self.conv_fro_aft_gcn)
            x = conv_list[0](x)
            x = conv_list[1](x)
            x = conv_list[1](x)
            x = conv_list[1](x)
            x = conv_list[1](x)
            x = conv_list[1](x)
            x, _ = self.st_gcn_scale1[3](x, self.A)
            x, _ = self.st_gcn_scale1[3](x, self.A)
            if self.opt.pool_rule=='conv_pool':
                x_scale2 = self.graph_conv_pool(x)
                x_scale2, _ = self.st_gcn_scale2[2](x_scale2.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, C[2], 1, T*num_parts
                x_scale2, _ = self.st_gcn_scale2[2](x_scale2, self.A_pool.clone())
                x_up_scale1 = self.de_conv_upsample(x_scale2)
                x = torch.cat((x, x_up_scale1), 1)
                x = self.conv_fro_aft_gcn[5](x)
            x = x.view(N, -1, T, J) # N, C, T ,J
            x = self.final_conv2(x)

        elif self.opt.framework == 'resgcn':
            gcn_list = list(self.st_gcn_modules)
            for i_gcn, gcn in enumerate(gcn_list):
                x, _ = gcn(x, self.A)  # (N * M), C, 1, (T*J)

            x = x.view(N, -1, T, J) # N, C, T ,J
            x = self.final_conv(x)

        elif self.opt.framework == 'two_scale_gcn':
            if self.opt.conv_position in ['before_1', 'before_2', 'before_3','before_1_after_1','before_1_after_2','before_2_after_1','before_2_after_2']:
                x = self.conv_fro_aft_gcn[0](x)
                x_conv1 = x
                if self.opt.conv_position in ['before_2', 'before_3','before_2_after_1','before_2_after_2']:
                    x = self.conv_fro_aft_gcn[1](x)
                    if self.opt.conv_position == 'before_3':
                        x = self.conv_fro_aft_gcn[1](x)
                if self.opt.two_scale_channels=='128-256-512':
                    x, _ = self.st_gcn_scale1[1](x, self.A)  # (N * M), C[0], 1, (T*J)
                    x, _ = self.st_gcn_scale1[2](x, self.A)
                else:
                    if self.opt.residual_every2gcn:
                        x, _ = self.st_2gcns_scale1[0](x, self.A)
                    else:
                        x, _ = self.st_gcn_scale1[3](x, self.A)  # (N * M), C[0], 1, (T*J)
                        x_gcn1 = x
                        x, _ = self.st_gcn_scale1[3](x, self.A)
            else:
                if not self.opt.gcn1_pernodeff2:
                    x, _ = self.st_gcn_scale1[0](x, self.A)  # (N * M), C[0], 1, (T*J)
                else:
                    x, _ = self.st_gcn_scale1[4](x, self.A)
                if self.opt.two_scale_channels=='128-256-512':
                    x, _ = self.st_gcn_scale1[1](x, self.A)  # C[1]
                else:
                    x, _ = self.st_gcn_scale1[3](x, self.A)
            if self.opt.scale1_3gcn:
                if self.opt.two_scale_channels=='128-256-512':
                    x, _ = self.st_gcn_scale1[2](x, self.A)
                else:
                    x, _ = self.st_gcn_scale1[3](x, self.A)
                    x, _ = self.st_gcn_scale1[3](x, self.A)
            if self.opt.pool_rule in ['max_pool', 'avg_pool']:
                for i in range(len(self.graph.part)):
                    num_node_part = len(self.graph.part[i])
                    x_i = x[:, :, :, self.graph.part[i]] #multi-frame yao gai
                    if self.opt.pool_rule == 'max_pool':
                        x_i = self.graph_max_pool(x_i, (1, num_node_part))
                    elif self.opt.pool_rule == 'avg_pool':
                        x_i = self.graph_avg_pool(x_i, (1, num_node_part))
                    x_scale2 = torch.cat((x_scale2, x_i), -1) if i > 0 else x_i  # N, C[1], T, num_parts
            elif self.opt.pool_rule=='conv_pool':
                if self.opt.graph_pool6:
                    x_scale2 = self.graph_conv_pool(x)

            if self.opt.pool_rule=='conv_pool':
                if self.opt.two_scale_channels=='128-256-512':
                    x_scale2, _ = self.st_gcn_scale2[1](x_scale2.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, C[2], 1, T*num_parts
                    x_scale2, _ = self.st_gcn_scale2[1](x_scale2, self.A_pool.clone())  # N, C[2], 1, T*num_parts
                else:
                    if self.opt.residual_every2gcn:
                        x_scale2, _ = self.st_2gcns_scale2[0](x_scale2.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())
                    else:
                        x_scale2, _ = self.st_gcn_scale2[2](x_scale2.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, C[2], 1, T*num_parts
                        x_gcn1_pool = x_scale2
                        if not self.opt.scale2_1gcn:
                            x_scale2, _ = self.st_gcn_scale2[2](x_scale2, self.A_pool.clone())
            else:
                if self.opt.two_scale_channels=='128-256-512':
                    x_scale2, _ = self.st_gcn_scale2[0](x_scale2.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, C[2], 1, T*num_parts
                    x_scale2, _ = self.st_gcn_scale2[1](x_scale2, self.A_pool.clone())  # N, C[2], 1, T*num_parts
                else:
                    x_scale2, _ = self.st_gcn_scale2[3](x_scale2.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, C[2], 1, T*num_parts
                    x_gcn1_pool = x_scale2
                    x_scale2, _ = self.st_gcn_scale2[2](x_scale2, self.A_pool.clone())

            if self.opt.scale2_3gcn:
                if self.opt.two_scale_channels=='128-256-512':
                    x_scale2, _ = self.st_gcn_scale2[1](x_scale2, self.A_pool.clone())
                else:
                    x_scale2, _ = self.st_gcn_scale2[2](x_scale2, self.A_pool.clone())

            if self.opt.pool_rule=='conv_pool':
                x_up_scale1 = self.de_conv_upsample(x_scale2) #inter_channel[0]
            else:
                x_scale2 = self.up_pre_conv(x_scale2)
                x_up_scale1 = torch.zeros((N * M, x_scale2.size(1), T, J)).cuda()  #C[1]
                for i in range(len(self.graph.part)):
                    num_node_part_i = len(self.graph.part[i])
                    if i < 5:
                        x_up_scale1[:, :, :, self.graph.part[i] ] = x_scale2[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node_part_i)
                    elif i ==5:
                        if not self.opt.graph_pool6_8unshared:
                            x_up_scale1[:, :, :, self.graph.part[i][1:3] ] = x_scale2[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node_part_i-1)
                        else:
                            x_up_scale1[:, :, :, self.graph.part[i] ] = x_scale2[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node_part_i)

            if self.opt.skip_connect_rule=='sum':
                if self.opt.extra_skip_connect=='sum_1':
                    x = x + x_up_scale1 + x_gcn1
                elif self.opt.extra_skip_connect=='sum_1_conv1':
                    x = x + x_up_scale1 + x_gcn1 + x_conv1
                else:
                    x = x + x_up_scale1    #C[0]
            elif self.opt.skip_connect_rule=='concat':
                if self.opt.extra_skip_connect=='concat_1':
                    x = torch.cat((x, x_up_scale1, x_gcn1),1)
                elif self.opt.extra_skip_connect=='concat_1_conv1':
                    x = torch.cat((x, x_up_scale1, x_gcn1, x_conv1),1)
                else:
                    x = torch.cat((x, x_up_scale1), 1) #2*C[0]
            elif self.opt.skip_connect_rule=='none':
                x = x_up_scale1  #C[0]

            if self.opt.skip_connect_rule in ['sum', 'none']:
                if self.opt.conv_position in ['after_1', 'after_2', 'after_3','before_1_after_1','before_1_after_2','before_2_after_1','before_2_after_2']:
                    if self.opt.after_channel=='2x':
                        if self.opt.two_scale_channels=='128-256-512':
                            x = self.conv_fro_aft_gcn[7](x)
                        elif self.opt.two_scale_channels=='128-384':
                            x = self.conv_fro_aft_gcn[6](x)
                        else:
                            x = self.conv_fro_aft_gcn[2](x)
                    elif self.opt.after_channel=='1x':
                        if self.opt.two_scale_channels=='128-256-512':
                            x = self.conv_fro_aft_gcn[3](x)  #5 22 6 02
                        else:
                            x = self.conv_fro_aft_gcn[1](x)
                    elif self.opt.after_channel=='3x':
                        if self.opt.two_scale_channels=='128-384':
                            x = self.conv_fro_aft_gcn[2](x)

                    if self.opt.conv_position in ['after_2', 'after3','before_1_after_2','before_2_after_2']:
                        if self.opt.after_channel=='2x':
                            if self.opt.two_scale_channels in ['128-256-512', '128-384']:
                                x = self.conv_fro_aft_gcn[5](x)
                            else:
                                x = self.conv_fro_aft_gcn[3](x)
                        elif self.opt.after_channel=='1x':
                            if self.opt.two_scale_channels in ['128-256-512']:
                                x = self.conv_fro_aft_gcn[3](x)
                            else:
                                x = self.conv_fro_aft_gcn[1](x)
                        elif self.opt.after_channel=='3x':
                            if self.opt.two_scale_channels in ['128-384']:
                                x = self.conv_fro_aft_gcn[3](x)
                        if self.opt.conv_position in ['after3']:
                            if self.opt.after_channel=='2x':
                                if self.opt.two_scale_channels in ['128-256-512', '128-384']:
                                    x = self.conv_fro_aft_gcn[5](x)
                                else:
                                    x = self.conv_fro_aft_gcn[3](x)
                            elif self.opt.after_channel=='1x':
                                if self.opt.two_scale_channels in ['128-256-512']:
                                    x = self.conv_fro_aft_gcn[3](x)
                                else:
                                    x = self.conv_fro_aft_gcn[1](x)
                            elif self.opt.after_channel=='3x':
                                if self.opt.two_scale_channels in ['128-384']:
                                    x = self.conv_fro_aft_gcn[3](x)
                    if self.opt.after_channel=='2x':
                        if self.opt.two_scale_channels in ['128-256-512', '128-384']:
                            x = self.final_conv2(x)
                        else:
                            x = self.final_conv1(x)
                    elif self.opt.after_channel=='1x':
                        if self.opt.two_scale_channels in ['128-256-512']:
                            x = self.final_conv1(x)
                        else:
                            x = self.final_conv(x)
                    elif self.opt.after_channel=='3x':
                        if self.opt.two_scale_channels in ['128-384']:
                            x = self.final_conv1(x)
                else:
                    if self.opt.two_scale_channels in ['128-256-512']:
                        x = self.final_conv1(x)
                    else:
                        x = self.final_conv(x)

            else:
                if self.opt.conv_position in ['after_1', 'after_2', 'after3','before_1_after_1','before_1_after_2','before_2_after_1','before_2_after_2']:
                    if self.opt.after_channel=='2x':
                        if self.opt.two_scale_channels in ['128-256-512', '128-384']:
                            if self.opt.extra_skip_connect=='concat_1':
                                x = self.conv_fro_aft_gcn[7](x)
                            elif self.opt.extra_skip_connect=='concat_1_conv1':
                                x = self.conv_fro_aft_gcn[10](x)
                            else:
                                x = self.conv_fro_aft_gcn[5](x)
                        else:
                            x = self.conv_fro_aft_gcn[3](x)
                    elif self.opt.after_channel=='1x':
                        if self.opt.two_scale_channels in ['128-256-512']:
                            x = self.conv_fro_aft_gcn[8](x)
                        elif self.opt.two_scale_channels in ['128-384']:
                            x = self.conv_fro_aft_gcn[4](x)
                        else:
                            x = self.conv_fro_aft_gcn[9](x)
                    elif self.opt.after_channel=='3x':
                        if self.opt.two_scale_channels in ['128-384']:
                            if self.opt.extra_skip_connect=='concat_1':
                                x = self.conv_fro_aft_gcn[3](x)
                            elif self.opt.extra_skip_connect=='concat_1_conv1':
                                x = self.conv_fro_aft_gcn[11](x)
                            else:
                                x = self.conv_fro_aft_gcn[8](x)

                    if self.opt.conv_position in ['after_2', 'after3','before_1_after_2','before_2_after_2']:
                        if self.opt.after_channel=='2x':
                            if self.opt.two_scale_channels in ['128-256-512', '128-384']:
                                x = self.conv_fro_aft_gcn[5](x)
                            else:
                                x = self.conv_fro_aft_gcn[3](x)
                        elif self.opt.after_channel=='1x':
                            if self.opt.two_scale_channels in ['128-256-512']:
                                x = self.conv_fro_aft_gcn[3](x)
                            else:
                                x = self.conv_fro_aft_gcn[1](x)
                        elif self.opt.after_channel=='3x':
                            if self.opt.two_scale_channels in ['128-384']:
                                x = self.conv_fro_aft_gcn[3](x)

                        if self.opt.conv_position in ['after3']:
                            if self.opt.after_channel=='2x':
                                if self.opt.two_scale_channels in ['128-256-512', '128-384']:
                                    x = self.conv_fro_aft_gcn[5](x)
                                else:
                                    x = self.conv_fro_aft_gcn[3](x)
                            elif self.opt.after_channel=='1x':
                                if self.opt.two_scale_channels in ['128-256-512']:
                                    x = self.conv_fro_aft_gcn[3](x)
                                else:
                                    x = self.conv_fro_aft_gcn[1](x)
                            elif self.opt.after_channel=='3x':
                                if self.opt.two_scale_channels in ['128-384']:
                                    x = self.conv_fro_aft_gcn[3](x)

                    if self.opt.after_channel=='2x':
                        if self.opt.two_scale_channels in ['128-256-512', '128-384']:
                            x = self.final_conv2(x)
                        else:
                            x = self.final_conv1(x)
                    elif self.opt.after_channel=='1x':
                        if self.opt.two_scale_channels in ['128-256-512']:
                            x = self.final_conv1(x)
                        else:
                            x = self.final_conv(x)
                    elif self.opt.after_channel=='3x':
                        if self.opt.two_scale_channels in ['128-384']:
                            x = self.final_conv1(x)
                else:
                    if self.opt.two_scale_channels in ['128-256-512', '128-384']:
                        x = self.final_conv2(x)
                    else:
                        x = self.final_conv1(x)

        elif self.opt.framework == 'three_scale_gcn':
            x, _ = self.st_gcn_scale1[0](x, self.A)  # (N * M), C[0], 1, (T*J)
            x, _ = self.st_gcn_scale1[1](x, self.A)  # C[1]
            if self.opt.scale1_3gcn:
                x, _ = self.st_gcn_scale1[2](x, self.A)

            for i in range(len(self.graph.part)):
                num_node_part = len(self.graph.part[i])
                x_i = x[:, :, :, self.graph.part[i]] #multi-frame yao gai
                x_i = self.graph_max_pool(x_i, (1, num_node_part))
                x_scale2 = torch.cat((x_scale2, x_i), -1) if i > 0 else x_i  # N, C[1], T, num_parts

            x_scale2, _ = self.st_gcn_scale2[0](x_scale2.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, C[2], 1, T*num_parts
            x_scale2, _ = self.st_gcn_scale2[1](x_scale2, self.A_pool.clone())  # N, C[2], 1, T*num_parts

            x_scale2 = x_scale2.view(N, -1, T, len(self.graph.part))  #  N, C[2], T, num_parts
            x_scale3 = self.graph_max_pool(x_scale2, (1, len(self.graph.part)))  # N, C[2], T, 1
            x_scale3 = self.conv_scale3(x_scale3)# N, C[2], T, 1

            x_up_scale2 = torch.cat((x_scale3.repeat(1, 1, 1, len(self.graph.part)), x_scale2), 1)  # N, 2*C[2], T, 5
            x_up_scale2 = self.conv_scale2(x_up_scale2) #N, C[1], T, 5

            x_up_scale1 = torch.zeros((N * M, x_up_scale2.size(1), T, J)).cuda()  #C[1]
            for i in range(len(self.graph.part)):
                num_node_part_i = len(self.graph.part[i])
                x_up_scale1[:, :, :, self.graph.part[i] ] = x_up_scale2[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node_part_i)

            x = torch.cat((x, x_up_scale1), 1) #2*C[1]
            x = self.final_conv2(x)  # N, 3, T, J

        x = x.view(N, M, -1, T, J).permute(0, 2, 3, 4, 1).contiguous()  # N, C, T, J, M

        if out_all_frame:
            x_out = x
        else:
            x_out = x[:, :, self.pad].unsqueeze(2)

        return x_out

class st_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, node_num, neigh_node_kinds, opt, perNodeFF2=False, residual=True):

        super().__init__()

        self.inplace = True
        self.momentum = 0.1
        self.resi = residual

        if opt.bn2d == True:
            self.batchnorm2d3 = nn.BatchNorm2d(out_channels, momentum=self.momentum)
        else:
            self.batchnorm2d3 = identity()

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

        self.st_gcn_layer = st_gcn_layer(in_channels, out_channels, node_num, neigh_node_kinds, opt, bias=True, perNodeFF2=opt.perNodeFF2, residual=residual)

        if perNodeFF2:
            if residual:
                self.per_node_FF2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), padding=0),
                    self.batchnorm2d3,
                    nn.Dropout(opt.dropout4), )
            else:
                self.per_node_FF2 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), padding=0),
                    self.batchnorm2d3,
                    self.activ_func,
                    nn.Dropout(opt.dropout4), )
        else:
            self.per_node_FF2 = identity()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1)),
                self.batchnorm2d3,
            )

    def forward(self, x, A):

        residual = self.residual(x)

        x, A = self.st_gcn_layer(x, A)

        x = self.per_node_FF2(x) + residual

        if self.resi:
            return self.activ_func(x), A
        else:
            return x, A


class two_st_gcns(nn.Module):

    def __init__(self, in_channels, out_channels, node_num, neigh_node_kinds, opt, residual=True):

        super().__init__()

        self.inplace = True
        self.momentum = 0.1

        if opt.bn2d == True:
            self.batchnorm2d4 = nn.BatchNorm2d(out_channels, momentum=self.momentum)
        else:
            self.batchnorm2d4 = identity()

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

        self.st_gcn_layer1 = st_gcn_layer(in_channels, out_channels, node_num, neigh_node_kinds, opt, perNodeFF2=True, residual=False)
        self.st_gcn_layer2 = st_gcn_layer(out_channels, out_channels, node_num, neigh_node_kinds, opt, perNodeFF2=False, residual=residual)

        if opt.perNodeFF2:
            self.per_node_FF2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), padding=0),
                self.batchnorm2d4,
                nn.Dropout(opt.dropout7), )
        else:
            self.per_node_FF2 = identity()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1)),
                self.batchnorm2d4, )

    def forward(self, x, A):

        residual = self.residual(x)

        x, A = self.st_gcn_layer1(x, A)
        x, A = self.st_gcn_layer2(x, A)

        x = x + residual

        return self.activ_func(x), A


class conv_before_after_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, opt, residual=True):

        super().__init__()

        self.inplace = True
        self.momentum = 0.1
        self.res = residual

        if opt.bn2d == True:
            self.batchnorm2d6 = nn.BatchNorm2d(out_channels, momentum=self.momentum)
        else:
            self.batchnorm2d6 = identity()

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

        if not residual:
            self.conv_bef_aft_gcn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0)),
                self.batchnorm2d6,
                self.activ_func,
                nn.Dropout(opt.dropout8))
        else:
            self.conv_bef_aft_gcn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0)),
                self.batchnorm2d6,
                nn.Dropout(opt.dropout8))

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1)),
                                          self.batchnorm2d6,)

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv_bef_aft_gcn(x) + residual

        if self.res:
            return self.activ_func(x)
        else:
            return x

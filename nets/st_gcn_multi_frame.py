import torch
import torch.nn as nn

from nets.utils.st_gcn_layer import st_gcn_layer
from nets.utils.graph_frames import Graph
from nets.utils.graph_frames_withpool_2 import Graph_pool
from nets.non_local_embedded_gaussian import NONLocalBlock2D


class Model(nn.Module): #

    def __init__(self, opt): #
        super().__init__()  #

        self.momentum = 0.1
        self.in_channels = opt.in_channels      #2
        self.out_channels = opt.out_channels    #3
        self.layout = opt.layout    #'hm36_gt'
        self.cat = True  #True: concatinate features;  False: add features
        self.inplace = True  #
        self.pad = opt.pad   #1 ?
        self.framework = opt.framework
        self.opt = opt

        # original graph
        self.graph = Graph(self.layout, opt=opt, pad=opt.pad)#
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda()
        # pooled graph
        self.graph_pool = Graph_pool(self.layout, opt=opt, pad=opt.pad)
        self.A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cuda()


        if self.framework == 'SimpleBaseLine':
            inter_channels = [128, 128, 256, 512]
            self.fc_out = inter_channels[-1]
            fc_unit = 512
        elif self.framework == 'hrgcn':
            inter_channels = [128, 128, 256, 512, 512] #
            self.fc_out = inter_channels[-1]  #
            fc_unit = 512;
            channel_nums = 384 #
        elif self.framework == 'resgcn':
            inter_channels = [128, 256, 256, 512, 512, 512]
            self.fc_out = inter_channels[2]
            fc_unit = 512
        elif self.framework =='ugcn':
            inter_channels = [128, 128, 256, 512]
            self.fc_out = inter_channels[-1]
            fc_unit = 256

        kernel_size = self.A.size(0)             #6 tensor.shape[0]与tensor.size(0)一样
        kernel_size_pool = self.A_pool.size(0)   #6

        self.input_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, self.momentum)  #2x17=34, 0.1

        self.st_gcn_ori_graph = nn.ModuleList((
            st_gcn(self.in_channels,  inter_channels[0], kernel_size, opt, self.A.size(-1), residual=False),#2,128,6
            st_gcn(inter_channels[0], inter_channels[1], kernel_size, opt, self.A.size(-1)),#128,128,6
            st_gcn(inter_channels[1], inter_channels[2], kernel_size, opt, self.A.size(-1)),#128,256,6
            st_gcn(inter_channels[2], inter_channels[2], kernel_size, opt, self.A.size(-1)),  ))

        self.st_gcn_pool_graph = nn.ModuleList((
            st_gcn(inter_channels[-2], fc_unit, kernel_size_pool, opt, self.A_pool.size(-1)),   #256,512,6
            st_gcn(fc_unit,            fc_unit, kernel_size_pool, opt, self.A_pool.size(-1)),  )) #512,512,6

        self.conv4 = nn.Sequential(
            nn.Conv2d(fc_unit, fc_unit, kernel_size=(3, 1), padding=(1, 0)),  #512,512
            nn.BatchNorm2d(fc_unit, momentum=self.momentum),#512
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.25))

        self.conv2 = nn.Sequential(
            nn.Conv2d(fc_unit*2, self.fc_out, kernel_size=(1, 1), padding=(0, 0)), #1024,256
            nn.BatchNorm2d(self.fc_out, momentum=self.momentum), #256
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.1))

        self.non_local = NONLocalBlock2D(in_channels=self.fc_out*2, sub_sample=False)  #256x2=512

        # fcn for final layer prediction
        if self.framework == 'SimpleBaseLine':
            fc_in = inter_channels[-1]+self.fc_out if self.cat else inter_channels[-1]     #256+256=512
        elif self.framework == 'hrgcn':
            fc_in = self.fc_out*3
        elif self.framework == 'resgcn':
            fc_in = fc_unit
        elif self.framework == 'ugcn':
            fc_in = 128

        self.fcn = nn.Sequential(
            nn.Dropout(0.1, inplace=True),
            nn.Conv2d(fc_in, self.out_channels, kernel_size=1))  #512,3

        self.fcn11 = nn.Sequential(
            nn.Conv2d(fc_in, self.out_channels, kernel_size=1))  #512,3

        self.fcn12 = nn.Sequential(
            nn.Conv2d(inter_channels[0], self.out_channels, kernel_size=1))  #512,3

        if self.framework == 'hrgcn':
            self.hr_gcn_ori_graph = nn.ModuleList((
                st_gcn(self.in_channels,  inter_channels[0], kernel_size, opt, self.A.size(-1), residual=False),#2,128,6
                st_gcn(inter_channels[0], inter_channels[0], kernel_size, opt, self.A.size(-1)),#128,128,6
                st_gcn(inter_channels[0], inter_channels[0], kernel_size, opt, self.A.size(-1)),#128,256,6
                st_gcn(inter_channels[0], inter_channels[0], kernel_size, opt, self.A.size(-1)),
                st_gcn(inter_channels[0], inter_channels[0], kernel_size, opt, self.A.size(-1)),
                st_gcn(inter_channels[0], inter_channels[0], kernel_size, opt, self.A.size(-1), residual=False),))

            self.hr_gcn_pool_graph = nn.ModuleList((
                st_gcn(inter_channels[2], inter_channels[2], kernel_size_pool, opt, self.A_pool.size(-1)),
                st_gcn(inter_channels[2], inter_channels[2], kernel_size_pool, opt, self.A_pool.size(-1)),
                st_gcn(inter_channels[2], inter_channels[2], kernel_size_pool, opt, self.A_pool.size(-1)),
                st_gcn(inter_channels[2], inter_channels[2], kernel_size_pool, opt, self.A_pool.size(-1)),))

            self.conv5 = nn.Sequential(
                nn.Conv2d(inter_channels[3], inter_channels[3], kernel_size=(3, 1), padding=(1, 0)),  #512,512
                nn.ELU(alpha=1.0, inplace=self.inplace),
                nn.Dropout(0.25))

            self.conv6 = nn.Sequential(
                nn.Conv2d(fc_unit*2, self.fc_out, kernel_size=(1, 1), padding=(0, 0)), #1024,256
            #    nn.BatchNorm2d(self.fc_out, momentum=self.momentum), #256
                nn.ELU(alpha=1.0, inplace=self.inplace),
                nn.Dropout(0.1))

            self.conv_pool_17to1 = nn.Conv2d(inter_channels[0], inter_channels[3], kernel_size=(1,17), stride=(1,1),padding=(0,0))
            self.bn2d_1to2 = nn.BatchNorm2d(inter_channels[2], momentum=self.momentum)
            self.bn2d_2to3 = nn.BatchNorm2d(inter_channels[3], momentum=self.momentum)

            self.conv_up3to1 = nn.Sequential(nn.Conv2d(inter_channels[3], inter_channels[0], kernel_size=(1,1), stride=(1,1), padding=(0,0)),
                                             )
            self.hr_non_local = NONLocalBlock2D(in_channels=inter_channels[3], sub_sample=False)  #256x2=512

            self.conv9 = nn.Sequential(
                nn.Conv2d(inter_channels[0], channel_nums, kernel_size=(1, 1), padding = (0, 0)),  #inter_channels[3]
           #     nn.BatchNorm2d(inter_channels[3], momentum=self.momentum),#512
                nn.ELU(alpha=1.0, inplace=self.inplace),
                nn.Dropout(0.15))

            self.conv10 = nn.Sequential(
                nn.Conv2d(channel_nums, channel_nums, kernel_size=(3, 1), padding = (1, 0)),  #512,512
             #   nn.BatchNorm2d(inter_channels[3], momentum=self.momentum),#512
                nn.ELU(alpha=1.0, inplace=self.inplace),
                nn.Dropout(0.15))

            self.conv15_res_31 = nn.Sequential(nn.Conv2d(inter_channels[0], channel_nums, kernel_size=(1, 1), padding = (0, 0)))
            self.conv15_res_11 = nn.Sequential(nn.Conv2d(inter_channels[0], inter_channels[3], kernel_size=(1, 1), padding = (0, 0)), nn.Dropout(0.05))
            self.conv13 = nn.Sequential(
                nn.Conv2d(inter_channels[0], inter_channels[0], kernel_size=(3, 1), padding = (1, 0)),  #512,512
                #     nn.BatchNorm2d(inter_channels[3], momentum=self.momentum),#512
                nn.ELU(alpha=1.0, inplace=self.inplace),
                nn.Dropout(0.2))

            self.conv12 = nn.Sequential(
                nn.Conv2d(channel_nums, channel_nums, kernel_size=(1, 1), padding = (0, 0)),  #512,512
                #   nn.BatchNorm2d(inter_channels[3], momentum=self.momentum),#512
                nn.Dropout(0.15) )

            self.conv14 = nn.Sequential(
                nn.Conv2d(inter_channels[0], inter_channels[0], kernel_size=(3, 1), padding = (1, 0)),
                nn.Dropout(0.2) )

            self.fcn2 = nn.Conv2d(channel_nums, self.out_channels, kernel_size=1)  #512,3
            self.fcn3 = nn.Conv2d(inter_channels[1], self.out_channels, kernel_size=1)  #512,3
        self.conv_pool_13 = nn.Conv2d(inter_channels[0], 2*inter_channels[0], kernel_size=(1,3), stride=(1,1), padding=(0,0))
        self.conv_pool_15 = nn.Conv2d(inter_channels[0], 2*inter_channels[0], kernel_size=(1,5), stride=(1,1), padding=(0,0))
        self.conv_pool_13_stgcn = nn.Conv2d(2*inter_channels[0], 2*inter_channels[0], kernel_size=(1,3), stride=(1,1), padding=(0,0))
        self.conv_pool_15_stgcn = nn.Conv2d(2*inter_channels[0], 2*inter_channels[0], kernel_size=(1,5), stride=(1,1), padding=(0,0))
        self.conv_pool_2to3_15 = nn.Conv2d(inter_channels[2], inter_channels[3], kernel_size=(1,5), stride=(1,1), padding=(0,0))
        self.conv_pool_2to3_15_stgcn = nn.Conv2d(inter_channels[3], inter_channels[3], kernel_size=(1,5), stride=(1,1), padding=(0,0))
        self.relu = nn.ELU(alpha=1.0, inplace=self.inplace)
        self.dropout1 = nn.Dropout(0.1)#0.15
        self.dropout2 = nn.Dropout(0.1)#0.25
        self.dropout3 = nn.Dropout(0.12)#0.1
        self.dropout4 = nn.Dropout(0.15) #0.1
        self.dropout5 = nn.Dropout(0.15) #0.05
        self.dropout6 = nn.Dropout(0.12) #0.25
        self.dropout7 = nn.Dropout(0.05)
        self.dropout8 = nn.Dropout(0.3)
        self.conv_kernel_31 = nn.Sequential(
            nn.Conv2d(inter_channels[3], inter_channels[3], kernel_size=(1, 1), padding = (0, 0)),
            #   nn.BatchNorm2d(inter_channels[3], momentum=self.momentum),#512
            )
        self.conv_kernel_11 = nn.Sequential(
            nn.Conv2d(inter_channels[3], inter_channels[3], kernel_size=(1, 1), padding = (0, 0)),)
        #   nn.BatchNorm2d(inter_channels[3], momentum=self.momentum)
        self.conv_up2to1 = nn.Sequential(nn.Conv2d(inter_channels[2], inter_channels[1], kernel_size=(1,1), stride=(1,1), padding=(0,0)),
        )
        self.conv_up3to2 = nn.Sequential(nn.Conv2d(inter_channels[3], inter_channels[2], kernel_size=(1,1), stride=(1,1), padding=(0,0)),
        )
        if self.framework == 'resgcn':
            self.resgcn_networks = nn.ModuleList((
                st_gcn(self.in_channels,  inter_channels[0], kernel_size, opt, self.A.size(-1), residual=False),#2,128,6
                st_gcn(inter_channels[0], inter_channels[1], kernel_size, opt, self.A.size(-1)),#128,128,6
                st_gcn(inter_channels[1], inter_channels[2], kernel_size, opt, self.A.size(-1)),#128,256,6
                st_gcn(inter_channels[2], inter_channels[3], kernel_size, opt, self.A.size(-1)),
                st_gcn(inter_channels[3], inter_channels[4], kernel_size, opt, self.A.size(-1)),
                st_gcn(inter_channels[4], inter_channels[5], kernel_size, opt, self.A.size(-1)),
                st_gcn(inter_channels[4], inter_channels[5], kernel_size, opt, self.A.size(-1)),
            ))
        if self.framework == 'ugcn':
            self.st_gcn_pool = nn.ModuleList((
                st_gcn(inter_channels[-2], fc_unit, kernel_size_pool, opt, self.A_pool.size(-1)),   #256,512,6
                st_gcn(fc_unit, fc_unit, kernel_size_pool, opt, self.A_pool.size(-1)),  #512,512,6
            ))
            self.ugcn_networks = nn.ModuleList((
                st_gcn(inter_channels[3], inter_channels[2], kernel_size_pool, opt, self.A_pool.size(-1)),
                st_gcn(fc_unit, inter_channels[0], kernel_size, opt, self.A.size(-1)),
            ))

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p,stride=None):
        if max(p) > 1:
            if stride is None:
                x = self.conv_pool_15(x)
                x = self.bnRelu_1to2(x)# B x F x V/p
            else:
                x = nn.MaxPool2d(kernel_size=p,stride=stride)(x)  # B x F x V/p
            return x
        else:
            return x

    def graph_max_pool_1to2(self, x):
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_i = x[:, :, :, self.graph.part[i]]
            if num_node == 3:
                x_i = self.conv_pool_13(x_i)
            else:
                x_i = self.conv_pool_15(x_i)
            x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i # Final to N, C, T, (NUM_SUB_PARTS)
      #  x_sub1 = self.bn2d_1to2(x_sub1)
        x_sub1 = self.dropout1(x_sub1)
        return x_sub1

    def graph_max_pool_1to2_stgcn(self, x):
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_i = x[:, :, :, self.graph.part[i]]
            if num_node == 3:
                x_i = self.conv_pool_13_stgcn(x_i)
            else:
                x_i = self.conv_pool_15_stgcn(x_i)
            x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i # Final to N, C, T, (NUM_SUB_PARTS)
        #  x_sub1 = self.bn2d_1to2(x_sub1)
        x_sub1 = self.dropout1(x_sub1)
        return x_sub1

    def graph_max_pool_1to3(self, x):
       # for i in range(len(self.graph.part)):
         #   num_node = len(self.graph.part[i])
        #    x_i = x[:, :, :, self.graph.part[i]]
        #    if num_node == 3 :
         #       x_i = self.conv_pool_13(x_i)
         #   else:
         #       x_i = self.conv_pool_15(x_i)
          #  x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i
      #  x = self.dropout1(x_sub1)
      #  x = self.conv_pool_2to3_15(x)
      #  x = self.dropout2(x)
        x = self.conv_pool_17to1(x)
        x = self.dropout8(x)
        return x

    def graph_max_pool_2to3(self, x):
        x = self.conv_pool_2to3_15(x)
     #   x = self.bn2d_2to3(x)
        x = self.dropout2(x)
        return x
    def graph_max_pool_2to3_stgcn(self, x):
        x = self.conv_pool_2to3_15_stgcn(x)
        x = self.dropout2(x)
        return x
    def graph_max_pool_1to3_2(self, x):
        x = self.conv_pool_17to1(x)
        x = nn.Dropout(0.5)(x)
        return x

    def graph_upsample_2to1(self, x, N, M, T, J):
        x = self.conv_up2to1(x)
        x = self.dropout7(x)
        x_up = torch.zeros((N * M, x.size(1), T, J)).cuda() #256,256,3,17
        for i in range(len(self.graph.part)):
            num_node = len(self.graph.part[i])
            x_up[:, :, :, self.graph.part[i]] = x[:, :, :, i].unsqueeze(-1).repeat(1, 1, 1, num_node)
        return x_up


    def forward(self, input_2d, out_all_frame=False):

        N, C, T, J, M = input_2d.size() # C:channels=2  T:frames=3   J:joints=17  M=1
        input_2d = input_2d.permute(0, 4, 3, 1, 2).contiguous() # N, M,J,C,T
        input_2d = input_2d.view(N * M, J * C, T) #256,34,3

       # input_2d = self.input_bn(input_2d)

        input_2d = input_2d.view(N, M, J, C, T).permute(0, 1, 3, 4, 2).contiguous()  #N,M,C,T,J
        x = input_2d.view(N * M, C, 1, -1)   # N,C,1,T*J

        if self.framework == 'SimpleBaseLine':
            gcn_list = list(self.st_gcn_ori_graph)
            for i_gcn, gcn in enumerate(gcn_list):#0-2
                x, _, _ = gcn(x, self.A) # (N * M), C, 1, (T*J)
                x = self.relu(x)
            x, _, _ = gcn_list[3](x, self.A) # (N * M), C, 1, (T*J)
            x = self.relu(x)
            x, _, _ = gcn_list[3](x, self.A) # (N * M), C, 1, (T*J)
            x = self.relu(x)
            x = x.view(N, -1, T, J) # N, C, T ,J
            x_sub1 = self.graph_max_pool_1to2_stgcn(x)

            x_sub1, _, _ = self.st_gcn_pool_graph[0](x_sub1.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
            x_sub1 = self.relu(x_sub1)
            x_sub1, _, _ = self.st_gcn_pool_graph[1](x_sub1, self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
            x_sub1 = self.relu(x_sub1)
            x_sub1, _, _ = self.st_gcn_pool_graph[1](x_sub1, self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
            x_sub1 = self.relu(x_sub1)
            x_sub1, _, _ = self.st_gcn_pool_graph[1](x_sub1, self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
            x_sub1 = self.relu(x_sub1)
            x_sub1 = x_sub1.view(N, -1, T, len(self.graph.part))   #3,5
            x_pool_1 = self.graph_max_pool_2to3_stgcn(x_sub1)   # N, 512, T, 1
            x_pool_1 = self.dropout6( self.relu( self.conv_kernel_31(x_pool_1) ))#256,c,3,1
            x_pool_1 = self.dropout3( self.conv_kernel_11(x_pool_1) )#  relu???
            x24_up_3to2 = (self.dropout4(self.relu(self.conv_up3to2(x_pool_1)))).repeat(1, 1, 1, len(self.graph.part))#?
            x_up = self.graph_upsample_2to1(x24_up_3to2.view(N, -1, T, len(self.graph.part)), N, M, T, J)
            x_up = self.relu(x_up)


            if self.opt.stgcn_nonlocal:
                x = self.non_local(x)  # N, 512, T, J-->512,3,17
            x = self.fcn12(x_up) # N, 3, T, J

            # output
            x = x.view(N, M, -1, T, J).permute(0, 2, 3, 4, 1).contiguous() # N, C, T, J, M
            if out_all_frame:
                x_out = x
            else:
                x_out= x[:, :, self.pad].unsqueeze(2)
            return x_out

        elif self.framework == 'hrgcn':   #tcnlayer=true?
            gcn_list = list(self.hr_gcn_ori_graph)
            gcn_list_pool = list(self.hr_gcn_pool_graph)
            x11, x11_gcn, _ = gcn_list[0](x, self.A)
            x11 = self.relu(x11)

            x21 = self.graph_max_pool_1to2(x11_gcn.view(N, -1, T, J))   #3,5
            x21 = self.relu(x21)
            x12, x12_gcn, _ = gcn_list[1](x11, self.A) #1,51
            x22, x22_gcn, _ = gcn_list_pool[0](x21.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone()) #1,15
            x22_pool = self.graph_max_pool_1to2(x12_gcn.view(N, -1, T, J)) #3,17-->3,5
            x12_up = self.graph_upsample_2to1(x22_gcn.view(N,-1,T,len(self.graph.part)), N, M, T, J)
            x12 = self.relu(x12 + x12_up.view(N,-1,1,T*J)) #1,51
            x22 = self.relu(x22 + x22_pool.view(N,-1,1,T*len(self.graph.part)) )#1,15


            x13, x13_gcn, _ = gcn_list[2](x12, self.A)
            x23, x23_gcn, _ = gcn_list_pool[1](x22, self.A_pool.clone()) #1,15
            x23_pool = self.graph_max_pool_1to2(x13_gcn.view(N, -1, T, J))  #3,5
            x13_up = self.graph_upsample_2to1(x23_gcn.view(N,-1,T,len(self.graph.part)), N, M, T, J)
            x13 = self.relu(x13 + x13_up.view(N,-1,1,T*J))
            x23 = self.relu(x23 + x23_pool.view(N,-1,1,T*len(self.graph.part)))
            x33_pool_2to3 = self.graph_max_pool_2to3(x23_gcn.view(N, -1, T, len(self.graph.part)))  # 3,5-->3,1
            x33_pool_1to3 = self.graph_max_pool_1to3(x13_gcn.view(N, -1, T, J))   # N, 512, T,
            x33 = self.relu(x33_pool_2to3 + x33_pool_1to3)  #3,1

            x14, x14_gcn, _ = gcn_list[3](x13, self.A)
            x24, x24_gcn, _ = gcn_list_pool[2](x23, self.A_pool.clone()) #1,15
            x34_gcn = self.dropout6( self.relu( self.conv_kernel_31(x33) ))#256,c,3,1
            x34 = self.dropout3( self.conv_kernel_11(x34_gcn) )#  relu
            x24_pool = self.graph_max_pool_1to2(x14_gcn.view(N, -1, T, J)) #
            x14_up_2to1 = self.graph_upsample_2to1(x24_gcn.view(N,-1,T,len(self.graph.part)), N, M, T, J)  #3,17
            x14_up_3to1 = (self.dropout5(self.conv_up3to1(x34_gcn))).repeat(1,1,1,J)
            x24_up_3to2 = (self.dropout4(self.conv_up3to2(x34_gcn))).repeat(1, 1, 1, len(self.graph.part))#
            x34_pool_1to3 = self.graph_max_pool_1to3(x14_gcn.view(N,-1,T,J))  #
            x34_pool_2to3 = self.graph_max_pool_2to3(x24_gcn.view(N, -1, T, len(self.graph.part)))
            x14 = self.relu(x14.view(N,-1,T,J) + x14_up_2to1 + x14_up_3to1)#
            x24 = self.relu(x24.view(N,-1,T,len(self.graph.part)) + x24_pool + x24_up_3to2)
            x34 = self.relu(x33 + x34 + x34_pool_1to3 + x34_pool_2to3)#

            x15, x15_gcn, _ = gcn_list[4](x14.view(N,-1,1,T*J), self.A)#
            x25, x25_gcn, _ = gcn_list_pool[3](x24.view(N,-1,1,T*len(self.graph.part)), self.A_pool.clone())
            x35_gcn = self.dropout6(self.relu(self.conv_kernel_31(x34)))
            x15_up_2to1 = self.graph_upsample_2to1(x25_gcn.view(N,-1,T,len(self.graph.part)), N, M, T, J)
            x15_up_3to1 = (self.dropout5(self.conv_up3to1(x35_gcn))).repeat(1,1,1,J)
            x15 = self.relu(x15.view(N,-1,T,J) + x15_up_2to1 + x15_up_3to1)#

            if self.opt.x15_to_x16 == 'gcn_conv':
                x16, x16_gcn, _ = gcn_list[1](x15.view(N,-1,1,T*J), self.A)
                x14fuse = self.fcn3(x16)
            elif self.opt.x15_to_x16 == 'gcn_conv_noRes':
                x16, x16_gcn, _ = gcn_list[5](x15.view(N,-1,1,T*J), self.A)
                x14fuse = self.fcn3(x16)
            elif self.opt.x15_to_x16 == 'conv128-512-512-conv12Res':
                x15 = self.conv9(x15)
                x16 = self.conv12(x15)
                x16 = x15+x16
                x15 = self.relu(x16)
                x14fuse = self.fcn2(x15)
            elif self.opt.x15_to_x16 == 'conv128-512-512-conv9Res31':
                x16 = self.conv9(x15)
                x15_res = self.conv15_res_31(x15)
                x16 = self.conv12(x16)
            #    x16 = self.conv12(x16)
                x16 = x15_res+x16
                x15 = self.relu(x16)
                x14fuse = self.fcn2(x15)
            elif self.opt.x15_to_x16 == 'conv128-512-512-conv9Res11':
                x16 = self.conv9(x15)
                x15_res = self.conv15_res_11(x15)
                x16 = self.conv12(x16)
                x16 = x15_res+x16
                x15 = self.relu(x16)
                x14fuse = self.fcn2(x15)
            elif self.opt.x15_to_x16 == 'conv128-512-512_noRes':
                x15 = self.conv9(x15)
                x15 = self.conv10(x15)#x15 = x15
                if self.opt.hrgcn_non_local:
                    x = self.hr_non_local(x15)  # N, 3C, T, V     ?x14
                    x = self.dropout4(x)
                    x14fuse = self.fcn2(x) # N, 3, T, V
                else:
                    x14fuse = self.fcn2(x15)  #x14
            elif self.opt.x15_to_x16 == 'conv128-128-128':
                x16 = self.conv13(x15)
                x16 = self.conv13(x16)
                x14fuse = self.fcn3(x16)
            elif self.opt.x15_to_x16 == 'conv128-128-128_conv13Res':
                x16 = self.conv13(x15)
                x16 = self.conv14(x16)
                x16 = self.relu(x15+x16)
                x14fuse = self.fcn3(x16)
            elif self.opt.x15_to_x16 == 'conv128-128-128_conv14Res':
                x15 = self.conv13(x15)
                x16 = self.conv14(x15)
                x16 = self.relu(x15+x16)
                x14fuse = self.fcn3(x16)
            # output
            x = x14fuse.view(N, M, -1, T, J).permute(0, 2, 3, 4, 1).contiguous() # N, C, T, V, M
            if out_all_frame:
                x_out = x
            else:
                x_out = x
            return x_out

        elif self.framework == 'resgcn':   #tcnlayer=true?
            gcn_list = list(self.resgcn_networks)
            for i_gcn, gcn in enumerate(gcn_list):#0-2
                x, _, _ = gcn(x, self.A) # (N * M), C, 1, (T*J)
            x = x.view(N, -1, T, J)
            if self.opt.stgcn_nonlocal:
                x = self.non_local(x)  # N, 512, T, J-->512,3,17
            x = self.fcn11(x) # N, 3, T, J
            x = x.view(N, M, -1, T, J).permute(0, 2, 3, 4, 1).contiguous() # N, C, T, J, M
            if out_all_frame:
                x_out = x
            else:
                x_out= x[:, :, self.pad].unsqueeze(2)
            return x_out

        elif self.framework =='ugcn':
            gcn_list = list(self.st_gcn_ori_graph)
            x, _, _ = gcn_list[0](x, self.A) # (N * M), C, 1, (T*J)
            x = self.relu(x)
          #  x, _, _ = gcn_list[2](x, self.A) # (N * M), C, 1, (T*J)
          #  x = self.relu(x)
            x = x.view(N, -1, T, J) # N, C, T ,J

            # Pooling
            x_sub1 = self.graph_max_pool_1to2(x)
            x_sub1, _, _ = self.st_gcn_pool_graph[0](x_sub1.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
           # x_sub1, _, _ = self.st_gcn_pool[1](x_sub1, self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
            x_sub1 = self.relu(x_sub1)
            x_sub1 = x_sub1.view(N, -1, T, len(self.graph.part))   #3,5

            x_pool_1 = self.graph_max_pool_2to3(x_sub1)   # N, 512, T, 1
            x_pool_1 = self.dropout6( self.relu( self.conv_kernel_31(x_pool_1) ))#256,c,3,1
            x_pool_1 = self.dropout3( self.conv_kernel_11(x_pool_1) )#  relu
            x24_up_3to2 = (self.dropout4(self.conv_up3to2(x_pool_1))).repeat(1, 1, 1, len(self.graph.part))#?
            x_up_sub = torch.cat((x24_up_3to2, x_sub1), 1)  # N, 512, T, 5
            ugcn_list = list(self.ugcn_networks)
            x_up_sub, _, _ = ugcn_list[0](x_up_sub.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone()) #N, C, T, 5    1024->256
            x_up_sub = self.relu(x_up_sub)
            x_up = self.graph_upsample_2to1(x_up_sub.view(N, -1, T, len(self.graph.part)), N, M, T, J)
            x = torch.cat((x, x_up), 1)
            x, _, _ = ugcn_list[1](x.view(N, -1, 1, T*J), self.A)
            x = self.relu(x)
            if self.opt.stgcn_nonlocal:
                x = self.non_local(x.view(N, -1, T, J))  # N, 512, T, J-->512,3,17
            x = self.fcn11(x.view(N,-1,T,J)) # N, 3, T, J

                # output
            x = x.view(N, M, -1, T, J).permute(0, 2, 3, 4, 1).contiguous() # N, C, T, J, M
            if out_all_frame:
                x_out = x
            else:
                x_out = x[:, :, self.pad].unsqueeze(2)
            return x_out


class st_gcn(nn.Module):
    """
    `N` is a batch size,
    `K` is the kernel size
    `T` is a length of sequence,
    `V` is the number of graph nodes of each frame.
    """
    def __init__(self, in_channels, out_channels, kernel_size, opt, num_nodes, stride=1, dropout=0.05, residual=True):

        super().__init__() #?
        self.inplace = True
        self.opt = opt
        self.momentum = 0.1
        self.gcn = st_gcn_layer(in_channels, out_channels, kernel_size, opt) #2,128,6  ;128,128,6;128,256,6

        if opt.tcnlayer:
            self.tcn = nn.Sequential(
          #      nn.BatchNorm2d(out_channels, momentum=self.momentum),    #128,0.1;128;256
          #      nn.ReLU(inplace=self.inplace),  #True
          #      nn.Dropout(0.05),
                nn.Conv2d( out_channels,    #128;128;256
                           out_channels,    #128;128;256
                           (1, 1),
                           (stride, 1),#(1,1)
                            padding = 0,),
           #     nn.BatchNorm2d(out_channels, momentum=self.momentum),    #128,0.1;128;256
                nn.Dropout(dropout, inplace=self.inplace),    #0.05
            )
            self.tcn2 = nn.Conv2d(out_channels, out_channels,(1,1),(stride,1),padding=0 )
        else:
            self.tcn = lambda x: x
        if opt.tcnlayer == True:
            self.bn_relu_drop = nn.Sequential(
       #      nn.BatchNorm2d(out_channels, momentum=self.momentum),    #128,0.1;128;256
             nn.ELU(alpha=1.0, inplace=self.inplace),  #True
             nn.Dropout(0.03), )
        else:
            self.bn_relu_drop = nn.Sequential(
         #       nn.BatchNorm2d(out_channels, momentum=self.momentum),    #128,0.1;128;256
                nn.Dropout(0.05),)
        if not residual:   #false ;true;true
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1): #;;128,256
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
         #       nn.BatchNorm2d(out_channels, momentum=self.momentum),
            )

        self.relu = nn.ELU(alpha=1.0, inplace=self.inplace)


        if opt.channel_sharedweights:
            self.learnable_correltion = nn.Parameter(torch.randn(num_nodes, num_nodes))
            self.learnable_correltion2 = nn.Parameter(torch.randn(num_nodes, num_nodes))
        else:
            self.learnable_correltion = nn.Parameter(torch.randn(out_channels, num_nodes, num_nodes))  #128,128,256
            self.learnable_correltion2 = nn.Parameter(torch.randn(out_channels, num_nodes, num_nodes))
        self.convlayer = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), dilation=(1, 1), bias=True) #128,128;128,256;
    def forward(self, x, A):

        res = self.residual(x)

        if self.opt.dynamic_correlation_weights:
            x = self.convlayer(x)  #256,128,1,51
            if self.opt.learn_adjacency:
                a = self.learnable_correltion
                a = (nn.Dropout(0.88)(a))*(1-0.88)
            else:
                aa = torch.sum(A, dim=0)  #6,51,51-->51,51
                a = self.learnable_correltion.mul(aa)  #128,51,51.mul -->128,51,51
            if self.opt.channel_sharedweights:
                x = torch.einsum('nctv,vw->nctw', (x, a))
            else:
                x = torch.einsum('nctv,cvw->nctw', (x, a))
            x_gcn = self.bn_relu_drop(x)
        else:
            x_gcn, A = self.gcn(x, A)
        tcn_PA = False
        if not tcn_PA:
            x = self.tcn(x_gcn) + res
        else:
            x = self.tcn2(x_gcn)
            a2 = self.learnable_correltion2.mul(aa)
            if self.opt.channel_sharedweights:
                x = torch.einsum('nctj,jw->nctw', (x, a2))
            else:
                x = torch.einsum('nctj,cjw->nctw', (x, a2))
            x = nn.Dropout(0.08)(x) + res

        return x, x_gcn, A

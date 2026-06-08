import numpy as np
import torch
import pandas as pd

class correlation_statistic():
    def __init__(self, train_data, test_data, opt):
        if opt.pose_2d_corre_stat:
            for i, video_3d_gt in train_data.poses_train_2d.items():
                all_3dgt_relative = np.concatenate((all_3dgt_relative, video_3d_gt), axis=0) if not i in [('S1', 'Phoning 1',0)] else video_3d_gt
            for i, video_3d_gt in test_data.poses_test_2d.items():
                all_3dgt_relative = np.concatenate((all_3dgt_relative, video_3d_gt), axis=0)
        else:
            for i, video_3d_gt in train_data.poses_train_3d.items():
                all_3dgt_relative = np.concatenate((all_3dgt_relative, video_3d_gt), axis=0) if not i in [('S1', 'Phoning 1',0)] else video_3d_gt
            b=1
        #    for i, video_3d_gt in test_data.poses_test_3d.items():
             #   all_3dgt_relative = np.concatenate((all_3dgt_relative, video_3d_gt), axis=0)

        all_3dgt_relative = torch.from_numpy(all_3dgt_relative).cuda()#nd(2103096,17,3)
        if opt.relative3D_0_joint:
            all_3dgt_relative[:,0:1,:] = 0 #
        N = all_3dgt_relative.size(0)

        eps=0.000001
        min_X_17, min_X_index_17 = torch.min(all_3dgt_relative[..., 0], dim=0) #17,   17
        min_X, min_X_Jindex = torch.min(min_X_17, dim=0)
        max_X_17, max_X_index_17 = torch.max(all_3dgt_relative[..., 0], dim=0)
        max_X, max_X_Jindex = torch.max(max_X_17, dim=0)

        min_Y_17, min_Y_index_17 = torch.min(all_3dgt_relative[..., 1], dim=0)
        min_Y, min_Y_Jindex = torch.min(min_Y_17, dim=0)
        max_Y_17, max_Y_index_17 = torch.max(all_3dgt_relative[..., 1], dim=0)
        max_Y, max_Y_Jindex = torch.max(max_Y_17, dim=0)

        if not opt.pose_2d_corre_stat:
            min_Z_17, min_Z_index_17 = torch.min(all_3dgt_relative[..., 2], dim=0)
            min_Z, min_Z_Jindex = torch.min(min_Z_17, dim=0)
            max_Z_17, max_Z_index_17 = torch.max(all_3dgt_relative[..., 2], dim=0)
            max_Z, max_Z_Jindex = torch.max(max_Z_17, dim=0)
        if not opt.pose_2d_corre_stat:
            min_X, min_Y, min_Z, max_X, max_Y, max_Z = min_X-eps, min_Y-eps, min_Z-eps, max_X+eps, max_Y+eps, max_Z+eps
        else:
            min_X, min_Y, max_X, max_Y = min_X-eps, min_Y-eps, max_X+eps, max_Y+eps

        if not opt.pose_2d_corre_stat:
            min_XYZ = torch.tensor([min_X,min_Y,min_Z]).cuda()
            discrete_degree = 4 #1/2/
            X_interval, Y_interval, Z_interval = (max_X -min_X)/(opt.n_joints*discrete_degree), (max_Y - min_Y)/(opt.n_joints*discrete_degree), (max_Z -min_Z)/(opt.n_joints*discrete_degree)
            interval_XYZ = torch.tensor([X_interval, Y_interval, Z_interval]).cuda()
            discrete_xyz = (all_3dgt_relative-min_XYZ) // interval_XYZ #0-16    N,J,3
            discrete_xyz_flatten = (discrete_xyz[...,0]+1)+(discrete_xyz[...,1]+1)*(opt.n_joints*discrete_degree)+(discrete_xyz[...,2]+1)*(opt.n_joints*discrete_degree)*(opt.n_joints*discrete_degree)
        else:
            min_XYZ = torch.tensor([min_X,min_Y]).cuda()
            discrete_degree = 1 #1/2/
            X_interval, Y_interval = (max_X -min_X)/(opt.n_joints*discrete_degree), (max_Y - min_Y)/(opt.n_joints*discrete_degree)
            interval_XYZ = torch.tensor([X_interval, Y_interval]).cuda()
            discrete_xyz = (all_3dgt_relative-min_XYZ) // interval_XYZ #0-16    N,J,3
            discrete_xyz_flatten = (discrete_xyz[...,0]+1)+(discrete_xyz[...,1]+1)*(opt.n_joints*discrete_degree)

    # ij_dis_corre_coord = torch.zeros((opt.n_joints,opt.n_joints,N,2)).cuda() #0?

        MI=torch.zeros(opt.n_joints,opt.n_joints).cuda()

        for i in range (opt.n_joints):
            for j in range (opt.n_joints): #???????????????????????????????????????????i+1
              #  ij_flatten_concat= torch.cat( [discrete_xyz_flatten[:,i:i+1], discrete_xyz_flatten[:,j:j+1]],dim=-1) #N,2  ij_dis_corre_coord[i,j,:,:]
                ij_elements_number = torch.unique(torch.cat( [discrete_xyz_flatten[:,i:i+1], discrete_xyz_flatten[:,j:j+1]],dim=-1), sorted=False, return_counts=True, dim=0) #tensor[m,2]  tensor[m]   m-different elements ,number of each element
                p_ij = ij_elements_number[1]/N

         #       i_elements_index_num = torch.unique(ij_elements_number[0][:,0], sorted=False, return_inverse=True, return_counts=True, dim=0) #(26,1); (N) ;(26)
          #      j_elements_index_num = torch.unique(ij_elements_number[0][:,1], sorted=False, return_inverse=True, return_counts=True, dim=0)#498   1840  498  bu chong fu de yuansu;  1840 fen bie de lei bie shu([0,497])
                p_i=torch.zeros(p_ij.shape).cuda()
                p_j=torch.zeros(p_ij.shape).cuda()
                for m in range(ij_elements_number[0].size(0)):
                    p_i[m] = torch.sum(p_ij[torch.where(ij_elements_number[0][:,0]==ij_elements_number[0][:,0][m])[0]])
                    p_j[m] = torch.sum(p_ij[torch.where(ij_elements_number[0][:,1]==ij_elements_number[0][:,1][m])[0]])
                MI[i, j] = torch.sum(p_ij * torch.log(p_ij/(p_i*p_j)))

        pd.DataFrame( MI.cpu().numpy()).to_csv('MI.csv')
        a=1
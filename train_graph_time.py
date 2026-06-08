from tqdm import tqdm
from utils.utils1 import *
import data.common.eval_cal as eval_cal
import torch
import torch.nn as nn
import math



def step(split, opt, actions, dataLoader, model, optimizer=None):

    num_data_all = 0
    various_losses_sum_avg = {'loss_mpjpe':AccumLoss(), 'loss_diff':AccumLoss(), 'loss_bl':AccumLoss(), 'loss_ba':AccumLoss(), 'loss_ba_n1':AccumLoss(),
                              'loss_ba_n2': AccumLoss(), 'loss_ba_n3': AccumLoss(), 'loss_ba_n4': AccumLoss(), 'loss_ba_n5': AccumLoss(),'loss_ba_n6': AccumLoss(),
                              'loss_ba_n7': AccumLoss(), 'loss_sum': AccumLoss()}
    mpjpe_J0_0_sum_avg = AccumLoss()
    mpjpe_J0_0_one_epoch = {'xyz': 0.0, 'pose_refine_output': 0.0}

    if opt.dataset == 'h36m':
        limb_center = [2, 5, 11, 14] if opt.keypoints.startswith('sh') else [2, 5, 12, 15]
        limb_terminal = [3, 6, 12, 15] if opt.keypoints.startswith('sh') else [3,6,13,16]
        joints_left = [4, 5, 6, 10, 11, 12]  if opt.keypoints.startswith('sh') else [4,5,6,11,12,13]
        joints_right = [1, 2, 3, 13, 14, 15] if opt.keypoints.startswith('sh') else [1, 2, 3, 14, 15, 16]

        action_test_error_sum_avg = define_error_list(actions)
        action_test_error_sum_avg_post_out = define_error_list(actions)

    model_st_gcn = model['st_gcn']
    model_pose_refine = model['pose_refine']

    if split == 'train':
        model_st_gcn.train()
        if opt.out_all:  #when train,opt.out_all decides the output frames, when test,outout 1 frame    in def input_augmentation,out_all_frame = False
            out_all_frame = True
        else:
            out_all_frame = False
    else:
        model_st_gcn.eval()
        out_all_frame = False

    torch.cuda.synchronize()


    for i, data in enumerate(tqdm(dataLoader, 0)):

        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind, index, start_3d, flip = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split,[input_2D, gt_3D, batch_cam, scale, bb_box]) #cpu转成GPU

        N = input_2D.size(0)   #256
        num_data_all += N    #0+256

        out_target = gt_3D.clone().view(N, -1, opt.n_joints, opt.out_channels) #[256,3,17,3].view(256,-1,17,3)=256,3,17,3
        if opt.target_0joint_0:
            out_target[:, :, 0] = 0   #

        gt_3D = gt_3D.view(N, -1, opt.n_joints, opt.out_channels)

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D


        if opt.test_flip_augment and split =='test':
            input_2D, output_3D = input_augmentation(input_2D, model_st_gcn, joints_left, joints_right)
        else:
            input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4) #[256,3,17,2].view()=[256，3，17，2，1].permute=256，2，3，17，1(3到第二位，原来的倒数第二维变成现在的第二维)
            output_3D = model_st_gcn(input_2D, out_all_frame)

        _, C, T, J, M = output_3D.size()

        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, J, C) #permute->N,T,J,M,C.view->N,T,J,C
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, T, J, C) #N,T,J,C

        if output_3D.size(1) > 1:
            output_3D_single = output_3D[:, opt.pad].unsqueeze(1) # N,1,J,C
        else:
            output_3D_single = output_3D

        pred_bv = torch.zeros([N, T, J-1, C]).cuda()  # predicted_bone_vector
        targ_bv = torch.zeros([N, T, J-1, C]).cuda()  # target_bone_vector

        father_nodes = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        for i in range(0, J-1): #0-15
            pred_bv[:, :, i:i+1, :] =  output_3D[:, :, i+1:i+2, :] -  output_3D[:, :, father_nodes[i]:father_nodes[i]+1, :]
            targ_bv[:, :, i:i+1, :] = out_target[:, :, i+1:i+2, :] - out_target[:, :, father_nodes[i]:father_nodes[i]+1, :]

        if targ_bv.size(1) > 1:
            targ_bv_single = targ_bv[:, opt.pad].unsqueeze(1)
        else:
            targ_bv_single = targ_bv

        if pred_bv.size(1) > 1:
            pred_bv_single = pred_bv[:, opt.pad].unsqueeze(1)
        else:
            pred_bv_single = pred_bv

        if split == 'train':
            pred_out = output_3D
            pred_bv = pred_bv
            out_target = out_target
            targ_bv = targ_bv
        elif split == 'test':
            pred_out = output_3D_single
            pred_bv = pred_bv_single
            out_target = out_target_single
            targ_bv = targ_bv_single


        if opt.uvd_to_xyz and not opt.input_unproj_3dVector:
            input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, J, 2)  #[N,C,T,J,1].permute[N,T,J,C,1].view-->[N,T,J,C]
            if opt.crop_uv:
                pred_uv = back_to_ori_uv(input_2D, bb_box)
            else:
                pred_uv = input_2D #[256,3,17,2]

            uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)  #pred-uv+pred-Z+(fx,fy,cx,cy)--->pred xyz  N 1 J C

        if opt.pose_refine: #true
            pose_refine_output_3D = model_pose_refine(output_3D_single, xyz)

            loss_pose_refine_mpjpe = eval_cal.mpjpe(pose_refine_output_3D, out_target_single)
        else:
            loss_pose_refine_mpjpe = torch.zeros(1).cuda()

        loss_mpjpe = eval_cal.mpjpe(pred_out, out_target)

        if opt.bl_sy_pnl:
            if not opt.pose_refine:
                loss_bl_sy = eval_cal.sym_penalty(opt.dataset, opt.keypoints, pred_out)
                loss_pose_refine_bl_sy =  torch.zeros(1).cuda()
            elif opt.pose_refine:
                loss_bl_sy = eval_cal.sym_penalty(opt.dataset, opt.keypoints, pred_out)
                loss_pose_refine_bl_sy = eval_cal.sym_penalty(opt.dataset, opt.keypoints, pose_refine_output_3D)
        else:
            loss_bl_sy=torch.zeros(1).cuda()
            loss_pose_refine_bl_sy =  torch.zeros(1).cuda()

        if opt.pad == 0 or split == 'test' or opt.out_all==False or not opt.frames_diff_pnl:
            loss_diff = torch.zeros(1).cuda()
        else:
            weight_diff = 4 * torch.ones(output_3D[:, :-1, :].size()).cuda() # N,T-1,J,C  4
            weight_diff[:, :, limb_center] = 2.5 #2,5,12,15
            weight_diff[:, :, limb_terminal] = 1#3,6,13,16
            diff = (output_3D[:, 1:] - output_3D[:, :-1]) * weight_diff  #N,T-1,J,C
            mse = nn.MSELoss(size_average=True).cuda()
            loss_diff = mse(diff, Variable(torch.zeros(diff.size()), requires_grad=False).cuda())

        b_d = len(pred_bv.shape)-1 #bone vector 3d coordiate dimension

        if opt.bl_pnl:
            if opt.error_rule == 'F1':  #pred_bv: N T J-1 C
                loss_bl = torch.mean( torch.abs(torch.norm(pred_bv, dim=b_d) - torch.norm(targ_bv, dim=b_d))) #torch.mean(N,T,J-1)
            elif opt.error_rule == 'F2':
                loss_bl = torch.mean(((torch.norm(pred_bv, dim=b_d) - torch.norm(targ_bv, dim=b_d))**2))
        else:
            loss_bl = torch.zeros(1).cuda()

        if opt.ba_pnl:
            if opt.ba_def == 'vector_product_cosa': #a[0，Π],
                loss_ba = torch.mean((1-(torch.sum(torch.mul(pred_bv, targ_bv),dim=3) / (torch.norm(pred_bv,dim=b_d) * torch.norm(targ_bv, dim=b_d)))))

            elif opt.ba_def == 'vector_product_a':
                loss_ba = torch.mean(torch.acos((torch.sum(torch.mul(pred_bv, targ_bv),dim=3)/(torch.norm(pred_bv,dim=b_d) * torch.norm(targ_bv, dim=b_d)))))

            elif opt.ba_def == 'cosa_|1+cosb|':
                pred_bv_X_PM = torch.ones(N, T, J-1).cuda()   # bv-X coordiate:Plus or Minus
                pred_bv_X_PM[pred_bv[:, :, :, 0]<0] = -1      #x>=0-->1, x<0 -->-1
                targ_bv_X_PM = torch.ones(N, T, J-1).cuda()
                targ_bv_X_PM[targ_bv[:, :, :, 0]<0] = -1

                pred_ba_cosb = pred_bv_X_PM * (1+(pred_bv[:,:,:,1] / torch.sqrt(pred_bv[:,:,:,0]**2 + pred_bv[:,:,:,1]**2))) #|1+cosb|  N,T,J-1
                targ_ba_cosb = targ_bv_X_PM * (1+(targ_bv[:,:,:,1] / torch.sqrt(targ_bv[:,:,:,0]**2 + targ_bv[:,:,:,1]**2)))

                if opt.error_rule == 'F1':
                    loss_ba_cosa = torch.mean(torch.abs(((pred_bv[:,:,:,2]/torch.norm(pred_bv, dim=b_d)) - (targ_bv[:,:,:,2]/torch.norm(targ_bv, dim=b_d)))))
                    loss_ba_cosb = torch.mean(torch.abs(pred_ba_cosb-targ_ba_cosb))
                elif opt.error_rule == 'F2':
                    loss_ba_cosa = torch.mean(((pred_bv[:,:,:,2]/torch.norm(pred_bv, dim=b_d)) - (targ_bv[:,:,:,2]/torch.norm(targ_bv, dim=b_d)))**2)
                    loss_ba_cosb = torch.mean((pred_ba_cosb-targ_ba_cosb)**2)
                loss_ba = loss_ba_cosa + loss_ba_cosb
        else:
            loss_ba = torch.zeros(1).cuda()

        if opt.ba_n1_pnl:
            bv_n1_pairs = [(0,1),(1,2),(0,3),(3,4),(4,5),(0,6),(3,6),(6,7),(7,8),(8,9),(7,10),(8,10),(10,11),(11,12),(7,13),(8,13),(13,14),(14,15)] #1-step neighborhood bv pairs, 0-15
            n1 = len(bv_n1_pairs)
            pred_bv_n1_cosa = torch.zeros(N,T,n1).cuda()  # n1=18  cosa between the predicted 1-step neighbor bone vector pairs
            targ_bv_n1_cosa = torch.zeros(N,T,n1).cuda()

            pred_bv_n1_cosa_dif = torch.zeros(N,T,n1).cuda()  #difference between the  cosa of predicted 1-step neighbor bone vector pairs
            targ_bv_n1_cosa_dif = torch.zeros(N,T,n1).cuda()

            pred_bv_n1_i_cosb = torch.zeros(N,T,n1).cuda()  #cosb of bone_i,where (bone_i,bone_j) is 1-step neighborhood bone pair
            pred_bv_n1_j_cosb = torch.zeros(N,T,n1).cuda()
            targ_bv_n1_i_cosb = torch.zeros(N,T,n1).cuda()
            targ_bv_n1_j_cosb = torch.zeros(N,T,n1).cuda()

            if opt.ba_def == 'vector_product_cosa':
                for n,(i,j) in enumerate(bv_n1_pairs):
                    pred_bv_n1_cosa[:,:,n] = torch.sum(torch.mul(pred_bv[:,:,i,:], pred_bv[:,:,j,:]),dim=2)/(torch.norm(pred_bv[:,:,i,:], dim=b_d-1) * torch.norm(pred_bv[:,:,j,:], dim=b_d-1))
                    targ_bv_n1_cosa[:,:,n] = torch.sum(torch.mul(targ_bv[:,:,i,:], targ_bv[:,:,j,:]),dim=2)/(torch.norm(targ_bv[:,:,i,:], dim=b_d-1) * torch.norm(targ_bv[:,:,j,:], dim=b_d-1))
                if opt.error_rule == 'F1':
                    loss_ba_n1 = torch.mean(torch.abs(pred_bv_n1_cosa - targ_bv_n1_cosa))
                elif opt.error_rule == 'F2':
                    loss_ba_n1 = torch.mean((pred_bv_n1_cosa - targ_bv_n1_cosa)**2)

            elif opt.ba_def == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(bv_n1_pairs):

                    pred_bv_n1_cosa_dif[:,:,n] = (pred_bv[:,:,i,2]/torch.norm(pred_bv[:,:,i,:],dim=b_d-1))-(pred_bv[:,:,j,2]/torch.norm(pred_bv[:,:,j,:],dim=b_d-1))
                    targ_bv_n1_cosa_dif[:,:,n] = (targ_bv[:,:,i,2]/torch.norm(targ_bv[:,:,i,:],dim=b_d-1))-(targ_bv[:,:,j,2]/torch.norm(targ_bv[:,:,j,:],dim=b_d-1))

                    pred_bv_n1_i_cosb[:,:,n] = pred_bv_X_PM[:,:,i] * (1+(pred_bv[:,:,i,1] / torch.sqrt(pred_bv[:,:,i,0]**2 + pred_bv[:,:,i,1]**2)))
                    pred_bv_n1_j_cosb[:,:,n] = pred_bv_X_PM[:,:,j] * (1+(pred_bv[:,:,j,1] / torch.sqrt(pred_bv[:,:,j,0]**2 + pred_bv[:,:,j,1]**2)))
                    targ_bv_n1_i_cosb[:,:,n] = targ_bv_X_PM[:,:,i] * (1+(targ_bv[:,:,i,1] / torch.sqrt(targ_bv[:,:,i,0]**2 + targ_bv[:,:,i,1]**2)))
                    targ_bv_n1_j_cosb[:,:,n] = targ_bv_X_PM[:,:,j] * (1+(targ_bv[:,:,j,1] / torch.sqrt(targ_bv[:,:,j,0]**2 + targ_bv[:,:,j,1]**2)))

                if opt.error_rule == 'F1':
                    loss_ba_n1_cosa = torch.mean(torch.abs(pred_bv_n1_cosa_dif - targ_bv_n1_cosa_dif))
                    loss_ba_n1_cosb = torch.mean(torch.abs((pred_bv_n1_i_cosb-pred_bv_n1_j_cosb)-(targ_bv_n1_i_cosb-targ_bv_n1_j_cosb)))
                elif opt.error_rule == 'F2':
                    loss_ba_n1_cosa = torch.mean((pred_bv_n1_cosa_dif - targ_bv_n1_cosa_dif)**2)
                    loss_ba_n1_cosb = torch.mean(((pred_bv_n1_i_cosb-pred_bv_n1_j_cosb)-(targ_bv_n1_i_cosb-targ_bv_n1_j_cosb))**2)
                loss_ba_n1 = loss_ba_n1_cosa + loss_ba_n1_cosb
        else:
            loss_ba_n1 = torch.zeros(1).cuda()

        if opt.ba_n2_pnl:
            bv_n2_pairs = [(0,2),(0,4),(0,7),(1,3),(1,6),(3,5),(3,7),(4,6),(6,8),(6,10),(6,13),(7,9),(7,11),(7,14),(8,11),(8,14),(9,10),(9,13),(10,12),(10,14),(13,15),(11,13)]
            n2 = len(bv_n2_pairs)
            pred_bv_n2_cosa = torch.zeros(N,T,n2).cuda()
            targ_bv_n2_cosa = torch.zeros(N,T,n2).cuda()

            pred_bv_n2_cosa_dif = torch.zeros(N,T,n2).cuda()
            targ_bv_n2_cosa_dif = torch.zeros(N,T,n2).cuda()

            pred_bv_n2_i_cosb = torch.zeros(N,T,n2).cuda()  #cosb of bone_i,where (bone_i,bone_j) is 1-step neighborhood bone pair
            pred_bv_n2_j_cosb = torch.zeros(N,T,n2).cuda()
            targ_bv_n2_i_cosb = torch.zeros(N,T,n2).cuda()
            targ_bv_n2_j_cosb = torch.zeros(N,T,n2).cuda()

            if opt.ba_def == 'vector_product_cosa':
                for n,(i,j) in enumerate(bv_n2_pairs):
                    pred_bv_n2_cosa[:,:,n] = torch.sum(torch.mul(pred_bv[:,:,i,:], pred_bv[:,:,j,:]),dim=2)/(torch.norm(pred_bv[:,:,i,:], dim=b_d-1) * torch.norm(pred_bv[:,:,j,:], dim=b_d-1))
                    targ_bv_n2_cosa[:,:,n] = torch.sum(torch.mul(targ_bv[:,:,i,:], targ_bv[:,:,j,:]),dim=2)/(torch.norm(targ_bv[:,:,i,:], dim=b_d-1) * torch.norm(targ_bv[:,:,j,:], dim=b_d-1))
                if opt.error_rule == 'F1':
                    loss_ba_n2 = torch.mean(torch.abs(pred_bv_n2_cosa - targ_bv_n2_cosa))
                elif opt.error_rule == 'F2':
                    loss_ba_n2 = torch.mean((pred_bv_n2_cosa - targ_bv_n2_cosa)**2)

            elif opt.ba_def == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(bv_n2_pairs):

                    pred_bv_n2_cosa_dif[:,:,n] = (pred_bv[:,:,i,2]/torch.norm(pred_bv[:,:,i,:],dim=b_d-1))-(pred_bv[:,:,j,2]/torch.norm(pred_bv[:,:,j,:],dim=b_d-1))
                    targ_bv_n2_cosa_dif[:,:,n] = (targ_bv[:,:,i,2]/torch.norm(targ_bv[:,:,i,:],dim=b_d-1))-(targ_bv[:,:,j,2]/torch.norm(targ_bv[:,:,j,:],dim=b_d-1))

                    pred_bv_n2_i_cosb[:,:,n] = pred_bv_X_PM[:,:,i] * (1+(pred_bv[:,:,i,1] / torch.sqrt(pred_bv[:,:,i,0]**2 + pred_bv[:,:,i,1]**2)))
                    pred_bv_n2_j_cosb[:,:,n] = pred_bv_X_PM[:,:,j] * (1+(pred_bv[:,:,j,1] / torch.sqrt(pred_bv[:,:,j,0]**2 + pred_bv[:,:,j,1]**2)))
                    targ_bv_n2_i_cosb[:,:,n] = targ_bv_X_PM[:,:,i] * (1+(targ_bv[:,:,i,1] / torch.sqrt(targ_bv[:,:,i,0]**2 + targ_bv[:,:,i,1]**2)))
                    targ_bv_n2_j_cosb[:,:,n] = targ_bv_X_PM[:,:,j] * (1+(targ_bv[:,:,j,1] / torch.sqrt(targ_bv[:,:,j,0]**2 + targ_bv[:,:,j,1]**2)))

                if opt.error_rule == 'F1':
                    loss_ba_n2_cosa = torch.mean(torch.abs(pred_bv_n2_cosa_dif - targ_bv_n2_cosa_dif))
                    loss_ba_n2_cosb = torch.mean(torch.abs((pred_bv_n2_i_cosb-pred_bv_n2_j_cosb)-(targ_bv_n2_i_cosb-targ_bv_n2_j_cosb)))
                elif opt.error_rule == 'F2':
                    loss_ba_n2_cosa = torch.mean((pred_bv_n2_cosa_dif - targ_bv_n2_cosa_dif)**2)
                    loss_ba_n2_cosb = torch.mean(((pred_bv_n2_i_cosb-pred_bv_n2_j_cosb)-(targ_bv_n2_i_cosb-targ_bv_n2_j_cosb))**2)
                loss_ba_n2 = loss_ba_n2_cosa + loss_ba_n2_cosb
        else:
            loss_ba_n2 = torch.zeros(1).cuda()

        if opt.ba_n3_pnl:
            bv_n3_pairs = [(0,5),(0,8),(0,10),(0,13),(1,4),(1,7),(2,3),(2,6),(3,8),(3,10),(3,13),(4,7),(5,6),(6,9),(6,11),(6,14),(7,12),(7,15),(8,12),(8,15),(9,11),(9,14),(10,15),(11,14),(12,13)]
            n3 = len(bv_n3_pairs)
            pred_bv_n3_cosa = torch.zeros(N,T,n3).cuda()
            targ_bv_n3_cosa = torch.zeros(N,T,n3).cuda()

            pred_bv_n3_cosa_dif = torch.zeros(N,T,n3).cuda()
            targ_bv_n3_cosa_dif = torch.zeros(N,T,n3).cuda()

            pred_bv_n3_i_cosb = torch.zeros(N,T,n3).cuda()  #cosb of bone_i,where (bone_i,bone_j) is 1-step neighborhood bone pair
            pred_bv_n3_j_cosb = torch.zeros(N,T,n3).cuda()
            targ_bv_n3_i_cosb = torch.zeros(N,T,n3).cuda()
            targ_bv_n3_j_cosb = torch.zeros(N,T,n3).cuda()

            if opt.ba_def == 'vector_product_cosa':
                for n,(i,j) in enumerate(bv_n3_pairs):
                    pred_bv_n3_cosa[:,:,n] = torch.sum(torch.mul(pred_bv[:,:,i,:], pred_bv[:,:,j,:]),dim=2)/(torch.norm(pred_bv[:,:,i,:], dim=b_d-1) * torch.norm(pred_bv[:,:,j,:], dim=b_d-1))
                    targ_bv_n3_cosa[:,:,n] = torch.sum(torch.mul(targ_bv[:,:,i,:], targ_bv[:,:,j,:]),dim=2)/(torch.norm(targ_bv[:,:,i,:], dim=b_d-1) * torch.norm(targ_bv[:,:,j,:], dim=b_d-1))
                if opt.error_rule == 'F1':
                    loss_ba_n3 = torch.mean(torch.abs(pred_bv_n3_cosa - targ_bv_n3_cosa))
                elif opt.error_rule == 'F2':
                    loss_ba_n3 = torch.mean((pred_bv_n3_cosa - targ_bv_n3_cosa)**2)

            elif opt.ba_def == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(bv_n3_pairs):

                    pred_bv_n3_cosa_dif[:,:,n] = (pred_bv[:,:,i,2]/torch.norm(pred_bv[:,:,i,:],dim=b_d-1))-(pred_bv[:,:,j,2]/torch.norm(pred_bv[:,:,j,:],dim=b_d-1))
                    targ_bv_n3_cosa_dif[:,:,n] = (targ_bv[:,:,i,2]/torch.norm(targ_bv[:,:,i,:],dim=b_d-1))-(targ_bv[:,:,j,2]/torch.norm(targ_bv[:,:,j,:],dim=b_d-1))

                    pred_bv_n3_i_cosb[:,:,n] = pred_bv_X_PM[:,:,i] * (1+(pred_bv[:,:,i,1] / torch.sqrt(pred_bv[:,:,i,0]**2 + pred_bv[:,:,i,1]**2)))
                    pred_bv_n3_j_cosb[:,:,n] = pred_bv_X_PM[:,:,j] * (1+(pred_bv[:,:,j,1] / torch.sqrt(pred_bv[:,:,j,0]**2 + pred_bv[:,:,j,1]**2)))
                    targ_bv_n3_i_cosb[:,:,n] = targ_bv_X_PM[:,:,i] * (1+(targ_bv[:,:,i,1] / torch.sqrt(targ_bv[:,:,i,0]**2 + targ_bv[:,:,i,1]**2)))
                    targ_bv_n3_j_cosb[:,:,n] = targ_bv_X_PM[:,:,j] * (1+(targ_bv[:,:,j,1] / torch.sqrt(targ_bv[:,:,j,0]**2 + targ_bv[:,:,j,1]**2)))

                if opt.error_rule == 'F1':
                    loss_ba_n3_cosa = torch.mean(torch.abs(pred_bv_n3_cosa_dif - targ_bv_n3_cosa_dif))
                    loss_ba_n3_cosb = torch.mean(torch.abs((pred_bv_n3_i_cosb-pred_bv_n3_j_cosb)-(targ_bv_n3_i_cosb-targ_bv_n3_j_cosb)))
                elif opt.error_rule == 'F2':
                    loss_ba_n3_cosa = torch.mean((pred_bv_n3_cosa_dif - targ_bv_n3_cosa_dif)**2)
                    loss_ba_n3_cosb = torch.mean(((pred_bv_n3_i_cosb-pred_bv_n3_j_cosb)-(targ_bv_n3_i_cosb-targ_bv_n3_j_cosb))**2)
                loss_ba_n3 = loss_ba_n3_cosa + loss_ba_n3_cosb
        else:
            loss_ba_n3 = torch.zeros(1).cuda()

        if opt.ba_n4_pnl:
            bv_n4_pairs = [(0,9),(0,11),(0,14),(1,5),(1,8),(1,10),(1,13),(2,4),(2,7),(3,9),(3,11),(3,14),(4,8),(4,10),(4,13),(5,7),(6,12),(6,15),(9,12),(9,15),(11,15),(12,14)]
            n4 = len(bv_n4_pairs)
            pred_bv_n4_cosa = torch.zeros(N,T,n4).cuda()
            targ_bv_n4_cosa = torch.zeros(N,T,n4).cuda()

            pred_bv_n4_cosa_dif = torch.zeros(N,T,n4).cuda()
            targ_bv_n4_cosa_dif = torch.zeros(N,T,n4).cuda()

            pred_bv_n4_i_cosb = torch.zeros(N,T,n4).cuda()  #cosb of bone_i,where (bone_i,bone_j) is 1-step neighborhood bone pair
            pred_bv_n4_j_cosb = torch.zeros(N,T,n4).cuda()
            targ_bv_n4_i_cosb = torch.zeros(N,T,n4).cuda()
            targ_bv_n4_j_cosb = torch.zeros(N,T,n4).cuda()

            if opt.ba_def == 'vector_product_cosa':
                for n,(i,j) in enumerate(bv_n4_pairs):
                    pred_bv_n4_cosa[:,:,n] = torch.sum(torch.mul(pred_bv[:,:,i,:], pred_bv[:,:,j,:]),dim=2)/(torch.norm(pred_bv[:,:,i,:], dim=b_d-1) * torch.norm(pred_bv[:,:,j,:], dim=b_d-1))
                    targ_bv_n4_cosa[:,:,n] = torch.sum(torch.mul(targ_bv[:,:,i,:], targ_bv[:,:,j,:]),dim=2)/(torch.norm(targ_bv[:,:,i,:], dim=b_d-1) * torch.norm(targ_bv[:,:,j,:], dim=b_d-1))
                if opt.error_rule == 'F1':
                    loss_ba_n4 = torch.mean(torch.abs(pred_bv_n4_cosa - targ_bv_n4_cosa))
                elif opt.error_rule == 'F2':
                    loss_ba_n4 = torch.mean((pred_bv_n4_cosa - targ_bv_n4_cosa)**2)

            elif opt.ba_def == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(bv_n4_pairs):

                    pred_bv_n4_cosa_dif[:,:,n] = (pred_bv[:,:,i,2]/torch.norm(pred_bv[:,:,i,:],dim=b_d-1))-(pred_bv[:,:,j,2]/torch.norm(pred_bv[:,:,j,:],dim=b_d-1))
                    targ_bv_n4_cosa_dif[:,:,n] = (targ_bv[:,:,i,2]/torch.norm(targ_bv[:,:,i,:],dim=b_d-1))-(targ_bv[:,:,j,2]/torch.norm(targ_bv[:,:,j,:],dim=b_d-1))

                    pred_bv_n4_i_cosb[:,:,n] = pred_bv_X_PM[:,:,i] * (1+(pred_bv[:,:,i,1] / torch.sqrt(pred_bv[:,:,i,0]**2 + pred_bv[:,:,i,1]**2)))
                    pred_bv_n4_j_cosb[:,:,n] = pred_bv_X_PM[:,:,j] * (1+(pred_bv[:,:,j,1] / torch.sqrt(pred_bv[:,:,j,0]**2 + pred_bv[:,:,j,1]**2)))
                    targ_bv_n4_i_cosb[:,:,n] = targ_bv_X_PM[:,:,i] * (1+(targ_bv[:,:,i,1] / torch.sqrt(targ_bv[:,:,i,0]**2 + targ_bv[:,:,i,1]**2)))
                    targ_bv_n4_j_cosb[:,:,n] = targ_bv_X_PM[:,:,j] * (1+(targ_bv[:,:,j,1] / torch.sqrt(targ_bv[:,:,j,0]**2 + targ_bv[:,:,j,1]**2)))

                if opt.error_rule == 'F1':
                    loss_ba_n4_cosa = torch.mean(torch.abs(pred_bv_n4_cosa_dif - targ_bv_n4_cosa_dif))
                    loss_ba_n4_cosb = torch.mean(torch.abs((pred_bv_n4_i_cosb-pred_bv_n4_j_cosb)-(targ_bv_n4_i_cosb-targ_bv_n4_j_cosb)))
                elif opt.error_rule == 'F2':
                    loss_ba_n4_cosa = torch.mean((pred_bv_n4_cosa_dif - targ_bv_n4_cosa_dif)**2)
                    loss_ba_n4_cosb = torch.mean(((pred_bv_n4_i_cosb-pred_bv_n4_j_cosb)-(targ_bv_n4_i_cosb-targ_bv_n4_j_cosb))**2)
                loss_ba_n4 = loss_ba_n4_cosa + loss_ba_n4_cosb
        else:
            loss_ba_n4 = torch.zeros(1).cuda()

        if opt.ba_n5_pnl:
            bv_n5_pairs = [(0,12),(0,15),(1,9),(1,11),(1,14),(2,5),(2,8),(2,10),(2,13),(3,12),(3,15),(4,9),(4,11),(4,14),(5,8),(5,10),(5,13),(12,15)]
            n5 = len(bv_n5_pairs)
            pred_bv_n5_cosa = torch.zeros(N,T,n5).cuda()
            targ_bv_n5_cosa = torch.zeros(N,T,n5).cuda()

            pred_bv_n5_cosa_dif = torch.zeros(N,T,n5).cuda()
            targ_bv_n5_cosa_dif = torch.zeros(N,T,n5).cuda()

            pred_bv_n5_i_cosb = torch.zeros(N,T,n5).cuda()  #cosb of bone_i,where (bone_i,bone_j) is 1-step neighborhood bone pair
            pred_bv_n5_j_cosb = torch.zeros(N,T,n5).cuda()
            targ_bv_n5_i_cosb = torch.zeros(N,T,n5).cuda()
            targ_bv_n5_j_cosb = torch.zeros(N,T,n5).cuda()

            if opt.ba_def == 'vector_product_cosa':
                for n,(i,j) in enumerate(bv_n5_pairs):
                    pred_bv_n5_cosa[:,:,n] = torch.sum(torch.mul(pred_bv[:,:,i,:], pred_bv[:,:,j,:]),dim=2)/(torch.norm(pred_bv[:,:,i,:], dim=b_d-1) * torch.norm(pred_bv[:,:,j,:], dim=b_d-1))
                    targ_bv_n5_cosa[:,:,n] = torch.sum(torch.mul(targ_bv[:,:,i,:], targ_bv[:,:,j,:]),dim=2)/(torch.norm(targ_bv[:,:,i,:], dim=b_d-1) * torch.norm(targ_bv[:,:,j,:], dim=b_d-1))
                if opt.error_rule == 'F1':
                    loss_ba_n5 = torch.mean(torch.abs(pred_bv_n5_cosa - targ_bv_n5_cosa))
                elif opt.error_rule == 'F2':
                    loss_ba_n5 = torch.mean((pred_bv_n5_cosa - targ_bv_n5_cosa)**2)

            elif opt.ba_def == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(bv_n5_pairs):

                    pred_bv_n5_cosa_dif[:,:,n] = (pred_bv[:,:,i,2]/torch.norm(pred_bv[:,:,i,:],dim=b_d-1))-(pred_bv[:,:,j,2]/torch.norm(pred_bv[:,:,j,:],dim=b_d-1))
                    targ_bv_n5_cosa_dif[:,:,n] = (targ_bv[:,:,i,2]/torch.norm(targ_bv[:,:,i,:],dim=b_d-1))-(targ_bv[:,:,j,2]/torch.norm(targ_bv[:,:,j,:],dim=b_d-1))

                    pred_bv_n5_i_cosb[:,:,n] = pred_bv_X_PM[:,:,i] * (1+(pred_bv[:,:,i,1] / torch.sqrt(pred_bv[:,:,i,0]**2 + pred_bv[:,:,i,1]**2)))
                    pred_bv_n5_j_cosb[:,:,n] = pred_bv_X_PM[:,:,j] * (1+(pred_bv[:,:,j,1] / torch.sqrt(pred_bv[:,:,j,0]**2 + pred_bv[:,:,j,1]**2)))
                    targ_bv_n5_i_cosb[:,:,n] = targ_bv_X_PM[:,:,i] * (1+(targ_bv[:,:,i,1] / torch.sqrt(targ_bv[:,:,i,0]**2 + targ_bv[:,:,i,1]**2)))
                    targ_bv_n5_j_cosb[:,:,n] = targ_bv_X_PM[:,:,j] * (1+(targ_bv[:,:,j,1] / torch.sqrt(targ_bv[:,:,j,0]**2 + targ_bv[:,:,j,1]**2)))

                if opt.error_rule == 'F1':
                    loss_ba_n5_cosa = torch.mean(torch.abs(pred_bv_n5_cosa_dif - targ_bv_n5_cosa_dif))
                    loss_ba_n5_cosb = torch.mean(torch.abs((pred_bv_n5_i_cosb-pred_bv_n5_j_cosb)-(targ_bv_n5_i_cosb-targ_bv_n5_j_cosb)))
                elif opt.error_rule == 'F2':
                    loss_ba_n5_cosa = torch.mean((pred_bv_n5_cosa_dif - targ_bv_n5_cosa_dif)**2)
                    loss_ba_n5_cosb = torch.mean(((pred_bv_n5_i_cosb-pred_bv_n5_j_cosb)-(targ_bv_n5_i_cosb-targ_bv_n5_j_cosb))**2)
                loss_ba_n5 = loss_ba_n5_cosa + loss_ba_n5_cosb
        else:
            loss_ba_n5 = torch.zeros(1).cuda()

        if opt.ba_n6_pnl:
            bv_n6_pairs = [(1,12),(1,15),(2,9),(2,11),(2,14),(4,12),(4,15),(5,9),(5,11),(5,14)]
            n6 = len(bv_n6_pairs)
            pred_bv_n6_cosa = torch.zeros(N,T,n6).cuda()
            targ_bv_n6_cosa = torch.zeros(N,T,n6).cuda()

            pred_bv_n6_cosa_dif = torch.zeros(N,T,n6).cuda()
            targ_bv_n6_cosa_dif = torch.zeros(N,T,n6).cuda()

            pred_bv_n6_i_cosb = torch.zeros(N,T,n6).cuda()  #cosb of bone_i,where (bone_i,bone_j) is 1-step neighborhood bone pair
            pred_bv_n6_j_cosb = torch.zeros(N,T,n6).cuda()
            targ_bv_n6_i_cosb = torch.zeros(N,T,n6).cuda()
            targ_bv_n6_j_cosb = torch.zeros(N,T,n6).cuda()

            if opt.ba_def == 'vector_product_cosa':
                for n,(i,j) in enumerate(bv_n6_pairs):
                    pred_bv_n6_cosa[:,:,n] = torch.sum(torch.mul(pred_bv[:,:,i,:], pred_bv[:,:,j,:]),dim=2)/(torch.norm(pred_bv[:,:,i,:], dim=b_d-1) * torch.norm(pred_bv[:,:,j,:], dim=b_d-1))
                    targ_bv_n6_cosa[:,:,n] = torch.sum(torch.mul(targ_bv[:,:,i,:], targ_bv[:,:,j,:]),dim=2)/(torch.norm(targ_bv[:,:,i,:], dim=b_d-1) * torch.norm(targ_bv[:,:,j,:], dim=b_d-1))
                if opt.error_rule == 'F1':
                    loss_ba_n6 = torch.mean(torch.abs(pred_bv_n6_cosa - targ_bv_n6_cosa))
                elif opt.error_rule == 'F2':
                    loss_ba_n6 = torch.mean((pred_bv_n6_cosa - targ_bv_n6_cosa)**2)

            elif opt.ba_def == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(bv_n6_pairs):

                    pred_bv_n6_cosa_dif[:,:,n] = (pred_bv[:,:,i,2]/torch.norm(pred_bv[:,:,i,:],dim=b_d-1))-(pred_bv[:,:,j,2]/torch.norm(pred_bv[:,:,j,:],dim=b_d-1))
                    targ_bv_n6_cosa_dif[:,:,n] = (targ_bv[:,:,i,2]/torch.norm(targ_bv[:,:,i,:],dim=b_d-1))-(targ_bv[:,:,j,2]/torch.norm(targ_bv[:,:,j,:],dim=b_d-1))

                    pred_bv_n6_i_cosb[:,:,n] = pred_bv_X_PM[:,:,i] * (1+(pred_bv[:,:,i,1] / torch.sqrt(pred_bv[:,:,i,0]**2 + pred_bv[:,:,i,1]**2)))
                    pred_bv_n6_j_cosb[:,:,n] = pred_bv_X_PM[:,:,j] * (1+(pred_bv[:,:,j,1] / torch.sqrt(pred_bv[:,:,j,0]**2 + pred_bv[:,:,j,1]**2)))
                    targ_bv_n6_i_cosb[:,:,n] = targ_bv_X_PM[:,:,i] * (1+(targ_bv[:,:,i,1] / torch.sqrt(targ_bv[:,:,i,0]**2 + targ_bv[:,:,i,1]**2)))
                    targ_bv_n6_j_cosb[:,:,n] = targ_bv_X_PM[:,:,j] * (1+(targ_bv[:,:,j,1] / torch.sqrt(targ_bv[:,:,j,0]**2 + targ_bv[:,:,j,1]**2)))

                if opt.error_rule == 'F1':
                    loss_ba_n6_cosa = torch.mean(torch.abs(pred_bv_n6_cosa_dif - targ_bv_n6_cosa_dif))
                    loss_ba_n6_cosb = torch.mean(torch.abs((pred_bv_n6_i_cosb-pred_bv_n6_j_cosb)-(targ_bv_n6_i_cosb-targ_bv_n6_j_cosb)))
                elif opt.error_rule == 'F2':
                    loss_ba_n6_cosa = torch.mean((pred_bv_n6_cosa_dif - targ_bv_n6_cosa_dif)**2)
                    loss_ba_n6_cosb = torch.mean(((pred_bv_n6_i_cosb-pred_bv_n6_j_cosb)-(targ_bv_n6_i_cosb-targ_bv_n6_j_cosb))**2)
                loss_ba_n6 = loss_ba_n6_cosa + loss_ba_n6_cosb
        else:
            loss_ba_n6 = torch.zeros(1).cuda()

        if opt.ba_n7_pnl:
            bv_n7_pairs = [(2,12),(2,15),(5,12),(5,15)]
            n7 = len(bv_n7_pairs)
            pred_bv_n7_cosa = torch.zeros(N,T,n7).cuda()
            targ_bv_n7_cosa = torch.zeros(N,T,n7).cuda()

            pred_bv_n7_cosa_dif = torch.zeros(N,T,n7).cuda()
            targ_bv_n7_cosa_dif = torch.zeros(N,T,n7).cuda()

            pred_bv_n7_i_cosb = torch.zeros(N,T,n7).cuda()  #cosb of bone_i,where (bone_i,bone_j) is 1-step neighborhood bone pair
            pred_bv_n7_j_cosb = torch.zeros(N,T,n7).cuda()
            targ_bv_n7_i_cosb = torch.zeros(N,T,n7).cuda()
            targ_bv_n7_j_cosb = torch.zeros(N,T,n7).cuda()

            if opt.ba_def == 'vector_product_cosa':
                for n,(i,j) in enumerate(bv_n7_pairs):
                    pred_bv_n7_cosa[:,:,n] = torch.sum(torch.mul(pred_bv[:,:,i,:], pred_bv[:,:,j,:]),dim=2)/(torch.norm(pred_bv[:,:,i,:], dim=b_d-1) * torch.norm(pred_bv[:,:,j,:], dim=b_d-1))
                    targ_bv_n7_cosa[:,:,n] = torch.sum(torch.mul(targ_bv[:,:,i,:], targ_bv[:,:,j,:]),dim=2)/(torch.norm(targ_bv[:,:,i,:], dim=b_d-1) * torch.norm(targ_bv[:,:,j,:], dim=b_d-1))
                if opt.error_rule == 'F1':
                    loss_ba_n7 = torch.mean(torch.abs(pred_bv_n7_cosa - targ_bv_n7_cosa))
                elif opt.error_rule == 'F2':
                    loss_ba_n7 = torch.mean((pred_bv_n7_cosa - targ_bv_n7_cosa)**2)

            elif opt.ba_def == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(bv_n7_pairs):

                    pred_bv_n7_cosa_dif[:,:,n] = (pred_bv[:,:,i,2]/torch.norm(pred_bv[:,:,i,:],dim=b_d-1))-(pred_bv[:,:,j,2]/torch.norm(pred_bv[:,:,j,:],dim=b_d-1))
                    targ_bv_n7_cosa_dif[:,:,n] = (targ_bv[:,:,i,2]/torch.norm(targ_bv[:,:,i,:],dim=b_d-1))-(targ_bv[:,:,j,2]/torch.norm(targ_bv[:,:,j,:],dim=b_d-1))

                    pred_bv_n7_i_cosb[:,:,n] = pred_bv_X_PM[:,:,i] * (1+(pred_bv[:,:,i,1] / torch.sqrt(pred_bv[:,:,i,0]**2 + pred_bv[:,:,i,1]**2)))
                    pred_bv_n7_j_cosb[:,:,n] = pred_bv_X_PM[:,:,j] * (1+(pred_bv[:,:,j,1] / torch.sqrt(pred_bv[:,:,j,0]**2 + pred_bv[:,:,j,1]**2)))
                    targ_bv_n7_i_cosb[:,:,n] = targ_bv_X_PM[:,:,i] * (1+(targ_bv[:,:,i,1] / torch.sqrt(targ_bv[:,:,i,0]**2 + targ_bv[:,:,i,1]**2)))
                    targ_bv_n7_j_cosb[:,:,n] = targ_bv_X_PM[:,:,j] * (1+(targ_bv[:,:,j,1] / torch.sqrt(targ_bv[:,:,j,0]**2 + targ_bv[:,:,j,1]**2)))

                if opt.error_rule == 'F1':
                    loss_ba_n7_cosa = torch.mean(torch.abs(pred_bv_n7_cosa_dif - targ_bv_n7_cosa_dif))
                    loss_ba_n7_cosb = torch.mean(torch.abs((pred_bv_n7_i_cosb-pred_bv_n7_j_cosb)-(targ_bv_n7_i_cosb-targ_bv_n7_j_cosb)))
                elif opt.error_rule == 'F2':
                    loss_ba_n7_cosa = torch.mean((pred_bv_n7_cosa_dif - targ_bv_n7_cosa_dif)**2)
                    loss_ba_n7_cosb = torch.mean(((pred_bv_n7_i_cosb-pred_bv_n7_j_cosb)-(targ_bv_n7_i_cosb-targ_bv_n7_j_cosb))**2)
                loss_ba_n7 = loss_ba_n7_cosa + loss_ba_n7_cosb
        else:
            loss_ba_n7 = torch.zeros(1).cuda()

        loss_sum = loss_mpjpe + loss_pose_refine_mpjpe + opt.co_diff*loss_diff + opt.co_bl_sy * loss_bl_sy + opt.co_bl_sy_pore*loss_pose_refine_bl_sy+ \
                   opt.co_bl*loss_bl + opt.co_ba*loss_ba + opt.co_ba_n1*loss_ba_n1 + opt.co_ba_n2*loss_ba_n2 + opt.co_ba_n3*loss_ba_n3 + \
                   opt.co_ba_n4*loss_ba_n4 + opt.co_ba_n5*loss_ba_n5 + opt.co_ba_n6*loss_ba_n6 + opt.co_ba_n7*loss_ba_n7

        various_losses_sum_avg['loss_mpjpe'].update(loss_mpjpe.detach().cpu().numpy() * N, N) #detach使require_grad变False
        various_losses_sum_avg['loss_diff'].update(loss_diff.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_bl'].update(loss_bl.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_ba'].update(loss_ba.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_ba_n1'].update(loss_ba_n1.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_ba_n2'].update(loss_ba_n2.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_ba_n3'].update(loss_ba_n3.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_ba_n4'].update(loss_ba_n4.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_ba_n5'].update(loss_ba_n5.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_ba_n6'].update(loss_ba_n6.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_ba_n7'].update(loss_ba_n7.detach().cpu().numpy() * N, N)
        various_losses_sum_avg['loss_sum'].update(loss_sum.detach().cpu().numpy() * N, N)


        if split == 'train':
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            if not opt.target_0joint_0 and opt.relative3D_0_16=='0joint':
                out_target[:, :, 0, :] = 0
            if opt.relative3D_0_16=='0joint':
                pred_out[:, :, 0, :] = 0
            elif opt.relative3D_0_16=='7joint':
                pred_out[:, :, 7, :] = 0
            mpjpe_J0_0 = eval_cal.mpjpe(pred_out, out_target).item()  #set pred 0 joint 3d pose=0,then calculate mpjpe
            mpjpe_J0_0_sum_avg.update(mpjpe_J0_0*N, N)

        elif split == 'test':
            if not opt.target_0joint_0 and opt.relative3D_0_16=='0joint':
                out_target[:, :, 0, :] = 0
            if opt.relative3D_0_16=='0joint':
                pred_out[:, :, 0, :] = 0
            elif opt.relative3D_0_16=='7joint':
                pred_out[:, :, 7, :] = 0
            action_test_error_sum_avg = eval_cal.test_error_per_action(pred_out, out_target, action, action_test_error_sum_avg, opt.dataset, show_protocol2=opt.show_protocol2)

            if opt.pose_refine:
                pose_refine_output_3D[:, :, 0, :] = 0
                action_test_error_sum_avg_post_out = eval_cal.test_error_per_action(pose_refine_output_3D, out_target, action,
                                                                    action_test_error_sum_avg_post_out, opt.dataset, show_protocol2=opt.show_protocol2)

    #after 1 epoch of training or test
    if split == 'train':
        mpjpe_J0_0_one_epoch['xyz'] = mpjpe_J0_0_sum_avg.avg #
        print('when J0!=0, loss_mpjpe for all training data: %f mm' % (various_losses_sum_avg['loss_mpjpe'].avg*1000))
        print('when J0 =0, loss_mpjpe for all training data: %f mm' % (mpjpe_J0_0_one_epoch['xyz']*1000))
        print('weighted_sum of various losses for all training data: %f' % (various_losses_sum_avg['loss_sum'].avg))

    elif split == 'test':
        if not opt.pose_refine:
            mean_mpjpe_of_15actions = print_test_error(opt.dataset, action_test_error_sum_avg, opt.show_protocol2) # when test, mpjpe_J0_0 is printed per action
            mpjpe_J0_0_one_epoch['xyz'] = mean_mpjpe_of_15actions

        elif opt.pose_refine:
            print('-----pose refine output-----')
            mean_mpjpe_of_15actions = print_test_error(opt.dataset, action_test_error_sum_avg_post_out, opt.show_protocol2)
            mpjpe_J0_0_one_epoch['pose_refine_output'] = mean_mpjpe_of_15actions

   #whether train or test, the printing format of various structure error is the same
    print('loss_diff for all train/test data: %f' % (various_losses_sum_avg['loss_diff'].avg))
    if opt.bl_pnl:
        print('loss_bl for all train/test data: %f mm' % (various_losses_sum_avg['loss_bl'].avg))
    if opt.ba_pnl:
        print('loss ba for all train/test data: %f mm' % (various_losses_sum_avg['loss_ba'].avg))
    if opt.ba_n1_pnl:
        print('loss_ba_n1 for all train/test data: %f ' % (various_losses_sum_avg['loss_ba_n1'].avg))
    if opt.ba_n2_pnl:
        print('loss_ba_n2 for all train/test data: %f ' % (various_losses_sum_avg['loss_ba_n2'].avg))
    if opt.ba_n3_pnl:
        print('loss_ba_n3 for all train/test data: %f ' % (various_losses_sum_avg['loss_ba_n3'].avg))
    if opt.ba_n4_pnl:
        print('loss_ba_n4 for all train/test data: %f ' % (various_losses_sum_avg['loss_ba_n4'].avg))
    if opt.ba_n5_pnl:
        print('loss_ba_n5 for all train/test data: %f ' % (various_losses_sum_avg['loss_ba_n5'].avg))
    if opt.ba_n6_pnl:
        print('loss_ba_n6 for all train/test data: %f ' % (various_losses_sum_avg['loss_ba_n6'].avg))
    if opt.ba_n7_pnl:
        print('loss_ba_n7 for all train/test data: %f ' % (various_losses_sum_avg['loss_ba_n7'].avg))

    return mpjpe_J0_0_one_epoch, various_losses_sum_avg

def train(opt, actions, train_dataloader, model, optimizer):
    return step('train', opt, actions, train_dataloader, model, optimizer)

def val(opt, actions, val_dataloader, model):
    return step('test', opt, actions, val_dataloader, model)

def input_augmentation(input_2D, model_st_gcn, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape # T:frame  J:joints  C: channels    M=1
    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) #N, C, T, J , M
    input_2D_unflip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) #N, C, T, J , M

    output_3D_flip = model_st_gcn(input_2D_flip, out_all_frame=False)  #N, C, T, J, M
    output_3D_flip[:, 0] *= -1
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]

    output_3D_unflip = model_st_gcn(input_2D_unflip, out_all_frame=False)

    output_3D = (output_3D_unflip + output_3D_flip) / 2
    input_2D = input_2D_unflip

    return input_2D, output_3D
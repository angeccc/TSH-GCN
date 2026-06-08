import torch
from torch.autograd import Variable
import numpy as np
import os


class AccumLoss(object):
    """
    for initialize and accumulate loss/err
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def get_varialbe(split, target):
    num = len(target)  #5
    var = []

    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor) #
            var.append(temp)
    else:
        with torch.no_grad(): #
            for i in range(num):
                temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
                var.append(temp)

    return var


def get_uvd2xyz(uvd, gt_3D, cam):
    """
    pred-uv+pred-Z+(fx,fy,cx,cy)--->pred xyz
    """
    N, T, J,_ = uvd.size()  #uvd and gt_3D: [N,T,J,C]  T=1

    dec_out_all = uvd.view(-1, T, J, 3).clone()
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, J, 1).clone()# [N,T,1,C].repeat=[N,T,J,C], all 0-joint 3d-gt
    enc_in_all = uvd[:, :, :, :2].view(-1, T, J, 2).clone()  #[N,T,J,2]  uv  in range(0,0,1,1)

    cam_f_all = cam[..., :2].view(-1,1,1,2).repeat(1,T,J,1) #[N,1,1,2].repeat=[N,T,J,2]   fx,fy focal-length
    cam_c_all = cam[..., 2:4].view(-1,1,1,2).repeat(1,T,J,1)#[N,T,J,2] cx,cy camera-center

    # change to global
    z_global = dec_out_all[:, :, :, 2]# [N,T,J]  pred Z
    z_global[:, :, 0] = root[:, :, 0, 2] #[256,1,17,3][0,2]=[256,1]  0-joint 3d-gt-Z replace 0-joint pred-Z
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]  # N,T,J-1,1  1-16joint pred-Z + 0-joint gt-Z     Z-global:world-coordiate system
    z_global = z_global.unsqueeze(-1)  # N,T,J,1
    
    uv = enc_in_all - cam_c_all  # N,T,J,2    u-cx,v-cy
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  #(uZ,vZ)/(fx,fy)  [N,T,J,2]
    xyz_global = torch.cat((xy, z_global), -1)  #N,T,J,3  (uZ/fx, vZ/fy, z-global)
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, J, 1))# minus 0-joint 3d coordiate

    return xyz_offset

def print_test_error(data_type,action_test_error_sum_avg,show_protocol2=False):
    if data_type =='h36m':
        mean_error =  print_test_error_per_action(action_test_error_sum_avg, show_protocol2)

    return mean_error

def print_error_directly(action_error_sum):

    error = action_error_sum.avg * 1000.0
    print('Error:%f mm' % (error))
    return error

def print_test_error_per_action(action_test_error_sum_avg, show_protocol2=False):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if show_protocol2:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm")) #===Action=== ==p#1 mm== =p#2 mm=     0，1，2，第一个加Action共12个字符
        for action,value in action_test_error_sum_avg.items():  #'Directions'   {'p1':AccumLoss, 'p2':AccumLoss}
            print("{0:<12} ".format(action), end="")

            for j in range(1,3): #1, 2
                mean_error_each['p'+str(j)] = action_test_error_sum_avg[action]['p'+str(j)].avg * 1000.0 #
                mean_error_all['p'+str(j)].update(mean_error_each['p'+str(j)], 1)

            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'],mean_error_each['p2']))
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg))

    else:
        print("{0:=^12} {1:=^6}".format("p#1 Action", "mm"))
        for action,value in action_test_error_sum_avg.items():
            print("{0:<12} ".format(action), end="")

            mean_error_each['p1'] = action_test_error_sum_avg[action]['p1'].avg * 1000.0
            print("{0:>6.2f}".format(mean_error_each['p1']))
            mean_error_all['p1'].update(mean_error_each['p1'], 1)
        print("{0:<12} {1:>6.2f}".format("Average", mean_error_all['p1'].avg))

    return mean_error_all['p1'].avg

def print_boneError(action_error_sum):
    mean_boneError_each = {'boneLengthError': 0.0, 'boneAngleError': 0.0, 'neighBoneAngleError': 0.0, 'twoStepNeighBoneAngleError': 0.0, 'threeStepNeighBoneAngleError': 0.0, 'fourStepNeighBoneAngleError': 0.0, 'fiveStepNeighBoneAngleError': 0.0, 'sixStepNeighBoneAngleError': 0.0, 'sevenStepNeighBoneAngleError': 0.0}
    mean_boneError_all = {'boneLengthError': AccumLoss(), 'boneAngleError': AccumLoss(), 'neighBoneAngleError': AccumLoss(), 'twoStepNeighBoneAngleError': AccumLoss(), 'threeStepNeighBoneAngleError': AccumLoss(), 'fourStepNeighBoneAngleError': AccumLoss(), 'fiveStepNeighBoneAngleError': AccumLoss(), 'sixStepNeighBoneAngleError': AccumLoss(), 'sevenStepNeighBoneAngleError': AccumLoss()}


    print("{0:=^21} {1:=^6}".format("boneLenErr Action", "mm"))
    for action,value in action_error_sum.items():
        print("{0:<12} ".format(action), end="")
        mean_boneError_each['boneLengthError'] = action_error_sum[action]['boneLengthError'].avg * 1000.0
        print("{0:>6.2f}".format(mean_boneError_each['boneLengthError']))
        mean_boneError_all['boneLengthError'].update(mean_boneError_each['boneLengthError'], 1)
    print("{0:<12} {1:>6.2f}".format("Average", mean_boneError_all['boneLengthError'].avg))

    print("{0:=^23}".format("boneAngleErr Action"))
    for action,value in action_error_sum.items():
        print("{0:<12} ".format(action), end="")
        mean_boneError_each['boneAngleError'] = action_error_sum[action]['boneAngleError'].avg
        print("{0:>6.2f}".format(mean_boneError_each['boneAngleError']))
        mean_boneError_all['boneAngleError'].update(mean_boneError_each['boneAngleError'], 1)
    print("{0:<12} {1:>6.2f}".format("Average", mean_boneError_all['boneAngleError'].avg))

    return mean_boneError_all['boneLengthError'].avg

def print_error_xyz(action_error_sum_xyz):
    mean_error_xyz_sum = np.zeros([3])
    print("{0:=^12} {1:=^6} {2:=^6} {3:=^6}".format("p#1 Action", "x", "y","z"))
    for action, value in action_error_sum_xyz.items():
        print("{0:<12} ".format(action), end="")
        mean_error_xyz = np.array(action_error_sum_xyz[action][1:4]) /action_error_sum_xyz[action][0] * 1000.0
        mean_error_xyz_sum += mean_error_xyz
        print("{0:>6.2f} {1:>6.2f} {2:>6.2f}".format(mean_error_xyz[0],mean_error_xyz[1],mean_error_xyz[2]))
    mean_error_xyz_sum/= float(len(action_error_sum_xyz))
    print("{0:<12} {1:>6.2f}{2:>6.2f}{3:>6.2f}".format("Average", mean_error_xyz_sum[0], mean_error_xyz_sum[1], mean_error_xyz_sum[2]))


def get_loss_sum(pre_list,value_list,num_data):
    """

    :param pre_list: [loss_sum1,loss_sum2,...]
    :param value_list: [loss_value_1,loss_value_2...]
    :param num_data: number of data in this batch
    :return:
    """
    num_list = len(pre_list)
    for i in range(num_list):
        pre_list[i] = pre_list[i] + value_list[i].detach() * num_data
    return pre_list

def save_model(previous_saved_model_complete_path, save_dir,epoch, output_type, test_mpjpe_current_epoch, model, model_name):
    if os.path.exists(previous_saved_model_complete_path): # save_dir: ./results/x_frame/xx_gcn/xx_pose_refine/cpn/
        os.remove(previous_saved_model_complete_path)

    torch.save(model.state_dict(), '%s/model_%s_%d_eva_%s_%d.pth' % (save_dir, model_name, epoch, output_type, test_mpjpe_current_epoch * 100))
    previous_saved_model_complete_path = '%s/model_%s_%d_eva_%s_%d.pth' % (save_dir, model_name,epoch, output_type, test_mpjpe_current_epoch * 100)
    return previous_saved_model_complete_path

def define_error_list(actions):
    action_test_error_sum_avg = {} # dict.update
    action_test_error_sum_avg.update({actions[i]: {'p1':AccumLoss(),'p2':AccumLoss(),'bl_err':AccumLoss(),'ba_err':AccumLoss(),'ba_n1_err':AccumLoss(),
                                                   'ba_n2_err':AccumLoss(), 'ba_n3_err': AccumLoss(), 'ba_n4_err': AccumLoss(), 'ba_n5_err': AccumLoss(),
                                                   'ba_n6_err': AccumLoss(), 'ba_n7_err': AccumLoss()} for i in range(len(actions))})
    return action_test_error_sum_avg

def back_to_ori_uv(cropped_uv,bb_box):
    """
    bb_box: N,4   0,0,1,1
    for cropped uv, back to origial uv to help do the uvd->xyz operation
    :return:
    """
    N, T, J,_ = cropped_uv.size()
    uv = (cropped_uv+1) * (bb_box[:, 2:].view(N, 1, 1, 2)/2.0) + bb_box[:, 0:2].view(N, 1, 1, 2) #(N,1,1,2) 0.5,0.5    +(N,1,1,2) 0,0
    return uv
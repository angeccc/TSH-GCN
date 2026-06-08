import torch
import numpy as np


def mpjpe(predicted, target):
    assert predicted.shape == target.shape #N T J C
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def test_error_per_action(pred_out, out_target, action, action_test_error_sum_avg, data_type, show_protocol2=False):
    if data_type == 'h36m':
        action_test_error_sum_avg = mpjpe_per_action(pred_out, out_target, action, action_test_error_sum_avg)

        if show_protocol2:
            action_test_error_sum_avg = p_mpjpe_per_action(pred_out, out_target, action, action_test_error_sum_avg)

    return action_test_error_sum_avg

def mjmpe_directly(predicted, target,action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    # action_error_sum[0] += num
    # action_error_sum[1] += num * torch.mean(dist).item()
    action_error_sum.update(torch.mean(dist).item() * num, num)
    return action_error_sum

def mjmpe_by_action_subject(predicted, target,action,action_error_sum,subject):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    if len(set(list(action))) == 1 and len(set(list(subject))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name][subject][0] += num
        action_error_sum[action_name][subject][1] += num*torch.mean(dist).item()
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name][subject][0] += 1
            action_error_sum[action_name][subject][1] += dist[i].item()
    return action_error_sum

def mpjpe_per_action(pred_3D, out_target, action, action_test_error_sum_avg):
    assert pred_3D.shape == out_target.shape
    num = pred_3D.size(0) # N
    dist = torch.mean(torch.norm(pred_3D - out_target, dim=len(out_target.shape) - 1),dim=len(out_target.shape) - 2)  # MPJPE->(N,1)

    if len(set(list(action))) == 1: #
        end_index = action[0].find(' ') #
        if end_index != -1: #
            action_name = action[0][:end_index] #
        else:
            action_name = action[0]

        action_test_error_sum_avg[action_name]['p1'].update(torch.mean(dist).item()*num, num)
    else:  #
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_test_error_sum_avg[action_name]['p1'].update(dist[i].item(), 1)

    return action_test_error_sum_avg

def p_mpjpe_per_action(pred_3D, out_target,action,action_test_error_sum_avg):
    assert pred_3D.shape == out_target.shape
    num = pred_3D.size(0)
    pred = pred_3D.detach().cpu().numpy().reshape(-1, pred_3D.shape[-2], pred_3D.shape[-1]) # N, J, C
    gt = out_target.detach().cpu().numpy().reshape(-1, out_target.shape[-2], out_target.shape[-1])
    dist = p_mpjpe(pred,gt)
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_test_error_sum_avg[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_test_error_sum_avg[action_name]['p2'].update(np.mean(dist), 1)

    return action_test_error_sum_avg

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1),axis=len(target.shape) - 2)

def sym_penalty(dataset, keypoints, pred_out):
    loss_sym_BL = 0

    if dataset == 'h36m':

        if keypoints.startswith('sh'):
            left_bone = [(0,4),(4,5),(5,6),(8,10),(10,11),(11,12)]
            right_bone = [(0,1),(1,2),(2,3),(8,13),(13,14),(14,15)]
        else:
            left_bone_pairs = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13)]
            right_bone_pairs = [(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)]

        for (i_left,j_left),(i_right,j_right) in zip(left_bone_pairs, right_bone_pairs):
            left_bone = pred_out[:,:,i_left]-pred_out[:,:,j_left] #N T J C-->N,T,C  C->bone coordiate
            right_bone = pred_out[:, :, i_right] - pred_out[:, :, j_right]
            loss_sym_BL += torch.mean(torch.norm(left_bone, dim=- 1) - torch.norm(right_bone, dim=- 1))

    return loss_sym_BL

def bone_length_penalty(dataset,keypoints,pred_out):
    """
    get penalty for the consistency of the bone length in sequences
    :return:
    """
    loss_bone = 0
    if pred_out.size(1) == 1:
        return 0
    if dataset == 'h36m' and keypoints is not 'sh_ft_h36m':
        bone_id = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13),(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)
            ,(0,7),(7,8),(8,9),(9,10)]
        for (i,j) in bone_id:
            bone = torch.norm(pred_out[:,:,i]-pred_out[:,:,j],dim=-1)#N*T
            loss_bone += torch.sum(torch.var(bone,dim=1))#N

    return loss_bone

def bone_error_by_action(boneErrorType, loss_bone_batch, action, action_error_sum):
    num = len(action)
    if len(set(list(action))) == 1: #256('Directions 1',....)
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index] #'Directions'
        else:
            action_name = action[0]

        action_error_sum[action_name][boneErrorType].update(torch.mean(loss_bone_batch).item()*num, num) #{'Directions':{'p1':AccumLoss, 'p2':AccumLoss}, ...,'WalkTogether':{ , } }
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name][boneErrorType].update(loss_bone_batch[i].item(), 1)
    return action_error_sum
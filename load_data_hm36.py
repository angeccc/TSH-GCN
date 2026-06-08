import os
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt

from data.common.camera import *
from data.common.utils import deterministic_random
from data.common.generator import ChunkedGenerator


class Fusion(data.Dataset):
    def __init__(self, opt, dataset, data_root_path, train=True):

        self.data_type = opt.dataset   #'h36m'
        self.train = train
        self.keypoints_name = opt.keypoints #'cpn_ft_h36m_dbb'
        self.data_path = data_root_path #'/home/chuanjiang/Projects/STGCN/STGCN/data/dataset/'
        self.train_list = opt.subjects_train.split(',')  #['S1', 'S5', 'S6', 'S7', 'S8']
        self.test_list = opt.subjects_test.split(',')    #['S9','s11']
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')   #None
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.crop_uv = opt.crop_uv
        self.test_flip_aug = opt.test_flip_augment
        self.pad = opt.pad
        self.input_unproj_3dVector = opt.input_unproj_3dVector

        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list, opt)
            self.cameras_train, self.poses_train_3d, self.poses_train_2d = self.fetch(dataset, self.train_list, subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.chunk_length, self.cameras_train, self.poses_train_3d, self.poses_train_2d,
                                              opt.chunk_length, pad=self.pad,    flip_aug=opt.flip_horizontal_augment, reverse_aug=opt.reverse_augment,
                                              kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left, joints_right=self.joints_right,
                                              out_all=opt.out_all)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list, opt)
            self.cameras_test, self.poses_test_3d, self.poses_test_2d = self.fetch(dataset, self.test_list, subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.chunk_length,  self.cameras_test, self.poses_test_3d, self.poses_test_2d,
                                              opt.chunk_length, pad=self.pad,     flip_aug=False, reverse_aug=False, #
                                              kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left, joints_right=self.joints_right)
            self.key_index = self.generator.saved_index #('S9','Directions 1',0):[0, 2356]
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list, opt):
        print('Preparing data...')
        for subject in folder_list: #['S1', 'S5', 'S6', 'S7', 'S8']
            print('load %s' % subject)
            for action in dataset[subject].keys():
                anim = dataset[subject][action] #

                positions_3d = []
                multiViewUv_proj_distort = []
                multiViewUv_proj_undistort = []
                multiViewBoneVector = []
                multiViewBoneLength = []
                multiViewBoneLengthSum = []
                multiViewDepth_order = []

                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])  #

                    uv_proj_distort = project_to_2d( torch.tensor(pos_3d), torch.tensor( np.tile(cam['intrinsic'],(pos_3d.shape[0],1)) ) ) #
                    uv_proj_undistort = project_to_2d_linear( torch.tensor(pos_3d), torch.tensor( np.tile(cam['intrinsic'],(pos_3d.shape[0],1)) ) ) #
            #        uv_proj_undistort1 = np.matmul( np.tile(cam['intrinsic_matrix'][np.newaxis, np.newaxis], (pos_3d.shape[0], pos_3d.shape[1], 1, 1) ),\
            #                                       np.expand_dims(pos_3d/(np.tile(pos_3d[:, :, 2:3], (1, 1, 3)) ), axis=3))
                    #(1,1,3,3)  tile(3304,17,1,1)-->3304,17,3,3  tile(Z)-->3304,17,3  X/Z, Y/Z, 1  matmul( (3304,17,3,3),(3304,17,3,1) )
                #
                    multiViewUv_proj_distort.append(uv_proj_distort)
                    multiViewUv_proj_undistort.append(uv_proj_undistort)

                    if opt.relative3D_0_16=='0joint':
                        pos_3d[:, 1:] -= pos_3d[:, :1]  #
                    elif opt.relative3D_0_16=='7joint':
                        pos_3d[:, :] -= pos_3d[:, 7:8]

                    if self.keypoints_name.startswith('sh'):
                        pos_3d = np.delete(pos_3d, obj=9, axis=1)# remove neck for sh 2D detection

                    parentNodes = dataset._skeleton._parents.tolist()
                    parentNodes[0] = 0 #[0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
                    pos_3d_0center = np.zeros(pos_3d.shape) #3304,17,3
                    pos_3d_0center[:,1:] = pos_3d[:, 1:]
                    singleViewBoneVector = pos_3d_0center - pos_3d_0center[:, parentNodes, :]
                    singleViewBoneLength = np.linalg.norm(singleViewBoneVector, axis=2)
                    singleViewBoneLengthSum = np.sum(singleViewBoneLength, axis=1)

                    obtain_depth_order = 0
                    if obtain_depth_order:
                        depth_order = np.zeros((pos_3d.shape[0], pos_3d.shape[1], pos_3d.shape[1])) #3304,17,17
                        depth_order_discrete = np.zeros((pos_3d.shape[0], pos_3d.shape[1], pos_3d.shape[1]))#3304.17.17
                        for i in range(pos_3d.shape[1]): #17
                            for j in range(pos_3d.shape[1]):
                                depth_order[:, i, j] = pos_3d_0center[:, i, 2] - pos_3d_0center[:, j, 2]

                        depth_threhold = 0.1 # 0.05/0.1/0.12/0.2 and so on
                        fix_depth_interval = 0.1 #30/0.1/0.2/0.15/....
                       # dynamic_depth_interval = []
                        depth_order_discrete[depth_order < -depth_threhold] = (depth_order[depth_order < -depth_threhold] + depth_threhold) // fix_depth_interval
                        depth_order_discrete[depth_order > depth_threhold] = (depth_order[depth_order > depth_threhold] + fix_depth_interval - depth_threhold) // fix_depth_interval

                        relative_depth_joint_pairs = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), (7, 0), (8, 7), (9, 8), (10, 9), (11, 8), (12, 11), (13, 12), (14, 8), (15, 14), (16, 15),  #16 edges
                                                  (6, 3), (5, 2), (4, 1), (11, 14), (12, 15), (13, 16), #dui chen
                                                  (3, 1), (6, 4), (13, 11), (16, 14), (8, 0), (10, 8)] # partial 2-step
                        relative_matrix = np.zeros((pos_3d.shape[1], pos_3d.shape[1])) #17,17

                        relative_depth_partial = []
                        for (i, j) in relative_depth_joint_pairs:
                            relative_matrix[i, j] = 1
                            relative_depth_partial.append(depth_order[:, i, j]) #append(3304)
                        relative_depth_partial = np.stack(relative_depth_partial, axis=1)  #3304,28p
                        name = subject + '_' + action + '_' + cam['id'] +'.csv'  #'S1_Sitting 1_55011271.csv'
                        if not os.path.exists('relative_depth_partial'):
                            os.makedirs('relative_depth_partial')
                        np.savetxt('relative_depth_partial/'+name, relative_depth_partial, delimiter=',')
                        #dataframe = pd.DataFrame(relative_depth_partial)
                        #dataframe.hist()
                        multiViewDepth_order.append(depth_order)

                    multiViewBoneVector.append(singleViewBoneVector)
                    multiViewBoneLength.append(singleViewBoneLength)
                    multiViewBoneLengthSum.append(singleViewBoneLengthSum)
                    positions_3d.append(pos_3d) #POS_3D

                anim['positions_3d'] = positions_3d #
                anim['multiViewUv_proj_distort'] = multiViewUv_proj_distort
                anim['multiViewUv_proj_undistort'] = multiViewUv_proj_undistort
                anim['multiViewBoneVector'] = multiViewBoneVector
                anim['multiViewBoneLength'] = multiViewBoneLength
                anim['multiViewBoneLengthSum'] = multiViewBoneLengthSum
                anim['multiViewDepthorder'] = multiViewDepth_order

        #np.save('human36m_3Dgt.npy', np.array(list(dataset._data.items())))


        print('Loading 2D detections...')
        keypoints = np.load(self.data_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz', allow_pickle=True) #'/home...dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz'

        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']  #[[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
        if self.keypoints_name.startswith('sh'):
            self.kps_left, self.kps_right = [4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]
            self.joints_left, self.joints_right = [4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]
        else:
            self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1]) # [4, 5, 6, 11, 12, 13],[1, 2, 3, 14, 15, 16]
            self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(  #[4, 5, 6, 11, 12, 13],[1, 2, 3, 14, 15, 16]
                                                  dataset.skeleton().joints_right())
        keypoints = keypoints['positions_2d'].item() #{ S1:{act1:[nd,nd,nd,nd],...,act30:[nd,nd,nd,nd]},S5,S6,S7,S8,S9,S11} nd:(2636,17,2)

        for subject in folder_list: #S1,5,6,7,8
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject) #subject->{}
            for action in dataset[subject].keys():   #
                assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)

                for cam_idx in range(len(keypoints[subject][action])):   #4
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0] #3304
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length #3305>3304
                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

                assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])  #4

        for subject in folder_list:
            for action in keypoints[subject]:

                if opt.keypoints == 'gt' or opt.keypoints == 'cpn_ft_h36m_dbb':
                    multiView_2dPose = keypoints[subject][action]
                elif opt.keypoints == 'gt_proj_distort':
                    multiView_2dPose = dataset[subject][action]['multiViewUv_proj_distort']
                elif opt.keypoints == 'gt_proj_undistort':
                    multiView_2dPose = dataset[subject][action]['multiViewUv_proj_undistort']

                for cam_idx, kps in enumerate(multiView_2dPose):
                    cam = dataset.cameras()[subject][cam_idx]

                    kps_unproj_3dVector = np.zeros((kps.shape[0], kps.shape[1], 3), dtype=float)
                    kps_unproj_3dVector[:, :, 0:2] = kps
                    kps_unproj_3dVector[:, :, 2:3] = np.ones((kps.shape[0], kps.shape[1], 1))
                    kps_unproj_3dVector = np.matmul(np.linalg.inv(np.tile(cam['intrinsic_matrix'], (kps.shape[0], kps.shape[1], 1, 1))), np.expand_dims(kps_unproj_3dVector, axis=3))

                    if self.crop_uv == 1:  #
                        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])

                    if self.keypoints_name.startswith('sh'):
                        permute_index = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10]
                        kps = kps[:, permute_index, :]

                    if self.input_unproj_3dVector:
                        if cam_idx < 4:
                            keypoints[subject][action][cam_idx] = kps_unproj_3dVector.reshape((kps.shape[0], kps.shape[1], 3))
                        else:
                            keypoints[subject][action].append(kps_unproj_3dVector.reshape((kps.shape[0], kps.shape[1], 3)))
                    else:
                        if cam_idx < 4:
                            keypoints[subject][action][cam_idx] = kps
                        else:
                            keypoints[subject][action].append(kps)
        return keypoints

    def fetch(self, dataset, subjects, subset=1):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None: #False
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                multiView_2dpose = self.keypoints[subject][action]
                for i in range(len(multiView_2dpose)):
                    out_poses_2d[(subject, action, i)] = multiView_2dpose[i]

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(multiView_2dpose), 'Camera count mismatch' #4=4

                    for i, cam in enumerate(cams):  #
                        if 'intrinsic' in cam:      #true
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if 'positions_3d' in dataset[subject][action]:  #true
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(multiView_2dpose), 'Camera count mismatch'  #4=4

                    for i in range(len(poses_3d)):  # 4
                        out_poses_3d[(subject, action, i)] = poses_3d[i]  #

    #
        if subset < 1:  # 1,false
            for key in out_poses_2d.keys():  # ('S1', 'Phoning 1', 0)
                n_frames = int(round(len(out_poses_2d[key]) // self.downsample * subset) * self.downsample) #(2636//downsample*subset) *downsample
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key]))) #de-ra(0,2636-n_frame+1, '2636')   236
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:self.downsample] #[236: 236+1318: downsample]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:self.downsample]
        elif self.downsample > 1: #false

            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::self.downsample] #[::2]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::self.downsample]

        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):

        sequence_name, start_3dFrame, end_3dFrame, flip, reverse = self.generator.pairs[index]

        cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(sequence_name, start_3dFrame, end_3dFrame, flip, reverse)

        if self.train == False and self.test_flip_aug: #
            #
            #
            _, _, input_2D_flip_aug, _, _,_ = self.generator.get_batch(sequence_name, start_3dFrame, end_3dFrame, flip=True, reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_flip_aug,axis=0)),0) #

        if flip == True:
            flip = 'flip=True'
        else:
            flip = 'flip=False'

        bb_box = np.array([0, 0, 1, 1])
        scale = np.float(1.0)

        return cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind, index, start_3dFrame, flip
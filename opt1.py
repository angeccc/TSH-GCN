import argparse
import os
import math

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        # ===============================================================
        #                     Dataset options
        # ===============================================================
        self.parser.add_argument('--dataset', type=str, default='h36m', help='dataset name')
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use {gt||cpn_ft_h36m_dbb||gt_proj_distort||gt_proj_undistort}')
        self.parser.add_argument('--layout', type=str, default='hm36_gt', help='3d dataset used')
        self.parser.add_argument('--crop_uv', type=int, default=1, help='if crop_uv to center and do normalization')
        self.parser.add_argument('--data_root_path', type=str, default='/root/autodl-tmp/1view_SSE/data/dataset/', help='dataset root path/'
                                        '/root/autodl-tmp/1view_SSE/data/dataset/')
        self.parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST', help='actions to train/test on, separated by comma, or * for all actions')
        self.parser.add_argument('--chunk_length', default=1, type=int, metavar='N', help='output frames for each example unit')
        self.parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='discrete sampling per n frames')
        self.parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='constant partial sampling by fraction')
        self.parser.add_argument('--reverse_augment', type=bool, default=False, help='if reverse the video to augment data')
        self.parser.add_argument('--flip_horizontal_augment', type=bool, default=True, help='training data flipping horizontally')
        self.parser.add_argument('--test_flip_augment', type=bool, default=True, help='flip and fuse the input and output for test data')
        self.parser.add_argument('--comput_joint_corre_statis', type=bool, default=False, help='if comput the correlation between joints')
        self.parser.add_argument('--pose_2d_corre_stat', type=bool, default=False, help='if comput the correlation between 2d joints')
        self.parser.add_argument('--relative3D_0_16', type=str, default='0joint', help='1-16 joints are 0-joint-relative') #train =true
        self.parser.add_argument('--relative3D_0_joint', type=bool, default=True, help='0 joints are 0-joint-relative') #pose_2d_corre_stat false
        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--pro_train', type=int, default=1, help='if proceed train process')
        self.parser.add_argument('--pro_test', type=int, default=1, help='if proceed test process')
        self.parser.add_argument('--epoch_num', type=int, default=200, help='number of epochs')
        self.parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
        self.parser.add_argument('--learning_rate', type=float, default=2e-3)
        self.parser.add_argument('--large_decay_epoch', type=int, default=5, help='give a large lr decay after how manys epochs')
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float, metavar='LR', help='learning rate decay per epoch')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='SGD or Adam')
        self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer for SGD')
        self.parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
        self.parser.add_argument('--target_0joint_0', type=bool, default=True, help='if train-target 0 joint is 0,default is true') #pose_2d_corre_stat false
    # ===============================================================
        #                     Save/reload trained model options
        # ===============================================================
        self.parser.add_argument('--save_trained_model', type=int, default=1, help='if save trained model')
        self.parser.add_argument('-previous_smallest_test_mpjpe', type=float, default=math.inf, help='initial smallest test_mpjpe')
        self.parser.add_argument('-previous_saved_model_complete_path', type=str, default='', help='saved model complete path in a certain previous epoch')
        self.parser.add_argument('-previous_saved_refine_model_complete_path', type=str, default='', help='saved model complete path in a certain previous epoch')

        self.parser.add_argument('--reload_trained_stgcn', type=int, default=0, help='if continue from last time')
        self.parser.add_argument('--reload_trained_pose_refine', type=int, default=0, help='if continue from last time')
        self.parser.add_argument('--trained_model_dir', type=str, default='/root/autodl-tmp/1view_SSE/results/1_frame/two_scale_gcn/no_pose_refine/cpn/', help='previous trained model folder /home/chuanjiang/Projects/1view_SSE/results/1_frame/st_gcn/no_pose_refine/gt/')
        self.parser.add_argument('--trained_stgcn_model', type=str, default='model_two_scale_gcn_40_eva_xyz_5065.pth', help='trained model name')
        self.parser.add_argument('--trained_pose_refine_model', type=str, default='model_pose_refine_10_eva_post_.pth', help='trained model name')
        # ===============================================================
        #                     Loss function or test error options
        # ===============================================================
        self.parser.add_argument('--error_rule', default='F1', type=str, metavar='NAME', help='error:absF1/mseF2')
        self.parser.add_argument('--ba_def', type=str, default='cosa_|1+cosb|', help='how to define BoneAngle:cosa_|1+cosb|;vector_product_cosa;vector_product_a --> error')
        self.parser.add_argument('--bl_sy_pnl', type=int, default=0, help='if add symmetric bone length penalty ')
        self.parser.add_argument('--frames_diff_pnl', type=int, default=0, help='if add difference penalty between frames')
        self.parser.add_argument('--bl_pnl', type=int, default=1, help='if add bone_length_penalty')  #from bl_pnl to ba_n7_pnl,must be 1 to comput various structure losses
        self.parser.add_argument('--ba_pnl', type=int, default=1, help='if add bone_angle_penalty')   #just change the co_bl to...
        self.parser.add_argument('--ba_n1_pnl', type=int, default=1, help='if add neighbourBoneAngle_penalty')
        self.parser.add_argument('--ba_n2_pnl', type=int, default=1, help='if add two-step neighborhood bone_angle_penalty')
        self.parser.add_argument('--ba_n3_pnl', type=int, default=1, help='if add three-step neighborhood bone_angle_penalty')
        self.parser.add_argument('--ba_n4_pnl', type=int, default=1, help='if add four-step neighborhood bone_angle_penalty')
        self.parser.add_argument('--ba_n5_pnl', type=int, default=1, help='if add five-step neighborhood bone_angle_penalty')
        self.parser.add_argument('--ba_n6_pnl', type=int, default=1, help='if add six-step neighborhood bone_angle_penalty')
        self.parser.add_argument('--ba_n7_pnl', type=int, default=1, help='if add seven-step neighborhood bone_angle_penalty')

        self.parser.add_argument('--co_bl_sy', type=float, default=0.00, help='coefficient of symmetric bone length loss')
        self.parser.add_argument('--co_bl_sy_pore', type=float, default=0.00, help='coefficient of symmetric bone length loss for pose-refine output')
        self.parser.add_argument('--co_diff', type=float, default=0)
        self.parser.add_argument('--co_bl', type=float, default=0.007)#1
        self.parser.add_argument('--co_ba', type=float, default=0.003)#1
        self.parser.add_argument('--co_ba_n1', type=float, default=0.001)#0.1
        self.parser.add_argument('--co_ba_n2', type=float, default=0.001)
        self.parser.add_argument('--co_ba_n3', type=float, default=0.0009)
        self.parser.add_argument('--co_ba_n4', type=float, default=0.0004)
        self.parser.add_argument('--co_ba_n5', type=float, default=0.000)
        self.parser.add_argument('--co_ba_n6', type=float, default=0.000)
        self.parser.add_argument('--co_ba_n7', type=float, default=0.000)
        self.parser.add_argument('--show_protocol2', type=bool, default=True, help='if show p-mpjpe when testing')
        # ===============================================================
        #                     Graph options
        # ===============================================================
        self.parser.add_argument('--n_joints', type=int, default=17, help='number of joints')
        self.parser.add_argument('--partialSym', type=bool, default=True, help='if add partial sym pairs')
        self.parser.add_argument('--mulitistep_neigh', type=str, default='1kind', help='the number of categories of multi-step neigh nodes:none/1kind/2kinds')
        self.parser.add_argument('--graph_pool5_neigh_node_adjust', type=bool, default=True, help='if adjust neighbor_node of the graph_pool5')
        self.parser.add_argument('--graph_pool6', type=bool, default=True, help='if pool_graph has 6 nodes')
        self.parser.add_argument('--graph_pool6_8unshared', type=bool, default=False, help='in pool_graph, if node 8 is unshared,if true,means0,7,8+9,10, if false ,0,7,8+8,9,10')
        self.parser.add_argument('--terminal_connect', type=str, default='no', help='terminal joints connect rule no/13-6_16-3/13-3_16-6/13-6')
        self.parser.add_argument('--refine_nei_nodes', type=bool, default=True, help='if refine categories of neighbor nodes')
        self.parser.add_argument('--pool_graph_02_13', type=bool, default=False, help='if connect 02_13 in pool_graph')
        self.parser.add_argument('--pool_graph_01_23', type=bool, default=False, help='if connect 01_23 in pool_graph')
        self.parser.add_argument('--pool_graph_node4_sym', type=bool, default=True, help='if pool_graph_node4_sym share weight')
        self.parser.add_argument('--pool_graph_node4_3nei', type=bool, default=False, help='if pool_graph_node4_sym share weight')
    # ===============================================================
        #                     Network in/output options
        # ===============================================================
        self.parser.add_argument('--input_unproj_3dVector', type=bool, default=False, help='if input unprojected 3d Vector')
        self.parser.add_argument('--in_channels', type=int, default=2, help='input channels of model: 2')
        self.parser.add_argument('--output_type', type=str, default='xyz', help='xyz/pose_refine_output/uvd/time')
        self.parser.add_argument('--out_channels', type=int, default=3, help='output channels of model: 3')
        self.parser.add_argument('--uvd_to_xyz', type=bool, default=False, help='calculate xyz from uvd')
        self.parser.add_argument('--pad', type=int, default=0)  # 2*pad+1_frame graph
        self.parser.add_argument('--out_all', type=bool, default=True, help='output 1 frame or all frames when training')
        # ===============================================================
        #                     Network options
        # ===============================================================
        self.parser.add_argument('--framework', default='two_scale_gcn', type=str, metavar='NAME', help='framework to run{resgcn/two_scale_gcn/conv_gcn/three_scale_gcn}')
        self.parser.add_argument('--pose_refine', type=bool, default=False, help='if use pose_refine model')
        self.parser.add_argument('--corre_weight_diversity', type=str, default='SeveralKinds_1trunk+2sym_limbs+2_5_3_6_pool_trunklimb', help='completely_unshared/SeveralKinds_1node'
        'SeveralKinds_1node/SeveralKinds_1trunk+6sym_limbs/shared/SeveralKinds_1trunk+2limbs/SeveralKinds_1trunk+4limbs'
        'SeveralKinds_7center_0_1_2_3_4/SeveralKinds_7center_0_1_2_3_4_updown/SeveralKinds_1trunk+4limbs_1_2_3/SeveralKinds_1trunk+2sym_limbs+2_5_3_6/'
        'SeveralKinds_1trunk+4sym_limbs+3_6/SeveralKinds_1trunk+141114+2_5_3_6/0789_10_1114_14_2_5_3_6/SeveralKinds_1trunk+2sym_limbs+2_5_3_6_pool_trunk+limb'
        'SeveralKinds_1trunk+2sym_limbs+2_5_3_6_pool_trunklimb/')
        self.parser.add_argument('--channels_share_corre_weights', type=bool, default=False, help='if multi-channel input features share learnable correlation weight matrix')
        self.parser.add_argument('--residual_every2gcn', type=bool, default=False, help='if true, add residual module every two gcns,perNodeFF2 must be false')
        self.parser.add_argument('--perNodeFF', type=bool, default=True, help='if adding perNodeFF before st-gcn-layer')
        self.parser.add_argument('--perNodeFF2', type=bool, default=False, help='if adding perNodeFF after st-gcn-layer')
        self.parser.add_argument('--scale1_3gcn', type=bool, default=False, help='if perform 3 st-gcn-layers on scale1-graph')
        self.parser.add_argument('--scale2_1gcn', type=bool, default=False, help='if perform 1 st-gcn-layers on scale2-graph')
        self.parser.add_argument('--scale2_3gcn', type=bool, default=False, help='if perform 3 st-gcn-layers on scale2-graph')
        self.parser.add_argument('--conv_position', type=str, default='before_1_after_1', help='before_1/before_2/before_3/after_1/after_2/after_3/before_x_after_x/ if add conv layers before or after gcn layers or both position have conv layers')
        self.parser.add_argument('--gcn1_pernodeff2', type=bool, default=False, help='if add pernodeFF2 layer to gcn1 module')
        self.parser.add_argument('--after_channel', type=str, default='2x', help='if conv layer after gcn layers have 1x/2xC[0]/3x channels')
        self.parser.add_argument('--bn2d', type=bool, default=False, help='if adding BN2d after each convolution layer')
        self.parser.add_argument('--input_bn1d', type=bool, default=False, help='if perform BN1d on input data')
        self.parser.add_argument('--activation_func', type=str, default='elu_1', help='activation_function to be used:relu/elu_1/elu_0.1/selu/tanh')
        self.parser.add_argument('--two_scale_channels', type=str, default='128-384', help='128-384/128-256/160-320/128-256-512/2 scale of graph feature channels')
        self.parser.add_argument('--pool_rule', type=str, default='conv_pool', help='graph pool rule: max_pool/avg_pool/conv_pool')
        self.parser.add_argument('--skip_connect_rule', type=str, default='concat', help='skip_connect rule: concat/sum/none')
        self.parser.add_argument('--extra_skip_connect', type=str, default='=_1_', help='extra skip connect: sum_1/sum_1_conv1/concat_1/concat_1_conv1/')
        self.parser.add_argument('--gcn_residual', type=bool, default=True, help='if add residual layer to gcn module')
        self.parser.add_argument('--conv_residual', type=bool, default=True, help='if add residual layer to conv module')

        self.parser.add_argument('--dropout1', type=float, default=0) #0.25
        self.parser.add_argument('--dropout2', type=float, default=0)  #0.1
        self.parser.add_argument('--dropout3', type=float, default=0)  #0.1
        self.parser.add_argument('--dropout4', type=float, default=0.0)  #0.05
        self.parser.add_argument('--dropout5', type=float, default=0.2)  #0.05
        self.parser.add_argument('--dropout6', type=float, default=0)  #0.05
        self.parser.add_argument('--dropout7', type=float, default=0)  #0.05
        self.parser.add_argument('--dropout8', type=float, default=0.013)  #0.05
        self.parser.add_argument('--dropout9', type=float, default=0.026)  #0.05
        self.parser.add_argument('--dropout10', type=float, default=0)  #0.05
        self.parser.add_argument('--dropout11', type=float, default=0)  #0.05
        self.parser.add_argument('--dropout12', type=float, default=0.026)  #0.05
        self.parser.add_argument('--dropout13', type=float, default=0.0)  #0.05
        self.parser.add_argument('--dropout14', type=float, default=0.0)  #0.05
        self.parser.add_argument('--dropout_pose_ref', type=float, default=0) #0.5

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt) if not name.startswith('_'))


        if self.opt.dataset == 'h36m':
            self.opt.subjects_train = 'S1,S5,S6,S7,S8'
            self.opt.subjects_test = 'S9,S11'


        self.opt.save_dir = './results/'+'%d_frame/'%(self.opt.pad*2+1) +self.opt.framework + '/'+ \
        '%spose_refine/'%('' if self.opt.pose_refine else 'no_')


        if self.opt.keypoints == 'cpn_ft_h36m_dbb':
            self.opt.save_dir += 'cpn/'
        elif self.opt.keypoints == 'gt':
            self.opt.save_dir += 'gt/'
    #add file
        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)


        file_name = os.path.join(self.opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        print(self.opt)

        return self.opt

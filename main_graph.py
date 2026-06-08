from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os  #
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from opt1 import opts
import random
import time
import logging
import pickle
import torch
import torch.utils.data
import torch.nn as nn
import sys
import time
import h5py
import copy
import re

import numpy as np
from six.moves import xrange
import socket

from utils.data_utils import define_actions
from utils.utils1 import save_model
import torch.optim as optim
from nets.pose_refine import pose_refine
from train_graph_time import train, val
from data.load_data_hm36 import Fusion
from data.common.correlation_statistics import correlation_statistic

model = {}
opt = opts().parse()

if opt.pad > 0:
    from nets.st_gcn_multi_frame import Model
else:
    from nets.st_gcn_single_frame import Model

opt.manualSeed = 1
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(opt.save_dir, 'train_test.log'), level=logging.INFO)
logging.info('======================================================')

# load model
model['st_gcn'] = Model(opt).cuda()
model['pose_refine'] = pose_refine(opt).cuda()  # .cuda

# load data
data_root_path = opt.data_root_path
if opt.dataset == 'h36m':
    dataset_path = data_root_path + 'data_3d_' + opt.dataset + '.npz'
    from data.common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path, opt)

actions = define_actions(opt.actions)

if opt.pro_train:
    train_data = Fusion(opt=opt, train=True, dataset=dataset, data_root_path=data_root_path)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers), pin_memory=False)
if opt.pro_test:
    test_data = Fusion(opt=opt, train=False, dataset=dataset, data_root_path=data_root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers), pin_memory=False)

if opt.comput_joint_corre_statis:
    joints_statis_corre = correlation_statistic(train_data,test_data,opt)

#set optimizer
all_param = []
for i_model in model:  #{stgcn: , postrefine:}
    all_param += list(model[i_model].parameters())

lr = opt.learning_rate

if opt.optimizer == 'SGD':
    optimizer_all = optim.SGD(all_param, lr=lr, momentum=0.9, nesterov=True, weight_decay=opt.weight_decay)
elif opt.optimizer == 'Adam': #true
    optimizer_all = optim.Adam(all_param, lr=lr, amsgrad=True)

#Reload trained model
stgcn_dict = model['st_gcn'].state_dict()  #
if opt.reload_trained_stgcn == 1:
    preTrained_stgcn_dict = torch.load(os.path.join(opt.trained_model_dir, opt.trained_stgcn_model))
    for name, key in stgcn_dict.items():  #'data_bn.weight',tensor
        if name.startswith('A') == False:
            stgcn_dict[name] = preTrained_stgcn_dict[name]
    model['st_gcn'].load_state_dict(stgcn_dict)

pose_refine_dict = model['pose_refine'].state_dict()
if opt.reload_trained_pose_refine == 1:
    preTrained_pose_refine_dict = torch.load(os.path.join(opt.trained_model_dir, opt.trained_pose_refine_model)) #'model_pose_refine_10_eva_post_4870.pth'
    for name, key in pose_refine_dict.items():
        pose_refine_dict[name] = preTrained_pose_refine_dict[name]
    model['pose_refine'].load_state_dict(pose_refine_dict)


#training and test process
mpjpe_train_test_each_epoch = []
BL_train_test_each_epoch = []
BA_train_test_each_epoch = []
BA_n1_train_test_each_epoch = []
BA_n2_train_test_each_epoch = []
BA_n3_train_test_each_epoch = []
BA_n4_train_test_each_epoch = []
BA_n5_train_test_each_epoch = []
BA_n6_train_test_each_epoch = []
BA_n7_train_test_each_epoch = []

for epoch in range(1, opt.epoch_num):   #200
    print('======>>>>> Online epoch: #%d <<<<<======' % (epoch))
    torch.cuda.synchronize()  #

    if opt.pro_train == 1:
        timer = time.time()
        print('======>>>>> training <<<<<======')
        print('frame_number: %d' %(2*opt.pad+1))  #3
        print('Network architecture to run %s:' %opt.framework)  #st_gcn
        print('learning rate %f' % (lr))

        mpjpe_J0_0_one_epoch,various_losses_sum_avg = train(opt, actions, train_dataloader, model, optimizer_all)
        mpjpe_train_test_each_epoch.append(mpjpe_J0_0_one_epoch['xyz'])
        BL_train_test_each_epoch.append(various_losses_sum_avg['loss_bl'].avg)
        BA_train_test_each_epoch.append(various_losses_sum_avg['loss_ba'].avg)
        BA_n1_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n1'].avg)
        BA_n2_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n2'].avg)
        BA_n3_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n3'].avg)
        BA_n4_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n4'].avg)
        BA_n5_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n5'].avg)
        BA_n6_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n6'].avg)
        BA_n7_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n7'].avg)

        timer = time.time() - timer
        timer = timer / len(train_data) #Fusion.__len__()
        print('==> time to train 1 sample = %f (ms)' % (timer * 1000))  #ms

    if opt.pro_test == 1:
        timer = time.time()
        print('======>>>>> test<<<<<======')
        print('frame_number: %d' %(2*opt.pad+1))
        print('Network architecture to run %s:' %opt.framework)

        mpjpe_J0_0_one_epoch,various_losses_sum_avg = val(opt, actions, test_dataloader, model)
        mpjpe_train_test_each_epoch.append(mpjpe_J0_0_one_epoch['xyz'])
        BL_train_test_each_epoch.append(various_losses_sum_avg['loss_bl'].avg)
        BA_train_test_each_epoch.append(various_losses_sum_avg['loss_ba'].avg)
        BA_n1_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n1'].avg)
        BA_n2_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n2'].avg)
        BA_n3_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n3'].avg)
        BA_n4_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n4'].avg)
        BA_n5_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n5'].avg)
        BA_n6_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n6'].avg)
        BA_n7_train_test_each_epoch.append(various_losses_sum_avg['loss_ba_n7'].avg)

        timer = time.time() - timer
        timer = timer / len(test_data)
        print('==> time to test 1 sample = %f (ms)' % (timer * 1000))

        if opt.output_type == 'xyz':
            test_mpjpe_current_epoch = mpjpe_J0_0_one_epoch['xyz'] #mean test mpjpe of 15 actions
        elif opt.output_type == 'pose_refine_output':
            test_mpjpe_current_epoch = mpjpe_J0_0_one_epoch['pose_refine_output']

        if opt.save_trained_model and test_mpjpe_current_epoch < opt.previous_smallest_test_mpjpe:
            opt.previous_saved_model_complete_path = save_model(opt.previous_saved_model_complete_path, opt.save_dir, epoch, opt.output_type,
                                                                test_mpjpe_current_epoch, model['st_gcn'], opt.framework)

            if opt.pose_refine:
                opt.previous_saved_refine_model_complete_path = save_model(opt.previous_saved_refine_model_complete_path, opt.save_dir, epoch,
                                                        opt.output_type, test_mpjpe_current_epoch, model['pose_refine'], opt.framework+'_pose_refine')
            opt.previous_smallest_test_mpjpe = test_mpjpe_current_epoch

    if epoch % opt.large_decay_epoch == 0:
        for param_group in optimizer_all.param_groups:
            param_group['lr'] *= 0.5
            lr *= 0.5
    else:
        for param_group in optimizer_all.param_groups:
            param_group['lr'] *= opt.lr_decay
            lr *= opt.lr_decay

    if epoch % 5 == 0:
        print(mpjpe_train_test_each_epoch)
        print(BL_train_test_each_epoch)
        print(BA_train_test_each_epoch)
        print(BA_n1_train_test_each_epoch)
        print(BA_n2_train_test_each_epoch)
        print(BA_n3_train_test_each_epoch)
        print(BA_n4_train_test_each_epoch)
        print(BA_n5_train_test_each_epoch)
        print(BA_n6_train_test_each_epoch)
        print(BA_n7_train_test_each_epoch)
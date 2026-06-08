import torch
import torch.nn as nn

from torch.autograd import Variable


class pose_refine(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.inplace = True
        out_seqlen = 1
        fc_in = opt.out_channels * 2 * out_seqlen * opt.n_joints  #3x2x1x17
        fc_out = opt.in_channels * opt.n_joints    #2x17
        fc_unit = 1024

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

        self.pose_refine = nn.Sequential(
            nn.Linear(fc_in, fc_unit),    #102,1024
            self.activ_func,
            nn.Dropout(opt.dropout_pose_ref, inplace=True),
            nn.Linear(fc_unit, fc_out), #1024,34
            nn.Sigmoid())


    def forward(self, x, x_1):

        N, T, J,_ = x.size()  # _ 3     x_1: N,T,J,C
        x_in = torch.cat((x, x_1), -1)  #N,T,J,6  pred_3d concat xyz
        x_in = x_in.view(N, -1) # N,T*J*6

        score = self.pose_refine(x_in).view(N,T,J,2) #N,34.view-->N T J 2
        score_cm = Variable(torch.ones(score.size()), requires_grad=False).cuda() - score   #  1-score
        x_out = x.clone()  # pred-3d  (Xp, Yp, Zp)
        x_out[:, :, :, :2] = score * x[:, :, :, :2] + score_cm * x_1[:, :, :, :2] # N T J 2  [Xp*score1 + x*(1-score1), Yp*score2+ y*(1-score2), Zp]

        return x_out
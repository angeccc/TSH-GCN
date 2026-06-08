import numpy as np


class ChunkedGenerator:
    """ refined from https://github.com/facebookresearch/VideoPose3D

    batch_size: opt.batchsize，//chunk_length
    chunk_length,pad,out_all: padding,chunk_length+2*pad，out_all=F

    causal_shift -- a symmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator

    """

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0, shuffle=False, random_seed=1234, flip_aug=False, reverse_aug=False,
                 kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all = False):

        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d)) #600=600
        assert cameras is None or len(cameras) == len(poses_2d)   #600=600

        pairs = []
        self.saved_index = {}
        start_index = 0

        for key in poses_2d.keys(): #('S1', 'Directions 1', 0)
            n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length  #(1383+1-1)//1=1383
            offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2  #(1383x1-1383)//2=0

            bounds = np.arange(n_chunks + 1) * chunk_length - offset    #ndarray(0-1383)
            keys = np.tile( np.array(key).reshape([1,3]), (len(bounds) - 1,1)) #nd(1383,3)  ('S1', 'Directions 1', 0)
            flip_augment_vector = np.full(len(bounds) - 1, False, dtype=bool) #ndarray
            reverse_augment_vector = np.full(len(bounds) - 1, False, dtype=bool) #

            pairs += list(zip(keys, bounds[:-1], bounds[1:], flip_augment_vector,reverse_augment_vector)) #zip合并  [ (nd[S1,act1,'0'], 0, 1, False, False),..., (nd[S1,act1,'0'],1382,1383,False,False) ]

            if reverse_aug:   #false
                pairs += list(zip(keys, bounds[:-1], bounds[1:], flip_augment_vector, ~reverse_augment_vector))

            if flip_aug:   #true
                if reverse_aug:  #false
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~flip_augment_vector, ~reverse_augment_vector))
                else:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~flip_augment_vector, reverse_augment_vector))

            end_index = start_index + poses_3d[key].shape[0] #0+1383=1383
            self.saved_index[key] = [start_index, end_index]  #key:('S1', 'Directions 1', 0), [0,1383]
            start_index = start_index + poses_3d[key].shape[0] #1383

        #
        if cameras is not None: #true
            self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))  #256,9,
        if poses_3d is not None: #true
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1])) #256,1,17,3
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-2], poses_2d[key].shape[-1])) #256,3,17,2

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size #256
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.flip_aug = flip_aug  #true
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.flip_aug

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state


    def get_batch(self, sequence_name, start_3dFrame, end_3dFrame, flip, reverse):

        subject, action, cam_index = sequence_name
        seq_name = (subject, action, int(cam_index))
        seq_2d = self.poses_2d[seq_name].copy()

        start_2dFrame = start_3dFrame - self.pad - self.causal_shift
        end_2dFrame = end_3dFrame + self.pad - self.causal_shift
        low_2dFrame = max(start_2dFrame, 0) #
        high_2dFrame = min(end_2dFrame, seq_2d.shape[0]) #

        pad_left_2d = low_2dFrame - start_2dFrame #
        pad_right_2d = end_2dFrame- high_2dFrame
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2dFrame:high_2dFrame], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),'edge')#seq_2d[0:1] array:(1,17,2)
                                   # ( (2,0),(0,0),(0,0) )
        else:
            self.batch_2d = seq_2d[low_2dFrame:high_2dFrame]

        if flip:  #
            self.batch_2d[ :, :, 0] *= -1
            self.batch_2d[ :, self.kps_left + self.kps_right] = self.batch_2d[ :, self.kps_right + self.kps_left]

        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy() #

        # 3D poses
        seq_3d = self.poses_3d[seq_name].copy()
        if self.out_all:              #
            low_3dFrame = low_2dFrame
            high_3dFrame = high_2dFrame
            pad_left_3d = pad_left_2d
            pad_right_3d = pad_right_2d
        else:
            low_3dFrame = max(start_3dFrame, 0)   #
            high_3dFrame = min(end_3dFrame, seq_3d.shape[0])
            pad_left_3d = low_3dFrame - start_3dFrame
            pad_right_3d = end_3dFrame - high_3dFrame
        if pad_left_3d != 0 or pad_right_3d != 0:
            self.batch_3d = np.pad(seq_3d[low_3dFrame:high_3dFrame], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
        else:
            self.batch_3d = seq_3d[low_3dFrame:high_3dFrame]

        if flip:
            self.batch_3d[ :, :, 0] *= -1
            self.batch_3d[ :, self.joints_left + self.joints_right] = self.batch_3d[ :, self.joints_right + self.joints_left]
        if reverse:
            self.batch_3d = self.batch_3d[::-1].copy()

        # Cameras
        self.batch_cam = self.cameras[seq_name].copy()
        if flip:
            self.batch_cam[ 2] *= -1  #
            self.batch_cam[ 7] *= -1  #


        if self.cameras is None and self.poses_3d is None:
            return None, None, self.batch_2d.copy(), action, subject, int(cam_index)
        elif self.cameras is None and self.poses_3d is not None:
            return np.zeros(9), self.batch_3d.copy(), self.batch_2d.copy(), action, subject, int(cam_index)
        elif self.poses_3d is None:
            return self.batch_cam, None, self.batch_2d.copy(), action, subject, int(cam_index)
        else:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d.copy(), action, subject, int(cam_index)
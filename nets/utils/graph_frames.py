import numpy as np

class Graph():
    def __init__(self, layout, opt, pad=0, max_hop=1):
        self.max_hop = max_hop #
        self.seqlen = 2*pad+1   #
        self.opt = opt
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=self.max_hop)
        self.dist_center = self.get_distance_to_center(layout) #
        self.get_adjacency() #

    def get_distance_to_center(self, layout):
        dist_center = np.zeros(self.num_node) #51
        if layout == 'hm36_gt':
            for i in range(self.seqlen):   #0,1,2
                index_start = i*self.num_node_each #17
                dist_center[index_start+0: index_start+7] = [1, 2, 3, 4, 2, 3, 4]
                dist_center[index_start+7: index_start+11] = [0, 1, 2, 3]
                dist_center[index_start+11: index_start+17] = [2, 3, 4, 2, 3, 4]
        return dist_center

    def __str__(self):
        return self.A

    def graph_link_between_frames(self, base):
        return [((front - 1) + i*self.num_node_each, (back - 1)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]

    def basic_layout(self, neighbour_base, sym_base):
        self.num_node = self.num_node_each * self.seqlen   #17x3=51

        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1) for j in range(self.num_node_each)]  #
        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)
                                  for j in range(self.num_node_each)]  #list[(0, 17), (1, 18), 34个,
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in range(self.seqlen - 1)
                               for j in range(self.num_node_each)]   #list[(17, 0), (18, 1), 34个

        self_link = [(i, i) for i in range(self.num_node)]    #list[(0, 0), (1, 1), (2, 2), (3, 3)....(50,50)

        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)    #16x3=48

        self.sym_link_all = self.graph_link_between_frames(sym_base)

        return self_link, time_link

    def get_edge(self, layout):
        if layout == 'hm36_gt':
            self.num_node_each = 17
            neighbour_base = [(1, 2), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 1), (9, 8), (10, 9), (11, 10), (12, 9),
                              (13, 12), (14, 13), (15, 9), (16, 15), (17, 16)]   #list 16个边,17个点
            if self.opt.terminal_connect=='13-6_16-3':
                neighbour_base=neighbour_base+[(14,7),(17,4)]
            elif self.opt.terminal_connect=='13-3_16-6':
                neighbour_base=neighbour_base+[(14,4),(17,7)]

            if not self.opt.partialSym:
                sym_base = [(7, 4), (6, 3), (5, 2), (12, 15), (13, 16), (14, 17)]
            else:
                sym_base = [(4, 7), (14, 17)] #(14,17),(3,6),(13,16)(5, 2), (12, 15),

            if self.opt.mulitistep_neigh=='1kind':
                multi_step_neigh_base_1 = [(15, 2),(5,12)]#,(17,10),(14,10)
                self.multi_step_link_all_1 = self.graph_link_between_frames(multi_step_neigh_base_1)
                self.multi_step_link_all_2 = []
            elif self.opt.mulitistep_neigh=='2kinds':
                multi_step_neigh_base_1 = [(5, 12),(15,2),(5,9)]
                multi_step_neigh_base_2 = [(2, 9)]#,(8,12),(8,15)
                self.multi_step_link_all_1 = self.graph_link_between_frames(multi_step_neigh_base_1)
                self.multi_step_link_all_2 = self.graph_link_between_frames(multi_step_neigh_base_2)
            else:
                self.multi_step_link_all_1 = []
                self.multi_step_link_all_2 = []

            self_link, time_link = self.basic_layout(neighbour_base, sym_base)   #

            self.la, self.ra = [11, 12, 13], [14, 15, 16]
            self.ll, self.rl = [4, 5, 6], [1, 2, 3]
            if self.opt.graph_pool6:
                if self.opt.graph_pool6_8unshared:
                    self.cb1, self.cb2= [0, 7, 8], [9, 10]
                else:
                    self.cb1, self.cb2= [0, 7, 8], [8, 9, 10]
                self.part = [self.la, self.ra, self.ll, self.rl, self.cb1, self.cb2]
            else:
                self.cb = [0, 7, 8, 9, 10]
                self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + self.multi_step_link_all_1 + self.multi_step_link_all_2 + time_link   #

            self.center = 8 - 1  # center node

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self):
        valid_hop = range(0, self.max_hop+1, 1)   #range(0, 2,1)
        adjacency = np.zeros((self.num_node, self.num_node))   #51x51
        for hop in valid_hop:   #0,1
            adjacency[self.hop_dis == hop] = 1   #
        normalize_adjacency = normalize_digraph(adjacency)  #

        A = []
        for hop in valid_hop:      #range(0, 2)
            A_root    = np.zeros((self.num_node, self.num_node))
            A_close   = np.zeros((self.num_node, self.num_node))
            A_further = np.zeros((self.num_node, self.num_node))
            A_sym     = np.zeros((self.num_node, self.num_node))
            A_forward = np.zeros((self.num_node, self.num_node))
            A_back    = np.zeros((self.num_node, self.num_node))
            A_multistep1 = np.zeros((self.num_node, self.num_node))
            A_multistep2 = np.zeros((self.num_node, self.num_node))
            A_adjcent3 = np.zeros((self.num_node, self.num_node))

            for i in range(self.num_node): #51
                for j in range(self.num_node): #51
                    if self.hop_dis[j, i] == hop:
                        if (j, i) in self.sym_link_all or (i, j) in self.sym_link_all:
                            A_sym[j, i] = normalize_adjacency[j, i]
                        elif (j, i) in self.multi_step_link_all_1 or (i, j) in self.multi_step_link_all_1:
                            A_multistep1[j, i] = normalize_adjacency[j, i]
                        elif (j, i) in self.multi_step_link_all_2 or (i, j) in self.multi_step_link_all_2:
                            A_multistep2[j, i] = normalize_adjacency[j, i]
                        elif (j, i) in self.time_link_forward:
                            A_forward[j, i] = normalize_adjacency[j, i]
                        elif (j, i) in self.time_link_back:
                            A_back[j, i] = normalize_adjacency[j, i]
                        elif self.dist_center[j] == self.dist_center[i]: #hop=0
                            A_root[j, i] = normalize_adjacency[j, i]
                        elif self.dist_center[j] > self.dist_center[i]:
                            A_close[j, i] = normalize_adjacency[j, i]
                        else:
                            A_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(A_root)
            else:
                if self.opt.refine_nei_nodes:
                    A_sym[4, 0] = A_close[4, 0]
                    A_close[4, 0] = 0.000
                    A_further[8, 7] = A_close[8, 7]
                    A_close[8, 7] = 0.000
                    A_sym[9, 8] = A_close[9, 8]
                    A_close[9, 8] = 0.000
                    A_adjcent3[14,8]=A_close[14,8]
                    A_close[14,8]=0

                A.append(A_close)
                A.append(A_further)
                A.append(A_sym)
                if self.opt.refine_nei_nodes:
                    A.append(A_adjcent3)
                if self.seqlen > 1:
                    A.append(A_forward)
                    A.append(A_back)
                if self.opt.mulitistep_neigh=='1kind':
                    A.append(A_multistep1)
                if self.opt.mulitistep_neigh=='2kinds':
                    A.append(A_multistep1)
                    A.append(A_multistep2)

        A = np.stack(A)         #6,51,51
        self.A = A

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))    #51x51
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    for d in range(max_hop, -1, -1): #1,0
        hop_dis[arrive_mat[d]] = d    #
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0] #51
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
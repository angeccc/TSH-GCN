import numpy as np

class Graph_pool():
    def __init__(self, layout, opt, pad=0, max_hop=1):
        self.max_hop = max_hop
        self.seqlen = 2*pad+1
        self.opt = opt
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.dist_center = self.get_distance_to_center(layout)
        self.get_adjacency()

    def get_distance_to_center(self, layout):
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                if self.opt.graph_pool6:
                    dist_center[index_start+0:index_start+4] = [1, 1, 1, 1]
                    dist_center[index_start+5] = 1
                else:
                    dist_center[index_start+0:index_start+4] = [1, 1, 1, 1]
                dist_center[index_start+4] = 0
        return dist_center

    def __str__(self):
        return self.A

    def graph_link_between_frames(self, base):
        return [((front) + i*self.num_node_each, (back )+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]

    def get_edge(self, layout):
        if layout == 'hm36_gt':
            if self.opt.graph_pool6:
                self.num_node_each = 6
                if not self.opt.pool_graph_02_13:
                    neighbour_base = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (1, 5), (4, 5)]
                else:
                    neighbour_base = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 5), (1, 5), (4, 5),(0,2),(1,3)]
            else:
                self.num_node_each = 5
                neighbour_base = [(0, 4), (1, 4), (2, 4), (3, 4)]
            self.num_node = self.num_node_each * self.seqlen   #5x3=15

            self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)
            if self.opt.pool_graph_01_23:
                sym_base = [(0, 1), (2, 3)] #(0, 1), (2, 3)
            else:
                sym_base = [ ]
            self.sym_link_all = self.graph_link_between_frames(sym_base)

            time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1) for j in range(self.num_node_each)]
            self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)
                         for j in range(self.num_node_each)]
            self.time_link_back = [((i+1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in range(self.seqlen - 1)
                         for j in range(self.num_node_each)]

            self_link = [(i, i) for i in range(self.num_node)]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link

            self.center = 5-1

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        A = []
        for hop in valid_hop:
            A_root = np.zeros((self.num_node, self.num_node))
            A_close = np.zeros((self.num_node, self.num_node))
            A_further = np.zeros((self.num_node, self.num_node))
            A_sym = np.zeros((self.num_node, self.num_node))
            A_forward = np.zeros((self.num_node, self.num_node))
            A_back = np.zeros((self.num_node, self.num_node))
            A_adjcent3 = np.zeros((self.num_node, self.num_node))
            A_adjcent4 = np.zeros((self.num_node, self.num_node))
            A_multistep = np.zeros((self.num_node, self.num_node))

            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] == hop:
                        if (j, i) in self.sym_link_all or (i, j) in self.sym_link_all:
                            A_sym[j, i] = normalize_adjacency[j, i]
                        elif (j, i) in self.time_link_forward:
                            A_forward[j, i] = normalize_adjacency[j, i]
                        elif (j, i) in self.time_link_back:
                            A_back[j, i] = normalize_adjacency[j, i]
                        elif self.dist_center[j] == self.dist_center[i]:
                            A_root[j, i] = normalize_adjacency[j, i]
                        elif self.dist_center[j] > self.dist_center[i]:
                            A_close[j, i] = normalize_adjacency[j, i]
                        else:
                            A_further[j, i] = normalize_adjacency[j, i]

            if hop == 0:
                A.append(A_root)
            else:
                if self.opt.graph_pool5_neigh_node_adjust and not self.opt.graph_pool6:
                    A_further[1,4]=A_close[1,4]
                    A_sym[2,4]=A_close[2,4]
                    A_adjcent3[3,4]=A_close[3,4]
                    A_close[1:4,4]=0
                if self.opt.graph_pool6:
                    A_close[5,0:2] = A_root[5,0:2]
                    A_close[0:2,5] = A_root[0:2,5]
                    A_close[5,4] = 0
                    A_close[2:4, 4] = 0.000
                    A_further[2:4, 4] = A_close[0,4]
                    A_adjcent3[5,4] = A_close[0,4]
                    if self.opt.refine_nei_nodes:
                        A_close[0,5]=0
                        A_adjcent3[0,5]=0.25
                        if not self.opt.pool_graph_node4_sym and not self.opt.pool_graph_node4_3nei:
                            A_close[1,4]=0
                            A_sym[1,4]=1/6
                            A_further[3,4]=0
                            A_adjcent4[3,4]=1/6
                        if self.opt.pool_graph_node4_3nei:
                            A_close[2:4,4]=A_further[2:4,4]
                            A_further[2:4,4] = 0
                            
                    if self.opt.pool_graph_02_13:
                        A_multistep[2,0]=A_root[2,0]
                        A_multistep[0,2]=A_root[0,2]
                        A_multistep[1,3]=A_root[1,3]
                        A_multistep[3,1]=A_root[3,1]

                A.append(A_sym)
                A.append(A_close)
                A.append(A_further)
                A.append(A_adjcent3)

                if self.opt.graph_pool6 and self.opt.refine_nei_nodes:
                    A.append(A_adjcent4)
                if self.opt.pool_graph_02_13:
                    A.append(A_multistep)
                if self.seqlen > 1:
                    A.append(A_forward)
                    A.append(A_back)

        A = np.stack(A)
        self.A = A

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
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
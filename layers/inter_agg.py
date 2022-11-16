from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


def weight_inter_agg(num_relations, neigh_feats, embed_dim, alpha, n, cuda):
    
    """
    Weight inter-relation aggregator
    :param num_relations: number of relations in the graph
    :param neigh_feats: intra_relation aggregated neighbor embeddings for each aggregation
    :param embed_dim: the dimension of output embedding
    :param alpha: weight paramter for each relation
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    """

    neigh_h = neigh_feats.t()

    w = F.softmax(alpha, dim = 1)
    
    if cuda:
        aggregated = torch.zeros(size=(embed_dim, n)).cuda() #
    else:
        aggregated = torch.zeros(size=(embed_dim, n))

    for r in range(num_relations):

        aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1,n), neigh_h[:, r*n:(r+1)*n])

    return aggregated.t()


class InterAgg(nn.Module, ABC):

    """
    the fraud-aware convolution module
    Inter aggregation layer
    """

    def __init__(self, embed_dim, adj_lists, intraggs, cuda = False):

        """
        Initialize the inter-relation aggregator
        :param features: the input embeddings for all nodes
        :param embed_dim: the dimension need to be aggregated
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intraggs: the intra-relation aggregatore used by each single-relation graph
        :param cuda: whether to use GPU
        """

        super(InterAgg, self). __init__()

        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]
        self.intra_agg2 = intraggs[1]
        self.intra_agg3 = intraggs[2]
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.intra_agg1.cuda = cuda
        self.intra_agg2.cuda = cuda
        self.intra_agg3.cuda = cuda

        if self.cuda:
            self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim*2, 3)).cuda()

        else:
            self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim*2, 3))

        init.xavier_uniform_(self.alpha)

    @abstractmethod
    def get_batch_feature(self, unique_nodes, train_flag):
        raise NotImplementedError()

    
    def forward(self, nodes, train_flag=True):
        
        """
        nodes: a list of batch node ids
        """
        
        if (isinstance(nodes,list)==False):
            nodes = nodes.cpu().numpy().tolist()

        to_neighs = []

        #adj_lists = [relation1, relation2, relation3]
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        #to_neighs: [[set, set, set], [set, set, set], [set, set, set]]
        
        #find unique nodes and their neighbors used in current batch   #set(nodes)
        unique_nodes =  set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),set.union(*to_neighs[2], set(nodes)))
        
        #id mapping
        unique_nodes_new_index = {n: i for i, n in enumerate(list(unique_nodes))}
        

        # unique_nodes = list(unique_nodes)
        # unique_nodes.sort()
        batch_features = self.get_batch_feature(unique_nodes, train_flag)

        #get neighbor node id list for each batch node and relation
        r1_list = [set(to_neigh) for to_neigh in to_neighs[0]] # [[set],[set],[ser]]  //   [[list],[list],[list]]
        r2_list = [set(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [set(to_neigh) for to_neigh in to_neighs[2]]

        center_nodes_new_index = [unique_nodes_new_index[int(n)] for n in nodes]################
        '''
        if self.cuda and isinstance(nodes, list):
            self_feats = self.features(torch.cuda.LongTensor(nodes))
        else:
            self_feats = self.features(index)
        '''

        #center_feats = self_feats[:, -self.embed_dim:]
        
        self_feats = batch_features[center_nodes_new_index]

        r1_feats = self.intra_agg1.forward(batch_features[:, -self.embed_dim:], nodes, r1_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])
        r2_feats = self.intra_agg2.forward(batch_features[:, -self.embed_dim:], nodes, r2_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])
        r3_feats = self.intra_agg3.forward(batch_features[:, -self.embed_dim:], nodes, r3_list, unique_nodes_new_index, self_feats[:, -self.embed_dim:])

        neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim = 0)

        n=len(nodes)
        
        attention_layer_outputs = weight_inter_agg(len(self.adj_lists), neigh_feats, self.embed_dim * 2, self.alpha, n, self.cuda)

        result = torch.cat((self_feats, attention_layer_outputs), dim = 1)

        return result


class InterAgg1(InterAgg):

    def __init__(self, mlp, embed_dim, adj_lists, intraggs, cuda = False):
        super(InterAgg1, self). __init__(embed_dim, adj_lists, intraggs, cuda)
        self.mlp = mlp

    def get_batch_feature(self, unique_nodes, train_flag):
        if self.cuda:
            batch_features = self.mlp(torch.cuda.LongTensor(list(unique_nodes)), train_flag)
        else:
            batch_features = self.mlp(torch.LongTensor(list(unique_nodes)), train_flag)
        return batch_features


class InterAgg2(InterAgg):

    def __init__(self, agg1, embed_dim, adj_lists, intraggs, cuda = False):
        super(InterAgg2, self). __init__(embed_dim, adj_lists, intraggs, cuda)
        self.agg1 = agg1

    def get_batch_feature(self, unique_nodes, train_flag):
        if self.cuda:
            batch_features = self.agg1(torch.cuda.LongTensor(list(unique_nodes)), train_flag)
        else:
            batch_features = self.agg1(torch.LongTensor(list(unique_nodes)), train_flag)
        return batch_features
    
    
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import math

from layers.sampler import Sampler
from layers.mlp import FraudreMLP
from layers.inter_agg import InterAgg
from layers.intra_agg import IntraAgg

class Fraudre(nn.Module):

    def __init__(self, K, num_classes, embed_dim, agg, prior, adj_lists=None):
        super(Fraudre, self).__init__()

        """
        Initialize the model
        :param K: the number of CONVOLUTION layers of the model
        :param num_classes: number of classes (2 in our paper)
        :param embed_dim: the output dimension of MLP layer
        :agg: the inter-relation aggregator that output the final embedding
        :lambad 1: the weight of MLP layer (ignore it)
        :prior:prior
        """

        self.agg = agg
        #self.lambda_1 = lambda_1

        self.K = K #how many layers
        self.prior = prior
        self.xent = nn.CrossEntropyLoss()
        self.embed_dim = embed_dim
        self.fun = nn.LeakyReLU(0.3)
        if adj_lists is not None:
            self.sampler = Sampler(adj_lists)


        # self.weight_mlp = nn.Parameter(torch.FloatTensor(self.embed_dim, num_classes)) #Default requires_grad = True
        self.weight_model = nn.Parameter(torch.FloatTensor((int(math.pow(2, K+1)-1) * self.embed_dim), 64))

        self.weight_model2 = nn.Parameter(torch.FloatTensor(64, num_classes))

        
        # init.xavier_uniform_(self.weight_mlp)
        init.xavier_uniform_(self.weight_model)
        init.xavier_uniform_(self.weight_model2)

    def forward(self, nodes, train_flag = True):

        embedding = self.agg(nodes, train_flag)

        scores_model = embedding.mm(self.weight_model)
        scores_model = self.fun(scores_model)
        scores_model = scores_model.mm(self.weight_model2)
        #scores_model = self.fun(scores_model)

        # scores_mlp = embedding[:, 0: self.embed_dim].mm(self.weight_mlp)
        # scores_mlp = self.fun(scores_mlp)

        # return scores_model, scores_mlp
        return scores_model
        #dimension, the number of center nodes * 2
    
    def to_prob(self, nodes, train_flag=False):

        # scores_model, scores_mlp = self.forward(nodes, train_flag)
        scores_model = self.forward(nodes, train_flag)
        scores_model = torch.sigmoid(scores_model)
        return scores_model


    def loss(self, nodes, labels, train_flag=True):

        #the classification module

        scores_model = self.forward(nodes, train_flag)

        scores_model = scores_model + torch.log(self.prior)
        loss_model = self.xent(scores_model, labels.squeeze())
        #loss_mlp = self.xent(scores_mlp, labels.squeeze())
        final_loss = loss_model #+ self.lambda_1 * loss_mlp
        return final_loss


def create_fraudre(args, data):
    # Calculate the prior
    y_train = data.stream_y_train
    num_1 = len(np.where(y_train == 1)[0])
    num_2 = len(np.where(y_train == 0)[0])
    p0 = (num_1 / (num_1 + num_2))
    p1 = 1 - p0
    prior = np.array([p1, p0])
    if args.cuda:
        prior = (torch.from_numpy(prior +1e-8)).cuda()
    else:
        prior = (torch.from_numpy(prior +1e-8))

    # Initialize model input
    fradure_input = nn.Embedding(data.features.shape[0], data.features.shape[1])
    fradure_input.weight = nn.Parameter(torch.FloatTensor(data.features), requires_grad=False)
    if args.cuda:
        fradure_input.cuda()

    # Define input graph topology
    adj_lists = [data.streams_relations['upu'], data.streams_relations['usu'], data.streams_relations['uvu']]

    # The first neural network layer (ego-feature embedding module)
    mlp = FraudreMLP(fradure_input, data.features.shape[1], args.embed_dim, cuda=args.cuda)

    # First convolution layer
    intra1_1 = IntraAgg(cuda=args.cuda)
    intra1_2 = IntraAgg(cuda=args.cuda)
    intra1_3 = IntraAgg(cuda=args.cuda)
    agg1 = InterAgg(lambda nodes: mlp(nodes), args.embed_dim, adj_lists, [intra1_1, intra1_2, intra1_3], cuda=args.cuda)

    # Second convolution layer
    intra2_1 = IntraAgg(cuda=args.cuda)
    intra2_2 = IntraAgg(cuda=args.cuda)
    intra2_3 = IntraAgg(cuda=args.cuda)
    agg2 = InterAgg(lambda nodes: agg1(nodes), args.embed_dim*2, adj_lists, [intra2_1, intra2_2, intra2_3], cuda=args.cuda)
    
    gnn_model = Fraudre(2, 2, args.embed_dim, agg2, prior, data.adj_lists)

    return gnn_model
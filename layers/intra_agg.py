import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


class IntraAgg(nn.Module):

	"""
	the fraud-aware convolution module
	Intra Aggregation Layer
	"""

	def __init__(self, cuda = False):

		super(IntraAgg, self).__init__()

		self.cuda = cuda

	def forward(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):

		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param embedding: embedding of all nodes in a batch
		:param neighbor_lists: neighbor node id list for each batch node in one relation # [[list],[list],[list]]
		:param unique_nodes_new_index
		"""

		#find unique nodes
		unique_nodes_list = list(set.union(*neighbor_lists))

		#id mapping
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		mask = Variable(torch.zeros(len(neighbor_lists), len(unique_nodes)))
		
		column_indices = [unique_nodes[n] for neighbor_list in neighbor_lists for n in neighbor_list ]
		row_indices = [i for i in range(len(neighbor_lists)) for _ in range(len(neighbor_lists[i]))]

		mask[row_indices, column_indices] = 1


		num_neigh = mask.sum(1,keepdim=True)
		#mask = torch.true_divide(mask, num_neigh)
		num_neigh[num_neigh==0] = 1
		mask = torch.div(mask, num_neigh)

		neighbors_new_index = [unique_nodes_new_index[n] for n in unique_nodes_list ]

		embed_matrix = embedding[neighbors_new_index]
		
		embed_matrix = embed_matrix.cpu()

		_feats_1 = mask.mm(embed_matrix) 
		if self.cuda:
			_feats_1 = _feats_1.cuda()

		#difference 
		_feats_2 = self_feats - _feats_1
		return torch.cat((_feats_1, _feats_2), dim=1)

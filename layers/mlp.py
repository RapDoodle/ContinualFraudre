import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


class FraudreMLP(nn.Module):

	"""
	the ego-feature embedding module
	"""

	def __init__(self, features, input_dim, output_dim, cuda = False):

		super(FraudreMLP, self).__init__()

		self.features = features
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.cuda = cuda
		self.mlp_layer = nn.Linear(self.input_dim, self.output_dim)

	def forward(self, nodes):

		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(nodes))
		else:
			batch_features = self.features(torch.LongTensor(nodes))

		if self.cuda:
			self.mlp_layer.cuda()

		result = self.mlp_layer(batch_features)

		result = F.relu(result)


		return result
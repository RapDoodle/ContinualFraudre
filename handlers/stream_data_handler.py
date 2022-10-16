import os
import sys
import numpy as np 
import logging
import random
from collections import defaultdict

from .data_handler import DataHandler

class StreamDataHandler(DataHandler):

    def __init__(self):
        super(StreamDataHandler, self).__init__()

    def load(self, data_name, t):
        self.data_name = data_name
        self.t = t

        # Load features
        features_file_name = os.path.join('./data', data_name, 'features')
        self.features = np.loadtxt(features_file_name, delimiter=',')

        # Load labels
        labels_file_name = os.path.join('./data', data_name, 'labels')
        labels = np.loadtxt(labels_file_name, dtype = np.int64, delimiter=',')

        # Load train / valid nodes
        train_file_name = os.path.join('./data', data_name, 'train_nodes')
        self.train_all_nodes_list = np.loadtxt(train_file_name, dtype = np.int64, delimiter=',')
        valid_file_name = os.path.join('./data', data_name, 'val_nodes')
        self.valid_all_nodes_list = np.loadtxt(valid_file_name, dtype = np.int64, delimiter=',')

        # Load graph
        stream_dir_name = os.path.join('./data', data_name, 'streams')
        # self.nodes = set()
        self.train_cha_nodes_list, self.train_old_nodes_list = set(), set()
        self.valid_cha_nodes_list, self.valid_old_nodes_list = set(), set()
        self.adj_lists = defaultdict(set)
        
        begin_time = 0
        end_time = t
        for tt in range(0, len(os.listdir(os.path.join('./data', data_name, 'streams')))):
            stream_edges_file_name = os.path.join(stream_dir_name, str(tt), 'edges')
            with open(stream_edges_file_name) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split(',')
                    node1, node2 = int(info[0]), int(info[1])

                    # self.nodes.add(node1)
                    # self.nodes.add(node2)

                    if tt <= end_time and tt >= begin_time:
                        self._assign_node(node1, tt)
                        self._assign_node(node2, tt)

                        self.adj_lists[node1].add(node2)
                        self.adj_lists[node2].add(node1)
        
        # Generate node and label list
        self.labels = np.ones(len(labels), dtype=np.int64)
        # self.labels[labels[:, 0]] = labels[:, 1]
        self.labels[labels[:, 0]] = labels[:, 1]

        # Input & Output size
        self.feature_size = self.features.shape[1]
        self.label_size = np.unique(self.labels).shape[0]

        # Train & Valid data
        self.train_nodes = self.train_cha_nodes_list
        self.val_nodes = self.valid_cha_nodes_list.union(self.valid_old_nodes_list)

        
        self.train_nodes = list(self.train_nodes)
        self.val_nodes = list(self.val_nodes)
        self.train_cha_nodes_list, self.train_old_nodes_list = list(self.train_cha_nodes_list), list(self.train_old_nodes_list)
        self.valid_cha_nodes_list, self.valid_old_nodes_list = list(self.valid_cha_nodes_list), list(self.valid_old_nodes_list)
        
        self.train_size = len(self.train_nodes)
        self.valid_size = len(self.val_nodes)
        self.data_size = self.train_size + self.valid_size



    def _assign_node(self, node, tt):
        if node in self.train_all_nodes_list and tt == self.t:
            self.train_cha_nodes_list.add(node)
        elif node in self.train_all_nodes_list and tt < self.t:
            self.train_old_nodes_list.add(node)
        elif node in self.valid_all_nodes_list and tt == self.t:
            self.valid_cha_nodes_list.add(node)
        elif node in self.valid_all_nodes_list and tt < self.t:
            self.valid_old_nodes_list.add(node)
    
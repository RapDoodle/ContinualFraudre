import os
import sys
import pickle
import numpy as np 
import logging
import random
from collections import defaultdict

from .data_handler import DataHandler

class StreamDataHandler(DataHandler):

    def __init__(self, data_name=None, t=None, max_detect_size=None):
        super(StreamDataHandler, self).__init__()
        self.data_name = data_name
        self.t = t
        self.max_detect_size = max_detect_size

    def load(self, data_name=None, t=None, max_detect_size=None):
        if data_name is not None:
            self.data_name = data_name
        if t is not None:
            self.t = t
        if max_detect_size is not None:
            self.max_detect_size = max_detect_size

        # Load features
        features_file_name = os.path.join('./data', self.data_name, 'features')
        self.features = np.loadtxt(features_file_name, delimiter=',')

        # Load labels
        labels_file_name = os.path.join('./data', self.data_name, 'labels')
        labels = np.loadtxt(labels_file_name, dtype = np.int64, delimiter=',')

        # Load train / valid nodes
        train_file_name = os.path.join('./data', self.data_name, 'train_nodes')
        self.train_all_nodes_list = np.loadtxt(train_file_name, dtype = np.int64, delimiter=',')
        valid_file_name = os.path.join('./data', self.data_name, 'val_nodes')
        self.valid_all_nodes_list = np.loadtxt(valid_file_name, dtype = np.int64, delimiter=',')

        # Load graph
        stream_dir_name = os.path.join('./data', self.data_name, 'streams')
        # self.nodes = set()
        self.train_cha_nodes_list, self.train_old_nodes_list = set(), set()
        self.valid_cha_nodes_list, self.valid_old_nodes_list = set(), set()
        self.adj_lists = defaultdict(set)
        
        begin_time = 0 if self.max_detect_size is None else max(self.t - self.max_detect_size, 0)
        end_time = self.t
        for tt in range(0, len(os.listdir(os.path.join('./data', self.data_name, 'streams')))):
            stream_train_node_file_name = os.path.join(stream_dir_name, str(tt), 'train_nodes')
            stream_val_node_file_name = os.path.join(stream_dir_name, str(tt), 'val_nodes')

            if tt <= end_time and tt >= begin_time:
                stream_adj_list_file_name = os.path.join(stream_dir_name, str(tt), 'adj_list.pkl')
                if os.path.exists(stream_train_node_file_name):
                    train_nodes = np.loadtxt(stream_train_node_file_name, dtype = np.int64, delimiter=',').tolist()
                    val_nodes = np.loadtxt(stream_val_node_file_name, dtype = np.int64, delimiter=',').tolist()
                    if tt < self.t:
                        self.train_old_nodes_list.update(train_nodes)
                        self.valid_old_nodes_list.update(val_nodes)
                    elif tt == self.t:
                        self.train_cha_nodes_list.update(train_nodes)
                        self.valid_cha_nodes_list.update(val_nodes)
                    with open(stream_adj_list_file_name, 'rb') as fp:
                        new_adj_list = pickle.load(fp)
                    for node, node_adj_list in new_adj_list.items():
                        # Undirected information handled by preprocessing script
                        self.adj_lists[node].update(node_adj_list)
                else:
                    # Legacy
                    stream_edges_file_name = os.path.join(stream_dir_name, str(tt), 'edges')
                    with open(stream_edges_file_name) as fp:
                        for i, line in enumerate(fp):
                            info = line.strip().split(',')
                            node1, node2 = int(info[0]), int(info[1])

                            # self.nodes.add(node1)
                            # self.nodes.add(node2)
                            
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
    
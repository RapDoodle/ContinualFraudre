import os
import re
import random
import pickle
import numpy as np

from utils import read_stream_statistics
from .stream_data_handler import StreamDataHandler

class AmazonStreamDataHandler(StreamDataHandler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs):
        super().load(**kwargs)

    """
    Note: All stream retrieved using this method has indices starting from 0
    """
    def load_stream(self, t=None, reset_index=False):
        if t is not None:
            self.t = t

        stream_base_path = os.path.join('./data', self.data_name, 'streams', str(self.t))
        stream_statistics_path = os.path.join(stream_base_path, 'statistics')
        stream_adj_list_path = os.path.join(stream_base_path, 'adj_list.pkl')
        stream_features_path = os.path.join(stream_base_path, 'features')
        stream_train_nodes_path = os.path.join(stream_base_path, 'train_nodes')
        stream_val_nodes_path = os.path.join(stream_base_path, 'val_nodes')
        stream_labels_path = os.path.join(stream_base_path, 'labels')

        self.stream_stats = read_stream_statistics(stream_statistics_path)

        offset = self.stream_stats['lo'] if reset_index else 0
        
        # Read labels
        self.stream_labels = np.loadtxt(stream_labels_path, dtype = np.int64, delimiter=',')[:, 1]

        # Obtain train-test split
        self.stream_train_nodes = np.loadtxt(stream_train_nodes_path, dtype = np.int64, delimiter=',') - offset
        self.stream_val_nodes = np.loadtxt(stream_val_nodes_path, dtype = np.int64, delimiter=',') - offset

        # Load the features
        self.stream_features = np.loadtxt(stream_features_path, delimiter=',')
        # self.stream_X_train = features[self.train_nodes, :]
        # self.stream_X_val = features[self.val_nodes, :]
        self.stream_y_train = self.stream_labels[np.array(self.stream_train_nodes) - (0 if reset_index else self.stream_stats['lo'])]
        self.stream_y_val = self.stream_labels[np.array(self.stream_val_nodes) - (0 if reset_index else self.stream_stats['lo'])]

        # Load adjacency list
        offset = (self.stream_stats['lo'] if reset_index else 0)
        with open(stream_adj_list_path, 'rb') as fp:
            adj_list = pickle.load(fp)
            out_adj_list = {}
            for node, adj in adj_list.items():
                out_adj_list[node-offset] = (np.array(adj_list[node])-offset).tolist()
            self.stream_adj_list = out_adj_list

        # print(self.X_train.shape, self.X_val.shape, self.y_train.shape, self.y_val.shape)

    """
    Load relations for a given stream. 
    The loaded relations are stored in self.stream_relations[relation]
    * Note: Only the specified stream is loaded.
    * Note: All stream retrieved using this method has indices starting from 0
    Must call load_stream first.
    """
    def load_stream_relations(self, t=None, relations=['upu', 'usu', 'uvu']):
        if t is not None:
            self.t = t
        if not hasattr(self, 'stream_stats'):
            raise Exception('need to invoke load_stream() first')

        lo = self.stream_stats['lo']

        self.stream_relations = {}
        stream_base_path = os.path.join('./data', self.data_name, 'streams', str(self.t))
        
        for relation in relations:
            relation_path = os.path.join(stream_base_path, f'{relation}_adj_list.pkl')
            with open(relation_path, 'rb') as fp:
                adj_list = pickle.load(fp)
                out_adj_list = {}
                for node, adj in adj_list.items():
                    out_adj_list[node-lo] = (np.array(adj_list[node])-lo).tolist()
                self.stream_relations[relation] = out_adj_list

    """
    Load relations for multiple streams. 
    The loaded relations are stored in self.streams_relations[relation]
    * Note: The original index are preserved
    Must call load_stream first.
    """
    def load_streams_relations(self, t=None, max_detect_size=None, relations=['upu', 'usu', 'uvu']):
        if t is not None:
            self.t = t
        if max_detect_size is not None:
            self.max_detect_size = max_detect_size

        self.streams_relations = {}
        for relation in relations:
            self.streams_relations[relation] = {}
        
        begin_time = 0 if self.max_detect_size is None else max(self.t - self.max_detect_size, 0)
        end_time = self.t
        for i in range(begin_time, end_time+1):
            stream_base_path = os.path.join('.', 'data', self.data_name, 'streams', str(i))
            stream_statistics_path = os.path.join(stream_base_path, 'statistics')
            stream_stats = read_stream_statistics(stream_statistics_path)
            for relation in relations:
                for j in range(stream_stats['lo'], stream_stats['hi']+1):
                    if j not in self.streams_relations[relation]:
                        self.streams_relations[relation][j] = set()
                relation_path = os.path.join(stream_base_path, f'{relation}_adj_list.pkl')
                with open(relation_path, 'rb') as fp:
                    adj_list = pickle.load(fp)
                for node, adj in adj_list.items():
                    self.streams_relations[relation][node].update(adj_list[node])
        
        # Convert all sets to lists
        for relation in relations:
            for node, adj_list in self.streams_relations[relation].items():
                self.streams_relations[relation][node] = list(self.streams_relations[relation][node])

    """
    Sample a graph with random-walk-with-restrart
    Arguments:
        root: The root node. Default: a random node
        pr: The probability of restarting in any step
        max_steps: The maximum number of steps to walk
    """
    def sameple_graph(self, root=None, pr=0.2, max_steps=100):
        if not hasattr(self, 'stream_stats'):
            raise Exception('need to invoke load_stream() first')

        if root is None:
            stream_range = self.stream_stats['hi'] - self.stream_stats['lo'] + 1
            root_node = random.randint(0, stream_range-1)
        else:
            root_node = root
        curr_node = root_node

        edges = set()
        nodes = set()

        # Add the root node
        nodes.add(root_node)

        for i in range(max_steps):
            # Determine whether to restart
            if random.random() < pr:
                curr_node = root_node
                continue
            curr_node_adj_list_len = len(self.stream_adj_list[curr_node])
            if curr_node_adj_list_len > 0:
                next_node = self.stream_adj_list[curr_node][random.randint(0, curr_node_adj_list_len-1)]
                # Add the edge i->j (i < j)
                if next_node != curr_node:
                    # In case the adjacency list contains self-loop
                    edges.add((curr_node, next_node) if curr_node < next_node else (next_node, curr_node))
                    nodes.add(next_node)
                curr_node = next_node
        
        adj_list = {}
        for node in nodes:
            adj_list[node] = []
        for node1, node2 in edges:
            adj_list[node1].append(node2)
            adj_list[node2].append(node1)
        nodes_list = list(nodes)
        return nodes_list, adj_list, self.stream_features[nodes_list]


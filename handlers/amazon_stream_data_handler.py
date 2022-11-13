import os
import re
import random
import pickle
import numpy as np

from .stream_data_handler import StreamDataHandler

class AmazonStreamDataHandler(StreamDataHandler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs):
        super().load(**kwargs)

    """
    Note: All stream retrieved using this method has indices starting from 0
    """
    def load_stream(self, t=None):
        if t is not None:
            self.t = t

        stream_base_path = os.path.join('./data', self.data_name, 'streams', str(self.t))
        stream_statistics_path = os.path.join(stream_base_path, 'statistics')
        stream_adj_list_path = os.path.join(stream_base_path, 'adj_list.pkl')
        stream_features_path = os.path.join(stream_base_path, 'features')
        stream_train_nodes_path = os.path.join(stream_base_path, 'train_nodes')
        stream_val_nodes_path = os.path.join(stream_base_path, 'val_nodes')
        stream_labels_path = os.path.join(stream_base_path, 'labels')
        with open(stream_statistics_path, 'r') as f:
            stats = f.read().split('\n')
        self.stream_stats = {}
        for stat in stats:
            splitted = stat.split('=')
            if len(splitted) == 2:
                key = splitted[0]
                val = splitted[1]
                # Convert to int when possible
                try:
                    val = int(val)
                except:
                    pass
                self.stream_stats[key] = val
        
        # Read labels
        self.stream_labels = np.loadtxt(stream_labels_path, dtype = np.int64, delimiter=',')[:, 1]

        # Obtain train-test split
        self.stream_train_nodes = np.loadtxt(stream_train_nodes_path, dtype = np.int64, delimiter=',') - self.stream_stats['lo']
        self.stream_val_nodes = np.loadtxt(stream_val_nodes_path, dtype = np.int64, delimiter=',') - self.stream_stats['lo']

        # Load the features
        self.stream_features = np.loadtxt(stream_features_path, delimiter=',')
        # self.stream_X_train = features[self.train_nodes, :]
        # self.stream_X_val = features[self.val_nodes, :]
        self.stream_y_train = self.stream_labels[self.stream_train_nodes]
        self.stream_y_val = self.stream_labels[self.stream_val_nodes]

        # Load adjacency list
        lo = self.stream_stats['lo']
        with open(stream_adj_list_path, 'rb') as fp:
            adj_list = pickle.load(fp)
            out_adj_list = {}
            for node, adj in adj_list.items():
                out_adj_list[node-lo] = (np.array(adj_list[node])-lo).tolist()
            self.stream_adj_list = out_adj_list

        # print(self.X_train.shape, self.X_val.shape, self.y_train.shape, self.y_val.shape)

    """
    Load relations for a given stream. 
    The loaded relations are stored in self.relations[relation]
    * Note: Only the specified stream is loaded.
    * Note: All stream retrieved using this method has indices starting from 0
    Must call load_stream first.
    """
    def load_stream_relations(self, t=None, relations=['upu', 'usv', 'uvu']):
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


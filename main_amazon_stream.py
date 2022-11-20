import sys
import os
import torch
import random
import logging
import time
import math
import numpy as np 
from tqdm import tqdm
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score, precision_score

import utils
from models.graph_sage import GraphSAGE
from models.ewc import EWC
from models.fraudre import Fraudre
from models.fraudre import create_fraudre
from handlers.stream_data_handler import StreamDataHandler
from handlers.amazon_stream_data_handler import AmazonStreamDataHandler
from handlers.model_handler import ModelHandler
from extensions import detection
from extensions import memory_handler

MODEL_PKL_FILE = 'fraudre.pkl'


def train(data, model, args):
    # Model training
    times = []
    it = range(args.num_epochs) if args.log_only else tqdm(range(args.num_epochs))
    for epoch in it:
        losses = 0
        start_time = time.time()

        nodes = data.train_nodes
        np.random.shuffle(nodes)
        for batch in range(len(nodes) // args.batch_size):
            batch_nodes = nodes[batch * args.batch_size : (batch + 1) * args.batch_size]
            # print(batch_nodes)
            batch_labels = torch.LongTensor(data.labels[np.array(batch_nodes)]).to(args.device)

            model.optimizer.zero_grad()
            
            loss = model.loss(batch_nodes, batch_labels, args.skip_ewc)
            loss.backward()
            model.optimizer.step()

            loss_val = loss.data.item()
            losses += loss_val * len(batch_nodes)
            if (np.isnan(loss_val)):
                logging.error('Loss is NaN')
                sys.exit()
        
        if epoch % 10 == 0:
            logging.debug('--------- Epoch: ' + str(epoch) + ' ' + str(np.round(losses / data.train_size, 10)))
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.round(np.mean(times), 6)
    logging.info("Average epochs time: " + str(avg_time))
    return avg_time


def run(args, t):
    # Data loader
    data = AmazonStreamDataHandler(data_name=args.data, t=t, max_detect_size=args.max_detect_size)
    data.load()
    data.load_stream()
    data.load_streams_relations()
    
    if args.normalize:
        data.features = utils.normalize(data.features)
    logging.info(f'Number of nodes: {len(data.adj_lists)}, edges: {sum([len(v) for v in data.adj_lists.values()]) / 2}')
    logging.info(f'Data: {data.data_name}; Size: {data.data_size}; Data range: [{data.lo}, {data.hi}]; Train size: {len(data.train_nodes)}; Valid size: {len(data.val_nodes)}')


    if args.new_ratio > 0.0 and t > 0:
        data.train_new_nodes_list = detection.detect(data, t, args)
        data.train_nodes = list(set(data.train_nodes + data.train_new_nodes_list)) if len(data.train_new_nodes_list) > 0 else data.train_nodes
    else:
        detect_time = 0
    
    # FRADURE definition & initialization
    gnn_model = create_fraudre(args, data)
    if args.cuda:
        gnn_model.cuda()

    # fradure = GraphSAGE(layers, data.features, data.adj_lists, args)

    # Load model parameters from file
    if t > 0:
        model_handler_pre = ModelHandler(os.path.join(args.save_path, str(t - 1)))
        if not model_handler_pre.not_exist():
            gnn_model.load_state_dict(model_handler_pre.load(args.model_pkl_file))
        
    if t > 0:
        ewc_model = EWC(gnn_model, args.ewc_lambda, args.ewc_type).to(args.device)

        # Whether to use memory to store important nodes
        if args.memory_size > 0:
            memory_h = memory_handler.load('M', args)
            # Forget control
            important_nodes_list = [i for i in memory_h.memory if (i >= data.lo and i <= data.hi)]
            data.train_nodes = list(set(data.train_nodes + important_nodes_list))
            logging.info('Important Data Size: ' + str(len(important_nodes_list)) + ' / ' + str(len(data.train_nodes)))
        else:
            important_nodes_list = data.train_old_nodes_list
            
        # Calculate weight importance
        ewc_model.register_ewc_params(important_nodes_list, torch.LongTensor(data.labels[important_nodes_list]).to(args.device))
    else:
        if args.memory_size > 0:
            memory_h = memory_handler.MemoryHandler(args)
        ewc_model = gnn_model.to(args.device)
            
    # Train
    # ewc_model.optimizer = torch.optim.SGD(gnn_model.parameters(), lr = args.learning_rate)
    ewc_model.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.learning_rate, weight_decay=args.lambda_1)
    avg_time = train(data, ewc_model, args)

    # Evaluate only on the current stream
    logging.info('Evaluating on current stream')
    evaluate(args, t, data=data, gnn_model=gnn_model, val_nodes=data.stream_val_nodes, stream_mode=True)

    # Model save
    model_handler_cur = ModelHandler(os.path.join(args.save_path, str(t)))
    model_handler_cur.save(gnn_model.state_dict(), args.model_pkl_file)

    # Memory save
    if args.memory_size > 0:
        train_output = gnn_model.to_prob(data.train_nodes).cpu().detach().numpy()
        memory_h.update(data.train_nodes, x=train_output, y=data.labels, adj_lists=data.adj_lists)
        memory_handler.save(memory_h, 'M')

    return avg_time, gnn_model, data


def evaluate(args, t, data=None, gnn_model=None, val_nodes=None, stream_mode=False):
    if not stream_mode and (data is None or (args.val_stream_size is not None and data.max_detect_size != args.val_stream_size)):
        # Data loader
        max_detect_size = args.max_detect_size if args.val_stream_size is None else args.val_stream_size
        data = AmazonStreamDataHandler(data_name=args.data, t=t, max_detect_size=max_detect_size)
        data.load()
        data.load_stream()
        data.load_streams_relations()
        if args.normalize:
            data.features = utils.normalize(data.features)
        gnn_model = create_fraudre(args, data)
        if args.cuda:
            gnn_model.cuda()
    if gnn_model is None:
        gnn_model = create_fraudre(args, data)
        if args.cuda:
            gnn_model.cuda()

    # Load model
    model_handler_cur = ModelHandler(os.path.join(args.save_path, str(t)))
    gnn_model.load_state_dict(model_handler_cur.load(args.model_pkl_file))

    if val_nodes is None:
        val_nodes = data.val_nodes

    if len(val_nodes) == 0:
        return 0, 0

    valid_output = gnn_model.to_prob(val_nodes).data.cpu().numpy().argmax(axis=1)
    f1, acc = utils.node_classification(data.labels[val_nodes], valid_output, '')    

    return f1, acc


if __name__ == "__main__":
    args = utils.parse_argument()
    if args.eval:
        logging.basicConfig(level = logging.WARNING, format = '%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s') # filename = 'log', 
    utils.print_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = utils.check_device(args.cuda)
    args.model_pkl_file = MODEL_PKL_FILE
    args.train_type = 'amazon_stream'

    # f1, acc, time, detect_time
    print_ans = ['', '', '', '']
    avg_ans = [0.0, 0.0, 0.0, 0.0]

    t_num = len(os.listdir(os.path.join('./data', args.data, 'streams')))
    args.save_path = os.path.join('./res', args.data)
    
    t_begin = 0
    if args.check_point is not None:
        t_begin = args.check_point
    # The training for each stream
    for t in range(t_begin, t_num):
        logging.info('-------- Time ' + str(t) + ' --------')
        if args.eval == False:
            if args.cuda:
                torch.cuda.empty_cache()
            b, gnn_model, data = run(args, t)
            logging.info('Evaluating on all streams')
            a = evaluate(args, t, data=data, gnn_model=gnn_model)
        else:
            b = 0
            a = evaluate(args, t)
        print_ans[0] += str(a[0]) + '\t'
        print_ans[1] += str(a[1]) + '\t'
        print_ans[2] += str(b) + '\t'
        avg_ans[0] += a[0]
        avg_ans[1] += a[1]
        avg_ans[2] += b


    print('F1:\t', print_ans[0])
    print('Accuracy:\t', print_ans[1])
    print('Time:\t', print_ans[2])
    print(np.round(avg_ans[0] / (t_num-t_begin), 6), np.round(avg_ans[1] / (t_num-t_begin), 6), np.round(avg_ans[2] / (t_num-t_begin), 6))

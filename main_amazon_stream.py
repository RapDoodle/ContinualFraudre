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
    for epoch in tqdm(range(args.num_epochs)):
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
    logging.info('Data: ' + data.data_name + '; Data size: ' + str(data.data_size) \
    + '; Train size: ' + str(len(data.train_nodes)) \
    + '; Valid size: ' + str(len(data.val_nodes)))


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
            important_nodes_list = memory_h.memory
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

    # Model save
    model_handler_cur = ModelHandler(os.path.join(args.save_path, str(t)))
    model_handler_cur.save(gnn_model.state_dict(), args.model_pkl_file)

    # Memory save
    if args.memory_size > 0:
        train_output = gnn_model.to_prob(data.train_nodes).cpu().detach().numpy()
        memory_h.update(data.train_nodes, x=train_output, y=data.labels, adj_lists=data.adj_lists)
        memory_handler.save(memory_h, 'M')

    return avg_time


def evaluate(args, t):
    # Data loader
    data = AmazonStreamDataHandler(data_name=args.data, t=t, max_detect_size=args.max_detect_size)
    data.load()
    if args.normalize:
        data.features = utils.normalize(data.features)

    data.load_stream()
    data.load_streams_relations()
    gnn_model = create_fraudre(args, data)
    if args.cuda:
        gnn_model.cuda()

    # Load model
    model_handler_cur = ModelHandler(os.path.join(args.save_path, str(t)))
    gnn_model.load_state_dict(model_handler_cur.load(args.model_pkl_file))

    val_nodes = data.val_nodes

    if len(val_nodes) == 0:
        return 0, 0

    # valid_output = gnn_model.forward(val_nodes).data.cpu().numpy().argmax(axis=1)
    # f1, acc = utils.node_classification(data.labels[val_nodes], valid_output, '')
    gnn_prob = gnn_model.to_prob(val_nodes, train_flag = False)

    auc_gnn = roc_auc_score(data.labels[val_nodes], gnn_prob.data.cpu().numpy()[:,1].tolist())
    precision_gnn = precision_score(data.labels[val_nodes], gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    a_p = average_precision_score(data.labels[val_nodes], gnn_prob.data.cpu().numpy()[:,1].tolist())
    recall_gnn = recall_score(data.labels[val_nodes], gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    f1 = f1_score(data.labels[val_nodes], gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

    #print(gnn_prob.data.cpu().numpy().argmax(axis=1))

    print(f"GNN auc: {auc_gnn:.4f}")
    print(f"GNN precision: {precision_gnn:.4f}")
    print(f"GNN a_precision: {a_p:.4f}")
    print(f"GNN Recall: {recall_gnn:.4f}")
    print(f"GNN f1: {f1:.4f}")
    acc = 0

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
    
    # The training for each stream
    for t in range(0, t_num):
        logging.info('-------- Time ' + str(t) + ' --------')
        if args.eval == False:
            if args.cuda:
                torch.cuda.empty_cache()
            b = run(args, t)
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
    print(np.round(avg_ans[0] / t_num, 6), np.round(avg_ans[1] / t_num, 6), np.round(avg_ans[2] / t_num, 6))

import os
import re
import json
import pickle
import random
import datetime
import argparse
import logging
import shutil
import collections
import multiprocessing
from itertools import combinations

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Configure the logger
FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

def y_n_choice(msg='Do you want to continue?', default_choice=False):
    default_choice_str = 'yes' if default_choice else 'no'
    while (True):
        choice = input(f'{msg} (default: {default_choice_str}) [Y/n] ')
        # Case: default
        if choice == '':
            return default_choice
        # Case: confirm
        if choice.lower() == 'y':
            return True
        # Case: reject
        if choice.lower() == 'n':
            return False
        # Case: unkown, continue


def create_dataset_sorted_by_time(args):
    fn_comps = args.filename.split('.')
    out_fn = f'{fn_comps[-2]}_sorted.{fn_comps[-1]}'
    args.sorted_filename = out_fn
    args.sorted_filepath = os.path.join(args.origin_path, out_fn)

    if not os.path.exists(args.sorted_filepath):
        logging.info(f'Creating a sorted version of {args.filename}...')
        unix_review_time_extract_regex = re.compile(r'(?<=(\"unixReviewTime\": ))\d{1,}')

        unix_review_times = []
        index_mapping = {}

        with open(os.path.join(args.origin_path, args.filename), 'r') as f:
            index_mapping[0] = 0
            i = 1
            while (line := f.readline().rstrip()):
                unix_review_time_match = re.search(unix_review_time_extract_regex, line)
                unix_review_times.append(unix_review_time_match.group(0))

                # Currently at the beginning of the next line
                # Mark down the position for quick seek
                index_mapping[i] = f.tell()
                i = i + 1

        # Obtain the data sorted by unixReviewTime
        sorted_idx = [x for _, x in sorted(zip(unix_review_times, [i for i in range(len(unix_review_times))]))]

        with open(args.sorted_filepath, 'w') as f_out:
            with open(os.path.join(args.origin_path, args.filename), 'r') as f_in:
                for idx in sorted_idx:
                    f_in.seek(index_mapping[idx])
                    f_out.write(f_in.readline())

        logging.info(f'Successfully created {out_fn}')


def nlp_parse(args):
    fn_comps = args.filename.split('.')
    out_fn = f'{fn_comps[-2]}_parsed.{fn_comps[-1]}'
    args.parsed_filename = out_fn
    args.parsed_filepath = os.path.join(args.origin_path, out_fn)

    if not os.path.exists(args.parsed_filepath):
        logging.info(f'Parsing dataset {args.filename} for natural language processing...')

        # Download stopwords
        nltk.download('stopwords')

        logging.info(f'Reading {args.sorted_filepath}')
        full_df = pd.read_json(args.sorted_filepath, lines=True)
        n = full_df.shape[0]
        
        ps = PorterStemmer()
        corpus = []
        stopwords_set = set(stopwords.words('english'))
        for index, row in full_df.iterrows():
            if index % 1000 == 0:
                print(f'Processing {index}/{n}', end='\r')
            review = f"{row['summary']} {row['reviewText']}"
            review = re.sub('[^a-zA-z]', ' ', review)
            review = review.lower()
            words_list = review.split()
            words_list = [ps.stem(word) for word in words_list if not word in stopwords_set]
            review = ' '.join(words_list)
            corpus.append(review)
        full_df['review'] = corpus
        # full_df.drop(columns=['reviewText', 'summary'], inplace=True)
        full_df.drop(columns=['summary'], inplace=True)
        
        logging.info(f'Writing {args.parsed_filename}')
        full_df.to_json(args.parsed_filepath, lines=True, orient='records')
        logging.info(f'{args.parsed_filename} has been saved successfully.')


def compute_features_list(curr_df, label, lo, hi, args, _):
    if hi - lo <= 0:
        return
    logging.info(f'Computing feature list for {label} [{lo}, {hi}] (size: {hi-lo})')
    # Check path exists
    curr_stream_path = os.path.join(args.dataset_stream_path, label)
    if not os.path.exists(curr_stream_path):
        os.mkdir(curr_stream_path)
    stream_features_list_path = os.path.join(curr_stream_path, 'features_list')

    if not os.path.exists(stream_features_list_path):
        vectorizer = TfidfVectorizer(max_features=args.num_features)
        corpus = curr_df['review'].values
        vectorizer.fit(corpus)

        with open(stream_features_list_path, 'w') as f:
            f.write('\n'.join(vectorizer.get_feature_names()))
    
    logging.info(f'Feature list for {label} is ready.')
    return True


def create_adj_list(edges_set, lo, hi, output_path):
    adj_list = {}
    for v_id in range(lo, hi+1):
        adj_list[v_id] = []
    for v1, v2 in edges_set:
        adj_list[v1].append(v2)
        adj_list[v2].append(v1)
    with open(output_path, 'wb') as f:
        pickle.dump(adj_list, f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_stream(curr_df, label, lo, hi, args, _):
    if hi - lo <= 0:
        return
    logging.info(f'Preprocessing block {label} [{lo}, {hi}] (size: {hi-lo})')

    stream_path = os.path.join(args.dataset_path, 'streams', label)
    
    if not os.path.exists(stream_path):
        os.mkdir(stream_path)

    stream_statistics_path = os.path.join(stream_path, 'statistics')
    stream_edges_path = os.path.join(stream_path, 'edges')
    stream_edges_adj_list_path = os.path.join(stream_path, 'adj_list.pkl')
    stream_features_path = os.path.join(stream_path, 'features')
    stream_features_list_path = os.path.join(stream_path, 'features_list')
    stream_labels_path = os.path.join(stream_path, 'labels')
    stream_train_nodes_path = os.path.join(stream_path, 'train_nodes')
    stream_val_nodes_path = os.path.join(stream_path, 'val_nodes')

    stats = []
    # Record label, lo, hi
    stats.append(f'label={label}\n')
    stats.append(f'lo={lo}\n')
    stats.append(f'hi={hi}\n')
    
    """
    Arguments:
        i: vertex index i
        j: vertex index j
        add_to_local: add to edges_set_local
    """
    def add_edge(i, j, add_to_local=False):
        i = int(i)
        j = int(j)
        pair = (i, j) if i <= j else (j, i)
        edges_set.add(pair)
        if add_to_local:
            edges_set_local.add(pair)

    """
    Reset edges_set_local
    """
    def reset_edges_local():
        edges_set_local = set()
    
    """
    U-P-U: connects users reviewing at least one same product
    """
    def upu():
        reset_edges_local()
        upu_adj_list_path = os.path.join(stream_path, 'upu_adj_list.pkl')

        products = list(set(curr_df['asin'].values))
        n = len(products)
        if n > 0:
            i = 0
            count = 0
            for product in products:
                if i % 500 == 0:
                    print(f'UPU {i}/{n}', end='\r')
                i = i + 1
                node_indices = list(curr_df[curr_df.asin == product].index)
                for node_i, node_j in combinations(node_indices, 2):
                    assert(node_i >= lo)
                    assert(node_i < hi)
                    assert(node_j >= lo)
                    assert(node_j < hi)
                    
                    count = count + 1
                    add_edge(node_i, node_j, True)
            logging.info(f'[{label}] UPU Discovered {count} relations')
            stats.append(f'num_upu_relations={count}\n')
        else:
            logging.info(f'[{label}] UPU Skipped due of 0 product size')
        # Create adjacency list for upu
        create_adj_list(edges_set_local, lo, hi, upu_adj_list_path)
    
    """
    Connects users with top mutual review text similarities (measured by TF-IDF) among all users.
    """
    def uvu():
        reset_edges_local()
        uvu_adj_list_path = os.path.join(stream_path, 'uvu_adj_list.pkl')

        # Use all words to calculate the TF-IDF score
        logging.info(f'[{label}] UVU Calculating TF-IDF scores...')
        vectorizer = TfidfVectorizer()
        corpus = curr_df['review'].values
        tfidf_matrix = vectorizer.fit_transform(corpus)
        cosine_sim = np.triu(cosine_similarity(tfidf_matrix, tfidf_matrix), k=1)
        min_sim_score = np.percentile(cosine_sim[cosine_sim > 0], args.corpus_sim_percentile)
        corpus_indices = np.argwhere(cosine_sim > min_sim_score)
        count = 0
        for ci1, ci2 in corpus_indices:
            if abs(curr_df.iloc[ci1].overall - curr_df.iloc[ci2].overall) <= args.sim_review_max_diff:
                add_edge(lo + ci1, lo + ci2, True)
                count = count + 1
        logging.info(f'[{label}] UVU Discovered {count} relations')
        stats.append(f'num_uvu_relations={count}\n')
        # Create adjacency list for uvu
        create_adj_list(edges_set_local, lo, hi, uvu_adj_list_path)

    """
    Connects users having at least one same star rating within a specified interval 
    """       
    def usv():
        reset_edges_local()
        usv_adj_list_path = os.path.join(stream_path, 'usv_adj_list.pkl')

        # Sliding window approach
        # if hi - lo <= 0:
        i, j, n = 0, 0, curr_df.shape[0]
        i_time, j_time = -1, curr_df.iloc[0]['unixReviewTime'] + args.usv_interval

        # Move j to the appropriate position (initial offset)
        while j < n and curr_df.iloc[j]['unixReviewTime'] < j_time:
            j = j + 1
        if j > 0:
            j = j - 1
        j_time = curr_df.iloc[j]['unixReviewTime']
        prev_j_time = -1
        count = 0
        while j < n:
            if j % 100 == 0:
                print(f'USV {j}/{n}', end='\r')
            i_time = j_time - args.usv_interval
            # Move i to the correct position
            while i < j and curr_df.iloc[i]['unixReviewTime'] < i_time:
                i = i + 1

            assert(curr_df.iloc[j]['unixReviewTime'] - curr_df.iloc[i]['unixReviewTime'] <= args.usv_interval)
            # Create edges for all nodes in the range
            if j_time > prev_j_time and j - i >= 1:
                nodes = [x for x in range(i, j+1)]
                for ni1, ni2 in combinations(nodes, 2):
                    if int(curr_df.iloc[ni1]['overall']) == int(curr_df.iloc[ni2]['overall']):
                        count = count + 1
                        add_edge(lo+ni1, lo+ni2, True)

            if j + 1 < n:
                prev_j_time = j_time
                j_time = curr_df.iloc[j+1]['unixReviewTime']

            j = j + 1
        logging.info(f'[{label}] USV Discovered {count} relations')
        stats.append(f'num_usv_relations={count}\n')
        # Create adjacency list for usv
        create_adj_list(edges_set_local, lo, hi, usv_adj_list_path)
    
    def feature_schema_01():
        stream_features_list_path = os.path.join(stream_path, 'features_list')
        with open(stream_features_list_path, 'r') as f:
            features_list = f.read().split('\n')
        if len(features_list) > args.total_features:
            logging.error('Exceeding the maximum number of allowed features')
        
        tfidf_features = {}
        for i, feature in enumerate(features_list):
            tfidf_features[feature] = i
        
        # Append to features
        features = np.zeros((hi-lo, args.total_features), dtype=int)
        corpus = curr_df['review'].values
        for i, line in enumerate(corpus):
            line_words = line.split()
            for word in line_words:
                if word in tfidf_features and tfidf_features[word] < args.total_features:
                    features[i][tfidf_features[word]] = 1
        with open(stream_features_path, 'w') as feature_fp:
            np.savetxt(feature_fp, features, delimiter=',', fmt='%s', comments='')
        
        assert(i == hi-lo - 1)

    def feature_schema_sentence_embedding():
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Our sentences we like to encode
        sentences = curr_df['reviewText'].values
        embeddings = model.encode(sentences)
        with open(stream_features_path, 'w') as feature_fp:
            np.savetxt(feature_fp, embeddings, delimiter=',', fmt='%s', comments='')

    # Edges
    if not os.path.exists(stream_edges_path):
        logging.info(f'[{label}] Determining the edges')
        edges_set = set()
        edges_set_local = set()

        if 'upu' in args.relation_policy:
            upu()
        if 'usv' in args.relation_policy:
            usv()
        if 'uvu' in args.relation_policy:
            uvu()
        
        # Output as v1,v2 (v1 < v2) for each edge
        with open(stream_edges_path, 'w') as f:
            f.write("".join([f'{v1},{v2}\n' for v1, v2 in list(edges_set)]))

        # Create the adjacency list representation
        create_adj_list(edges_set, lo, hi, stream_edges_adj_list_path)

        
    # Train-test split
    if not os.path.exists(stream_train_nodes_path) or not os.path.exists(stream_val_nodes_path): 
        logging.info(f'[{label}] Splitting training and validation set')
        node_indices = [i for i in range(lo, hi)]
        try:
            random.Random(args.random_seed+int(label)).shuffle(node_indices)
        except:
            logging.error(f'Unable to convert {label} to int')
        split_index = int((hi-lo) * args.train_ratio)
        train_indices = node_indices[0:split_index]
        val_indices = node_indices[split_index:]
        with open(stream_train_nodes_path, 'w') as f:
            for node_index in train_indices:
                f.write(f'{node_index}\n')
        with open(stream_val_nodes_path, 'w') as f:
            for node_index in val_indices:
                f.write(f'{node_index}\n')

    # Write labels
    if not os.path.exists(stream_labels_path):
        logging.info(f'[{label}] Determining labels')
        labels = []
        for ri, row in curr_df.iterrows():
            if row['helpful'][1] == 0:
                y_label = 1 if args.fradulent_by_default else 0
            else:
                y_label = 1 if ((row['helpful'][0] / row['helpful'][1]) <= args.fradulent_threshold) else 0
            labels.append(f'{ri},{y_label}\n')
        with open(stream_labels_path, 'w') as f:
            f.write("".join(labels))

    # Features
    if not os.path.exists(stream_features_path):
        logging.info(f'[{label}] Processing node features')
        if args.feature_schema == '01':
            feature_schema_01()
        elif args.feature_schema == 'sentence_embeddings':
            feature_schema_sentence_embedding()
        else:
            raise Exception('Unknown feature schema')
    
    # Write statistics
    with open(stream_statistics_path, 'w') as f:
        f.write(''.join(stats))

    # Success
    return True


def concat_interleave_stream_nodes(path1, path2, lo, hi):
    out_nodes = []
    def partition(path):
        with open(path, 'r') as f:
            stream_nodes = f.read().strip().split('\n')
        for stream_node in stream_nodes:
            stream_node = int(stream_node)
            if stream_node >= lo and stream_node <= hi:
                out_nodes.append(stream_node)
    partition(path1)
    partition(path2)
    return out_nodes


def preprocess_dataset(args):
    # Read from the sorted and cleaned dataset
    full_df = pd.read_json(args.parsed_filepath, lines=True)
    full_df_size = full_df.shape[0]

    args.dataset_path = os.path.join(args.output_path, args.dataset_name)
    args.dataset_stream_path = os.path.join(args.dataset_path, 'streams')
    if os.path.exists(args.dataset_path):
        if y_n_choice(f'Dataset {args.dataset_name} already exists. Do you want to remove it?', default_choice=False):
            logging.info(f'Removing existing path {args.dataset_path}')
            shutil.rmtree(args.dataset_path, ignore_errors=False)
    if not os.path.exists(args.dataset_path):
        os.mkdir(args.dataset_path)
    if not os.path.exists(args.dataset_stream_path):
        os.mkdir(args.dataset_stream_path)
    
    # Calculate the intervals
    intervals = []
    initial_size = int(full_df_size * args.initial_portion)
    if args.initial_portion > 0:
        intervals.append((0, initial_size))
    stream_size = full_df_size - initial_size
    stream_block_size = int(stream_size / args.num_streams)
    lo = initial_size
    for i in range(args.num_streams):
        if i == args.num_streams - 1:
            hi = full_df_size
        else:
            hi = min((lo + stream_block_size), full_df_size)
        intervals.append((lo, hi))
        lo = hi
        
    if len(intervals) <= 0:
        logging.error('Empty list of intervals')
        exit()
        
    def generate_and_append_interval_func_args(lo, hi, interleave_flag=False):
        func_args.append((full_df.iloc[lo:hi], str(len(func_args)), lo, hi, args, interleave_flag))

    # Calculate the arguments for each parallel call
    func_args = []
    for i in range(0, len(intervals)-1):
        lo = intervals[i][0]
        hi = intervals[i][1]
        generate_and_append_interval_func_args(lo, hi, interleave_flag=False)
        if not args.no_interleave:
            interleave_lo = int((lo+hi) / 2)
            interleave_hi = int((intervals[i+1][0] + intervals[i+1][1]) / 2)
            generate_and_append_interval_func_args(interleave_lo, interleave_hi, interleave_flag=True)
    # The last interval
    generate_and_append_interval_func_args(intervals[-1][0], intervals[-1][1], interleave_flag=False)

    # Execute on multiple processes
    pool = multiprocessing.Pool(args.num_workers)
    exec_result = pool.starmap(compute_features_list, func_args)

    if args.feature_schema == '01':
        tfidf_features = []
        for func_arg in func_args:
            label = func_arg[1]
            stream_features_list_path = os.path.join(args.dataset_path, 'streams', label, 'features_list')
            with open(stream_features_list_path, 'r') as f:
                curr_features = f.read().split('\n')

            if len(tfidf_features) == 0:
                new_features = curr_features
            else:
                prev_set = set(tfidf_features)
                curr_set = set(curr_features)
                new_features = list(curr_set.difference(prev_set.intersection(curr_set)))
                logging.info(f'Newly added words {str(new_features)}')

            tfidf_features.extend(new_features)
            curr_feature_size = len(tfidf_features)
            
            if curr_feature_size > args.total_features:
                logging.warning('Warning: exceeding maximum number of features')

            with open(stream_features_list_path, 'w') as f:
                f.write('\n'.join(tfidf_features[0:min(args.total_features, curr_feature_size)]))

    exec_result = pool.starmap(generate_stream, func_args)
    
    # Merge features, labels, train_nodes, and val_nodes 
    features_path = os.path.join(args.dataset_path, 'features')
    labels_path = os.path.join(args.dataset_path, 'labels')
    train_nodes_path = os.path.join(args.dataset_path, 'train_nodes')
    val_nodes_path = os.path.join(args.dataset_path, 'val_nodes')

    features_fp = open(features_path, 'w')
    labels_fp = open(labels_path, 'w')
    train_nodes_fp = open(train_nodes_path, 'w')
    val_nodes_fp = open(val_nodes_path, 'w')

    logging.info('Merging streams')
    for i, func_arg in enumerate(func_args):
        label = func_arg[1]
        interleave_flag = func_arg[5]

        stream_path = os.path.join(args.dataset_path, 'streams', label)
        stream_train_nodes_path = os.path.join(stream_path, 'train_nodes')
        stream_val_nodes_path = os.path.join(stream_path, 'val_nodes')

        # Fix: Interleaving streams should not have independent train_nodes and val_nodes
        if interleave_flag:
            prev_label = func_args[i-1][1]
            next_label = func_args[i+1][1]

            stream_lo = func_arg[2]
            stream_hi = func_arg[3]

            prev_stream_path = os.path.join(args.dataset_path, 'streams', prev_label)
            prev_stream_train_nodes_path = os.path.join(prev_stream_path, 'train_nodes')
            prev_stream_val_nodes_path = os.path.join(prev_stream_path, 'val_nodes')
            next_stream_path = os.path.join(args.dataset_path, 'streams', next_label)
            next_stream_train_nodes_path = os.path.join(next_stream_path, 'train_nodes')
            next_stream_val_nodes_path = os.path.join(next_stream_path, 'val_nodes')

            train_nodes = concat_interleave_stream_nodes(prev_stream_train_nodes_path, next_stream_train_nodes_path, stream_lo, stream_hi)
            val_nodes = concat_interleave_stream_nodes(prev_stream_val_nodes_path, next_stream_val_nodes_path, stream_lo, stream_hi)

            with open(stream_train_nodes_path, 'w') as f:
                for node_index in train_nodes:
                    f.write(f'{node_index}\n')
            with open(stream_val_nodes_path, 'w') as f:
                for node_index in val_nodes:
                    f.write(f'{node_index}\n')
            continue

        stream_path = os.path.join(args.dataset_path, 'streams', label)
        stream_features_path = os.path.join(stream_path, 'features')
        stream_labels_path = os.path.join(stream_path, 'labels')

        with open(stream_features_path, 'r') as f:
            features_fp.write(f.read())
        with open(stream_labels_path, 'r') as f:
            labels_fp.write(f.read())
        with open(stream_train_nodes_path, 'r') as f:
            train_nodes_fp.write(f.read())
        with open(stream_val_nodes_path, 'r') as f:
            val_nodes_fp.write(f.read())
        
    features_fp.close()
    labels_fp.close()
    train_nodes_fp.close()
    val_nodes_fp.close()

    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        '-f',
        metavar='DIR',
        required=True,
        type=str,
        help='The filename to the original Amazon dataset. Original datasets should be placed in /data/origin'
    )
    parser.add_argument(
        '--dataset-name',
        '-n',
        metavar='STRING',
        required=True,
        type=str,
        help='The dataset\'s output name.'
    )
    parser.add_argument(
        '--initial-portion',
        type=float,
        default=0.0,
        help='The portion for the initial training set.'
    )
    parser.add_argument(
        '--num-streams',
        type=int,
        default=25,
        help='The number of streams.'
    )
    parser.add_argument(
        '--output-path',
        metavar='DIR',
        default='./data',
        type=str,
        help='Path to store the dataset.'
    )
    parser.add_argument(
        '--fradulent-threshold',
        type=float,
        default=0.2,
        help='Upper bound of the percentage of "up" votes required to be deemed as fradulent. The argument should be between 0 and 1'
    )
    parser.add_argument(
        '--fradulent-by-default',
        action='store_true',
        default=False,
        help='Consider a review as fradulent if no vote is provided.'
    )
    parser.add_argument(
        '--no-interleave',
        action='store_true',
        default=False,
        help='Disable generating interleaves between streams.'
    )
    parser.add_argument(
        '--relation-policy',
        metavar='STRING',
        default='upu,usv,uvu',
        type=str,
        help='Relation schema separated by comma (,).'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed.'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='The percentage of train data.'
    )
    parser.add_argument(
        '--feature-schema',
        metavar='STRING',
        default='01',
        type=str,
        help='Schema used to extract features.',
        choices=['01', 'sentence_embeddings']
    )
    parser.add_argument(
        '--num-features',
        type=int,
        default=100,
        help='Number of features for each TF-IDF vectorizer fit.'
    )
    parser.add_argument(
        '--total-features',
        type=int,
        default=1000,
        help='Total number of features.'
    )
    parser.add_argument(
        '--corpus-sim-percentile',
        type=float,
        default=95,
        help='Corpus similarity percentile. Should be a float between 0 and 100.0.'
    )
    parser.add_argument(
        '--sim-review-max-diff',
        type=int,
        default=1,
        help='Tolerance for difference in rating.'
    )
    parser.add_argument(
        '--usv-interval',
        type=int,
        default=259200,
        help='USV interval. Unit: Unix epoch time.'
    )
    parser.add_argument(
        '--origin-path',
        metavar='DIR',
        default='./data/origin',
        type=str,
        help='Path to the origin dataset.'
    )
    parser.add_argument(
        '--num-workers',
        default=None,
        type=int,
        help='Path to the origin dataset.'
    )
    args = parser.parse_args()

    # Preprocess arguments
    # relation_policy
    args.relation_policy = args.relation_policy.split(',')
    for policy in args.relation_policy:
        if policy not in ['upu', 'usv', 'uvu']:
            logging.error(f'Unknown relation policy named {policy}')
    # num_workers
    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()
        logging.info(f'Default number of workers set to {args.num_workers}')

    create_dataset_sorted_by_time(args)
    nlp_parse(args)

    preprocess_dataset(args)


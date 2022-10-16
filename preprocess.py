import os
import re
import random
import datetime
import argparse
import logging
import shutil
from itertools import combinations

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

# FILENAME = 'Musical_Instruments_5.json'
# FILENAME = 'Apps_for_Android_5.json'
# DATASET_NAME = 'amazon_android_apps'
# INITIAL_PORTION = 0.0 # A number between 0 and 1
# STREAMS = 75 -> NUM_STREAMS

# FRAUDULENT_THRESHOLD = 0.2
# FRAUDULENT_ON_DEFAULT = False

# RELATION_POLICY = ['upu', 'usv', 'uvu']
# RANDOM_SEED = 42
# TRAIN_RATIO = 0.8

# FEATURE_SCHEMA = '01'

# MAX_FEATURES = 500

# CORPUS_SIM_PERCENTILE = 99.95
# SIM_REVIEW_MAX_DIFF = 1

# USV_INTERVAL = 259200


# helpful_vote_regex = re.compile(r'(?<=(\"helpful\": \[))\d{1,}, \d{1,}(?=(\]))')
# int_extract_regex = re.compile(r'\d{1,}')


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
        full_df.drop(columns=['reviewText', 'summary'], inplace=True)
        
        logging.info(f'Writing {args.parsed_filename}')
        full_df.to_json(args.parsed_filepath, lines=True, orient='records')
        logging.info(f'{args.parsed_filename} has been saved successfully.')


def preprocess_dataset(args):
    # Read from the sorted and cleaned dataset
    full_df = pd.read_json(args.parsed_filepath, lines=True)
    full_df_size = full_df.shape[0]

    args.dataset_path = os.path.join(args.output_path, args.dataset_name)
    args.dataset_stream_patht = os.path.join(args.dataset_path, 'streams')
    if os.path.exists(args.dataset_path):
        if y_n_choice(f'Dataset {args.dataset_name} already exists. Do you want to remove it?', default_choice=False):
            logging.info(f'Removing existing path {args.dataset_path}')
            shutil.rmtree(args.dataset_path, ignore_errors=False)
        else:
            logging.info(f'Operation aborted.')
            exit()
    os.mkdir(args.dataset_path)
    os.mkdir(args.dataset_stream_patht)

    features_path = os.path.join(args.dataset_path, 'features')
    labels_path = os.path.join(args.dataset_path, 'labels')
    train_nodes_path = os.path.join(args.dataset_path, 'train_nodes')
    val_nodes_path = os.path.join(args.dataset_path, 'val_nodes')

    def generate_stream(label, lo, hi, tfidf_features=None):
        if hi - lo <= 0:
            return
        logging.info(f'Preprocessing block {label} [{lo}, {hi}] (size: {hi-lo})')
        curr_df = full_df.iloc[lo:hi]

        stream_path = os.path.join(args.dataset_path, 'streams', label)
        os.mkdir(stream_path)

        stream_edges_path = os.path.join(stream_path, 'edges')
        stream_features_path = os.path.join(stream_path, 'features')
        stream_labels_path = os.path.join(stream_path, 'labels')
        stream_train_nodes_path = os.path.join(stream_path, 'train')
        stream_val_nodes_path = os.path.join(stream_path, 'val')
        
        # Edges
        logging.info('Determining the edges')
        edges = []
        edges_set = set()
        
        def add_edge(i, j):
            pair = (i, j) if i <= j else (j, i)
            if pair not in edges_set:
                edges_set.add(pair)
                edges.append(f'{i},{j}\n')
        
        """
        U-P-U: connects users reviewing at least one same product
        """
        def upu():
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
                        add_edge(node_i, node_j)
                logging.info(f'UPU Discovered {count} relations')
            else:
                logging.info('UPU Skipped due of 0 product size')
        
        """
        Connects users with top mutual review text similarities (measured by TF-IDF) among all users.
        """
        def uvu():
            # Use all words to calculate the TF-IDF score
            logging.info('UVU Calculating TF-IDF scores...')
            vectorizer = TfidfVectorizer()
            corpus = curr_df['review'].values
            tfidf_matrix = vectorizer.fit_transform(corpus)
            cosine_sim = np.triu(cosine_similarity(tfidf_matrix, tfidf_matrix), k=1)
            min_sim_score = np.percentile(cosine_sim[cosine_sim > 0], args.corpus_sim_percentile)
            corpus_indices = np.argwhere(cosine_sim > min_sim_score)
            count = 0
            for ci1, ci2 in corpus_indices:
                if abs(curr_df.iloc[ci1].overall - curr_df.iloc[ci2].overall) <= args.sim_review_max_diff:
                    add_edge(lo + ci1, lo + ci2)
                    count = count + 1
            logging.info(f'UVU Discovered {count} relations')

        """
        Connects users having at least one same star rating within a specified interval 
        """       
        def usv():
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
                if i % 500 == 0:
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
                            add_edge(lo+ni1, lo+ni2)

                if j + 1 < n:
                    prev_j_time = j_time
                    j_time = curr_df.iloc[j+1]['unixReviewTime']

                j = j + 1
            logging.info(f'USV Discovered {count} relations')
            
        if 'upu' in args.relation_policy:
            upu()
        if 'usv' in args.relation_policy:
            usv()
        if 'uvu' in args.relation_policy:
            uvu()
                
        with open(stream_edges_path, 'a') as f:
            f.write("".join(edges))
        
        def feature_schema_01():
            if tfidf_features is None:
                raise Exception('Argument `tfidf_features` does not contain a dictionary')
            vectorizer = TfidfVectorizer(max_features=args.max_features)
            corpus = curr_df['review'].values
            vectorizer.fit(corpus)
            
            if len(tfidf_features) == 0:
                new_features = vectorizer.get_feature_names()
            else:
                prev_set = set(tfidf_features.keys())
                curr_set = set(vectorizer.get_feature_names())
                new_features = list(curr_set.difference(prev_set.intersection(curr_set)))
                logging.debug('Newly added words', new_features)
            
            curr_feature_size = len(tfidf_features)
            for new_feature_index, word in enumerate(new_features):
                tfidf_features[word] = curr_feature_size + new_feature_index
            
            if len(tfidf_features) > args.total_features:
                logging.warning('Warning: exceeding maximum number of features')
                
            # Append to features
            features = np.zeros((hi-lo, args.total_features), dtype=int)
            for i, line in enumerate(corpus):
                line_words = line.split()
                for word in line_words:
                    if word in tfidf_features and tfidf_features[word] < args.total_features:
                        features[i][tfidf_features[word]] = 1
            with open(features_path, 'a') as feature_fp:
                np.savetxt(feature_fp, features, delimiter=',', fmt='%s', comments='')
            with open(stream_features_path, 'a') as feature_fp:
                np.savetxt(feature_fp, features, delimiter=',', fmt='%s', comments='')
            
            assert(i == hi-lo - 1)

        logging.info('Processing node features')
        if args.feature_schema == '01':
            feature_schema_01()
        else:
            raise Exception('Unknown feature schema')
            
        # Train-test split
        logging.info('Splitting training and validation set')
        node_indices = [i for i in range(lo, hi)]
        random.shuffle(node_indices)
        split_index = int((hi-lo) * args.train_ratio)
        train_indices = node_indices[0:split_index]
        val_indices = node_indices[split_index:]
        with open(train_nodes_path, 'a') as f:
            for node_index in train_indices:
                f.write(f'{node_index}\n')
        with open(stream_train_nodes_path, 'a') as f:
            for node_index in train_indices:
                f.write(f'{node_index}\n')
        with open(val_nodes_path, 'a') as f:
            for node_index in val_indices:
                f.write(f'{node_index}\n')
        with open(stream_val_nodes_path, 'a') as f:
            for node_index in val_indices:
                f.write(f'{node_index}\n')

        # Write labels
        logging.info('Determining labels')
        labels = []
        for ri, row in curr_df.iterrows():
            if row['helpful'][1] == 0:
                y_label = 1 if args.fradulent_by_default else 0
            else:
                y_label = 1 if ((row['helpful'][0] / row['helpful'][1]) <= args.fradulent_threshold) else 0
            labels.append(f'{ri},{y_label}\n')
        with open(labels_path, 'a') as f:
            f.write("".join(labels))
        with open(stream_labels_path, 'a') as f:
            f.write("".join(labels))
    
    initial_size = int(full_df_size * args.initial_portion)
    tfidf_features = {}
    if args.initial_portion > 0:
        generate_stream('0', 0, initial_size, tfidf_features=tfidf_features)
    stream_size = full_df_size - initial_size
    stream_block_size = int(stream_size / args.num_streams)
    lo = initial_size
    for i in range(args.num_streams):
        if i == args.num_streams - 1:
            hi = full_df_size
        else:
            hi = min((lo + stream_block_size), full_df_size)
        if args.initial_portion > 0:
            set_label = str(i+1)
        else:
            set_label = str(i)
        generate_stream(set_label, lo, hi, tfidf_features=tfidf_features)
        lo = hi


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
        choices=['01']
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=500,
        help='Maximum number of features.'
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
        default=99.95,
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
    args = parser.parse_args()

    # Preprocess arguments
    args.relation_policy = args.relation_policy.split(',')
    for policy in args.relation_policy:
        if policy not in ['upu', 'usv', 'uvu']:
            logging.error(f'Unknown relation policy named {policy}')

    create_dataset_sorted_by_time(args)
    nlp_parse(args)

    preprocess_dataset(args)


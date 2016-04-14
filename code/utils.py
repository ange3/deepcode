#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: utils.py
# @Author: Lisa Wang
# @created: Jan 29 2016
#
#==============================================================================
# DESCRIPTION:
# A place to put useful functions, e.g. for transforming data, printing, etc.
#==============================================================================
# CURRENT STATUS: In progress/ working! :) 
#==============================================================================
# USAGE: 
# import utils or from utils import *
#==============================================================================
#
###############################################################################

import numpy as np
import time
import pickle
import random
from constants import *
from sklearn.utils import shuffle


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# taken from lasagne mnist example.

# better version that is flexible in terms of input
# data is a list of matrices for a data set, for example [X, y]
# or [X, next_problem, y]
def iter_minibatches(data, batchsize, shuffle=False):
    X = data[0]
    num_samples = X.shape[0]
    if shuffle:
        indices = np.arange(num_samples)
        indices = shuffle(indices, random_state=0)
    for start_idx in range(0, num_samples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [d[excerpt] for d in data]


def vectorize_data(data_raw):
    """
    inputs: 
        - data_raw of shape (num_samples, num_problems)
    outputs:
        - X, which is the input to the RNN, shape(num_samples, num_timesteps, num_problems * 2)
        - next_problem: for each student, indicates which problem the student is solving next. 
                        one hot encoding : shape (num_samples, num_timesteps, num_problems)
        - truth: for each student, indicates whether student answers problem at next time step
               correctly. shape(num_samples, num_timesteps)

        notice that num_timesteps can be at most num_problems - 1, since we need 
        to predict on the last input, whether student answers next problem correctly.
    """

    num_samples = data_raw.shape[0]
    num_problems = data_raw.shape[1]
    num_timesteps = num_problems - 1

    X = np.zeros((num_samples, num_timesteps, num_problems * 2), dtype=np.bool)
    next_problem = np.zeros((num_samples, num_timesteps, num_problems), dtype=np.int)
    truth = np.zeros((num_samples, num_timesteps), dtype=np.int)

    # Create 3-dimensional input tensor with one-hot encodings for each sample
    # the dimension of each vector for a student i and time t is 2 * num_problems
    # where the first half corresponds to the correctly answered problems and the
    # second half to the incorrectly answered ones.
    for i in xrange(num_samples):
        
        # for the first time step. Done separately so we can populate the output 
        # tensor at the same time, which is shifted back by 1.

        for t in xrange(0,num_timesteps):
            p = t # since timestep t corresponds to problem p where t=p
            if data_raw[i,p] == 1:
                X[i, t, p] = 1 
            else:
                X[i, t, num_problems + p] = 1
            # this is a special case for the synthetic data set, where the next problem 
            # is just the current problem index + 1
            next_problem[i,t, p+1] = 1
            # truth tells us whether student i answers problem at next time step correctly
            truth[i,t] = data_raw[i,t+1] 
    # truth = np.copy(data_raw) 
    # truth = truth[:, 1:]
    return X, next_problem, truth


###############################################################################
# Below utils code for milestone 2: Predict a student's next AST within an HOC.
###############################################################################

def load_traj_counts(hoc_num):
  '''
  Loads txt file with trajectory frequency counts

  Input: hoc_num
  Output: map {traj_id: count}
  '''
  traj_counts = {}
  traj_count_filepath = TRAJ_COUNT_FILEPATH_PRE + str(hoc_num) + TRAJ_COUNT_FILEPATH_POST
  with open(traj_count_filepath, 'rb') as f:
    reader = csv.reader(f, dialect = 'excel' , delimiter = '\t')
    for row in reader:
      traj_counts[int(row[0])] = int(row[1])
  return traj_counts

  
def prepare_traj_data_for_rnn(raw_matrix):
    """
    inputs: 
        - raw_matrix of shape (num_traj, max_traj_len, num_asts)
    outputs:
        - X, which is the input to the RNN, shape(num_traj, num_timesteps, num_asts)
        - y: truth vector. shape(num_traj, num_timesteps, num_asts)
    """
    (num_traj, max_traj_len, num_asts) = raw_matrix.shape
    # notice that num_timesteps can be at most max_traj_len - 1, since we need 
    # to be able to predict on the last input and have a truth value.  
    num_timesteps = max_traj_len - 1

    X = np.copy(raw_matrix[:,:-1,:])

    # y = np.copy(raw_matrix[:,1:,:])

    # alternative version where y has shape (num_traj, num_timesteps)
    # and the values are the indices corresponding to the correct ast prediction
    # instead of one-hot encoding
    y = np.zeros((num_traj, num_timesteps))
    for n in xrange(num_traj):
        for t in xrange(num_timesteps):
            y[n,t] = np.argmax(raw_matrix[n,t+1,:])

    return X, y

def prepare_traj_data_for_rnn_using_embeddings(traj_mat, ast_embeddings, traj_row_to_ast_id_map, embed_ast_id_to_row_map):
    """
    inputs: 
        - raw_matrix of shape (num_traj, max_traj_len, num_asts)
    outputs:
        - X, which is the input to the RNN, shape(num_traj, num_timesteps, num_asts)
        - y: truth vector. shape(num_traj, num_timesteps, num_asts)
    """
    (num_traj, max_traj_len, num_asts) = traj_mat.shape
    # notice that num_timesteps can be at most max_traj_len - 1, since we need 
    # to be able to predict on the last input and have a truth value.  
    num_timesteps = max_traj_len - 1

    num_ast_embeddings, embed_dim = ast_embeddings.shape
    print num_asts
    print num_ast_embeddings

    X = np.zeros((num_traj, num_timesteps, embed_dim))
    y = np.zeros((num_traj, num_timesteps))
    for n in xrange(num_traj):
        for t in xrange(num_timesteps):
            ast_row = np.argmax(traj_mat[n,t,:])
            X[n,t,:] = get_embedding_for_ast(ast_row, ast_embeddings, traj_row_to_ast_id_map, embed_ast_id_to_row_map)
            y[n,t] = np.argmax(traj_mat[n,t+1,:])
    return X, y

def get_embedding_for_ast(traj_mat_ast_row, ast_embeddings, traj_row_to_ast_id_map, embed_ast_id_to_row_map):
    num_ast_embeddings, embed_dim = ast_embeddings.shape
    ast_id = int(traj_row_to_ast_id_map[traj_mat_ast_row])

    if ast_id != -1:
        embed_ast_row = int(embed_ast_id_to_row_map[ast_id])
        return ast_embeddings[embed_ast_row,:].reshape((embed_dim,))
    else:
        return np.zeros((embed_dim,))

    
def prepare_block_data_for_rnn(raw_matrix):
    """
    inputs: 
        - raw_matrix of shape (num_traj, max_traj_len, num_asts)
    outputs:
        - X, which is the input to the RNN, shape(num_asts, num_timesteps, num_blocks)
        - mask, which masks out all the values in the X tensor which don't
            correspond to any sequence. (since some sequences are shorter 
                than num_timesteps), shape(num_asts, num_timesteps)
        - y: truth vector. the correct next block.
               shape(num_asts, num_timesteps)
    """
    (num_asts, max_ast_len, num_blocks) = raw_matrix.shape
    # notice that num_timesteps can be at most max_traj_len - 1, since we need 
    # to be able to predict on the last input and have a truth value.  
    num_timesteps = max_ast_len - 1

    X = np.copy(raw_matrix[:,:-1,:])
    mask = np.ones((num_asts, num_timesteps)).astype('uint8')
    
    for n in xrange(num_asts):
        for t in xrange(num_timesteps-1):
            # if the ast block sequence already ended (as indicated by dummy block
            # at index 0, then mask out) We want to predict only up until the
            # the second to last elem in the sequence
            if X[n,t+1,0] == 1:
                mask[n,t] = 0
        if X[n,num_timesteps-1,0] == 1:
            mask[n,num_timesteps-1] = 0

    # y has shape (num_asts, num_timesteps)
    # and the values are the indices corresponding to the correct ast prediction
    # instead of one-hot encoding
    y = np.zeros((num_asts, num_timesteps))
    for n in xrange(num_asts):
        for t in xrange(num_timesteps):
            y[n,t] = np.argmax(raw_matrix[n,t+1,:])

    return X, mask, y

def convert_data_to_ast_ids(data, row_to_ast_id_map):
    '''
    INPUT:
    data = (X,y)
    X: (batchsize, num_timesteps, num_asts)
    y: (batchsize, num_timesteps)

    OUTPUT:
    X_ast_ids: (batchsize, num_timesteps), containing ast ids, which we can
            use to look up ast json files.
    y_ast_ids: (batchsize, num_timesteps)

    '''
    X, y = data
    batchsize, num_timesteps, num_asts = X.shape
    X_ast_ids = np.zeros((batchsize, num_timesteps))
    y_ast_ids = np.zeros((batchsize, num_timesteps))

    for n in xrange(batchsize):
        for t in xrange(num_timesteps):
            y_ast_ids[n,t] = row_to_ast_id_map[int(y[n,t])]
            X_ast_ids[n,t] = row_to_ast_id_map[np.argmax(X[n,t,:])]

    return X_ast_ids, y_ast_ids

def convert_truth_to_ast_ids(y, row_to_ast_id_map):
    '''
    INPUT:
    y: (batchsize, num_timesteps)

    OUTPUT:
    y_ast_ids: (batchsize, num_timesteps)
    '''
    batchsize, num_timesteps = y.shape
    y_ast_ids = np.zeros((batchsize, num_timesteps))

    for n in xrange(batchsize):
        for t in xrange(num_timesteps):
            y_ast_ids[n,t] = row_to_ast_id_map[int(y[n,t])]

    return y_ast_ids

def convert_ast_or_block_data_to_ids(X, y, row_to_id_map):

    '''
    INPUT:
    data = (X,y)
    X: (batchsize, num_timesteps, num_asts)
    y: (batchsize, num_timestep)

    OUTPUT:
    X_ast_ids: (batchsize, num_timesteps), containing ast ids, which we can
            use to look up ast json files.
    y_ast_ids: (batchsize, num_timesteps)

    '''
    batchsize, num_timesteps, _ = X.shape
    X_ids = np.zeros((batchsize, num_timesteps))
    y_ids = np.zeros((batchsize, num_timesteps))

    for n in xrange(batchsize):
        for t in xrange(num_timesteps):
            y_ids[n,t] = row_to_id_map[int(y[n,t])]
            X_ids[n,t] = row_to_id_map[np.argmax(X[n,t,:])]

    return X_ids, y_ids



def convert_pred_to_ast_ids(pred, row_to_ast_id_map):
    batchsize, num_timesteps, num_asts = pred.shape
    pred_ast_ids = np.zeros((batchsize, num_timesteps))
    for n in xrange(batchsize):
        for t in xrange(num_timesteps):
            pred_ast_ids[n,t] = row_to_ast_id_map[np.argmax(pred[n,t,:])]
    
    return pred_ast_ids

def get_train_val_test_split(data, split=(float(7)/8, float(1)/16, float(1)/16)):
    ''' 
    Input: tuple of numpy matrices, first dimension must have same size for 
    all matrices.
    Optional: split: a tuple with 3 values definining the split, the 3 numbers
    must sum up to 1. E.g. (0.5, 0.25, 0.25)
    return train_data, val_data, test_data, each of which is a tuple with the 
    same number of elements as data
    '''
    num_traj = data[0].shape[0]
    train_data = []
    val_data = []
    test_data = []
    for mat in data:
        mat_train = mat[: int(num_traj*split[0])]
        train_data.append(mat_train)
        mat_val =  mat[int(num_traj*split[0]) : int(num_traj*(split[0]+split[1]))]
        val_data.append(mat_val)
        mat_test = mat[int(num_traj*(split[0]+split[1])) : ]
        test_data.append(mat_test)

    train_data = tuple(train_data)
    val_data = tuple(val_data)
    test_data = tuple(test_data)

    return train_data, val_data, test_data

def load_traj_id_to_row_map(hoc_num):
    traj_row_to_id_map = pickle.load(open(TRAJ_ROW_MAP_FILEPATH_PRE + str(hoc_num) + MAP_SUFFIX, "rb"))
    print traj_row_to_id_map.items()[:20]
    traj_id_to_row_map = {v: k for k, v in traj_row_to_id_map.items()}
    # traj_id_to_row_map = pickle.load(open(TRAJ_ROW_MAP_FILEPATH_PRE + str(hoc_num) + MAP_SUFFIX, "rb"))
    return traj_id_to_row_map

def weight_traj_mat_by_counts(hoc_num, traj_mat):
    # deck is a list with trajectory IDs
    deck = np.load(DECK_MAT_PREFIX + str(hoc_num) + MAT_SUFFIX)
    deck = np.array([int(val) for val in deck])
    print 'deck shape: {}'.format(deck.shape)
    print deck[:10]

    traj_id_to_row_map = load_traj_id_to_row_map(hoc_num)
    # print traj_id_to_row_map.items()
    print traj_id_to_row_map[0]


    deck_row_id = np.array([traj_id_to_row_map[traj_id] for traj_id in deck])

    total_samples = len(deck) # should be equal to the number of total students
    print 'total in deck: {}'.format(total_samples)
    highest_row = np.max(deck)
    print 'highest row in deck: {}'.format(highest_row)
    num_traj, num_timesteps, num_asts = traj_mat.shape
    print 'number of distinct trajectories: {}'.format(num_traj)
    weighted_shuffled_traj_mat = np.zeros((total_samples, num_timesteps, num_asts))
    for i, traj_row in enumerate(deck):
        weighted_shuffled_traj_mat[i] = traj_mat[int(traj_row)]

    return weighted_shuffled_traj_mat

def load_dataset_predict_ast(hoc_num=2, data_sz=-1, use_embeddings=False):
    '''
    if use_embeddings is True:
        the X matrices contain embedding vectors 
        for each ast within a trajectory.
        so output shapes would be 
        X.shape = (num_samples, num_timesteps, embedding_size)
        y.shape = (num_samples, num_timesteps)
    else:
        output shapes would be 
        X.shape = (num_samples, num_timesteps, num_asts)
        y.shape = (num_samples, num_timesteps)

    '''
    print('Preparing network inputs and targets, and the ast maps...')
    hoc_num = str(hoc_num)
    data_set = 'hoc' + hoc_num

    # trajectories matrix for a single hoc exercise
    # shape (num_traj, max_traj_len, num_asts)
    # Note that ast_index = 0 corresponds to the <END> token,
    # marking that the student has already finished.
    # The <END> token does not correspond to an AST.
    traj_mat = np.load(TRAJ_MAP_PREFIX + hoc_num + MAT_SUFFIX)
    # if data_sz specified, reduce matrix. 
    # Useful to create smaller data sets for testing purposes.
    if data_sz != -1:
        traj_mat = traj_mat[:data_sz]
    # print 'Trajectory matrix shape {}'.format(traj_mat.shape)

    # shuffle the first dimension of the matrix
    traj_mat = shuffle(traj_mat, random_state=0)
    num_asts = traj_mat.shape[2]
    
    # Load AST ID to Row Map for trajectory matrix
    traj_ast_id_to_row_map = pickle.load(open(TRAJ_AST_MAP_PREFIX + hoc_num + MAP_SUFFIX, "rb" ))
    traj_row_to_ast_id_map = {v: k for k, v in traj_ast_id_to_row_map.items()}

    X, y = None, None
    if use_embeddings:
        # print(AST_EMBEDDINGS_PREFIX + str(hoc_num) + MAT_SUFFIX)
        ast_embeddings = np.load(AST_EMBEDDINGS_PREFIX_STEM + AST_EMBEDDINGS_VARIATION +  str(hoc_num) + MAT_SUFFIX)

        embed_ast_map_file = EMBED_AST_MAP_PREFIX + hoc_num + MAP_SUFFIX
        embed_row_to_ast_id_map = pickle.load(open(embed_ast_map_file, "rb"))
        embed_ast_id_to_row_map = {v: k for k, v in embed_row_to_ast_id_map.items()}

        ast_maps = {
            'traj_ast_id_to_row': traj_ast_id_to_row_map,
            'traj_row_to_ast_id' : traj_row_to_ast_id_map,
            'embed_id_to_row' : embed_ast_id_to_row_map,
            'embed_row_to_id' : embed_row_to_ast_id_map,
        }

        X, y = prepare_traj_data_for_rnn_using_embeddings(traj_mat, \
            ast_embeddings, traj_row_to_ast_id_map, embed_ast_id_to_row_map)
    else:
        ast_maps = {
            'traj_id_to_row': traj_ast_id_to_row_map,
            'traj_row_to_id' : traj_row_to_ast_id_map,
        }

        X, y = prepare_traj_data_for_rnn(traj_mat)

    print ("Inputs and targets done!")
    return  X, y, ast_maps, num_asts



def load_dataset_predict_block_all_hocs():
    '''
    wrapper function to load predict_block data for all hocs together, so we can
    train a single model for all hocs.
    '''
    X_all_hocs = []
    mask_all_hocs = []
    y_all_hocs = []
    hocs_samples_count = []
    split_indices =  []
    hoc_to_indices = {}
    total_count = 0
    for hoc in xrange(1,10):
        train_data, val_data, test_data, all_data, num_timesteps, num_blocks  = load_dataset_predict_block(hoc_num=hoc)
        X, mask, y = all_data
        X_all_hocs.append(X)
        mask_all_hocs.append(mask)
        y_all_hocs.append(y)
        hocs_samples_count.append(all_data[0].shape[0])
        total_count += all_data[0].shape[0]
        split_indices.append(total_count)
    
    # we don't need the last split index for np.split(), otherwise the last 
    # split will be an empty array
    del split_indices[-1]
    X_all_hocs_mat = reduce(lambda a,b: np.concatenate([a,b], axis=0), X_all_hocs)
    mask_all_hocs_mat = reduce(lambda a,b: np.concatenate([a,b], axis=0), mask_all_hocs)
    y_all_hocs_mat = reduce(lambda a,b: np.concatenate([a,b], axis=0), y_all_hocs)

    return X_all_hocs_mat, mask_all_hocs_mat, y_all_hocs_mat, split_indices


def load_dataset_predict_block(hoc_num=7, data_sz=-1):
    print('Preparing network inputs and targets, and the block maps for hoc {}'.format(hoc_num))
    hoc_num = str(hoc_num)
    data_set = 'hoc' + hoc_num
    
    ast_mat = np.load(BLOCK_MAT_PREFIX + hoc_num + BLOCK_LIMIT_TIMESTEPS +  MAT_SUFFIX)

    # if data_sz specified, reduce matrix. 
    # Useful to create smaller data sets for testing purposes.
    if data_sz != -1:
        ast_mat = ast_mat[:data_sz]
    # print 'Trajectory matrix shape {}'.format(ast_mat.shape)

    num_asts, max_ast_len, num_blocks = ast_mat.shape

    all_data = prepare_block_data_for_rnn(ast_mat)

    ast_mat = shuffle(ast_mat, random_state=0)
    train_mat = ast_mat[0:7*num_asts/8,:]
    val_mat =  ast_mat[7*num_asts/8: 15*num_asts/16 ,:]
    test_mat = ast_mat[15*num_asts/16:num_asts,:]

    train_data = prepare_block_data_for_rnn(train_mat)
    val_data = prepare_block_data_for_rnn(val_mat)
    test_data = prepare_block_data_for_rnn(test_mat)

    num_timesteps = train_data[0].shape[1]

    # print ("Inputs and targets done!")
    # return train_data, val_data, test_data, block_id_to_row_map, row_to_block_id_map, num_timesteps, num_blocks
    return train_data, val_data, test_data, all_data, num_timesteps, num_blocks


def save_ast_embeddings(ast_embeddings, hoc_num, description=''):
    if description != '':
        np.save(AST_EMBEDDINGS_PREFIX_STEM + description + '_' + str(hoc_num) + MAT_SUFFIX, ast_embeddings)
    else:
        np.save(AST_EMBEDDINGS_PREFIX_STEM + str(hoc_num) + MAT_SUFFIX, ast_embeddings)

def save_ast_embeddings_for_all_hocs(ast_embeddings, split_indices):
    """ input: matrix with embeddings for asts across all HOCs. 
        We need to split up this matrix by asts, using the split_indices list 
        we created when we concatenated the data across all hocs.
    """
    ast_embeddings_list = np.split(ast_embeddings, split_indices)
    for hoc in xrange(HOC_MIN, HOC_MAX + 1):
        save_ast_embeddings(ast_embeddings_list[hoc - 1], hoc)



def print_timestep_accuracies(avg_acc, acc_per_timestep_list):
    '''
    Print accuracies over multiple timesteps
    '''
    print '\tAverage Accuracy: {:.2f}%'.format(avg_acc*100)
    print '\tTimestep Accuracies:'
    for t, timestep_acc in enumerate(acc_per_timestep_list):
        print '\t\tTimestep: {}, Accuracy: {:.2f}%'.format(t, timestep_acc*100)


def print_accuracies(hoc_num, train_acc_map, val_acc_map, test_acc_map):
    '''
    Input:  hoc_num is an int representing hour of code problem number
          train_acc_map is a map {hoc_num: (train_acc, train_acc_list)} 
              where train_acc is a single float value (average accuracy over all timesteps)
                    train_acc_list is a list of accuracy values at every timestep
          val_acc_map {hoc_num: (val_acc, val_acc_list)}
          test_acc_map {hoc_num: (test_acc, test_acc_list)}

    No output
    '''
    print '*' * 15
    print 'HOC: {}'.format(str(hoc_num))

    train_acc, train_acc_list = train_acc_map[hoc_num]
    val_acc, val_acc_list = val_acc_map[hoc_num]
    test_acc, test_acc_list = test_acc_map[hoc_num]

    print 'Training Accuracies:'
    print_timestep_accuracies(train_acc, train_acc_list)  

    print '\n\nVal Accuracies:'
    print_timestep_accuracies(val_acc, val_acc_list) 

    print '\n\nTest Accuracies:' 
    print_timestep_accuracies(test_acc, test_acc_list) 

def smoothen_data(data, smooth_window=100):
    smooth = []
    for i in xrange(len(data)-smooth_window):
        smooth.append(np.mean(data[i:i+smooth_window]))

    for i in xrange(len(data)-smooth_window, len(data)):
        smooth.append(np.mean(data[i:len(data)]))
    return smooth


if __name__ == "__main__":
    print "You are running utils.py directly, so you must be testing it!"
    hoc_num = 2

    # X, y, ast_maps, num_asts = load_dataset_predict_ast(hoc_num=2, data_sz=-1, use_embeddings=False)
    # convert_data_to_ast_ids((X,y), ast_maps['traj_row_to_ast_id'])
    
    traj_mat = np.load(TRAJ_MAP_PREFIX + str(hoc_num) + MAT_SUFFIX)
    print 'traj_mat shape {}'.format(traj_mat.shape)
    deck_mat = weight_traj_mat_by_counts(hoc_num, traj_mat)
    print 'deck_mat shape {}'.format(deck_mat.shape)

    # load_dataset_predict_block_all_hocs()
    # X, y, ast_maps, num_asts = load_dataset_predict_ast(hoc_num=2, data_sz=-1, use_embeddings=False)
    # print X.shape
    # print y.shape
    # print num_asts
    # X, y, ast_maps, num_asts = load_dataset_predict_ast(hoc_num=2, data_sz=-1, use_embeddings=True)
    # print X.shape
    # print y.shape
    # print num_asts



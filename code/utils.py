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



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# taken from lasagne mnist example.

def iterate_minibatches(X, next_problem, truth, batchsize, shuffle=False):
    assert(X.shape[0] == truth.shape[0])
    assert(X.shape[0] == next_problem.shape[0])
    num_samples = X.shape[0]
    if shuffle:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
    for start_idx in range(0, num_samples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield X[excerpt], next_problem[excerpt], truth[excerpt]


# just a test to make sure that iterate_minibatches works!
# batchsize = 50
# for batch in utils.iterate_minibatches(X_train, next_problem_train, truth_train, batchsize, shuffle=False):
#     X_, next_problem_, truth_  = batch
#     print X_.shape
#     print next_problem_.shape


# To use on synthetic data set
def vectorize_syn_data(data_raw, num_timesteps):
    num_samples = data_raw.shape[0]
    num_problems = data_raw.shape[1]
    X = np.zeros((num_samples, num_timesteps, num_problems * 2), dtype=np.bool)
    y = np.zeros((num_samples, num_timesteps), dtype=np.int)

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
            y[i,t] = p + 1
    corr = np.copy(data_raw) # Correctness indicates which problem a student has answered correctly
    return X, y, corr


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


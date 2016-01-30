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
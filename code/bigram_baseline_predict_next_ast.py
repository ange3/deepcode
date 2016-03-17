#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: bigram_baseline_predict_next_ast.py
# @Author: Angela Sy, Lisa Wang
# @created: Feb 26 2016
#
#==============================================================================
# DESCRIPTION:
# This file calculates the baseline accuracy using simple bigrams
# for the task of predicting the next AST ID in a trajectory.
# Concretely, for every AST ID X, we take the most common bigram that starts with X --> (X, Y) and predict Y.
# This prediction is pre-calculated and defined in the bigram guesses map.
#
# BIGRAMS
# There are 2 types of bigram guesses we use: 
#    unweighted: each pair of AST IDs only contributes to the count of thair pair once
#    weighted: weights the count of each pair by the frequency of that trajectory
# For example, if trajectory 1 had a count of 100 students who used that trajectory,
#    unweighted: every bigram of AST IDs in trajectory 1 get count of 1
#    weighted: every bigram of AST IDs in trajectory 1 get count of 1 * 100
#
# PREDICTION ACCURACY
# Similarly, prediction accuracy can be unweighted or weighted in the same fashion.
#==============================================================================
# CURRENT STATUS: Complete
#==============================================================================
# USAGE: 
# From command line, run python bigram_baseline_predict_next_ast.py
#==============================================================================
#
###############################################################################


# Python libraries
import numpy as np
import random
import sys, os
import csv
from collections import Counter
import pickle
import operator

# our own modules
import utils
from visualize import *
from constants import *
from model_predict_ast import compute_accuracy_given_data_and_predictions
# from model_predict_ast import compute_corrected_acc_on_ast_rows_per_timestep



def count_bigrams(x, hoc_num, using_weighted_traj_counts=False):
  '''
  Input: x matrix (num_trajectories, num_timesteps, num_asts)
  Output: Map { AST ID: {every other AST ID: count} }

  Counts the number of times other AST IDs come after each AST ID
  '''
  num_trajectories, num_timesteps, num_asts = x.shape

  traj_count_weight = 1
  if using_weighted_traj_counts:
    map_traj_row_to_id = pickle.load(open(TRAJ_ROW_MAP_FILEPATH_PRE + str(hoc_num) + TRAJ_ROW_MAP_FILEPATH_POST, 'rb'))
    traj_count_map = utils.load_traj_counts(hoc_num)

  bigram_count_map = {}
  for n in xrange(num_trajectories):
    if using_weighted_traj_counts:
      traj_id = map_traj_row_to_id[n]
      traj_count_weight = traj_count_map[traj_id]
    for t in xrange(num_timesteps):
      if t == 0:
        continue  # continue skips this for loop iteration (i.e. we are not counting bigrams on the first timestep) 
      ast_timestep_t = np.argmax(x[n, t, :])
      prev_ast = np.argmax(x[n, t-1, :])
      if prev_ast not in bigram_count_map:
        bigram_count_map[prev_ast] = Counter()
      bigram_count_map[prev_ast][ast_timestep_t] += traj_count_weight

      # Add count for end token
      if t == num_timesteps:
        if END_TOKEN_AST_ID not in bigram_count_map:
          bigram_count_map[END_TOKEN_AST_ID] = Counter()
        bigram_count_map[ast_timestep_t][END_TOKEN_AST_ID] += traj_count_weight

  return bigram_count_map

def create_bigram_mapping(x, hoc_num, using_weighted_traj_counts):
  '''
  Input: x matrix (num_trajectories, num_timesteps, num_asts)
  Output: Map {AST row => predicted AST row using bigrams}

  Mapping from AST row to succeeding AST row that appears most frequently in training data
  '''
  print 'Creating bigram mappings..'
  bigram_map = {}  
  bigram_count_map = count_bigrams(x, hoc_num, using_weighted_traj_counts)  # returns { AST row: {every other AST row: count} }
  for ast_row_key in bigram_count_map.keys():
    count_map = bigram_count_map[ast_row_key]
    count_tuples = count_map.items()
    most_frequent_successor_ast_row = max(count_tuples, key=operator.itemgetter(1))[0]
    bigram_map[ast_row_key] = most_frequent_successor_ast_row
  print 'Done!'
  return bigram_map

def make_bigram_predictions(x, bigram_map):
  '''
  Input: x matrix (num_trajectories, num_timesteps, num_asts)
        bigram_map: {AST row => predicted AST row using bigrams}

  Output: predictions matrix (num_trajectories, num_timesteps, num_asts) 
  '''
  num_trajectories, num_timesteps, num_asts = x.shape

  predictions = np.zeros((num_trajectories, num_timesteps, num_asts))

  for n in xrange(num_trajectories):
    for t in xrange(num_timesteps):
      ast_row_key = np.argmax(x[n, t, :])
      if ast_row_key in bigram_map:
        predicted_ast_successor = bigram_map[ast_row_key]
        predictions[n, t, predicted_ast_successor] = 1
      # TO-DO: What to do if no bigram prediction?

  return predictions

def make_gold_predictions(x, hoc_num):
  '''
  Input: x matrix (num_trajectories, num_timesteps, num_asts)

  Predicts 'gold solution' AST ID = 0 for every timestep

  Output: predictions matrix (num_trajectories, num_timesteps, num_asts) 
  '''
  num_trajectories, num_timesteps, num_asts = x.shape

  predictions = np.zeros((num_trajectories, num_timesteps, num_asts))

  map_ast_id_to_row = pickle.load( open(AST_ROW_MAP_FILEPATH_PRE + str(hoc_num) + AST_ROW_MAP_FILEPATH_POST, 'rb'))
  gold_solution_ast_row = map_ast_id_to_row['0']

  for n in xrange(num_trajectories):
    for t in xrange(num_timesteps):
      predictions[n, t, gold_solution_ast_row] = 1

  return predictions

def compute_accuracy(x, y, bigram_map, using_bigrams):
  '''
  Input: x matrix (num_trajectories, num_timesteps, num_asts)
         y matrix (num_trajectories, num_timesteps)

  Output: average accuracy value, list of accuracy values per timestep
  '''
  if using_bigrams:
    prediction = make_bigram_predictions(x, bigram_map)
  else:
    prediction = make_gold_predictions(x, hoc_num)
  # acc = 0
  # acc_list = compute_corrected_acc_on_ast_rows_per_timestep(x, y, prediction)
  acc, acc_list = compute_accuracy_given_data_and_predictions(x, y, prediction, compute_acc_per_timestep_bool=True)
  return acc, acc_list


if __name__ == "__main__":
  '''
  Runs through given problems and returns accuracy scores using given method.

  Set the ff:
    START_HOC and END_HOC
    USING_BIGRAMS (bigrams or gold solution method for prediction)
    USING_WEIGHTED_TRAJ_COUNTS (weighted or unweighted bigram files)

  For each of train, val, and test sets for each HOC saves
  (1) a single accuracy float value for each HOC and
  (2) a list of accuracy-per-timestep values
  '''

  START_HOC = 1
  END_HOC = 9

  USING_BIGRAMS = True  # using bigrams or gold prediction
  USING_WEIGHTED_TRAJ_COUNTS = False  # computing bigrams using weighted trajectories or unweighted

  train_acc_map, val_acc_map, test_acc_map = {}, {}, {}

  for hoc_num in xrange(START_HOC, END_HOC+1):
    print 'Loading data for HOC {}...'.format(str(hoc_num))
    # train_data, val_data, test_data, ast_id_to_row_map, row_to_ast_id_map, num_timesteps, num_asts =load_dataset_predict_ast(hoc_num)
    X, y, ast_maps, num_asts = utils.load_dataset_predict_ast(hoc_num)
    train_data, val_data, test_data = utils.get_train_val_test_split((X, y))
    x_train, y_train = train_data
    print 'Sanity check, first values of x_train:'
    print x_train[:5,:5,:5]
    if USING_BIGRAMS:
      print 'INFO: Predicting using bigrams'
      bigram_map = create_bigram_mapping(x_train, hoc_num, USING_WEIGHTED_TRAJ_COUNTS)
    else:
      print 'INFO: Predicting gold solution'
      bigram_map = None
    print 'Calculating accuracies on Train Data...'
    train_acc, train_acc_list = compute_accuracy(x_train, y_train, bigram_map, USING_BIGRAMS)
    x_val, y_val = val_data
    print 'Calculating accuracies on Val Data...'
    val_acc, val_acc_list = compute_accuracy(x_val, y_val, bigram_map, USING_BIGRAMS)
    x_test, y_test = test_data
    print 'Calculating accuracies on Test Data...'
    test_acc, test_acc_list = compute_accuracy(x_test, y_test, bigram_map, USING_BIGRAMS)
    
    train_acc_map[hoc_num] = (train_acc, train_acc_list)
    val_acc_map[hoc_num] = (val_acc, val_acc_list)
    test_acc_map[hoc_num] = (test_acc, test_acc_list)

  print '-----' * 10
  print 'Printing list of accuracies'
  print 'HOC numbers:', START_HOC, ' to ', END_HOC
  print 'Bigram Accuracies:'
  for hoc_num in xrange(START_HOC, END_HOC+1):
    utils.print_accuracies(hoc_num, train_acc_map, val_acc_map, test_acc_map)

  filename = '../baseline_results/bigram_acc_per_timestep_train_test_val_maps.pickle'
  pickle.dump((train_acc_map, val_acc_map, test_acc_map), open (filename, 'wb'))
  print '-- SAVING'
  print filename



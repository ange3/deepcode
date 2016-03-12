#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: utils.py
# @Author: Angela Sy
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
# From command line, run  python bigram_baseline_predict_next_ast.py
#==============================================================================
#
###############################################################################


# Python libraries
import numpy as np
import random
import sys, os
import csv
import matplotlib.pyplot as plt
# allows plots to show inline in ipython notebook
# get_ipython().magic(u'matplotlib inline')

# our own modules
from utils import *
from visualize import *

# Predicted next AST based on bigrams
code_org_data_bigrams_guess_unweighted_map = {
    1 : "../../data/bigrams/AST_guesses_unweighted_1.csv",
    2 : "../../data/bigrams/AST_guesses_unweighted_2.csv",
    3 : "../../data/bigrams/AST_guesses_unweighted_3.csv",
    4 : "../../data/bigrams/AST_guesses_unweighted_4.csv",
    5 : "../../data/bigrams/AST_guesses_unweighted_5.csv",
    6 : "../../data/bigrams/AST_guesses_unweighted_6.csv",
    7 : "../../data/bigrams/AST_guesses_unweighted_7.csv",
    8 : "../../data/bigrams/AST_guesses_unweighted_8.csv",
    9 : "../../data/bigrams/AST_guesses_unweighted_9.csv",
}

code_org_data_bigrams_guess_weighted_map = {
    1 : "../../data/bigrams/AST_guesses_weighted_with_end_token_1.csv",
    2 : "../../data/bigrams/AST_guesses_weighted_with_end_token_2.csv",
    3 : "../../data/bigrams/AST_guesses_weighted_with_end_token_3.csv",
    4 : "../../data/bigrams/AST_guesses_weighted_with_end_token_4.csv",
    5 : "../../data/bigrams/AST_guesses_weighted_with_end_token_5.csv",
    6 : "../../data/bigrams/AST_guesses_weighted_with_end_token_6.csv",
    7 : "../../data/bigrams/AST_guesses_weighted_with_end_token_7.csv",
    8 : "../../data/bigrams/AST_guesses_weighted_with_end_token_8.csv",
    9 : "../../data/bigrams/AST_guesses_weighted_with_end_token_9.csv",
}

# Actual trajectories
truth_labels_trajectories_of_asts_map = {
    1 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_1.csv",
    2 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_2.csv",
    3 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_3.csv",
    4 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_4.csv",
    5 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_5.csv",
    6 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_6.csv",
    7 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_7.csv",
    8 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_8.csv",
    9 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_9.csv",
}

trajectory_count_map = {
  1: "../../data/trajectory_count_files/counts_1.txt",
  2: "../../data/trajectory_count_files/counts_2.txt",
  3: "../../data/trajectory_count_files/counts_3.txt",
  4: "../../data/trajectory_count_files/counts_4.txt",
  5: "../../data/trajectory_count_files/counts_5.txt",
  6: "../../data/trajectory_count_files/counts_6.txt",
  7: "../../data/trajectory_count_files/counts_7.txt",
  8: "../../data/trajectory_count_files/counts_8.txt",
  9: "../../data/trajectory_count_files/counts_9.txt",
}

END_TOKEN = -1

def predict_accuracy(DATA_SET_HOC, DATA_SZ_LIST, weighted_bigrams_bool = True, use_bigrams_prediction_bool = True):
  '''
  Predict accuracy of baseline
  1) Using Bigram Predictions
  2) Using gold guesses (i.e. Guess that the next AST is the correct solution where AST ID = 0)

  Note: Can run this over multiple number of samples (data size)
  '''

  # Create Bigrams Prediction Map
  bigrams_prediction_map = {}
  if weighted_bigrams_bool:
    print 'INFO: Using Weighted Bigrams'
    bigrams_filepath = code_org_data_bigrams_guess_weighted_map[DATA_SET_HOC]
  else:
    print 'INFO: Using Unweighted Bigrams'
    bigrams_filepath = code_org_data_bigrams_guess_unweighted_map[DATA_SET_HOC]
  with open(bigrams_filepath, 'rb') as bigrams_csv_file:
    bigram_reader = csv.reader(bigrams_csv_file, delimiter =',')
    for row in bigram_reader:
      bigrams_prediction_map[int(row[0])] = int(row[1])

  # Load Trajectory Counts (to produce weighted prediction accuracy)
  traj_counts = {}
  traj_count_filepath = trajectory_count_map[DATA_SET_HOC]
  with open(traj_count_filepath, 'rb') as f:
    reader = csv.reader(f, dialect = 'excel' , delimiter = '\t')
    for row in reader:
      traj_counts[int(row[0])] = int(row[1])

  # Load Truth Labels
  trajectories_asts_filepath = truth_labels_trajectories_of_asts_map[DATA_SET_HOC]

  # Predict using Bigrams and Calculate Accuracy vs Truth
  if use_bigrams_prediction_bool:
    print 'INFO: Predicting Using Bigrams'
  else:
    print 'INFO: Predicting Gold Solution'

  for DATA_SZ in DATA_SZ_LIST:
    num_samples = DATA_SZ
    print '*' * 10
    print 'DATA SET: HOC', DATA_SET_HOC
    print 'num samples: ', num_samples
    correct_count = 0
    total_count = 0

    with open(trajectories_asts_filepath, 'rb') as trajectories_csv_file:
      trajectories_reader = csv.reader(trajectories_csv_file, delimiter=',')
      for index, line in enumerate(trajectories_reader):
        if index == num_samples:
          break
        traj_id = int(line[0])
        # Modify trajectory count increments based on whether or not we are weighting bigrams
        if weighted_bigrams_bool:
          traj_count = traj_counts[traj_id]
        else:
          traj_count = 1
        line = line[1:] # ignore first element which is the trajectory ID
        line = np.unique(line)  # ignore consecutive duplicate ASTs in trajectory
        # Iterate through every timestep and compare predicted AST (predicted from previous AST) with actual AST. Including prediction of the end token.
        for timestep in xrange(len(line)+1):
          if timestep == len(line):
            ast = END_TOKEN  # artificially insert end token at last timestep for correct solution
          else:
            ast = int(line[timestep])
          if timestep == 0:  # no prediction for first timestep
            pass
          else:
            if use_bigrams_prediction_bool:
              # Predicting using Bigram Baseline
              if ast == 0:
                continue
              elif ast == prediction_prev_ast:
                correct_count += traj_count
              total_count += traj_count
            else:
              # Predicting using Gold Baseline (AST = 0)
              if ast == 0:
                correct_count += traj_count
              total_count += traj_count

          # Predicting using Bigram Baseline (handle special case)
          if use_bigrams_prediction_bool:
            if ast not in bigrams_prediction_map:  # reached the end of one student's trajectory
              continue
            # otherwise, predict the next AST of that student
            prediction_prev_ast = bigrams_prediction_map[ast]

    accuracy = float(correct_count)/total_count

    print 'Correct: ', correct_count
    print 'Total: ', total_count
    print 'Accuracy: ', accuracy

  return accuracy


if __name__ == "__main__":

  START_HOC = 1
  END_HOC = 9

  DATA_SET_HOC = xrange(START_HOC, END_HOC+1)
  # DATA_SZ_LIST = [100, 1000, 10000, 100000]
  DATA_SZ_LIST = [100000]

  acc_list = []
  for hoc in DATA_SET_HOC:
    acc = predict_accuracy(hoc, DATA_SZ_LIST, weighted_bigrams_bool = True, use_bigrams_prediction_bool = True)
    acc_list.append(acc)

  print '*' * 20
  print 'List of accuracies'
  print 'HOC numbers:', START_HOC, ' to ', END_HOC
  print 'Bigram Accuracies:', acc_list



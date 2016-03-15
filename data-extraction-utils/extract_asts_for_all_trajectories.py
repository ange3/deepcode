#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: extract_asts_for_all_trajectories.py
# @Author: Angela Sy
# @created: Jan 29 2016
#
#==============================================================================
# DESCRIPTION:
# Data extractor for CSV files with list of ASTs
# Creates 3-D numpy matrices with one row per trajectory sample.
# Each AST in this trajectory is encoded at a specific timestep, where each AST is a one-hot encoding at its AST ID index.
#   (num_trajectories, num_timesteps, num_ast)
# 
#==============================================================================
# CURRENT STATUS: Done 
#==============================================================================
# USAGE: 
# Set parameters in main function below (e.g. problem ID number range, clipping settings, etc.)
# In Terminal, run  python extract_asts_for_all_trajectories
#==============================================================================
#
###############################################################################


from os import listdir
from os.path import isfile, join

import numpy as np

import csv, pickle
import time


PATH_TO_ALL_PROBLEMS ='../data/hoc1-9/'

PATH_TO_TRAJECTORIES = '/trajectories/'
PATH_TO_AST_COUNTS = '/asts/counts.txt'

PATH_TO_LARRY_TRAJECTORIES = '../data/trajectory_ast_csv_files/'
LARRY_TRAJECTORIES_FILENAME_PRE = 'Trajectory_ASTs_'
LARRY_TRAJECTORIES_FILENAME_POST = '.csv'

PATH_TO_COUNTS = '../data/trajectory_count_files/'
COUNTS_FILENAME_PRE = 'counts_'
COUNTS_FILENAME_POST = '.txt'

PATH_TO_PROCESSED_DATA = '../processed_data/'

AST_ID_FOR_END_TOKEN = -1
ROW_INDEX_FOR_END_TOKEN = 0


def count_total_asts(hoc_num):
    '''
    Counts the number of unique ASTs present for this problem
    '''
    ast_counts_filename = PATH_TO_COUNTS + COUNTS_FILENAME_PRE + str(hoc_num) + COUNTS_FILENAME_POST
    counts = 0
    with open(ast_counts_filename, 'r') as ac_f:
        for l in ac_f:
            counts += 1
    return counts

def get_set_of_trajs_to_remove(hoc_num, freq_threshold, map_traj_to_count, clip_traj_bool_by_freq, verbose = True):
    '''
    Return set of Trajectory IDs with a count less than or equal to freq_threshold
    Args: hoc_num is the problem number
          freq is the count threshold to determine if the trajectory id will be put in the removal set
    '''

    if verbose and clip_traj_bool_by_freq:
        print '0) Cleaning trajectories'.format(str(hoc_num))
        print 'Get set of trajectories to remove for HOC {}...'.format(hoc_num)
        print 'Clip trajectories with count <= {}'.format(freq_threshold)

    filepath = PATH_TO_COUNTS + COUNTS_FILENAME_PRE + str(hoc_num) + COUNTS_FILENAME_POST

    traj_id_to_remove_set = set()

    with open(filepath, 'rb') as count_file:
        for index, line in enumerate(count_file):
            traj_id, count = line.split() # tokenize over white space
            map_traj_to_count[traj_id] = int(count)  # assume each traj_id is only listed once in file
            if int(count) <= freq_threshold:
                traj_id_to_remove_set.add(traj_id)

    if verbose and clip_traj_bool_by_freq:
        num_original = index
        num_removed = len(traj_id_to_remove_set)
        num_remaining = num_original-num_removed

        num_removed_times_frequency = 0
        for traj_id in traj_id_to_remove_set:
            num_removed_times_frequency += map_traj_to_count[traj_id]  # num_removed_times_frequency == num_removed when frequency_threshold = 1 since each traj_id for removal will have only been seen once
        num_remaining_times_frequency = num_original - num_removed_times_frequency
        print 'Original num trajectories:  {}'.format(num_original)
        print 'Clipped trajectories:  {}'.format(num_removed)
        print 'Remaining trajectories:  {}'.format(num_remaining)
        print 'Percentage trajectories remaining (unweighted by count of traj):  {}%'.format(float(num_remaining)/num_original*100)
        print 'Percentage trajectories remaining weighted by freq: {}%'.format(float(num_remaining_times_frequency)/num_original*100)
    return traj_id_to_remove_set

def extract_asts_for_one_hoc_from_larry_trajectories(hoc_num, clip_traj_bool_by_freq, clip_traj_freq, clip_traj_length, clip_num_traj = -1, verbose = True, save_traj_matrix = True):
    '''
    Extracts a trajectories matrix for one problem as a one-hot encoding of each AST's ID
    Output: (num_trajectories, num_timesteps, num_asts)
        where
            num_trajectories = clipped number of trajectories where trajectory has been seen at least a certain number of times (see get_set_of_trajs_to_remove method),
            num_timesteps = length of the longest trajectory
            num_asts = number of distinct AST IDs in the given trajectories

    clip_traj_bool_by_freq and clip_traj_freq determines removal of trajectories below a certain frequency
    clip_traj_length of -1 is sign post not to clip the trajectory lengths (which is the number of timesteps)
    '''
    # Setup variables
    filepath = PATH_TO_LARRY_TRAJECTORIES + LARRY_TRAJECTORIES_FILENAME_PRE + str(hoc_num) + LARRY_TRAJECTORIES_FILENAME_POST

    # Data structures
    map_traj_to_count = {}
    map_row_index_to_traj_id = {}

    unique_asts_set = set()

    raw_trajectories_list = []
    longest_trajectory_len = -1

    # Clip trajectory IDs with count < frequency threshold
    # Want to run this function whether or not clip_traj_bool_by_freq is True because it populates map_traj_to_count
    traj_id_to_remove_set = get_set_of_trajs_to_remove(hoc_num, clip_traj_freq, map_traj_to_count, clip_traj_bool_by_freq)

    # Extract all trajectories from file and place in raw_trajectories_list
    with open(filepath, 'rb') as traj_file:
        row_index = 0
        for index, line in enumerate(csv.reader(traj_file, delimiter=',')):
            # limit num trajectories loaded
            if clip_num_traj != -1 and row_index == clip_num_traj:
              break
            traj_id = line[0]
            if clip_traj_bool_by_freq and traj_id in traj_id_to_remove_set:
                # print 'skipping ', traj_id
                continue  # is there a faster way to ignore unwanted lines? load all then use set subtraction?
            if clip_traj_length == -1:
                raw_trajectory = np.array(list(line[1:]))
            else:
                raw_trajectory = np.array(list(line[1:1+clip_traj_length]))

            map_row_index_to_traj_id[row_index] = int(traj_id)

            # Clean up trajectories - remove duplicate ASTs
            raw_trajectory_clean = []
            prev_ast = -1
            for ast in raw_trajectory:
                if ast != prev_ast:
                    raw_trajectory_clean.append(ast)
                    unique_asts_set.add(ast)
                    prev_ast = ast

            # If we are not clipping by trajectory length, Get max trajectory length
            if clip_traj_length == -1 and len(raw_trajectory_clean) > longest_trajectory_len:
                longest_trajectory_len = raw_trajectory.shape[0]
                longest_trajectory_id = traj_id
            raw_trajectories_list.append(raw_trajectory_clean)
            row_index += 1

            # To test
            # if traj_id == '2345':
            #     print traj_id
            #     print raw_trajectory_clean
            #     print len(raw_trajectory_clean)

        # save map_row_index_to_traj_id file
        map_traj_row_filename = PATH_TO_PROCESSED_DATA + 'map_traj_row_' + str(hoc_num) + '.pickle'
        pickle.dump(map_row_index_to_traj_id, open( map_traj_row_filename, "wb" ))


    # Create empty trajectories_matrix
    num_trajectories = len(raw_trajectories_list)
    num_unique_asts = len(unique_asts_set)
    num_ast = num_unique_asts + 1  # adding 1 for the end token
    if clip_traj_length == -1:
        num_trajectory_length = longest_trajectory_len
    else:
        num_trajectory_length = clip_traj_length
    trajectories_matrix = np.zeros((num_trajectories, num_trajectory_length, num_ast))

    if verbose:
        print '1) Number of ASTs: {}'.format(num_ast)
        if True:
        # if clip_traj_bool_by_freq:
            print '-- Info: Removed ASTs since some trajectories were clipped'
            num_original = count_total_asts(hoc_num)
            num_removed = num_original-num_unique_asts
            print 'Original num ASTS: {}'.format(num_original)
            print 'Removed ASTS: {}'.format(num_removed)
            print 'Remaining ASTS: {}'.format(num_unique_asts)
            print 'Percentage ASTs remaining (unweighted by count of traj):  {}%'.format(float(num_unique_asts)/num_original*100)

        print '2) Number of Trajectories: {}'.format(num_trajectories)
        if clip_traj_length == -1:
            print 'Max Trajectory Length: {}, Num ASTs at Traj ID: {}'.format(longest_trajectory_len, longest_trajectory_id)
        else:
            print 'Clipped Trajectory Length: {}'.format(num_trajectory_length)


    # Fill in trajectories_matrix
    max_timesteps = 0

    # map AST ID to row index in trajectories_matrix
    map_ast_id_to_row_index = {}
    map_ast_id_to_row_index[AST_ID_FOR_END_TOKEN] = ROW_INDEX_FOR_END_TOKEN

    for traj_file_index, traj in enumerate(raw_trajectories_list):
        for timestep, ast in enumerate(traj):
            # print 'ast = {}'.format(ast)
            if ast not in map_ast_id_to_row_index:
                map_ast_id_to_row_index[ast] = len(map_ast_id_to_row_index)
            ast_index = map_ast_id_to_row_index[ast]
            trajectories_matrix[traj_file_index, timestep, ast_index] = 1
            if timestep == len(traj)-1: # if reached the last timestep for this trajectory, add end token for all remaining timesteps
                trajectories_matrix[traj_file_index, timestep+1:, ROW_INDEX_FOR_END_TOKEN] = 1

    if verbose:
        print '3) Trajectories matrix created and encoded'
        print 'Trajectories Matrix shape: {}'.format(trajectories_matrix.shape)

    if save_traj_matrix:
        # Clipping number of timesteps
        if clip_traj_length == -1:
            # Default
            traj_matrix_filename = PATH_TO_PROCESSED_DATA + 'traj_matrix_' + str(hoc_num)
            map_ast_row_filename = PATH_TO_PROCESSED_DATA + 'map_ast_row_' + str(hoc_num)
        else:
            # Place in special folder (Need to create this folder first in filesystem)
            traj_matrix_filename = PATH_TO_PROCESSED_DATA + str(clip_traj_length) + '_timesteps/traj_matrix_' + str(hoc_num)
            map_ast_row_filename = PATH_TO_PROCESSED_DATA + str(clip_traj_length) + '_timesteps/map_ast_row_' + str(hoc_num)

        # Clipping number of samples
        # if clip_num_traj != -1:
        #     traj_matrix_filename = traj_matrix_filename + '_samples_' + str(clip_num_traj
        #     map_ast_row_filename = map_ast_row_filename + '_samples_' + str(clip_num_traj)

        # Add file suffix
        traj_matrix_filename += '.npy'
        map_ast_row_filename += '.pickle'
        
        if verbose:
            print '--SAVE: Saving trajectories matrix and AST ID map'
            print 'Traj Matrix Filename: {}'.format(traj_matrix_filename)
            print 'Map AST ID to Row Index Filename: {}'.format(map_ast_row_filename)

        # Save
        np.save(traj_matrix_filename, trajectories_matrix)
        pickle.dump(map_ast_id_to_row_index, open( map_ast_row_filename, "wb" ))

        # Load
        # arr = np.load(traj_matrix_filename)
        # print arr.shape
        # map_test = pickle.load(open( map_ast_row_filename, "rb" ))
        # print ast, map_test[ast]


def extract_asts_for_all_hocs(start_problem_id, end_problem_id, clip_traj_bool, clip_traj_freq, clip_traj_length, clip_num_traj = -1, verbose=True, save_traj_matrix=True):
    '''
    Extracts trajectories matrix for all problems from START_PROBLEM_ID to END_PROBLEM_ID inclusive.
    Saves trajectories matrices to a numpy file and AST ID to row index maps to a pickle file.
    '''
    for hoc_num in xrange(start_problem_id, end_problem_id + 1):
        if verbose:
            print '*** INFO: HOC {} ***'.format(hoc_num)

        tic = time.clock()
        extract_asts_for_one_hoc_from_larry_trajectories(hoc_num, clip_traj_bool, clip_traj_freq, clip_traj_length, clip_num_traj, save_traj_matrix)
        toc = time.clock()

        if verbose:
            print 'Finished extracting ASTs from Problem {} in {}s'.format(hoc_num, toc-tic)

def test_extracted_traj_matrix(hoc_num):
    '''
    Testing for HOC 3's first trajectory
    traj_id: 10, trajectory: 12,0  (from csv file)

    Note: Testing not generalized to other HOCs
    '''
    traj_matrix_filename = PATH_TO_PROCESSED_DATA + 'traj_matrix_' + str(hoc_num) + '.npy'
    map_ast_row_filename = PATH_TO_PROCESSED_DATA + 'map_ast_row_' + str(hoc_num) + '.pickle'

    # Load
    traj_matrix = np.load(traj_matrix_filename)
    print 'traj_matrix shape: {}'.format(traj_matrix.shape)
    map_ast_id_to_row_index = pickle.load(open( map_ast_row_filename, "rb" ))
    print 'Number of ASTs: {}'.format(len(map_ast_id_to_row_index))

    # Print first trajectory in matrix
    print traj_matrix[0, :, :]
    print 'FOR HOC 3:'
    print 'first trajectory is 12'
    print 'map_ast_id_to_row_index[AST 12] = {}, map_ast_id_to_row_index[AST 0] = {}'.format(ap_ast_id_to_row_index['12'], map_ast_id_to_row_index['0'])

if __name__ == "__main__":
    START_PROBLEM_ID = 5
    END_PROBLEM_ID = 5

    CLIP_TRAJECTORY_BOOL = False
    CLIP_TRAJ_FREQ = 1
    CLIP_NUM_TRAJ = 10000  # set to -1 if not cutting number of trajectory samples

    SAVE_TRAJ_MATRIX = True

    CLIP_TRAJECTORY_LENGTH = 10
    extract_asts_for_all_hocs(START_PROBLEM_ID, END_PROBLEM_ID, CLIP_TRAJECTORY_BOOL, CLIP_TRAJ_FREQ, CLIP_TRAJECTORY_LENGTH, CLIP_NUM_TRAJ, save_traj_matrix=SAVE_TRAJ_MATRIX)

    # To test extracted trajectory matrix
    # hoc_num = 3
    # test_extracted_traj_matrix(hoc_num)
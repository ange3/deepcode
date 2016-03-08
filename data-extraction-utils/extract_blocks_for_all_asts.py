#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: extract_blocks_for_all_asts.py
# @Author: Angela Sy
# @created: Mar 2 2016
#
#==============================================================================
# DESCRIPTION: Extract matrix of code blocks represented as one-hot encodings.
#   (num_trajectories, num_timesteps, num_code_blocks)
# 
#==============================================================================
# CURRENT STATUS: In progress/ working! :) 
#==============================================================================
# USAGE: 
# run file in terminal
#==============================================================================
#
###############################################################################

import numpy as np
import csv, pickle
import time

PATH_TO_AST_TO_BLOCKS = '../data/ast_blocks_files/'
AST_TO_BLOCKS_FILENAME_PRE = 'AST_to_blocks_'
AST_TO_BLOCKS_FILENAME_POST = '.csv'

PATH_TO_PROCESSED_DATA = '../processed_data_asts/'

BLOCK_ID_FOR_END_TOKEN = 0
BLOCK_STRING_FOR_END_TOKEN = '-1'



def extract_blocks_for_one_hoc(hoc_num, clip_timesteps = -1, verbose = True, save_traj_matrix = True):
  '''
  Extracts a trajectories matrix for one problem as a one-hot encoding of each code block
  Output: (num_asts, num_timesteps, num_code_blocks)
      where 
          num_asts = number of ASTs used for this problem,
          num_timesteps = length of the longest AST
          num_code_blocks = number of distinct code blocks in the given trajectories

  '''

  filepath = PATH_TO_AST_TO_BLOCKS + AST_TO_BLOCKS_FILENAME_PRE + str(hoc_num) + AST_TO_BLOCKS_FILENAME_POST

  # Maps to save data
  map_ast_id_to_count = {}
  map_row_index_to_ast_id = {}
  map_code_block_string_to_block_id = {}
  map_code_block_string_to_block_id[BLOCK_STRING_FOR_END_TOKEN] = BLOCK_ID_FOR_END_TOKEN  # after end of last line of code
  map_row_ast_filename = PATH_TO_PROCESSED_DATA + 'map_row_index_to_ast_id_' + str(hoc_num) + '.pickle'
  map_ast_count_filename = PATH_TO_PROCESSED_DATA + 'map_ast_id_to_count_' + str(hoc_num) + '.pickle'
  map_block_string_to_id_filename = PATH_TO_PROCESSED_DATA + 'map_block_string_to_block_id_' + str(hoc_num) + '.pickle'

  # Setup data structures
  raw_asts_list = []
  longest_ast_len = -1
  unique_code_blocks_set = set()

  # Process file and save maps
  with open(filepath, 'rb') as ast_file:
    for index, line in enumerate(csv.reader(ast_file, delimiter=',')):
      ast_id = int(line[0])
      ast_count = int(line[1])
      raw_ast = line[2:]
      if clip_timesteps != -1:
        raw_ast = raw_ast[:clip_timesteps]

      map_ast_id_to_count[ast_id] = ast_count
      map_row_index_to_ast_id[index] = ast_id
      raw_asts_list.append(raw_ast)
      if len(raw_ast) > longest_ast_len:
        longest_ast_len = len(raw_ast)
      for code_block in raw_ast:
        unique_code_blocks_set.add(code_block)
    

    if verbose:
      print '-- SAVE: Saving Row Index to AST ID map and AST ID to Count map'
      print 'Map Row to AST File: {}'.format(map_row_ast_filename)
      print 'Map AST ID to Count File: {}'.format(map_ast_count_filename)

    pickle.dump(map_row_index_to_ast_id, open( map_row_ast_filename, "wb" ))
    pickle.dump(map_ast_id_to_count, open( map_ast_count_filename, "wb" ))

    # Testing pickled maps for HOC 2
    # map_row_ast_test = pickle.load( open(map_row_ast_filename, "rb") ) 
    # print map_row_ast_test[2]   # 10
    # map_ast_count_test = pickle.load( open(map_ast_count_filename, "rb") ) 
    # print map_ast_count_test[0]   # 497937


  # Create AST matrix
  num_asts = len(raw_asts_list)
  num_timesteps = longest_ast_len
  num_code_blocks = len(unique_code_blocks_set) + 1  # for empty block (when nothing left in the program)

  ast_matrix = np.zeros((num_asts, num_timesteps, num_code_blocks)) 
  print '-- INFO: AST Matrix:', ast_matrix.shape 

  for ast_index, ast in enumerate(raw_asts_list):
    for timestep, code_block_string in enumerate(ast):
      # Get block ID
      if code_block_string not in map_code_block_string_to_block_id:
        map_code_block_string_to_block_id[code_block_string] = len(map_code_block_string_to_block_id)
      block_id = map_code_block_string_to_block_id[code_block_string]
      ast_matrix[ast_index, timestep, block_id] = 1
      if timestep == len(ast)-1: # if reached the last timestep for this AST, add end token for all remaining timesteps
        ast_matrix[ast_index, timestep+1:, BLOCK_ID_FOR_END_TOKEN] = 1

  # Save Matrix
  if save_traj_matrix:
    ast_matrix_filename = PATH_TO_PROCESSED_DATA + 'ast_matrix_' + str(hoc_num) + '.npy'
    if clip_timesteps != -1:
        ast_matrix_filename = PATH_TO_PROCESSED_DATA + 'ast_matrix_' + str(hoc_num) + '_timesteps_' + str(clip_timesteps) + '.npy'

    if verbose:
      print '-- SAVE: Saving trajectories matrix and Block ID map'
      print 'AST Matrix Filename: {}'.format(ast_matrix_filename)
      print 'Map Block String to ID Filename: {}'.format(map_block_string_to_id_filename)

    # Save
    np.save(ast_matrix_filename, ast_matrix)
    pickle.dump(map_code_block_string_to_block_id, open( map_block_string_to_id_filename, "wb" ))


def extract_blocks_for_all_hocs(start_problem_id, end_problem_id, clip_timesteps = -1, verbose=True):
  '''
  Extracts trajectories matrix code blocks for all problems from START_PROBLEM_ID to END_PROBLEM_ID inclusive.
  Saves trajectories matrices to a numpy file and code block name to row index maps to a pickle file.
  '''

  for hoc_num in xrange(start_problem_id, end_problem_id + 1):
    if verbose:
        print '*** INFO: HOC {} ***'.format(hoc_num)

    tic = time.clock()
    extract_blocks_for_one_hoc(hoc_num, clip_timesteps)
    toc = time.clock()

    if verbose:
        print 'Finished extracting Code Blocks from Problem {} in {}s'.format(hoc_num, toc-tic)

def test_extracted_ast_matrix(hoc_num):
  '''
  Testing for HOC 2's first AST
  ast_id: 0, ast: program,maze_moveForward,maze_moveForward,maze_moveForward,end_program  (from csv file)

  Note: Testing not generalized to other HOCs
  '''

  ast_matrix_filename = PATH_TO_PROCESSED_DATA + 'ast_matrix_' + str(hoc_num) + '.npy'
  map_row_ast_filename = PATH_TO_PROCESSED_DATA + 'map_row_index_to_ast_id_' + str(hoc_num) + '.pickle'
  map_block_string_to_id_filename = PATH_TO_PROCESSED_DATA + 'map_block_string_to_block_id_' + str(hoc_num) + '.pickle'

  # Load
  ast_matrix = np.load(ast_matrix_filename)
  print 'ast_matrix shape: {}'.format(ast_matrix.shape)
  map_row_index_to_ast_id = pickle.load( open(map_row_ast_filename, "rb") )
  print 'Number of ASTs: {}'.format(len(map_row_index_to_ast_id))
  map_block_string_to_id = pickle.load( open(map_block_string_to_id_filename, "rb"))
  map_block_id_to_string = {v: k for k, v in map_block_string_to_id.items()}

  test_row_index = 0
  test_ast = ast_matrix[test_row_index, :, :]
  test_ast_id = map_row_index_to_ast_id[test_row_index]
  test_ast_block_string_list = []

  for block_id_one_hot_timestep in test_ast:
    block_id = np.argmax(block_id_one_hot_timestep)
    block_string = map_block_id_to_string[block_id]
    test_ast_block_string_list.append(block_string)

  print 'Test AST ID:', test_ast_id
  print test_ast_block_string_list



if __name__ == "__main__":
  START_PROBLEM_ID = 6
  END_PROBLEM_ID = 9
  CLIP_TIMESTEPS = -1

  extract_blocks_for_all_hocs(START_PROBLEM_ID, END_PROBLEM_ID, CLIP_TIMESTEPS)

  # Testing
  # test_extracted_ast_matrix(2)
  # test_extracted_ast_matrix(3)



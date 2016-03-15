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
#   Code command strings (code blocks) are transformed into command IDs (block IDs) 
#     corresponding to their row position in the block one-hot encoding.
#   Furthermore, for loop counts are ignored and a 'no_block' token is added for all 
#     timesteps after the end of a program (necessary since we are imposing a set number
#     of timesteps for each program).
# 
#   We use the following block string to block ID mapping as defined in the file 
#     processed_data_asts/map_block_string_to_block_id_master.pickle
#      {'controls_repeat': 6,
#       'end_loop': 7,
#       'end_program': 3,
#       'maze_moveForward': 2,
#       'maze_turnLeft': 5,
#       'maze_turnRight': 4,
#       'no_block': 0,
#       'program': 1}
#==============================================================================
# CURRENT STATUS: Done
#==============================================================================
# USAGE: 
# Set parameters in main function below (e.g. problem ID number range, clipping settings, etc.)
# In Terminal, run  python extract_blocks_for_all_asts
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
LOAD_MAP_BLOCK_STRING_TO_BLOCK_ID_FILE = '../processed_data_asts/map_block_string_to_block_id_master.pickle'
# LOAD_MAP_BLOCK_STRING_TO_BLOCK_ID_FILE = None

BLOCK_ID_FOR_END_TOKEN = 0
BLOCK_STRING_FOR_END_TOKEN = 'no_block'


def extract_blocks_for_one_hoc(hoc_num, clip_timesteps = -1, map_code_block_string_to_block_id = None, verbose = True, save_traj_matrix = True):
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
  map_row_ast_filename = PATH_TO_PROCESSED_DATA + 'map_row_index_to_ast_id_' + str(hoc_num) + '.pickle'
  map_ast_count_filename = PATH_TO_PROCESSED_DATA + 'map_ast_id_to_count_' + str(hoc_num) + '.pickle'
  map_block_string_to_id_filename = PATH_TO_PROCESSED_DATA + 'map_block_string_to_block_id_' + str(hoc_num) + '.pickle'

  # Setup data structures
  clean_asts_list = []
  longest_ast_len = -1
  if map_code_block_string_to_block_id is None:
    # Setup map: {block string => block ID} if not pre-loaded
    map_code_block_string_to_block_id = {}
    map_code_block_string_to_block_id[BLOCK_STRING_FOR_END_TOKEN] = BLOCK_ID_FOR_END_TOKEN  # after end of last line of code

  # Process file and save maps
  with open(filepath, 'rb') as ast_file:
    for index, line in enumerate(csv.reader(ast_file, delimiter=',')):
      ast_id = int(line[0])
      ast_count = int(line[1])
      raw_ast = line[2:]

      # Test AST
      # test_ast_id = 0
      # if ast_id == test_ast_id:
      #   test_ast_row = index
      #   print 'TESTING'
      #   print 'AST', ast_id
      #   print 'Raw Trajectory'
      #   print raw_ast

      map_ast_id_to_count[ast_id] = ast_count
      map_row_index_to_ast_id[index] = ast_id
      clean_ast = []
      for code_block in raw_ast:
        if not code_block.isdigit():
          clean_ast.append(code_block)

      if len(clean_ast) > longest_ast_len:
        longest_ast_len = len(clean_ast)

      if clip_timesteps != -1:
        clean_ast = clean_ast[:clip_timesteps]

      clean_asts_list.append(clean_ast)

      # Test AST
      # if ast_id == test_ast_id:
      #   test_ast_row = index
      #   print 'Clean Trajectory clipped'
      #   print clean_ast

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
  num_asts = len(clean_asts_list)
  if clip_timesteps != -1:
    num_timesteps = clip_timesteps
  else:
    num_timesteps = longest_ast_len
  num_code_blocks = len(map_code_block_string_to_block_id)

  ast_matrix = np.zeros((num_asts, num_timesteps, num_code_blocks)) 
  print '-- INFO: AST Matrix:', ast_matrix.shape 

  for ast_index, ast in enumerate(clean_asts_list):
    timestep = 0
    for code_block_string in ast:

      # Skip blocks that are integers and not commands
      if code_block_string.isdigit():
        continue

      # Get block ID
      if code_block_string not in map_code_block_string_to_block_id:
        map_code_block_string_to_block_id[code_block_string] = len(map_code_block_string_to_block_id)
      block_id = map_code_block_string_to_block_id[code_block_string]

      # Test AST
      # if ast_index == test_ast_row:
      #   map_block_id_to_string = {v: k for k, v in map_code_block_string_to_block_id.items()}
      #   print block_id, map_block_id_to_string[block_id]

      # Insert one-hot encoding of block ID for this time step
      ast_matrix[ast_index, timestep, block_id] = 1

      # Increment timestep after a block ID one-hot encoding has been added to ast_matrix
      timestep += 1 

      if code_block_string == 'end_program': # if reached the last timestep for this AST, add end token for all remaining timesteps
        ast_matrix[ast_index, timestep:, BLOCK_ID_FOR_END_TOKEN] = 1

  # Test AST
  # print ast_matrix[test_ast_row]

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
    print '--' * 10
    print 'HOC', str(hoc_num)
    print map_code_block_string_to_block_id
    print '--' * 10


def extract_blocks_for_all_hocs(start_problem_id, end_problem_id, clip_timesteps = -1, verbose=True):
  '''
  Extracts trajectories matrix code blocks for all problems from START_PROBLEM_ID to END_PROBLEM_ID inclusive.
  Saves trajectories matrices to a numpy file and code block name to row index maps to a pickle file.
  '''
  if LOAD_MAP_BLOCK_STRING_TO_BLOCK_ID_FILE:
    map_code_block_string_to_block_id = pickle.load(open(LOAD_MAP_BLOCK_STRING_TO_BLOCK_ID_FILE, 'rb'))
  else:
    map_code_block_string_to_block_id = None
  for hoc_num in xrange(start_problem_id, end_problem_id + 1):
    if verbose:
        print '*** INFO: HOC {} ***'.format(hoc_num)

    tic = time.clock()
    extract_blocks_for_one_hoc(hoc_num, clip_timesteps, map_code_block_string_to_block_id)
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
  START_PROBLEM_ID = 1
  END_PROBLEM_ID = 9
  CLIP_TIMESTEPS = -1

  extract_blocks_for_all_hocs(START_PROBLEM_ID, END_PROBLEM_ID, CLIP_TIMESTEPS)

  # Testing
  # test_extracted_ast_matrix(2)
  # test_extracted_ast_matrix(3)



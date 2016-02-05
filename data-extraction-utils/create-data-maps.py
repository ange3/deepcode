#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: preprocess.py
# @Author: Angela Sy
# @created: Feb 04 2016
#
#==============================================================================
# DESCRIPTION: Creating maps of given data and save as pickle files
# 
#==============================================================================
# CURRENT STATUS: In progress
#==============================================================================
# USAGE:
# Run in Terminal:  python create-data-maps.py
#==============================================================================
#
###############################################################################

import numpy as np
import csv, pickle

DATA_FOLDER = '/Volumes/ANGELA SY/From Chris - senior project knowledge tracing/hoc1-9'
START_AT_NUM_PROBLEM = 9
END_AT_NUM_PROBLEM = 9


# Helper function that extracts {trajectory ID: final AST ID}
def extract_final_AST_id_from_trajectory(problemNum, trajectoryID):
  trajectory_filename = DATA_FOLDER + '/hoc' + str(problemNum) + '/trajectories/' + str(trajectoryID) + '.txt'
  ast_list = list(csv.reader(open(trajectory_filename,"rb"),delimiter=','))
  final_ast_id = ast_list[len(ast_list)-1]
  return final_ast_id

# Args: map {student ID: final AST ID}, numpy array row with info read from data_array
def extract_student_id_final_AST_id_map(student_id_final_AST_id_map, trajectory_id_final_AST_id_map, line, problemNum):
  student_id = line[0]
  trajectory_id = line[1]
  if trajectory_id not in trajectory_id_final_AST_id_map:
    final_ast_id = extract_final_AST_id_from_trajectory(problemNum, trajectory_id)
    trajectory_id_final_AST_id_map[trajectory_id] = final_ast_id
  final_ast_id = trajectory_id_final_AST_id_map[trajectory_id]
  student_id_final_AST_id_map[student_id] = final_ast_id

if __name__ == '__main__':
  problem_to_student_AST_maps = {}
  for problemNum in range(START_AT_NUM_PROBLEM,END_AT_NUM_PROBLEM+1):
    # Set up data structures
    student_id_final_AST_id_map = {}
    problem_to_student_AST_maps[str(problemNum)] = student_id_final_AST_id_map

    trajectory_id_final_AST_id_map = {}

    # Load data
    id_map_filename = DATA_FOLDER + '/hoc' + str(problemNum) + '/trajectories/idMap.txt'
    print 'Loading data from: {}'.format(id_map_filename)
    data_array = np.array(list(csv.reader(open(id_map_filename,"rb"),delimiter=',')))

    for index,line in enumerate(data_array):
      if index%50000 == 0:
        print 'Progress: processed {} *50k students'.format(index/50000)
      extract_student_id_final_AST_id_map(student_id_final_AST_id_map, trajectory_id_final_AST_id_map, line, problemNum)

    # save student id to final AST id map for THIS problem to pickle file
    pickle_filename = '../data/maps/' + str(problemNum) + '_students_to_asts.pickle'
    with open(pickle_filename, 'w') as f:
      pickle.dump(student_id_final_AST_id_map, f)

    # test pickle file
    # print 'original'
    # print student_id_final_AST_id_map
    # with open(pickle_filename, 'r') as f:
    #   test_map = pickle.load(f)
    #   print 'pickle values'
    #   print test_map



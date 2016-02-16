from os import listdir
from os.path import isfile, join

import numpy as np

import csv, pickle
import time


PATH_TO_ALL_PROBLEMS ='../data/hoc1-9/'

PATH_TO_TRAJECTORIES = '/trajectories/'
PATH_TO_AST_COUNTS = '/asts/counts.txt'

PATH_TO_PROCESSED_DATA = '../processed_data/'

START_PROBLEM_ID = 1
END_PROBLEM_ID = 1

CLIP_TRAJECTORY_LENGTH = 20


def count_total_asts(hoc_num):
    '''
    Counts the number of unique ASTs present for this problem
    '''
    ast_counts_filename = PATH_TO_ALL_PROBLEMS + 'hoc' + str(hoc_num) + PATH_TO_AST_COUNTS
    counts = 0
    with open(ast_counts_filename, 'r') as ac_f:
        for l in ac_f:
            counts += 1
    return counts


def extract_asts_for_one_hoc(hoc_num):
    '''
    Extracts a trajectories matrix for one problem as a one-hot encoding of each AST's ID
    Output: (num_trajectories, num_timesteps, num_asts)
        where num_timesteps is the length of the longest trajectory
        where num_asts is the number of distinct ASTs
    '''
    traj_folder_path = PATH_TO_ALL_PROBLEMS + 'hoc' + str(hoc_num) + PATH_TO_TRAJECTORIES
    trajectory_files = [f for f in listdir(traj_folder_path) if isfile(join(traj_folder_path, f))]
    raw_trajectories_list = []
    longest_trajectory_len = -1
    for tr_f in trajectory_files:
        if tr_f not in ['counts.txt', 'idMap.txt']:
            with open(join(traj_folder_path,tr_f), 'r') as tr_file:
                 raw_trajectory = np.array(list(csv.reader(tr_file,delimiter=',')))
                 # print raw_trajectory.shape[0]
                 # if raw_trajectory.shape[0] > longest_trajectory_len:
                 #    print tr_f
                 #    print raw_trajectory.shape[0]
                 
                 longest_trajectory_len = max(longest_trajectory_len, raw_trajectory.shape[0])
                 raw_trajectories_list.append(raw_trajectory)

    num_asts = count_total_asts(hoc_num)     
    trajectories_matrix = np.zeros((len(trajectory_files), longest_trajectory_len, num_asts))
    max_timesteps = 0
    for traj_file_index, traj in enumerate(raw_trajectories_list):
        prev_ast = -1
        timestep = 0
        for ast in traj:
            # print 'ast = {}'.format(ast)
            ast_val = int(ast[0])
            if ast_val != prev_ast:
                # print 'ast_val = {}'.format(ast_val)
                trajectories_matrix[traj_file_index, timestep, ast_val] = 1
                timestep += 1
                prev_ast = ast_val
        if timestep > max_timesteps:
            max_timesteps = timestep
            max_timesteps_traj = traj
    print 'Max timesteps = {}'.format(max_timesteps)
    # print 'Trajectory = {}'.format(max_timesteps_traj)
    print 'orrginial trag length = {}'.format(len(max_timesteps_traj))

    return trajectories_matrix


def extract_asts_for_all_hocs():
    '''
    Extracts trajectories matrix for all problems and saves them to a numpy file
    '''
    for hoc_num in xrange(START_PROBLEM_ID, END_PROBLEM_ID + 1):
        tic = time.clock()
        trajectories_matrix = extract_asts_for_one_hoc(hoc_num)
        toc = time.clock()
        print 'Finished extracting ASTs from Problem {} in {}s'.format(hoc_num, toc-tic)
        print trajectories_matrix.shape
        print trajectories_matrix[:10, :10, :10]
        # save_matrix_filename = PATH_TO_PROCESSED_DATA + 'hoc' + str(hoc_num) + '_trajectories'
        # np.save(save_matrix_filename, trajectories_matrix)
        # print 'Saved to {}'.format(save_matrix_filename)


if __name__ == "__main__":
    extract_asts_for_one_hoc(1)
    # extract_asts_for_all_hocs()



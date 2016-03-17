#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: debug_utils.py
# @Author: Lisa Wang
# @created: Mar 15 2016
#
#==============================================================================
# DESCRIPTION:
# A place to put useful functions for debugging and sanity checking, e.g.
# printing out the contents of a matrix in readable form (e.g. the actual 
# code blocks within a student's program)
#==============================================================================
# CURRENT STATUS: In progress/ working! :) 
#==============================================================================
# USAGE: 
# import debug_utils or from debug_utils import *
#==============================================================================
#
###############################################################################

import numpy as np
import time
import pickle
import random
from constants import *
from sklearn.utils import shuffle
from utils import *


def print_sample_program(hoc_num=7, ast_id=0):
    hoc_num = str(hoc_num)
    embed_ast_map_file = EMBED_AST_MAP_PREFIX + hoc_num + MAP_SUFFIX
    embed_row_to_ast_id_map = pickle.load(open(embed_ast_map_file, "rb"))
    embed_ast_id_to_row_map = {v: k for k, v in embed_row_to_ast_id_map.items()}
    ast_row = embed_ast_id_to_row_map[ast_id]
    print 'printing program sequence for hoc {} and ast id {}'.format(hoc_num, ast_id)
    
    block_string_to_row_map = pickle.load(open(BLOCK_STRING_TO_BLOCK_ROW_MAP, "rb" ))
    block_row_to_string_map = {v: k for k, v in block_string_to_row_map.items()}
    ast_mat = np.load(BLOCK_MAT_PREFIX + hoc_num + BLOCK_LIMIT_TIMESTEPS +  MAT_SUFFIX)
    num_asts, max_ast_len, num_blocks = ast_mat.shape
    program = []
    for t in xrange(max_ast_len):
        block_row = np.argmax(ast_mat[ast_row,t,:])
        block_string = block_row_to_string_map[block_row]
        program.append(block_string)
    print "dimension num_blocks {}".format(num_blocks)
    print program



def convert_to_block_strings(mat_with_block_rows):
    ''' Converts a matrix with block row ids to a matrix filled with the 
    corresponding block strings, e.g. 'move' 
    Input: np matrix (num_asts, num_timesteps)
    Output: np matrix with strings (num_asts, num_timesteps)
    '''
    num_samples, num_timesteps  = mat_with_block_rows.shape
    block_string_to_row_map = pickle.load(open(BLOCK_STRING_TO_BLOCK_ROW_MAP, "rb" ))
    block_row_to_string_map = {v: k for k, v in block_string_to_row_map.items()}
    mat_with_block_strings = np.empty((num_samples, num_timesteps), dtype=object)
    for i in xrange(num_samples):
        for t in xrange(num_timesteps):
            block_string = block_row_to_string_map[int(mat_with_block_rows[i,t])]
            # print block_string
            mat_with_block_strings[i,t] = block_string
    return mat_with_block_strings


    if __name__ == "__main__":
    print "You are running debug_utils.py directly, so you must be testing it!"
    for hoc in xrange(HOC_MIN, HOC_MAX):
        print_sample_program(hoc_num=hoc,ast_id=0)
#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: prepocessing_utils.py
# @Author: Lisa Wang
# @created: Mar 11 2016
#
#==============================================================================
# DESCRIPTION:
# A place to put preprocessing functions, which we usually only use once. 
# I separated these from functions in utils.py which we call more frequently.
#==============================================================================
# CURRENT STATUS: In progress/ working! :) 
#==============================================================================
# USAGE: 
# python preprocessing_utils.py
#==============================================================================
#
###############################################################################

import numpy as np
import time
import pickle
import random
from constants import *


def pad_ast_block_matrices():
    """ to make sure that all matrices across hocs have the same dimension for 
    num_blocks. This function was already used to pad matrices, left here 
    just for reference. """
    num_blocks_final = 8

    for hoc in xrange(1, 10):
        hoc_num = str(hoc)
        ast_mat = np.load(BLOCK_MAT_PREFIX + hoc_num + BLOCK_LIMIT_TIMESTEPS +  MAT_SUFFIX)
        num_asts, max_ast_len, num_blocks = ast_mat.shape
        if num_blocks < num_blocks_final:
            print "padding hoc {}".format(hoc)
            ast_mat = np.concatenate([ast_mat, np.zeros((num_asts, max_ast_len, num_blocks_final-num_blocks))], axis=2)
            print ast_mat.shape
            np.save(BLOCK_MAT_PREFIX + hoc_num + BLOCK_LIMIT_TIMESTEPS +  MAT_SUFFIX, ast_mat)


if __name__ == "__main__":
    print "You are running preprocessing_utils.py directly, so you must be testing it!"
    
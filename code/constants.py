#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: constants.py
# @Author: Lisa Wang
# @created: Mar 11 2016
#
#==============================================================================
# DESCRIPTION:
# A single place to put all constants. This makes it easier to coordinate
# constants such as filenames between different scripts.
# A lot of the data we are using is not public. The file paths assume a 
# certain file structure, so if you get the data, you need to make sure that
# paths work out.
#==============================================================================
# CURRENT STATUS: Always in progress, but also always working! ;) 
#==============================================================================
# USAGE: 
# from constants import *
#==============================================================================
#
###############################################################################

# Each trajectory matrix corresponds to one hoc exercise and is
# its own data set. Mixing data sets currently does not make much
# sense since the AST IDs don't persist betweeen different hoc's.

TRAJ_MAP_PREFIX = '../processed_data/ast_id_level/traj_matrix_'
TRAJ_AST_MAP_PREFIX = '../processed_data/ast_id_level/map_ast_row_'

BLOCK_MAT_PREFIX = '../processed_data/block_level/ast_matrix_'
BLOCK_LIMIT_TIMESTEPS = '_timesteps_20'

# BLOCK IDS ARE THE SAME AS BLOCK ROWS
BLOCK_STRING_TO_BLOCK_ROW_MAP = '../processed_data/block_level/map_block_string_to_block_id_master.pickle'

AST_EMBEDDINGS_PREFIX_STEM = '../processed_data/embeddings/ast_embeddings_hoc_'
AST_EMBEDDINGS_VARIATION = 'indiv_only_forward'
EMBED_AST_MAP_PREFIX = '../processed_data/block_level/map_row_index_to_ast_id_'

COUNTS_MAT_PREFIX = '../counts_by_set/deck_'

MAT_SUFFIX = '.npy'
MAP_SUFFIX = '.pickle'
# To specify which Hour of Code problems we are taking into account
# when we iterate over "all" of them. 
# there are a bunch of hocs we are not including yet, since they are still
# in the SQL database. Once we include them, we should change the following
# constants.
HOC_MIN = 1
HOC_MAX = 9
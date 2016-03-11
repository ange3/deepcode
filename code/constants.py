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
#==============================================================================
# CURRENT STATUS: In progress/ working! :) 
#==============================================================================
# USAGE: 
#  from constants import *
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
BLOCK_STRING_TO_BLOCK_ROW_MAP = '../processed_data/block_level/map_block_string_to_block_id.pickle'
MAT_SUFFIX = '.npy'
MAP_SUFFIX = '.pickle'

AST_EMBEDDINGS_PREFIX = '../processed_data/embeddings/ast_embeddings_hoc'
EMBED_AST_MAP_PREFIX = '../processed_data/block_level/map_row_index_to_ast_id_'

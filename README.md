# deepcode
Deep learning using Recurrent Neural Networks on student code submissions; focusing on LSTMs to predict student success

Senior Project
- Students: Angela, Lisa, Larry
- Advisor: Chris

# Folders

## data-extraction-utils
Contains the python files which pre-process the given CSV files with HOC data into numpy matrices used for the RNNs.
- extract_from_activities_csv extracts the number of attempted and correct student solutions from the activities.csv database dump for HOC 1-9
- extract_asts_for_all_trajectories.py extracts AST IDs from trajectories to create matrices of (num_trajectories, num_timesteps, num_ast). Can also be used to clip trajectories below a certain frequency. Note: Data files defined similar to 'data/trajectory_ast_csv_files/Trajectory_ASTs_1.csv'.
- extract_blocks_for_all_asts.py extracts code statements from ASTs to create matrices of (num_trajectories, num_timesteps, num_code_blocks). Note: Data files defined similar to 'data/ast_blocks_files/AST_to_blocks_1.csv'.
- info files (printed output from running extract_* files)
# What Will You Code Next? Deep Knowledge Tracing on Non-Binary Data
## Predicting Student Performance on Online Coding Exercises
Deep learning using Recurrent Neural Networks on student code submissions; focusing on LSTMs to predict student success.

In our research, we use deep learning to understand a student’s learning trajectory as they solve open-ended problems. With a robust understanding of student learning, we can ultimately provide personalized automated feedback to students at scale.

We perform 2 tasks using RNNs to predict a student's future performance on given questions:
(1) Binary - Using a student's past accuracy, predict if the student will get the next question right or wrong
(2) Non-Binary - Using a student's past submissions, predict the next step in the student's problem solving path (in the case of the Code.org data, we are predicting the next code program that the student will write)

Stanford Computer Science Senior Project
- Students: Angela, Lisa, Larry
- Advisor: Chris

## Folders
A description of our folders and a few of the main files in each folder.

### code
Contains code to run our RNN models and other helper files
- baselines contain the baseline models that serve as accuracy value benchmarks for our 2 tasks
- constants.py
- ipython notebooks run the Recurrent Neural Networks. We have 2 flavors of RNNs: one which predicts binary correct/wrong for each student solution (milestone_1_binary) and one which predicts the next AST in a student's problem solving path (lasagne_rnn_predict_next_ast)
- model* python files build and compile the Lasagne models and functions
- visualize.py creates loss and accuracy plots for our results

### data-extraction-utils
Contains the python files which pre-process the CSV files with code.org Hour of Code data into numpy matrices used for the RNNs.
- extract_from_activities_csv extracts the number of attempted and correct student solutions from the activities.csv database dump for HOC 1-9
- extract_asts_for_all_trajectories.py extracts AST IDs from trajectories to create matrices of (num_trajectories, num_timesteps, num_ast). Can also be used to clip trajectories below a certain frequency. Note: Data files defined similar to 'data/trajectory_ast_csv_files/Trajectory_ASTs_1.csv'.
- extract_blocks_for_all_asts.py extracts code statements from ASTs to create matrices of (num_trajectories, num_timesteps, num_code_blocks). Note: Data files defined similar to 'data/ast_blocks_files/AST_to_blocks_1.csv'.
- info files (printed output from running extract_* files)

### loss_plots
Images showing the loss and accuracy values of our RNNs

### syntheticDetailed
Synthetic generated data of students answering a series of questions
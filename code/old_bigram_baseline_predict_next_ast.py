

from itertools import groupby


# Predicted next AST based on bigrams
code_org_data_bigrams_guess_unweighted_map = {
    1 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_1.csv",
    2 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_2.csv",
    3 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_3.csv",
    4 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_4.csv",
    5 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_5.csv",
    6 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_6.csv",
    7 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_7.csv",
    8 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_8.csv",
    9 : "../../data/bigrams/AST_guesses_unweighted_with_end_token_9.csv",
}

code_org_data_bigrams_guess_weighted_map = {
    1 : "../../data/bigrams/AST_guesses_weighted_with_end_token_1.csv",
    2 : "../../data/bigrams/AST_guesses_weighted_with_end_token_2.csv",
    3 : "../../data/bigrams/AST_guesses_weighted_with_end_token_3.csv",
    4 : "../../data/bigrams/AST_guesses_weighted_with_end_token_4.csv",
    5 : "../../data/bigrams/AST_guesses_weighted_with_end_token_5.csv",
    6 : "../../data/bigrams/AST_guesses_weighted_with_end_token_6.csv",
    7 : "../../data/bigrams/AST_guesses_weighted_with_end_token_7.csv",
    8 : "../../data/bigrams/AST_guesses_weighted_with_end_token_8.csv",
    9 : "../../data/bigrams/AST_guesses_weighted_with_end_token_9.csv",
}

# Actual trajectories
truth_labels_trajectories_of_asts_map = {
    1 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_1.csv",
    2 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_2.csv",
    3 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_3.csv",
    4 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_4.csv",
    5 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_5.csv",
    6 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_6.csv",
    7 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_7.csv",
    8 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_8.csv",
    9 : "../../data/trajectory_ast_csv_files/Trajectory_ASTs_9.csv",
}

trajectory_count_map = {
  1: "../../data/trajectory_count_files/counts_1.txt",
  2: "../../data/trajectory_count_files/counts_2.txt",
  3: "../../data/trajectory_count_files/counts_3.txt",
  4: "../../data/trajectory_count_files/counts_4.txt",
  5: "../../data/trajectory_count_files/counts_5.txt",
  6: "../../data/trajectory_count_files/counts_6.txt",
  7: "../../data/trajectory_count_files/counts_7.txt",
  8: "../../data/trajectory_count_files/counts_8.txt",
  9: "../../data/trajectory_count_files/counts_9.txt",
}

def predict_accuracy(DATA_SET_HOC, DATA_SZ, weighted_bigrams_bool = True, use_bigrams_prediction_bool = True, accuracy_per_timestep = False):
  '''
  Predict accuracy of baseline
  1) Using Bigram Predictions
  2) Using gold guesses (i.e. Guess that the next AST is the correct solution where AST ID = 0)

  Note: Can run this over multiple number of samples (data size)

  Returns: Either a float accuracy score (average over all timesteps) or a list of accuracy scores over each timestep.
  '''

  # Store accuracy values for each timestep or an average for entire problem
  accuracy_map = {}  # {timestep => (num_correct, num_total)}
  max_num_timestep = 0

  if accuracy_per_timestep:
    print 'INFO: Calculating accuracy at every timestep'

  # Create Bigrams Prediction Map
  bigrams_prediction_map = {}
  if weighted_bigrams_bool:
    print 'INFO: Using Weighted Bigrams'
    bigrams_filepath = code_org_data_bigrams_guess_weighted_map[DATA_SET_HOC]
  else:
    print 'INFO: Using Unweighted Bigrams'
    bigrams_filepath = code_org_data_bigrams_guess_unweighted_map[DATA_SET_HOC]
  with open(bigrams_filepath, 'rb') as bigrams_csv_file:
    bigram_reader = csv.reader(bigrams_csv_file, delimiter =',')
    for row in bigram_reader:
      bigrams_prediction_map[int(row[0])] = int(row[1])

  # Load Trajectory Counts (to produce weighted prediction accuracy)
  traj_counts = {}
  traj_count_filepath = trajectory_count_map[DATA_SET_HOC]
  with open(traj_count_filepath, 'rb') as f:
    reader = csv.reader(f, dialect = 'excel' , delimiter = '\t')
    for row in reader:
      traj_counts[int(row[0])] = int(row[1])

  # Load Truth Labels
  trajectories_asts_filepath = truth_labels_trajectories_of_asts_map[DATA_SET_HOC]

  # Predict using Bigrams and Calculate Accuracy vs Truth
  if use_bigrams_prediction_bool:
    print 'INFO: Predicting Using Bigrams'
  else:
    print 'INFO: Predicting Gold Solution'

  num_samples = DATA_SZ
  print '*' * 10
  print 'DATA SET: HOC', DATA_SET_HOC
  print 'num samples: ', num_samples

  with open(trajectories_asts_filepath, 'rb') as trajectories_csv_file:
    trajectories_reader = csv.reader(trajectories_csv_file, delimiter=',')
    # Iterate through student trajectories for this problem
    for index, line in enumerate(trajectories_reader):
      if index == num_samples:
        break
      traj_id = int(line[0])

      # Modify trajectory count increments based on whether or not we are weighting bigrams
      if weighted_bigrams_bool:
        traj_count = traj_counts[traj_id]
      else:
        traj_count = 1
      
      # Clean up of trajectory line (list of ASTs)
      line = line[1:] # ignore first element which is the trajectory ID
      line = [x[0] for x in groupby(line)]  # ignore consecutive duplicate ASTs in trajectory
      # Testing
      # if traj_id == 10:
      #   print line
      
      # Iterate through every timestep and compare predicted AST (predicted from previous AST) with actual AST. Including prediction of the end token.
      for timestep in xrange(len(line)+1):
        # Artificially insert end token at last timestep for correct solution
        if timestep == len(line):
          ast = END_TOKEN  
        else:
          ast = int(line[timestep])
        # Testing
        # if traj_id == 10:
        #   print ast
        # Skip first time step: no prediction since we haven't seen a previous AST
        if timestep == 0:  
          pass
        else:
          # Initialize each timestep's counts to 0 correct and 0 total predictions
          if timestep not in accuracy_map:
            accuracy_map[timestep] = (0, 0)

          # Keep track of max number of timesteps
          if timestep > max_num_timestep:
            max_num_timestep = timestep

          # Predict!
          num_correct, num_total = accuracy_map[timestep]
          new_num_correct = num_correct
          new_num_total = num_total
          if use_bigrams_prediction_bool:
            # (1) Predicting using Bigram Baseline
            if ast == 0:
              continue
            elif ast == prediction_prev_ast:
              new_num_correct += traj_count
          else: 
            # (2) Predicting using Gold Baseline (AST = 0)
            if ast == END_TOKEN:
              new_num_correct += traj_count
          new_num_total += traj_count
          # Update accuracy counts for this timestep
          accuracy_map[timestep] = (new_num_correct, new_num_total)

        # Predicting using Bigram Baseline (handle special case)
        if use_bigrams_prediction_bool:
          if ast not in bigrams_prediction_map:  # reached the end of one student's trajectory
            continue
          # otherwise, predict the next AST of that student
          prediction_prev_ast = bigrams_prediction_map[ast]

  # print accuracy_map
  # Calculate accuracy at each timestep (num correct / num total) OR over all timesteps
  acc_per_timestep = []
  if not accuracy_per_timestep:
    num_correct_over_all_timesteps = 0
    num_total_over_all_timesteps = 0

  for timestep in xrange(1, max_num_timestep+1):
    num_correct, num_total = accuracy_map[timestep]
    accuracy = float(num_correct)/num_total
    acc_per_timestep.append(accuracy)
    if not accuracy_per_timestep:
      num_correct_over_all_timesteps += num_correct
      num_total_over_all_timesteps += num_total

  if not accuracy_per_timestep:
    accuracy = float(num_correct_over_all_timesteps)/num_total_over_all_timesteps
    print 'Correct: ', num_correct_over_all_timesteps
    print 'Total: ', num_total_over_all_timesteps
    print 'Accuracy: ', accuracy
    return accuracy

  return acc_per_timestep

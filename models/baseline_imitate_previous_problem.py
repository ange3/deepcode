# Predicts student will get next question correct if they got the previous question correct. 
class Baseline_imitate_previous_problem(Abstract_knowledge_model):
  def __init__(self, num_timesteps, num_problems):
    Abstract_knowledge_model.init(num_timesteps, num_problems)

  # no compute_loss_and_gradients function since there is no training necessary for this model

  # Arguments: X: shape(num_samples, num_timesteps, num_problems * 2)
  #            next_problem: shape(num_samples, num_timesteps, num_problems)
  # Returns: vector of students' probability of the next problem being correct 
    # shape(num_samples, num_timesteps, num_problems)
  def predict(self, x, next_problem):
    prediction = np.array((next_problem.shape))

    # naive version with for loop
    for student_index in X:
      student_results = X[student_index]  # (num_timesteps, num_problems * 2)
      pred_row = np.sum(student_results, axis=1)  # (num_timesteps), e.g. [1 0 1 0 0]
      # expand prediction row into matrix - predict 1 or 0 for all questions (for each timestep)
      pred_matrix = np.repeat(pred_row, num_problems, axis=1)
      prediction[student_index] = pred_matrix

  # write function to check accuracy in solver.py
  # solver has a train function that calls compute_loss_and_gradients
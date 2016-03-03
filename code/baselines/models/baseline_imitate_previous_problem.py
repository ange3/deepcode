# Import Abstract Knowledge Model
from abstract_knowledge_model import *
import numpy as np

# Predicts student will get next question correct if they got the previous question correct. 
class Baseline_imitate_previous_problem(Abstract_knowledge_model):
  def __init__(self, num_timesteps, num_problems):
    Abstract_knowledge_model.__init__(self, num_timesteps, num_problems)

  # no compute_loss_and_gradients function since there is no training necessary for this model

  # Arguments: X: shape(num_samples, num_timesteps, num_problems * 2)
  #            next_problem: shape(num_samples, num_timesteps, num_problems)
  # Returns: vector of students' probability of the next problem being correct 
    # shape(num_samples, num_timesteps)
  def predict(self, X, next_problem):
    num_samples = X.shape[0]
    prediction = np.zeros((num_samples, self.num_timesteps))

    # naive version with for loop
    for student_index, student_results in enumerate(X):
      # student_results: (num_timesteps, num_problems * 2)
      student_results_correct = student_results[:,:self.num_problems]  # only look at first p problems (representing correctly answered questions)
      # print student_results_correct[0]
      # print 'student_results_correct.shape ', student_results_correct.shape
      pred_row = np.sum(student_results_correct, axis=1)  # (num_timesteps), e.g. [1 0 1 0 0]
      # print pred_row[0]
      prediction[student_index] = pred_row

    return prediction

  # write function to check accuracy in solver.py
  # solver has a train function that calls compute_loss_and_gradients
# Import Abstract Knowledge Model
from abstract_knowledge_model import *
import numpy as np

# Use probability of num_correct/num_problems of all problems student has previously answered up until this current problem
# Use window_val to specify how many previous problem results to use to compute probability (like n-grams) --> we will look at window_val + 1 problems (including current problem)

class Baseline_prob_of_previous_problems(Abstract_knowledge_model):
  def __init__(self, num_timesteps, num_problems, window_bool = False, window_val = 0):
    Abstract_knowledge_model.__init__(self, num_timesteps, num_problems)
    self.window_bool = window_bool
    self.window_val = window_val

  # Returns: probabilities for each timestep  (num_samples, num_timesteps)
  def predict(self, X, next_problem):
    num_samples = X.shape[0]
    prediction = np.zeros((num_samples, self.num_timesteps))

    for student_index, student_results in enumerate(X):
      # student_results: (num_timesteps, num_problems * 2)
      student_results_correct = student_results[:,:self.num_problems]  # only look at first p problems (representing correctly answered questions)
      correct_row = np.sum(student_results_correct, axis=1)  # (num_timesteps)
      running_sum = 0
      for time_step, result in enumerate(correct_row):
        if self.window_bool:  # checking only windows
          start_index = time_step - self.window_val
          if start_index < 0:
            start_index = 0
          end_index = time_step + 1
          window = correct_row[start_index:end_index]
          # print window
          prob = float(np.sum(window))/(self.window_val+1)  # num_correct / num_total  within specified window frame of timestep
          # print prob
        else:
          running_sum += result
          prob = float(running_sum)/(time_step+1)  # num_correct / num_total  up until this timestep
        prediction[student_index][time_step] = prob

    return prediction
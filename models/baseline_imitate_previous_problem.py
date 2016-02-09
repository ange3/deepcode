# Predicts student will get next question correct if they got the previous question correct. 
class Baseline_imitate_previous_problem(Abstract_knowledge_model):
  def __init__(self, num_timesteps, num_problems):
    Abstract_knowledge_model.init(num_timesteps, num_problems)

  # no compute_loss_and_gradients function since there is no training necessary for this model

  # Returns: vector of students' probability of the next problem being correct (num_samples x num_timesteps)
  def predict(self, x, next_problem):
    pass

  # write function to check accuracy in solver.py
  # solver has a train function that calls compute_loss_and_gradients
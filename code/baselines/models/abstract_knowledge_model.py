class Abstract_knowledge_model():

  def __init__(self, num_timesteps, num_problems):
    self.num_timesteps = num_timesteps
    self.num_problems = num_problems

# Args: defined in utils.py (look at vectorize_data function)
# solver has a train function that calls compute_loss_and_gradients
  def compute_loss_and_gradients(self, x, next_problem, truth):
    # implement in sub-class
    pass

# Returns scores
  def predict(self, x, next_problem):
    pass

  def print_params(self):
    # implement in sub-class
    pass

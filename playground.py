# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, num_problems)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(num_problems, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((num_problems, 1)) # output bias

def lossFun(inputs, targets, correctness, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps, ps_denom = {}, {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((num_problems,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps_denom[t] = np.sum(np.exp(ys[t]))
    ps[t] = np.exp(ys[t]) / ps_denom[t] # probabilities for next chars

    # softmax (cross-entropy loss)
    if correctness[targets[t]] == 1:
      loss += -np.log(ps[t][targets[t],0]) 
    else:
      loss += -np.log(1-ps[t][targets[t],0]) 
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])


  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    if correctness[targets[t]] == 1:
      dy[targets[t]] -= 1 # backprop into y
    else:
      for p in xrange(num_problems):
        if p != targets[t]:
          dy[p] -= np.exp(ys[t][p]) / (ps_denom[t] - np.exp(ys[t][targets[t]]))


    
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def accuracy(ps, targets, correctness):
  """
  Computes the accuracy using the predictions at each time step.
  For each t, if probability of next problem is > 0.5 for correct, or <= 0.5 
  for incorrect, then count this as correct prediction.
  """

  num_correct = 0
  for t in xrange(num_timesteps):
    predicted_prob = ps[t][targets[t],0] 
    if (predicted_prob >= 0.5 and correctness[targets[t]] == 1) or (predicted_prob < 0.5 and correctness[targets[t]] == 0):
      num_correct += 1
  accuracy = num_correct / float(num_timesteps)

  return accuracy


def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((num_problems, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(num_problems), p=p.ravel())
    x = np.zeros((num_problems, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/num_problems)*num_timesteps # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps num_timesteps long)
  if p+num_timesteps+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+num_timesteps]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+num_timesteps+1]]

  # sample from the model now and then
  # if n % 100 == 0:
  #   sample_ix = sample(hprev, inputs[0], 200)
  #   txt = ''.join(ix_to_char[ix] for ix in sample_ix)
  #   print '----\n %s \n----' % (txt, )

  # forward num_timesteps characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += num_timesteps # move data pointer

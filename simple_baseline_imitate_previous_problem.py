# Python libraries
import numpy as np
import random
import sys
import csv
import matplotlib.pyplot as plt
# allows plots to show inline in ipython notebook
# get_ipython().magic(u'matplotlib inline')

# our own modules
from utils import *
from visualize import *
__import__('models/Baseline_imitate_previous_problem.py')


data_sets_map = {
    'synth':"syntheticDetailed/naive_c5_q50_s4000_v0.csv",
    'code_org' : "data/hoc_1-9_binary_input.csv"
}

# DATA_SET = 'code_org'
# DATA_SZ = 500000
DATA_SET = 'synth'
DATA_SZ = 1  # num_samples

# Read in the data set
# This function can be moved to utils.py
data_array = np.array(list(csv.reader(open(data_sets_map[DATA_SET],"rb"),delimiter=','))).astype('int')
data_array = data_array[:DATA_SZ]
num_samples = data_array.shape[0]
num_problems = data_array.shape[1]

# time steps is number of problems - 1 because we cannot predict on the last problem.
num_timesteps = num_problems - 1 

# No need to split data since no training needed
test_data = data_array

print('Vectorization...')
X_test, next_problem_test, truth_test = vectorize_data(test_data)
print ("Vectorization done!")
print X_test.shape
# print X_test

model = Baseline_imitate_previous_problem(Abstract_knowledge_model(num_timesteps, num_problems))
model.predict()

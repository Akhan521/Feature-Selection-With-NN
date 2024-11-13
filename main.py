# main.py

# Importing our utility functions and search functions.
from util import *
from forward_selection import forward_selection
from backward_elimination import backward_elimination

# Setting the number of features we're considering.
num_features = 4

# Running the forward selection algorithm.
#max_score, best_subset = forward_selection(num_features)

# Running the backward elimination algorithm.
max_score, best_subset = backward_elimination(num_features)

# main.py

# Importing our utility functions and search functions.
from util import *
from forward_selection import forward_selection

# Setting the number of features we're considering.
num_features = 4

# Running the forward selection algorithm.
max_score, best_subset = forward_selection(num_features)


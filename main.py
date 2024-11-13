# main.py

# Importing our utility functions and search functions.
from util import *
from forward_selection import forward_selection
from backward_elimination import backward_elimination

print('\nWelcome to Aamir Khan\'s Feature Selection Algorithm:')
print('-----------------------------------------------------\n')


# Setting the number of features we're considering.
num_features = int(input('Enter the number of features you\'d like to consider: '))

print('''\nSelect the algorithm you\'d like to run:\n
      1. Forward Selection\n
      2. Backward Elimination
      ''')
choice = input('Enter your choice: ')

if choice == '1':
    # Running the forward selection algorithm.
    max_score, best_subset = forward_selection(num_features)
else:
    # Running the backward elimination algorithm.
    max_score, best_subset = backward_elimination(num_features)

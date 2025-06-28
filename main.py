
# Our imports.
import numpy as np
from Validator import Validator
from Classifier import Classifier
from forward_selection import forward_selection
from backward_elimination import backward_elimination
import matplotlib.pyplot as plt

'''

    Datasets: Small, Large, Titanic

    Results (Best Feature Subset & Accuracy):

        1. Small Dataset:
            - Forward Selection:    [5,3] w/ 92% accuracy.
            - Backward Elimination: [3,5] w/ 92% accuracy.
        2. Large Dataset:
            - Forward Selection:    [27,1] w/ 95.5% accuracy.
            - Backward Elimination: [27]   w/ 84.7% accuracy.
        3. Titanic Dataset:
            - Forward Selection:    [2] w/ 78.01% accuracy.
            - Backward Elimination: [2] w/ 78.01% accuracy.

'''

print('\nWelcome to Aamir Khan\'s 1NN-Classifier Program:')
print('-----------------------------------------------------\n')

# Loading our text files into python using numpy (From numpy documentation).
small_dataset = np.loadtxt('datasets/small-test-dataset.txt')
large_dataset = np.loadtxt('datasets/large-test-dataset.txt')
titanic_dataset = np.loadtxt('datasets/titanic-clean.txt')
dataset = None # Our data of choice: small or large.
feature_subset = None # Our feature subset.

# Forward selection or backward elimination?
print('Which feature selection algorithm would you like to use?')
print('1. Forward Selection')
print('2. Backward Elimination')
choice = int(input('Enter your choice: '))

# Small or large dataset?
print('\nWhich dataset would you like to use?')
print('1. Small Dataset')
print('2. Large Dataset')
print('3. Titanic Dataset')
dataset_choice = int(input('Enter your choice: '))

if dataset_choice == 1:
    dataset = small_dataset
elif dataset_choice == 2:
    dataset = large_dataset
else:
    dataset = titanic_dataset

print(f'\nThis dataset has {dataset.shape[1] - 1} features and {dataset.shape[0]} samples.\n')

if choice == 1:
    # Using forward selection on the dataset.
    _, feature_subset = forward_selection(dataset.shape[1] - 1, dataset)
else:
    # Using backward elimination on the dataset.
    _, feature_subset = backward_elimination(dataset.shape[1] - 1, dataset)






# Our imports.
import numpy as np
from Validator import Validator
from Classifier import Classifier
from forward_selection import forward_selection
from backward_elimination import backward_elimination

print('\nWelcome to Aamir Khan\'s 1NN-Classifier Program:')
print('-----------------------------------------------------\n')

# Loading our text files into python using numpy (From numpy documentation).
small_dataset = np.loadtxt('datasets/small-test-dataset.txt')
large_dataset = np.loadtxt('datasets/large-test-dataset.txt')

# Our feature subsets of choice: [3, 5, 7] and [1, 15, 27] for small and large datasets, respectively.
small_feature_subset = [3, 5, 7]
large_feature_subset = [1, 15, 27]

# Our classifier - 1NN classifier.
classifier = Classifier()

# Our validator for the small dataset.
validator_small_dataset = Validator(small_feature_subset, classifier, small_dataset)
# Our validator for the large dataset.
validator_large_dataset = Validator(large_feature_subset, classifier, large_dataset)

# Evaluating our classifier on both datasets.
accuracy_small_dataset = validator_small_dataset.evaluate()
accuracy_large_dataset = validator_large_dataset.evaluate()

# Printing the accuracies of our classifier on both datasets.
print(f'Accuracy on the small dataset: {accuracy_small_dataset:.2f}')
print(f'Accuracy on the large dataset: {accuracy_large_dataset:.2f}\n')



import numpy as np
from Classifier import Classifier
from Validator import Validator
import time

'''
Defining our forward selection function:
    Input:
        The total number of features we're considering.
    Output:
        A node or subset of features with the maximum score.
'''
def forward_selection(num_features, dataset):
    # Initializing our variables.
    features = [i for i in range(1, num_features + 1)] # Our features {f1,...,fn}.
    max_score = 0                                      # The maximum score for our best subset.
    best_subset = []                                   # The best subset of features.
    current_best_subset = []                           # The current best subset for the given iteration.

    # Declaring our classifier and validator.
    classifier = Classifier()
    validator = Validator(current_best_subset, classifier, dataset)

    # Intializing our best score using no features.
    start = time.time()
    max_score = validator.evaluate()
    # Using no features, we'll evaluate the accuracy.
    print(f'Running NN w/ no features & Leave-One-Out validation, I get an accuracy of {max_score*100:.2f}%')
    print('\nBeginning search:')
    print('-------------------\n')
    # Now, we'll greedily add features to our current best subset.
    while(len(features) > 0):
        current_best_score = 0 # The current best score for the given iteration.
        current_subset = None
        # As a starting point for the given iteration, we need a copy of the current best subset.
        starting_point = current_best_subset
        # By the end of the iteration, we'll have a feature to remove.
        feature_to_remove = None
        # Iterating over all features:
        for feature in features:
            # Adding the current feature to our starting point.
            current_subset = starting_point + [feature]
            # Updating our validator with the current subset.
            validator = Validator(current_subset, classifier, dataset)
            # Evaluating the current subset.
            current_score = validator.evaluate()
            print(f"\tUsing feature(s) {current_subset}, accuracy is {current_score*100:.2f}%")
            # If the current score is the best, we'll update the best subset and max score.
            if current_score > max_score:
                max_score = current_score
                best_subset = current_subset
            # We'll also need to store our current best subset and current best score.
            if current_score > current_best_score:
                current_best_score = current_score
                current_best_subset = current_subset
                feature_to_remove = feature

        # Outputting our best subsets and scores after the current iteration.
        if current_best_subset == []:
            print(f'\nUsing no features is the current best, accuracy is {current_best_score*100:.2f}%')
        else:
            print(f'\nFeature set {current_best_subset} is the current best, accuracy is {current_best_score*100:.2f}%')
        if best_subset == []:
            print(f'Using no features is the overall best, accuracy is {max_score*100:.2f}%\n')
        else:
            print(f"Feature set {best_subset} is the overall best, accuracy is {max_score*100:.2f}%\n")
        # Updating our starting point for the next iteration.
        starting_point = current_best_subset
        # Removing the feature we've chosen from our list of features.
        features.remove(feature_to_remove)
    print('-' * 85)
    print(f'Total time = {time.time() - start:.2f} seconds')
    print(f'Finished search! The best feature subset is {best_subset}, which has an accuracy of {max_score*100:.2f}%\n')
    return max_score, best_subset

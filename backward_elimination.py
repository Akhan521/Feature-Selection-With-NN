# Importing our utility functions.
from util import *

'''
Defining our backward elimination function:
    Input:
        The total number of features we're considering.
    Output:
        A node or subset of features with the maximum score.
'''

def backward_elimination(num_features):
    features = [i for i in range(1, num_features + 1)]
    max_score = 0
    best_subset = None
    length = num_features
    # We'll create a powerset of our features that is in descending order.
    subsets = reversed(powerset(features))
    # Iterating over all subsets of features (starting from a subset with all features):
    for subset in subsets:
        current_score = evaluate(subset)
        current_length = len(subset)
        # If the subset is empty, we'll output a specific message.
        if current_length == num_features:
            print(f"\nUsing all features and random evaluation, I get an accuracy of {current_score*100:.2f}%")
            print('\nBeginning search.\n')
            length = current_length - 1 # The next subset will have one less feature.
        else:
            if current_length == length:
                print(f"\tUsing feature(s) {subset}, accuracy is {current_score*100:.2f}%")
            else:
                length = current_length
                print(f"\nFeature set {best_subset} was best, accuracy is {max_score*100:.2f}%\n")
                print(f"\tUsing feature(s) {subset}, accuracy is {current_score*100:.2f}%")
        # Updating the best subset and score if necessary.
        if current_score > max_score:
            max_score = current_score
            best_subset = subset
        
    print(f'\nFinished search! The best feature subset is {best_subset}, which has an accuracy of {max_score*100:.2f}%\n')
    return max_score, best_subset
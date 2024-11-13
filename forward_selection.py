# Importing our utility functions.
from util import *

'''
Defining our forward selection function:
    Input:
        The total number of features we're considering.
    Output:
        A node or subset of features with the maximum score.
'''
def forward_selection(num_features):
    features = [i for i in range(1, num_features + 1)]
    max_score = 0
    best_subset = None
    length = 1
    # We'll create a powerset of our features.
    subsets = powerset(features)
    # Iterating over all subsets of features:
    for subset in subsets:
        current_score = evaluate(subset)
        current_length = len(subset)
        # If the subset is empty, we'll output a specific message.
        if not subset:
            print(f"\nUsing no features and random evaluation, I get an accuracy of {current_score*100:.2f}%")
            print('\nBeginning search.\n')
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
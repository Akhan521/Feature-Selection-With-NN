import numpy as np

'''
Defining our backward elimination function:
    Input:
        The total number of features we're considering.
    Output:
        A node or subset of features with the maximum score.
'''

def backward_elimination(num_features):
    # Initializing our variables.
    features = [i for i in range(1, num_features + 1)] # Our features {f1,...,fn}.
    max_score = 0                                      # The maximum score for our best subset.
    best_subset = None                                 # The best subset of features.
    current_best_subset = features                     # The current best subset for the given iteration.

    # Intializing our best score:
    max_score = np.random.rand()
    # Using all features, we'll evaluate the accuracy.
    print(f"\nUsing all features and random evaluation, I get an accuracy of {max_score*100:.2f}%")
    print('\nBeginning search.\n')
    # Now, we'll greedily remove features from our current best subset.
    while(len(features) > 0):
        current_best_score = 0 # The current best score for the given iteration.
        current_subset = None
        # As a starting point for the given iteration, we need a copy of the current best subset.
        starting_point = current_best_subset
        # By the end of the iteration, we'll have a feature to remove.
        feature_to_remove = None
        # Iterating over all features:
        for feature in features:
            current_subset = [f for f in starting_point if f != feature]
            current_score = np.random.rand()
            if current_subset == []:
                print(f"\tUsing no features, accuracy is {current_score*100:.2f}%")
            else:
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
            print(f"Using no features is the overall best, accuracy is {max_score*100:.2f}%\n")
        else:
            print(f"Feature set {best_subset} is the overall best, accuracy is {max_score*100:.2f}%\n")
        # Updating our starting point for the next iteration.
        starting_point = current_best_subset
        # Removing the feature we've chosen from our list of features.
        features.remove(feature_to_remove)
        
    print(f'\nFinished search! The best feature subset is {best_subset}, which has an accuracy of {max_score*100:.2f}%\n')
    return max_score, best_subset
import random as rand

'''
Defining a powerset function:
    Input:
        A list of features.
    Output:
        A list of all subsets of the features.
'''
def powerset(features):
    # If the list is empty, return an empty list.
    if not features:
        return [[]]
    # Otherwise, iterate over all subsets.
    subsets = [[]]
    for i in range(len(features)):
        for j in range(len(subsets)):
            subsets.append(subsets[j] + [features[i]])
    # Sorting the subsets by length.
    subsets = sorted(subsets, key=len)
    # Return the powerset.
    return subsets
'''
Defining our evaluation function stub:
    Input: 
        A node/list representing a subset of features we're considering.
    Output:
        A random value for now.
'''
def evaluate(features):
    return rand.random()
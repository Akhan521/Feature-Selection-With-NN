
import numpy as np

# Our classifier class.
class Classifier:
    # A 2D numpy array representing our training data.
    training_data = None

    # Our train method.
    def train(self, training_data):
        # Setting our training data.
        self.training_data = training_data

    # Our test method -> returns the predicted label of the test data.
    def test(self, test_data):
        predicted_label = None
        nn = None
        nn_index = None
        nn_distance = float('inf')
        # Computing the distance from the test data to each training data point:
        for i, sample in enumerate(self.training_data):
            # Computing the Euclidean distance using numpy.
            distance = np.linalg.norm(test_data[1:] - sample[1:])
            # If we have a closer neighbor, we'll take this into account.
            if distance < nn_distance:
                nn = sample
                nn_index = i + 1 # Storing our nearest neighbor.
                nn_distance = distance
                predicted_label = sample[0]
        # After computing the distances to all training samples, we'll return the predicted label.
        #print(f'Nearest neighbor:  {nn}\n\tTraining sample {nn_index} w/ a distance of {nn_distance:.2f}.')
        return predicted_label
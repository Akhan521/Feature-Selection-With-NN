
import numpy as np

# Our validator class.
class Validator:
    # Our feature subset in the form of a list. We specify our features using their indices.
    feature_subset = None
    # Our NN-classifier.
    classifier = None
    # A 2D numpy array representing our training data.
    dataset = None

    def __init__(self, feature_subset, classifier, dataset):
        # Inserting 0 at the beginning of our feature subset to account for the label.
        self.feature_subset = [0] + feature_subset
        # Setting our classifier as the NN-classifier.
        self.classifier = classifier
        # Initially storing our dataset specified by the feature subset.
        self.dataset = dataset[:, self.feature_subset]
        # Normalizing our dataset using min-max normalization (From Wikipedia).
        mins = np.min(self.dataset[:, 1:], axis=0)
        maxs = np.max(self.dataset[:, 1:], axis=0)
        # Normalization formula: X = (X - min) / (max - min).
        self.dataset[:, 1:] = (self.dataset[:, 1:] - mins) / (maxs - mins)

    # Our validation method: Leave-One-Out Cross Validation.
    def evaluate(self):
        correct_classifications = 0 
        # The total number of samples or the number of rows in our dataset.
        N = self.dataset.shape[0]
        # Our validation data sample.
        validation_sample = None
        # Our Leave-One-Out loop:
        for i in range(N):
            # We'll first set aside the validation sample.
            validation_sample = self.dataset[i]
            # We'll use the remaining samples to train our classifier.
            selected_samples = [s for s in range(N) if s != i]
            self.classifier.train(self.dataset[selected_samples, :])
            # Testing our classifier on the validation sample.
            predicted_label = self.classifier.test(validation_sample)
            # If our predicted label matches the true label, we'll increment our correct classifications.
            if predicted_label == validation_sample[0]:
                correct_classifications += 1
        # Computing our accuracy and returning it.
        accuracy = correct_classifications / N
        return accuracy
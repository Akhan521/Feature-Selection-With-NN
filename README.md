# Nearest Neighbor Classifier with Feature Selection

**Feature selection** is the process of choosing a subset of relevant features (variables) from a dataset to improve model performance, reduce overfitting, and enhance interpretability. This project explores how feature selection impacts classification using a simple 1-Nearest Neighbor (1-NN) algorithm combined with **greedy feature selection techniques**, forward selection and backward elimination, on real-world and synthetic datasets.

## Project Summary

This project was started in order to learn more about feature selection techniques and why it matters in the context of building accurate classifiers using AI and machine-learning.
The goal is to evaluate subsets of features that maximize the classification accuracy of a nearest-neighbor algorithm using **leave-one-out cross-validation**. An NN-Classifier's accuracy is dependent on the subset of features that it operates on. Using feature selection techniques, we are able to determine the subset of features that gives our NN-classifier the greatest classification accuracy.

You can run the algorithm on datasets of varying sizes, including:

- `titanic-clean.txt` - Preprocessed Titanic survival dataset
- `small-test-dataset.txt` - A synthetic dataset for quick validation
- `large-test-dataset.txt` - A high-dimensional dataset to test scalability

## File Structure 📁

- `main.py`: Main file for running the program and selecting algorithms.
- `Classifier.py`: Contains the 1-NN classifier logic using Euclidean distance.
- `Validator.py`: Implements leave-one-out cross-validation to evaluate accuracy.
- `forward_selection.py`: Implements greedy forward feature selection.
- `backward_elimination.py`: Implements greedy backward feature elimination.
- `datasets/*.txt`: Datasets in whitespace-separated format.

## How the 1-NN Classifier Works

1. **Train/Test Phase**: A `Classifier` object stores training data and predicts the label for each test instance based on the closest training point. Note: although we refer to this as the training phase, there isn't actual model training involved. We simply compute the closest training point to each test instance/example.

2. **Validation Phase**: The `Validator` evaluates a feature subset using leave-one-out cross-validation, measuring the classifier's accuracy over all points.

3. **Feature Selection**:
   - **Forward Selection** begins with no features and adds the one that most improves accuracy at each step.
   - **Backward Elimination** starts with all features and removes the least useful one per iteration.

Example output from `backward_elimination.py`:
```
Using feature(s) [1, 3, 5], accuracy is 91.67%
Feature set [1, 3, 5] is the overall best, accuracy is 91.67%
```

## Sample Results

| Dataset                | Method               | Accuracy | Best Subset     |
|------------------------|----------------------|----------|-----------------|
| `titanic-clean.txt`    | Forward Selection    | 84.21%   | [2, 3]          |
| `small-test-dataset.txt` | Backward Elimination | 91.67%   | [1, 3, 5]       |
| `large-test-dataset.txt` | Forward Selection    | 86.75%   | [4, 12, 17, 20] |

## Getting Started / Running the Project

### 1. Open a Terminal  
On macOS/Linux, open the Terminal app.  
On Windows, open Command Prompt or PowerShell.

### 2. Clone the Repository  
```bash
git clone https://github.com/Akhan521/Feature-Selection-With-NN.git
cd Feature-Selection-With-NN
```

### 3. Run the Program  
```bash
python3 main.py
```

You'll be prompted to select:
- Dataset to load
- Feature selection algorithm (forward or backward)

## Lessons Learned

- **Leave-One-Out Cross-Validation** is extremely powerful for small datasets but computationally expensive for large ones.
- **Greedy Search** doesn't guarantee the global optimum but it's fast and often good enough.
- **Dimensionality Reduction** through feature selection is crucial for interpretability and overfitting prevention.

## Skills Demonstrated

- Implementation of classic ML algorithms from scratch (no sklearn!)
- Efficient use of NumPy for numerical computation
- Design and analysis of greedy algorithms
- Real application of feature engineering
- Interpretable model evaluation and reporting

## Potential Future Directions

- Add support for k-NN with `k > 1`
- Automate dataset normalization
- Swap Leave-one-out-CV for k-fold CV to improve performance on large datasets
- Plot decision boundaries and learning curves

## My Details

**Aamir Khan**  |  [LinkedIn](https://www.linkedin.com/in/aamir-khan-aak521/)  |  [GitHub](https://github.com/Akhan521)  

---
_This project was a deep dive into understanding how feature selection works with simple models like 1-NN and how cross-validation influences evaluation. I believe it reinforced both the value and limitations of building models from scratch._


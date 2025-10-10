### 34 -- Random Tree Forest
import random

### CLASS
class RandomForestClassifier():
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features=None):
        """
        Args:
            n_estimators: number of trees in the forest
            max_depth: maximum depth of each tree
            min_samples_split: minimum number of samples to split a node
            max_features: number of features to consider when looking for the best split (None = all)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feature_indices_list = []  # to store features used by each tree

    def bootstrap_sample(self, X, Y):
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        return [X[i] for i in indices], [Y[i] for i in indices]

    def select_features(self, X):
        n_features = len(X[0])
        if self.max_features is None:
            max_feats = n_features
        elif isinstance(self.max_features, float):
            max_feats = max(1, int(n_features * self.max_features))
        else:
            max_feats = self.max_features

        # Randomly select feature indices
        feature_indices = random.sample(range(n_features), max_feats)
        return feature_indices

    def _subset_features(self, X, feature_indices):
        return [[row[i] for i in feature_indices] for row in X]

    def fit(self, X, Y):
        self.trees = []
        self.feature_indices_list = []

        for _ in range(self.n_estimators):
            X_sample, Y_sample = self.bootstrap_sample(X, Y)
            feature_indices = self.select_features(X_sample)
            X_sample_subset = self._subset_features(X_sample, feature_indices)

            tree = DecisionTreeClassifier(
                minimum_samples_split=self.min_samples_split,
                maximum_depth=self.max_depth
            )
            tree.fit(X_sample_subset, Y_sample)
            self.trees.append(tree)
            self.feature_indices_list.append(feature_indices)

def majority_vote(self, predictions):
    vote_counts = {}
    for pred in predictions:
        if pred in vote_counts:
            vote_counts[pred] += 1
        else:
            vote_counts[pred] = 1
    max_votes = -1
    majority_class = None
    for label, count in vote_counts.items():
        if count > max_votes:
            max_votes = count
            majority_class = label
    return majority_class

    def predict_single(self, x):
        predictions = []
        for tree, feature_indices in zip(self.trees, self.feature_indices_list):
            x_subset = [x[i] for i in feature_indices]
            pred = tree.make_prediction(x_subset, tree.root)
            predictions.append(pred)
        return self.majority_vote(predictions)


    def predict(self, X):
        return [self.predict_single(x) for x in X]

# Author GCreus
# Done via pyzo

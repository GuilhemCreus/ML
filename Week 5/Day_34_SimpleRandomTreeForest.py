### 34 -- Random Tree Forest
import random

### CLASS
class RandomForestClassifier():
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features=None):
        """
        Initialize the Random Forest Classifier

        Args:
            n_estimators: number of trees in the forest
            max_depth: maximum depth of each tree
            min_samples_split: minimum number of samples required to split a node
            max_features: number of features to consider when looking for the best split (None = all)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []  # List to store trained decision trees
        self.feature_indices_list = []  # Stores feature indices used for each tree

    def bootstrap_sample(self, X, Y):
        """
        Generate a bootstrap sample (random sampling with replacement)

        Args:
            X: features
            Y: labels

        Returns:
            A tuple of (sampled X, sampled Y)
        """
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        return [X[i] for i in indices], [Y[i] for i in indices]

    def select_features(self, X):
        """
        Randomly select a subset of features for one tree

        Args:
            X: features (list of feature vectors)

        Returns:
            List of selected feature indices
        """
        n_features = len(X[0])
        # Determine how many features to select
        if self.max_features is None:
            max_feats = n_features  # Use all features
        elif isinstance(self.max_features, float):
            max_feats = max(1, int(n_features * self.max_features))  # Use a fraction of features
        else:
            max_feats = self.max_features  # Use fixed number of features

        # Randomly sample feature indices
        feature_indices = random.sample(range(n_features), max_feats)
        return feature_indices

    def _subset_features(self, X, feature_indices):
        """
        Keep only the selected features for each sample in X

        Args:
            X: full feature set
            feature_indices: indices of features to keep

        Returns:
            A new dataset with only selected features
        """
        return [[row[i] for i in feature_indices] for row in X]

    def fit(self, X, Y):
        """
        Train the random forest on data X and labels Y
        """
        self.trees = []
        self.feature_indices_list = []

        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, Y_sample = self.bootstrap_sample(X, Y)

            # Select features randomly
            feature_indices = self.select_features(X_sample)

            # Subset the data to selected features
            X_sample_subset = self._subset_features(X_sample, feature_indices)

            # Train a decision tree on the sampled data
            tree = DecisionTreeClassifier(
                minimum_samples_split=self.min_samples_split,
                maximum_depth=self.max_depth
            )
            tree.fit(X_sample_subset, Y_sample)

            # Store the trained tree and the feature indices it used
            self.trees.append(tree)
            self.feature_indices_list.append(feature_indices)

    def majority_vote(self, predictions):
        """
        Return the most common prediction (majority vote)

        Args:
            predictions: list of predictions from each tree

        Returns:
            The class label with the most votes
        """
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
        """
        Predict the label for a single data point

        Args:
            x: input sample (feature vector)

        Returns:
            Predicted class label
        """
        predictions = []

        # Each tree makes a prediction using only the features it was trained on
        for tree, feature_indices in zip(self.trees, self.feature_indices_list):
            x_subset = [x[i] for i in feature_indices]
            pred = tree.make_prediction(x_subset, tree.root)
            predictions.append(pred)

        # Return the majority prediction
        return self.majority_vote(predictions)

    def predict(self, X):
        """
        Predict the labels for a list of input samples

        Args:
            X: list of input samples

        Returns:
            List of predicted labels
        """
        return [self.predict_single(x) for x in X]

# Author GCreus
# Done via pyzo

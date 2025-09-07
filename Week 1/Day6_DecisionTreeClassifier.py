### Day 6 -- Simple Decision Tree
"""
Now that we have been working on the main linear regression algorithms, we will dive in another class of machine learning algorithm which is decision tree

In this code, we will try to implement a decision tree algorithm from scratch using object oriented programming because it will allow us to handle more efficiently the creation of a tree and to manage easily the nodes

Also, with object oriented programming, we will be able to easily create multiple instances of decision tree when need, for example, this will be useful when implementing random forest algorithms

As usual, we first import the main libraries
"""

import pandas as pd
import numpy as np # numpy for log2 and other useful functions
import matplotlib.pyplot as plt

### IMPORT THE DATA
"""
First, let's import the dataframe and check that the data is usable
"""
df = pd.read_csv("...your_path/CarDealerData.csv")

df.head()


### SETTING UP THE CLASS
"""
Before jumping into the class, we have to understand two important facts about decision trees :

1) Main features of decision tree are :
    -the nodes that could be a decision node or a leaf node
    -decision nodes are nodes where we will make a comparison (<; <=; >; >=) with a indepent variable value of a data point and a threshold, based on the result of the comparison we will send the data point deeper in the tree in the following left node or the following right node
    -leaf nodes are like end nodes, i.e there are the nodes without child tree and we don't make any comparison in this node, data points reach this node after making all the needed comparisons in the upper decision nodes

2) Information Gain and Entropy :
    -entropy of a node is equal to SUM(-p(i) * log(p(i))) with p(i) the probability of a data point to be classified rightfully at this node (i.e, this is number of rightfully classified data points at this node/all data points at this node)
    -information gain = entropy(parent) - SUM(wi * entropy(child(i))) with wi the weight of data points sent to the child(i) (amount of data points sent to child(i)/total datapoints at parent node); and child(i) the left or right child node

So the information gain of a whole tree is the information gain of the root which will itself call the information gain of the child nodes etc

With this in mind, we understand now what error metric we will seek to minimize
"""

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        '''
        constructor of an instance of the class

        Ags :
            feature_index = the index in the dataframe of the feature we will do the comparison
            threshold = limit value for comparison that will choose if we send the data point to the left child or right child
            left = left node child, it is like an adress that leads to the left node which is also a node; None if no left child
            right = same but for right child
            info_gain = the information gain of the node from the split
            value = value of the node (used for leaf nodes, typically the predicted class or regression value)

        Return a None
        '''

        self.predicted_value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

"""
Now we are going to create the decision tree class
This class will have functions taking data points as input; to prevent calling those functions for each data point every time, we will send as input a dataframe for X and Y (i.e a an array or list, here we will work with list)

By sending dataframe instead of data points, we will be able to easily send sub dataframe that are under the threshold by sorting the dataframe and spliting the dataframe according to the threshold
"""

class DecisionTreeClassifier():
    def __init__(self, minimum_samples_split : int = 2, maximum_depth : int = 10):
        """
        constructor of an instance of the class

        Ags :
            minimum_samples_split = a threshold on the number of datapoints at a node under which the node will automically considered a leaf due to unecessary further comparisons regarding the small portion of data to treat at that node
            maximum_depth = a threshold on the number of nodes between a leaf and the root (i.e the distance in term of nodes), if the distance is strictly higher than the threshold, the node will be considered a leaf

        Return a None
        """

        self.root = None # At first, we don't know nothing about what to compare in the root node

        self.minimum_samples_split = minimum_samples_split
        self.maximum_depth = maximum_depth



    def entropy(self, Y_rows : list[float]) -> float:
        """
        Ags :
            Y_rows = the Y values in a list format as input

        Return the entropy at that node
        """

        # We first need to check all the labels at the current node, current_labels will be a dict to stock labels and their occurences
        current_label_counts = {}

        for label in Y_rows:
            # If the label is undocumented, we add it in the current_labels list containing all the label and their occurence in Y_rows
            if label not in current_label_counts:
                current_label_counts[label] = 1
            else:
                current_label_counts[label] += 1


        entropy = 0
        total = len(Y_rows)
        for count in current_label_counts.values():
            p_label = count / total
            entropy += -p_label * np.log2(p_label)
        return entropy


    def information_gain(self, parent : list[float], left_child : list[float], right_child : list[float]) -> float:
        """
        Ags :
            parent = the actual Y values at the current node at which we are looking to calculate the information_gain
            left_child = the left child Y values
            right_child = the right child Y values


        Return the information gain at that node
        """
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        gain = self.entropy(parent) - (weight_left*self.entropy(left_child) + weight_right*self.entropy(right_child))
        return gain


    def calculate_leaf_value(self, Y_rows):
        """
        Ags :
            Y_rows = the Y values in a list format as input

        Return what predicted value should be at that node based on the what is the label the most represented at that node
        """
        current_label_counts = {}

        for label in Y_rows:
            # If the label is undocumented, we add it in the current_labels list containing all the label and their occurence in Y_rows
            if label not in current_label_counts:
                current_label_counts[label] = 1
            else:
                current_label_counts[label] += 1

        return max(current_label_counts, key=current_label_counts.get)


    def split(self, X : list[list[float]], Y_rows : list[float], feature_index : int, threshold : int) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
        """
        Args:
            X = list of input features and each input feature is itself a list of floats
            Y_rows = the Y values in a list format as input
            feature_index = the feature index to split on
            threshold = the threshold value for the split

        Returns the split for X and Y_rows based on a threshold for a specific feature index;
        return data in the following order : X_left, Y_left, X_right, Y_right
        """

        X_left, Y_left = [], []
        X_right, Y_right = [], []

        # We use zip here, zip() will associate Y_rows value to the X values by creating a list of tuple with two elements, the first will be a row of X, and the second will be the value inf Y_rows in the same row index
        for x_row, y_val in zip(X, Y_rows):
            if x_row[feature_index] <= threshold:
                X_left.append(x_row)
                Y_left.append(y_val)
            else:
                X_right.append(x_row)
                Y_right.append(y_val)

        return X_left, Y_left, X_right, Y_right

    def get_best_split(self, X : list[list[float]], Y_rows : list[float]) -> dict:
        """
        Args:
            X = list of input features and each input feature is itself a list of floats
            Y_rows = the Y values in a list format as input

        Return a dict containing the best information gain by looking at all information gain possible (brute force); the dict has the following attributes :
        - feature_index
        - threshold
        - X_left, Y_left, X_right, Y_right
        - info_gain
        """

        best_split = {}
        max_info_gain = -float("inf") # We start with the minimum float possible to prevent inputting by mistake a information gain that would be higher than every information gain the tree could provide

        number_features = len(X[0])

        for feature_index in range(number_features):
            # Brute force here, lot of computational complexity
            feature_values = [row[feature_index] for row in X]

            # We use sorted() here to prevent having duplicate
            unique_values = sorted(set(feature_values))

            for threshold in unique_values:
                X_left, Y_left, X_right, Y_right = self.split(X, Y_rows, feature_index, threshold)

                # We check here that X_left and X_right aren't empty
                if not X_left or not X_right:
                    continue

                current_information_gain = self.information_gain(Y_rows, Y_left, Y_right)
                if current_information_gain > max_info_gain:
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "X_left": X_left,
                        "Y_left": Y_left,
                        "X_right": X_right,
                        "Y_right": Y_right,
                        "info_gain": current_information_gain
                    }
                    max_info_gain = current_information_gain

        return best_split


    # And finally...
    def build_tree(self, X : list[list[float]], Y_rows : list[float], current_depth : int = 0) -> Node:
        """
        Args:
            X = list of input features and each input feature is itself a list of floats
            Y_rows = the Y values in a list format as input
            current_depth = the actual depth at which the node is located, this value is an input because the function is RECURSIVE so we pass the value of the depth from parent to child, otherwise the child couldn't know at what depth it is located

        Return a Tree with the best information gain
        """

        number_samples = len(Y_rows)

        # We check that we are within the authorized depth and sample size
        if number_samples >= self.minimum_samples_split and current_depth < self.maximum_depth:
            best = self.get_best_split(X, Y_rows)

            """
            We check that best existe, and by looking at get_best_split(), we can see that it could happen when the node has no child
            We also check that the best possible information gain is positive; if you remind yourself the formula of the information gain, you know that if the best gain is negative or zero, then no need to try to split further because it would tend to degrade the ability of the tree to correctly predict values
            """
            if best and best.get("info_gain", 0) > 0:
                left_subtree = self.build_tree(best["X_left"], best["Y_left"], current_depth + 1)
                right_subtree = self.build_tree(best["X_right"], best["Y_right"], current_depth + 1)

                return Node(
                    feature_index = best["feature_index"],
                    threshold = best["threshold"],
                    left = left_subtree,
                    right = right_subtree,
                    info_gain = best["info_gain"]
                )
        # If we don't match the limit in the first upper check, or that no split could improve the tree, we then create a leaf
        leaf_value = self.calculate_leaf_value(Y_rows)
        return Node(value = leaf_value)



    """
    We then create the two functions to fit the data and to make a prediction
    """



    def fit(self, X : list[list[float]], Y_rows : list[float]) -> None:
        """
        Args:
            X = list of input features and each input feature is itself a list of floats
            Y_rows = the Y values in a list format as input
            current_depth = the actual depth at which the node is located, this value is an input because the function is RECURSIVE so we pass the value of the depth from parent to child, otherwise the child couldn't know at what depth it is located

        Return a None but update the actual root of the current tree to create the best tree possible
        """

        self.root = self.build_tree(X, Y_rows)


    def make_prediction(self, singe_x_row : list[float], tree : Node) -> float:
        """
        Args:
            X = list of input features and each input feature is itself a list of floats
            Y_rows = the Y values in a list format as input
            current_depth = the actual depth at which the node is located, this value is an input because the function is RECURSIVE so we pass the value of the depth from parent to child, otherwise the child couldn't know at what depth it is located

        Return the predicted y value by looking at the leaf it landed on following all the comparisons in the tree
        """

        if tree.predicted_value!=None: # If we are on a leaf : return value of the leaf
            return tree.predicted_value

        # Else we look at the feature threshold of the Node and we compare it with the feature of X
        feature_val = singe_x_row[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(singe_x_row, tree.left)
        else:
            return self.make_prediction(singe_x_row, tree.right)


### CREATING A TREE INSTANCE AND FITTING THE DATA
Y = df['Purchased'].values.tolist()
X = df.iloc[:, :-1].values


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, Y)

"""
Everything seems to work, let's now remove the last two rows of our dataset and try to see if our model works well
"""

### LITTLE BIT OF TRAIN AND TEST
df2 = pd.read_csv("/Users/guilhemcreus/Desktop/Github/WEEK 1/CarDealerData.csv")

df_test = df2.tail(2)
df_train = df2.iloc[:-2]


# Training set
X_train = df_train.iloc[:, :-1].values.tolist()
Y_train = df_train['Purchased'].values.tolist()

# Test set
X_test = df_test.iloc[:, :-1].values.tolist()
Y_test = df_test['Purchased'].values.tolist()


decision_tree2 = DecisionTreeClassifier()
decision_tree2.fit(X_train, Y_train)

print(decision_tree2.make_prediction(X_test[0], decision_tree2.root), " is the predicted value, and we should have obtained", Y_test[0])

print(decision_tree2.make_prediction(X_test[1], decision_tree2.root), " is the predicted value, and we should have obtained", Y_test[1])

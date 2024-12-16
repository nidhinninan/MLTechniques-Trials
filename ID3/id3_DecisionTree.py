import numpy as np
from collections import Counter

# Training dataset
training_data = [
    {"ID": 1, "Color": "black", "Body Shape": "big", "Hair Type": "poodle", "Label": "danger"},
    {"ID": 2, "Color": "brown", "Body Shape": "big", "Hair Type": "smooth", "Label": "danger"},
    {"ID": 4, "Color": "black", "Body Shape": "small", "Hair Type": "poodle", "Label": "safe"},
    {"ID": 5, "Color": "brown", "Body Shape": "medium", "Hair Type": "smooth", "Label": "danger"},
    {"ID": 7, "Color": "brown", "Body Shape": "small", "Hair Type": "poodle", "Label": "danger"},
    {"ID": 8, "Color": "brown", "Body Shape": "small", "Hair Type": "smooth", "Label": "safe"},
    {"ID": 9, "Color": "brown", "Body Shape": "big", "Hair Type": "poodle", "Label": "danger"},
    {"ID": 10, "Color": "black", "Body Shape": "medium", "Hair Type": "poodle", "Label": "safe"},
    {"ID": 12, "Color": "black", "Body Shape": "small", "Hair Type": "smooth", "Label": "safe"},
]

# Encode categorical data
encoding = {
    "Color": {"black": 0, "brown": 1},
    "Body Shape": {"small": 0, "medium": 1, "big": 2},
    "Hair Type": {"poodle": 0, "smooth": 1},
    "Label": {"danger": 0, "safe": 1},
}

def encode_data(data):
    numerical_data = []
    for row in data:
        numerical_data.append([
            encoding["Color"][row["Color"]],
            encoding["Body Shape"][row["Body Shape"]],
            encoding["Hair Type"][row["Hair Type"]],
            encoding["Label"][row["Label"]],
        ])
    return np.array(numerical_data)

numerical_data = encode_data(training_data)

# Define functions for entropy and information gain
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    probabilities = [count / total for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def split_entropy(data, feature_index):
    total_samples = len(data)
    unique_values = np.unique(data[:, feature_index])
    weighted_entropy = 0

    for value in unique_values:
        subset = data[data[:, feature_index] == value]
        subset_labels = subset[:, -1]
        weighted_entropy += (len(subset) / total_samples) * entropy(subset_labels)

    return weighted_entropy

def information_gain(data, feature_index):
    parent_entropy = entropy(data[:, -1])
    child_entropy = split_entropy(data, feature_index)
    return parent_entropy - child_entropy

# Recursive function to build the ID3 tree
def build_tree(data, features):
    labels = data[:, -1]

    # If all labels are the same, return the label
    if len(np.unique(labels)) == 1:
        return {"label": labels[0]}

    # If no features left, return the majority label
    if len(features) == 0:
        majority_label = Counter(labels).most_common(1)[0][0]
        return {"label": majority_label}

    # Calculate information gain for each feature
    gains = [information_gain(data, i) for i in range(len(features))]
    best_feature_index = np.argmax(gains)
    best_feature = features[best_feature_index]

    # Split the data on the best feature
    tree = {"feature": best_feature, "branches": {}}
    unique_values = np.unique(data[:, best_feature_index])

    for value in unique_values:
        subset = data[data[:, best_feature_index] == value]
        subtree = build_tree(
            np.delete(subset, best_feature_index, axis=1),  # Remove the used feature
            features[:best_feature_index] + features[best_feature_index + 1:],  # Update features
        )
        tree["branches"][value] = subtree

    return tree

# Features in the dataset
features = ["Color", "Body Shape", "Hair Type"]

# Build the tree
decision_tree = build_tree(numerical_data, features)

# Print the tree
def print_tree(tree, depth=0):
    if "label" in tree:
        print("  " * depth + f"Label: {list(encoding['Label'].keys())[list(encoding['Label'].values()).index(tree['label'])]}")
    else:
        print("  " * depth + f"Feature: {tree['feature']}")
        for value, subtree in tree["branches"].items():
            print("  " * (depth + 1) + f"Value: {value}")
            print_tree(subtree, depth + 2)

print_tree(decision_tree)


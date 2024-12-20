# ID3 Algorithm

## Decision Tree Construction Using the ID3 Algorithm

This project implements the ID3 algorithm to construct a decision tree that classifies dogs into two categories: **danger** and **safe**. The training dataset includes various features such as color, body shape, and hair type. Calculations are performed step-by-step using entropy and information gain to determine the best splits. The test dataset is used to validate the constructed tree.


## Overview
The ID3 algorithm uses entropy to measure the impurity of a dataset and information gain to decide which feature provides the best split. This project ensures:

- Calculations are precise up to four decimal digits.
- If a tie occurs between features, the leftmost feature in the dataset is chosen.


## Training Dataset
| ID  | Color  | Body Shape | Hair Type | Label  |
|-----|--------|------------|-----------|--------|
| 1   | black  | big        | poodle    | danger |
| 2   | brown  | big        | smooth    | danger |
| 4   | black  | small      | poodle    | safe   |
| 5   | brown  | medium     | smooth    | danger |
| 7   | brown  | small      | poodle    | danger |
| 8   | brown  | small      | smooth    | safe   |
| 9   | brown  | big        | poodle    | danger |
| 10  | black  | medium     | poodle    | safe   |
| 12  | black  | small      | smooth    | safe   |

## Test Dataset
| ID  | Color  | Body Shape | Hair Type | Label  |
|-----|--------|------------|-----------|--------|
| 3   | brown  | medium     | poodle    | safe   |
| 6   | black  | big        | smooth    | danger |
| 11  | black  | medium     | smooth    | safe   |


## Entropy and Information Gain Calculations
1. **Parent Entropy Calculation:**
   - Formula: \( E = -\sum (p_i \times \log_2(p_i)) \)
   - Labels: `danger` and `safe`

2. **Weighted Mean of Entropy for Each Feature:**
   - Split dataset based on feature values.
   - Calculate the entropy for each subset.
   - Weight each entropy by the proportion of samples in the subset.

3. **Information Gain:**
   - Formula: \( IG = E_{parent} - E_{weighted\_mean} \)

4. **Feature Selection:**
   - Choose the feature with the highest information gain.
   - Resolve ties by selecting the leftmost feature in the dataset.

## ID3 Algorithm Implementation
The decision tree is built recursively:
1. Calculate entropy and information gain for all features.
2. Split on the feature with the highest information gain.
3. Repeat until all samples in a node belong to the same class or no features are left.

## Results
- A complete decision tree is generated and visualized.
- The tree provides predictions for the test dataset based on learned rules.

## Python Commands
The following Python commands were used to calculate entropy, information gain, and construct the decision tree:

1. **Entropy Calculation:**
   ```python
   def entropy(labels):
       total = len(labels)
       counts = Counter(labels)
       probabilities = [count / total for count in counts.values()]
       return -sum(p * np.log2(p) for p in probabilities if p > 0)
   ```

2. **Information Gain:**
   ```python
   def information_gain(data, feature_index):
       parent_entropy = entropy(data[:, -1])
       child_entropy = split_entropy(data, feature_index)
       return parent_entropy - child_entropy
   ```

3. **Tree Construction:**
   ```python
   def build_tree(data, features):
       # Base cases and recursive splitting
   ```

4. **Graphical Visualization:**
   ```python
   plt.figure(figsize=(12, 8))
   nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue")
   ```

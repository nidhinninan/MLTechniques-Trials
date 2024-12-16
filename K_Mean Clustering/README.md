# K-Means Clustering for 1D Data

This project demonstrates the K-Means clustering algorithm on a 1D dataset. The goal is to partition the dataset into three clusters using the squared Euclidean distance to assign points and compute the objective function. Cluster means are updated iteratively based on these assignments.


## Table of Contents
- [Problem Overview](#problem-overview)
- [Dataset](#dataset)
- [Algorithm Steps](#algorithm-steps)
- [Results](#results)
- [Usage](#usage)


## Problem Overview
Given the 1D dataset:  
**X = [1, 2, 2, 3, 6, 10, 11, 16, 18]**,  
partition the data into three clusters with initial means \( m_0 = 3 \), \( m_1 = 8 \), and \( m_2 = 15 \).  

The following tasks are performed:
1. Assign points to clusters based on squared Euclidean distance.
2. Update cluster means using the assigned points.
3. Compute the objective function (sum of squared distances within clusters).


## Dataset
### Initial Data Points
\[ 1, 2, 2, 3, 6, 10, 11, 16, 18 \]

### Initial Cluster Means
\[ m_0 = 3, m_1 = 8, m_2 = 15 \]


## Algorithm Steps
1. **Cluster Assignment**  
   Each data point is assigned to the cluster with the closest mean using the squared Euclidean distance.

2. **Update Means**  
   Recompute the mean of each cluster by averaging the points assigned to it.

3. **Compute Objective Function**  
   Calculate the total squared distance between each point and its cluster mean.


## Results
After running the algorithm:
- **Initial Cluster Means**: \[ 3, 8, 15 \]  
- **Cluster Assignments**: Computed for all data points.
- **Updated Cluster Means**: Updated based on current cluster assignments.
- **Objective Function (Error)**: Calculated to evaluate the quality of clustering.


## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Run the Python script:
   ```bash
   python kmeans_1d.py
   ```
3. View the output:
   - Initial and updated cluster means.
   - Cluster assignments for each data point.
   - Objective function value.


### Example Code Output
```
Initial Cluster Means: [ 3  8 15]
Cluster Assignments: [0 0 0 0 0 1 1 2 2]
Updated Cluster Means: [2.8 10.5 17. ]
Objective Function (Error): 36.9
```

This project provides a clear understanding of the K-Means algorithm for simple datasets.

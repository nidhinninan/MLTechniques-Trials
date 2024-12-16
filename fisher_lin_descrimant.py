import numpy as np
import matplotlib.pyplot as plt

# Define the data for two classes
C1 = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]])
C2 = np.array([[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]])


def compute_fisher_lda_optimal_vector(C1, C2):
    # Compute class means
    mean_C1 = np.mean(C1, axis=0)
    mean_C2 = np.mean(C2, axis=0)

    # Number of samples in each class
    n1 = len(C1)
    n2 = len(C2)

    # Step 3: Between-class scatter matrix (Sb)
    Sb = np.outer(mean_C1 - mean_C2, mean_C1 - mean_C2)

    # Compute within-class scatter matrix (Sw)
    # Use pooled covariance approach
    S_w1 = np.cov(C1.T) * (len(C1) - 1)
    S_w2 = np.cov(C2.T) * (len(C2) - 1)
    S_w = S_w1 + S_w2

    # Compute the inverse of Sw
    Sw_inv = np.linalg.inv(S_w)

    # Compute the difference between class means
    mean_diff = mean_C1 - mean_C2

    # Direct computation of optimal w
    # w is proportional to Sw_inv * (mean_C1 - mean_C2)
    w_direct = Sw_inv @ mean_diff

    # Normalize the vector
    w_normalized = w_direct / np.linalg.norm(w_direct)

    # Visualization and additional computations
    # Project data onto the optimal direction
    proj_C1 = C1 @ w_normalized
    proj_C2 = C2 @ w_normalized

    # Compute between-class and within-class variances
    mean_proj_C1 = np.mean(proj_C1)
    mean_proj_C2 = np.mean(proj_C2)
    between_class_variance = (mean_proj_C1 - mean_proj_C2) ** 2

    within_class_variance_C1 = np.var(proj_C1)
    within_class_variance_C2 = np.var(proj_C2)
    within_class_variance = (n1 * within_class_variance_C1 + n2 * within_class_variance_C2) / (n1 + n2)

    # Fisher Criterion (J)
    fisher_criterion = between_class_variance / within_class_variance

    # Additional Metrics
    class_separability = np.abs(mean_proj_C1 - mean_proj_C2)

    return {
        'optimal_w': w_direct,
        'normalized_w': w_normalized,
        'Sw_inverse': Sw_inv,
        'Sw' :S_w,
        'Sb' :Sb,
        'class_means_diff': mean_diff,
        'fisher_criterion' : fisher_criterion,
        'projected_data': {
            'C1': proj_C1,
            'C2': proj_C2
        },
        'projection_means': {
            'C1': mean_proj_C1,
            'C2': mean_proj_C2
        }
    }


# Compute Fisher LDA Optimal Vector
results = compute_fisher_lda_optimal_vector(C1, C2)

# Print Detailed Results
print("Fisher LDA Optimal Vector Derivation:")
print("\n1. Class Means:")
print("Class 1 Mean:", np.mean(C1, axis=0))
print("Class 2 Mean:", np.mean(C2, axis=0))

print("\n2. Difference Between Class Means (μ1 - μ2):")
print(results['class_means_diff'])

print("\n3. Inverse of Within-Class Scatter Matrix (Sw^-1):")
print(results['Sw_inverse'])
print(results['Sw'])
print(results['Sb'])

print("\n4. Optimal Projection Vector w:")
print("Unnormalized w (Sw^-1 * (μ1 - μ2)):", results['optimal_w'])
print("Normalized w:", results['normalized_w'])


print("\n5. Projection Details:")
print("Projected Mean of Class 1:", results['projection_means']['C1'])
print("Projected Mean of Class 2:", results['projection_means']['C2'])
print(results['fisher_criterion'])
# Visualization
plt.figure(figsize=(12, 5))

# Original Data Space
plt.subplot(121)
plt.scatter(C1[:, 0], C1[:, 1], color='blue', label='Class 1')
plt.scatter(C2[:, 0], C2[:, 1], color='red', label='Class 2')
plt.title('Original Data Space')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Projected Space
plt.subplot(122)
plt.scatter(results['projected_data']['C1'],
            np.zeros_like(results['projected_data']['C1']),
            color='blue', label='Class 1')
plt.scatter(results['projected_data']['C2'],
            np.zeros_like(results['projected_data']['C2']),
            color='red', label='Class 2')
plt.title('Projected Space')
plt.xlabel('Discriminant Direction')
plt.legend()

plt.tight_layout()
plt.show()

# Theoretical Verification
print("\n6. Theoretical Verification:")
print("Verifying: w ∝ Sw^-1 * (μ1 - μ2)")
print("Sw^-1 * (μ1 - μ2):", results['Sw_inverse'] @ results['class_means_diff'])
print("Optimal w:", results['optimal_w'])
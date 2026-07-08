import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    #Step 1: standardization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data-mean)/(std+1e-8)
    
    #Step 2:Covariance matrix calculation, how each feature is related to the other, always a square matrix
    cov_matrix = np.cov(standardized_data, rowvar=False)

    #Step 3: EigenVector and values calculation
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    W = sorted_eigenvectors[:, :k]

    for i in range(k):
        non_zero_indices = np.where(np.abs(W[:, i]) > 1e-8)[0]
        
        if len(non_zero_indices) > 0:
            first_idx = non_zero_indices[0]
            if W[first_idx, i] < 0:
                W[:, i] *= -1

    return np.round(W, 4)

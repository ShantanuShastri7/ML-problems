import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    theta = np.linalg.inv(X_arr.T @ X_arr) @ X_arr.T @ y_arr
    
    theta_rounded = np.round(theta, 4)
    
    return theta_rounded.tolist()
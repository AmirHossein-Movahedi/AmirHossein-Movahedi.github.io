import numpy as np


def gradient_descent(A, b, lr=None, tol=1e-6, max_iters=1000):
    m, n = A.shape
    x = np.zeros(n)  # Initialize x 

    # Compute optimal learning rate (1 / largest eigenvalue of A^T A)
    if lr is None:
        eigenvalues = np.linalg.eigvalsh(A.T @ A)
        lr = 1.0 / np.max(eigenvalues)  

    for _ in range(max_iters):
        grad = 2 * A.T @ (A @ x - b)  # Compute gradient
        x_new = x - lr * grad  # Gradient descent step

        # Check for convergence
        if np.linalg.norm(grad) < tol:  # Stop if gradient is small
            break
        if np.linalg.norm(x_new - x) < tol:  # Stop if x doesn't change much
            break

        x = x_new 
    return x


# Example Usage
A = np.array([[3, 2], [4, 1], [5, 3]]) 
b = np.array([1, 2, 3])  

x_opt = gradient_descent(A, b)
print("\nOptimal x:", x_opt)


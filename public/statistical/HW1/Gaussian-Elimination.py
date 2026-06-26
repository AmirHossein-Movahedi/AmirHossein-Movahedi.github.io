import numpy as np

def gaussian_elimination_partial_pivoting(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    
    # Forward Elimination
    for i in range(n):
        max_row_index = np.argmax(np.abs(A[i:, i])) + i
        if A[max_row_index, i] == 0:
            raise ValueError("Solution error")
        # Change the row
        A[[i, max_row_index]] = A[[max_row_index, i]]
        b[[i, max_row_index]] = b[[max_row_index, i]]
        
        # Remove matrix element 
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    
    return x



A = np.array([[2, 4.001],
                  [1, 2]])
b = np.array([1, 2])
    
solution = gaussian_elimination_partial_pivoting(A, b)
print(solution)



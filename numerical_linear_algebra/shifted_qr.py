#!/usr/bin/python3
"""

"""
import numpy as np

def wilkinson_shift(a, b, c):
    """ Compute Wilkinson shift for the 2x2 bottom-right block of a tridiagonal matrix.
        Given elements: a = T[l, l], b = T[l+1, l], c = T[l+1, l+1]. """
    delta = (a - c) / 2
    sign = np.sign(delta) if delta != 0 else 1
    mu = c - sign * b**2 / (abs(delta) + np.sqrt(delta**2 + b**2))
    return mu

def householder_vector(x):
    """ Constructs a Householder vector v to zero out all but the first element of x. """
    v = x.copy()
    v[0] += np.sign(x[0]) * np.linalg.norm(x)
    v /= np.linalg.norm(v)
    return v

def qr_iteration_tridiagonal(T, tol=1e-16, max_iter=1000):
    """ Performs QR iteration with Wilkinson shifts on a symmetric tridiagonal matrix T.
        Returns the matrix of eigenvalues (diagonal) and eigenvectors (Q). """
    m = T.shape[0]
    Q = np.eye(m)
    W = np.zeros((2, m-1))  # Matrix to store Householder vectors

    # Main loop to reduce T to nearly diagonal form
    l = m - 1
    iter_count = 0
    while l > 0 and iter_count < max_iter:
        iter_count += 1
        if abs(T[l, l-1]) < tol:
            # Deflate and work on a smaller subproblem
            T[l, l-1] = 0
            T[l-1, l] = 0
            l -= 1
            continue

        # Compute the Wilkinson shift using the bottom-right 2x2 block
        sigma = wilkinson_shift(T[l-1, l-1], T[l, l-1], T[l, l])

        # Shift T by subtracting sigma
        T -= sigma * np.eye(m)

        # QR decomposition of shifted matrix using Householder transformations
        for j in range(l):
            x = T[j:j+2, j]
            v = householder_vector(x)
            W[:, j] = v

            # Apply the transformation to T on the right side
            T[j:j+2, j:j+3] -= 2 * np.outer(v, v.T @ T[j:j+2, j:j+3])
            T[j+1, j] = 0  # Explicitly zero out subdiagonal element

            # Apply the transformation to Q (update eigenvector matrix)
            Q[:, j:j+2] -= 2 * np.outer(Q[:, j:j+2] @ v, v)

        # Reverse the shift by adding sigma back
        T += sigma * np.eye(m)

        # Accumulate RQ to complete the A = RQ + σI step
        for j in range(l):
            v = W[:, j]
            T[j:j+2, max(0, j-1):j+2] -= 2 * np.outer(v, v.T @ T[j:j+2, max(0, j-1):j+2])

        # Debugging: Print the current value of T[l, l-1]
        print(f"Iteration {iter_count}: T[{l}, {l-1}] = {T[l, l-1]}")

    if iter_count == max_iter:
        print("Warning: Maximum number of iterations reached.")

    # At the end of the process, T should be nearly diagonal, and Q should hold eigenvectors
    return Q, np.diag(T)

def create_tridiagonal_matrix(m, diag_value=2.0, off_diag_value=-1.0):
    """ Creates an m x m tridiagonal matrix with `diag_value` on the diagonal
        and `off_diag_value` on the sub- and superdiagonals. """
    A = np.diag([diag_value] * m)
    for i in range(m - 1):
        A[i, i+1] = off_diag_value
        A[i+1, i] = off_diag_value
    return A


if __name__ == "__main__":
    # Define matrix size
    m = 10
    A = create_tridiagonal_matrix(m)
    # Run QR iteration with Wilkinson shift on matrix A
    Q, D = qr_iteration_tridiagonal(A)
    # Define exact eigenvalues based on the analytical formula
    exact_eigenvalues = np.array([2 - 2 * np.cos(np.pi * k / (m + 1)) for k in range(1, m + 1)])

    # Forward Eigenvalue Error
    computed_eigenvalues = np.sort(D)  # Sort to match with exact eigenvalues
    forward_errors = np.abs(computed_eigenvalues - exact_eigenvalues)

    # Backward Error
    A_approx = Q @ np.diag(computed_eigenvalues) @ Q.T
    backward_error = np.linalg.norm(A_approx - A, ord=np.inf) / np.linalg.norm(A, ord=np.inf)

    # Orthogonality Error
    orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(m), ord=np.inf)

    # Print Errors
    print("Forward Eigenvalue Errors:", forward_errors)
    print("Max Forward Error:", np.max(forward_errors))
    print("Relative Backward Error:", backward_error)
    print("Orthogonality Error:", orthogonality_error)
#!/usr/bin/python3
import numpy as np
import scipy.sparse as sp
from typing import Sequence
from standard_lanczos import generate_matrix_sample, random_symmetric_matrix
from standard_lanczos import relative_ritz_pair_residual


def lanczos(A, m, b=None):
    """
    Lanczos iterative method to approximate eigenvalues and eigenvectors of a symmetric matrix A.

    Parameters:
    A (numpy.ndarray): Sparse symmetric matrix (n x n) over R
    m (int): Number of Lanczos iterations (size of the Krylov subspace)
    b (numpy.ndarray): Optional, initial vector for the first Lanczos vector (should be of size n)

    Returns:
    T_m (numpy.ndarray): Tridiagonal matrix of size (m x m) approximating A in the Krylov subspace
    Q_m (numpy.ndarray): Matrix of Lanczos vectors (n x m)
    """
    n = A.shape[0]
    if b is None:
        # Start with a random initial vector b of unit norm
        b = np.random.randn(n)
    
    # Initialize storage for alpha, beta, and Lanczos vectors
    alphas = np.zeros(m)
    betas = np.zeros(m - 1)
    Q_m = np.zeros((n, m))

    # First Lanczos vector
    beta_prev = 0
    q_prev = np.zeros(n)
    q_curr = b / np.linalg.norm(b)

    # Perform Lanczos iterations
    for j in range(m):
        # Store the current Lanczos vector
        Q_m[:, j] = q_curr

        # Apply A to the current Lanczos vector
        v = A @ q_curr

        # Compute and store alpha
        alpha = np.dot(q_curr, v)
        alphas[j] = alpha

        # Orthogonalize v against the current Lanczos vector
        v = v - (beta_prev * q_prev) - (alpha * q_curr)

        # Compute beta
        if j < m - 1:
            beta = np.linalg.norm(v)
            betas[j] = beta

            if beta < 1e-16:
                break

            # Normalize the new Lanczos vector
            q_prev = q_curr
            q_curr = v / beta
            beta_prev = beta

    # Construct the tridiagonal matrix T_m
    T_m = np.zeros((m, m))
    np.fill_diagonal(T_m, alphas)
    np.fill_diagonal(T_m[:-1, 1:], betas)
    np.fill_diagonal(T_m[1:, :-1], betas)

    return T_m, Q_m

def simple_restart_lanczos(A, m, max_restarts, tol=1e-6):
    """
    Lanczos method with simple restarting to find eigenvalues of A.

    Parameters:
    A (numpy.ndarray): Symmetric matrix (n x n)
    m (int): Number of Lanczos iterations before restart
    max_restarts (int): Maximum number of restarts
    tol (float): Tolerance for convergence of the Ritz values

    Returns:
    eigenvalues (numpy.ndarray): Approximate eigenvalues of A
    eigenvectors (numpy.ndarray): Approximate eigenvectors of A
    """
    n = A.shape[0]
    b = np.random.randn(n)
    b /= np.linalg.norm(b)

    for restart in range(max_restarts):
        # Run Lanczos algorithm for m steps
        T_m, Q_m = lanczos(A, m, b)
        
        # Compute eigenvalues and eigenvectors of T_m (Ritz values and vectors)
        eigvals_Tm, eigvecs_Tm = np.linalg.eigh(T_m)
        
        # Compute Ritz vectors (approximate eigenvectors of A)
        eigvecs_A = Q_m @ eigvecs_Tm

        # Calculate relative residuals for each Ritz pair
        residuals = relative_ritz_pair_residual(A, eigvals_Tm, eigvecs_A)

        # Check if any residuals are below the tolerance
        if np.all(residuals < tol):
            print(f"Converged after {restart+1} restarts.")
            return eigvals_Tm, eigvecs_A

        # Restart with the Ritz vector with the smallest residual
        min_residual_index = np.argmin(residuals)
        b = eigvecs_A[:, min_residual_index]

        print(f"Restarting with Ritz vector {min_residual_index} (residual: {residuals[min_residual_index]})")

    print("Reached maximum number of restarts.")
    return eigvals_Tm, eigvecs_A

if __name__ == "__main__":
    #A = generate_matrix_sample()
    A = random_symmetric_matrix(2000, density=0.01).toarray()
    m = 100
    max_restarts = 10
    tol = 1e-6

    eigenvalues, eigenvectors = simple_restart_lanczos(A, m, max_restarts, tol)
    residuals = relative_ritz_pair_residual(A, eigenvalues, eigenvectors)
    print(f"Relative Ritz Residuals: {residuals}")
    print("Eigenvalues:", eigenvalues)
    

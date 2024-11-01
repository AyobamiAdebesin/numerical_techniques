#!/usr/bin/env python3
""" Lanczos Iterative method for solving a standard eigenvalue problem """
import numpy as np
import scipy
import scipy.io as sio
import os
import sys
import time

from typing import Sequence, Tuple, Any, List, Mapping
import scipy.sparse as sp

def lanczos(A, m, b=None):
    """
    Extended Lanczos iterative method to approximate eigenvalues and eigenvectors of a symmetric matrix A,
    and construct Q_m, Q_{m+1}, T_m, and T_m tilde.

    Parameters:
    A (numpy.ndarray): Sparse symmetric matrix (n x n) over R
    m (int): Number of Lanczos iterations (size of the Krylov subspace)
    b (numpy.ndarray): Optional, initial vector for the first Lanczos vector (should be of size n)

    Returns:
    T_m (numpy.ndarray): Tridiagonal matrix of size (m x m) approximating A in the Krylov subspace
    T_m_tilde (numpy.ndarray): Extended tridiagonal matrix of size ((m+1) x m) approximating A in the Krylov subspace
    Q_m (numpy.ndarray): Matrix of Lanczos vectors (n x m)
    Q_m_plus_1 (numpy.ndarray): Matrix of Lanczos vectors (n x (m+1)) including the next Lanczos vector q_{n+1}
    """
    n = A.shape[0]
    if b is None:
        # Start with a random initial vector b of unit norm
        b = np.random.randn(n)
    
    # Initialize storage for alpha, beta, and Lanczos vectors
    alphas = np.zeros(m)
    betas = np.zeros(m)
    Q_m_plus_1 = np.zeros((n, m + 1))  # Store m+1 Lanczos vectors

    # First Lanczos vector
    beta_prev = 0
    q_prev = np.zeros(n)
    q_curr = b / np.linalg.norm(b)

    # Perform Lanczos iterations (including one extra iteration)
    for j in range(m + 1):  # We need m+1 Lanczos vectors
        # Store the current Lanczos vector
        Q_m_plus_1[:, j] = q_curr

        # Apply A to the current Lanczos vector
        v = A @ q_curr

        if j < m:  # For first m iterations, compute alpha and beta
            # Compute and store alpha (Rayleigh quotient)
            alpha = np.dot(q_curr, v)
            alphas[j] = alpha

            # Orthogonalize v against the current Lanczos vector
            v = v - (beta_prev * q_prev) - (alpha * q_curr)

            # Reorthogonalization (optional)
            v -= Q_m_plus_1[:, :j+1] @ (Q_m_plus_1[:, :j+1].T @ v)

            # Compute beta
            beta = np.linalg.norm(v)
            betas[j] = beta

            if beta < 1e-16:
                break  # Terminate if beta becomes too small

            # Normalize the new Lanczos vector
            q_prev = q_curr
            q_curr = v / beta
            beta_prev = beta
        else:
            # For the last iteration, we don't compute alphas or betas, just store the Lanczos vector q_{n+1}
            break

    # Construct the tridiagonal matrix T_m of size (m x m)
    T_m = np.zeros((m, m))
    np.fill_diagonal(T_m, alphas)
    np.fill_diagonal(T_m[:-1, 1:], betas[:m - 1])
    np.fill_diagonal(T_m[1:, :-1], betas[:m - 1])

    # Construct the extended tridiagonal matrix T_m_tilde of size (m+1 x m)
    T_m_tilde = np.zeros((m + 1, m))
    np.fill_diagonal(T_m_tilde[:m, :], alphas)
    np.fill_diagonal(T_m_tilde[:m, 1:], betas[:m - 1])
    np.fill_diagonal(T_m_tilde[1:, :m - 1], betas[:m - 1])

    # Add the last beta_n for q_{n+1}
    T_m_tilde[m, m - 1] = betas[m - 1] if m > 1 else betas[0]

    # Return the first m Lanczos vectors (Q_m), the first m+1 Lanczos vectors (Q_{m+1}),
    # the tridiagonal matrix T_m, and the extended tridiagonal matrix T_m tilde.
    Q_m = Q_m_plus_1[:, :m]

    return T_m, T_m_tilde, Q_m, Q_m_plus_1


def compute_eigenvalues_and_vectors(A: np.ndarray, m: int) -> Sequence:
    """
    Computes the approximate eigenvalues and eigenvectors of a symmetric matrix A using the Lanczos algorithm.

    Parameters:
    A (numpy.ndarray): Symmetric matrix (n x n)
    m (int): Number of Lanczos iterations

    Returns:
    eigenvalues (numpy.ndarray): Approximated eigenvalues
    eigenvectors (numpy.ndarray): Approximated eigenvectors
    """
    # Perform Lanczos iteration to get T_m and V_m
    T_m, T_m_tilde, Q_m, Q_m_plus_1 = lanczos(A, m)
    
    test_accuracy(A, Q_m, Q_m_plus_1, T_m_tilde)

    # Solve the eigenvalue problem for the tridiagonal matrix T_m
    eigvals_Tm, eigvecs_Tm = np.linalg.eigh(T_m)

    # Compute approximate eigenvectors of A
    eigvecs_A = Q_m @ eigvecs_Tm

    return (eigvals_Tm, eigvecs_A)

# def relative_ritz_pair_residual(A, eigenvalues, eigenvectors):
#     """ Calculate the relative residual for a ritz pair """
#     norm_A = np.linalg.norm(A)
#     residuals = []
#     for i in range(eigenvalues.shape[0]):
#         num = np.linalg.norm((A @ eigenvectors[:, i]) -( eigenvalues[i] * eigenvectors[:, i]))
#         den = (norm_A + np.abs(eigenvalues[i])) * np.linalg.norm(eigenvectors[:, i])
#         residuals.append(num/den)
#     return residuals

def relative_ritz_pair_residual(A, eigenvalues, eigenvectors):
    """Calculate the relative residuals for the Ritz pairs."""
    norm_A = np.linalg.norm(A)
    residual_vectors = A @ eigenvectors - eigenvectors * eigenvalues
    num = np.linalg.norm(residual_vectors, axis=0)
    eigvec_norms = np.linalg.norm(eigenvectors, axis=0)
    den = (norm_A + np.abs(eigenvalues)) * eigvec_norms
    residuals = num / den
    return residuals


def random_symmetric_matrix(n, density=0.1, random_state=None):
    """
    Generate a random sparse symmetric matrix.

    Args:
        n (int): Size of the matrix (n x n).
        density (float): Density of the non-zero elements.
        random_state (int, optional): Seed for the random number generator.

    Returns:
        scipy.sparse.csr_matrix: Random sparse symmetric matrix.
    """
    rng = np.random.default_rng(random_state)
    A = sp.random(n, n, density=density, format='csr', random_state=rng)
    A = (A + A.T) / 2
    return A

def test_accuracy(A, Q_m, Q_m_plus_1, T_m_tilde):
    """
    Test the accuracy of the Lanczos method.

    Args:
        A (numpy.ndarray): Symmetric matrix (n x n).
        Q (numpy.ndarray): Matrix of Lanczos vectors (n x m).
        T (numpy.ndarray): Tridiagonal matrix of size (m x m) approximating A in the Krylov subspace.
    """
    lhs = A @ Q_m
    rhs = Q_m_plus_1 @ T_m_tilde

    diff = np.linalg.norm(lhs - rhs)
    print(f"Accuracy test: {diff}")
    return
    

def generate_matrix_sample():
    diagonal_values  = np.arange(0, 2.01, 0.01)
    diagonal_values = np.append(diagonal_values, [2.5, 3.0])
    assert len(diagonal_values) == 203, "Diagonal values must be 203"
    A = np.diag(diagonal_values)
    return A


if __name__ == "__main__":
    #matrix_A = sio.mmread("./matrix_market/bcsstk11.mtx")
    #A = matrix_A.toarray()
    A = random_symmetric_matrix(2000, density=0.01).toarray()
    
    #A = generate_matrix_sample()
    A[0, 0] = -10
    print(f"Matrix A: {A.shape}")

    # Number of Lanczos iterations (size of Krylov subspace)
    m = 20

    eigenvalues, eigenvectors = compute_eigenvalues_and_vectors(A, m)
    print(eigenvalues)
    rel_residual = relative_ritz_pair_residual(A, eigenvalues, eigenvectors)
    print(f"Relative Ritz pair residual: {rel_residual}")
    

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
    Lanczos iterative method to approximate eigenvalues and eigenvectors of a symmetric matrix A.

    Parameters:
    A (numpy.ndarray): Sparse Symmetric matrix (n x n) over R
    m (int): Number of Lanczos iterations (size of the Krylov subspace)
    v0 (numpy.ndarray): Optional, initial guess for the first Lanczos vector (should be of size n)

    Returns:
    T_m (numpy.ndarray): Tridiagonal matrix of size (m x m) approximating A in the Krylov subspace
    V_m (numpy.ndarray): Matrix of Lanczos vectors (n x m)
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

        # Compute and store alpha (Rayleigh quotient)
        alpha = np.dot(q_curr, v)
        alphas[j] = alpha

        # Orthogonalize w against the current Lanczos vector
        v = v - (beta_prev * q_prev) - (alpha * q_curr)

        # reorthogonalization
        v -= Q_m[:, :j] @ (Q_m[:, :j].T @ v)

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


def check_condition_number(A: np.ndarray) -> float:
    cond_num = np.linalg.cond(A)

    return cond_num


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
    T_m, Q_m = lanczos(A, m)

    print(f"Q_m: {Q_m}")

    # Solve the eigenvalue problem for the tridiagonal matrix T_m
    eigvals_Tm, eigvecs_Tm = np.linalg.eigh(T_m)

    # Compute approximate eigenvectors of A
    eigvecs_A = Q_m @ eigvecs_Tm

    return (eigvals_Tm, eigvecs_A)


def compute_relative_error(A: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> float:
    """
    Compute the residual error

    Parameters:
    A (numpy.ndarray): Symmetric matrix (n x n)
    eigenvalues (numpy.ndarray): Approximated eigenvalues of A
    eigenvectors (numpy.ndarray): Approximated eigenvectors of A

    Returns:
        error (float): Residual error between A and the approximated eigenvalues and eigenvectors
    """
    rel_error = (np.linalg.norm(A @ eigenvectors - eigenvectors @ np.diag(eigenvalues))) / np.linalg.norm(A) * np.linalg.norm(eigenvectors)

    return rel_error


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

def generate_matrix_sample():
    diagonal_values  = np.arange(0, 2.01, 0.01)
    diagonal_values = np.append(diagonal_values, [2.5, 3.0])
    assert len(diagonal_values) == 203, "Diagonal values must be 203"
    A = np.diag(diagonal_values)
    return A


if __name__ == "__main__":
    # matrix_A = sio.mmread("./matrix_market/bcsstk11.mtx")
    # A = matrix_A.toarray()
    #A = random_symmetric_matrix(2000, density=0.01).toarray()
    
    A = generate_matrix_sample()
    print(f"Matrix A: {A.shape}")
    print(f"Condition number of A: {check_condition_number(A)}")

    # Number of Lanczos iterations (size of Krylov subspace)
    m = 20

    eigenvalues, eigenvectors = compute_eigenvalues_and_vectors(A, m)

    print(f"Shape of eigenvalues: {eigenvalues.shape}")
    print(f"Shape of eigenvectors: {eigenvectors.shape}")
    print(f"eigenvaluess: {eigenvalues}")
    e_vals, e_vecs = np.linalg.eigh(A)
    

    rel_error = compute_relative_error(A, eigenvalues, eigenvectors)
    print(f"Relative error: {rel_error}")

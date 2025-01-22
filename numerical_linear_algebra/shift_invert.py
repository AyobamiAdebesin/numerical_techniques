#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import os
import sys
import time

from typing import Sequence, Tuple, Any, List, Mapping
from standard_lanczos import generate_matrix_sample, relative_ritz_pair_residual, test_accuracy
from standard_lanczos import random_symmetric_matrix

def lanczos_shift_invert(A: np.ndarray, sigma: float, m: int) -> Tuple[float, np.ndarray]:      
    """
    Lanczos shift-invert method to find the eigenvalues close to a shift.

    Parameters:
    A (numpy.ndarray): Symmetric matrix (n x n)
    sigma (float): Shift value to focus on the smallest eigenvalue
    m (int): Number of Lanczos iterations

    Returns:
    smallest_eigenvalue (float): Approximated smallest eigenvalue of A
    eigenvector (numpy.ndarray): Approximated eigenvector corresponding to the smallest eigenvalue
    """
    n = A.shape[0]
    q0 = np.random.randn(n)
    q0 /= np.linalg.norm(q0)

    alphas = np.zeros(m)
    betas = np.zeros(m - 1)
    Q_m = np.zeros((n, m))

    q_prev = np.zeros(n)
    q_curr = q0

    for j in range(m):
        Q_m[:, j] = q_curr

        # Apply the shifted inverse of A
        w = np.linalg.solve(A - sigma * np.eye(n), q_curr)

        # Orthogonalize w against previous Lanczos vectors
        for k in range(j + 1):
            w -= np.dot(Q_m[:, k], w) * Q_m[:, k]

        # Compute alpha and beta
        alpha = np.dot(q_curr, w)
        alphas[j] = alpha

        if j < m - 1:
            beta = np.linalg.norm(w)
            betas[j] = beta

            if beta == 0:
                break

            q_prev = q_curr
            q_curr = w / beta

    # Construct the tridiagonal matrix T_m
    T_m = np.diag(alphas) + np.diag(betas, k=1) + np.diag(betas, k=-1)

    # Solve the eigenvalue problem for T_m
    eigvals_Tm, eigvecs_Tm = np.linalg.eigh(T_m)

    # Compute approximate eigenvectors of A
    eigvecs_A = Q_m @ eigvecs_Tm
    eigvals_A = sigma + 1 / eigvals_Tm

    # The smallest eigenvalue of A is the largest eigenvalue of T_m
    # smallest_eigenvalue = eigvals_A[-1]
    # eigenvector = eigvecs_A[:, -1]

    return eigvecs_A, eigvals_A


if __name__ == "__main__":
    #A = generate_matrix_sample()
    A = random_symmetric_matrix(2000, density=0.01).toarray()
    # Parameters
    sigma = 0.5
    m = 20

    # Compute the smallest eigenvalue and its corresponding eigenvector
    eigvecs, eigvals = lanczos_shift_invert(A, sigma, m)
    residuals = relative_ritz_pair_residual(A, eigenvalues=eigvals, eigenvectors=eigvecs)
    print(f"Eigenvalues: {eigvals}")
    print(f"Relative Residual: {residuals}")

    

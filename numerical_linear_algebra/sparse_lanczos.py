#!/usr/bin/python3

"""
This script contains the implementation of the Shift-Invert Lanczos algorithm
for finding the eigenvalues of sparse matrix near a shift. This implementation follows
from the pseudocode algorithm from the lecture notebook.

Author: Ayobami Adebesin
Date: 25th November 2024

Usage:
    python3 sparse_lanczos.py --reortho --refine

    where:
        --reortho - flag for reorthogonolization. Default is False
        --refine - flafg for iterative refinement. Default is False
    
    (./sparse_lanczos.py --reortho --refine on a unix system)
    
"""
""" """
import numpy as np
import scipy.sparse as sp
import argparse
from scipy.sparse import csr_matrix
from qr_iteration import create_tridiagonal_matrix
from typing import Callable, List, Sequence, Tuple
from collections import Counter


def make_mul_inv(A: csr_matrix, sigma: float, m: int) -> Callable:
    args = argument_parser()
    F = sp.linalg.splu(A - sigma * sp.eye(m))
    def mul_invA(B):
        Y = F.solve(B)
        if args.refine:
            R = A@Y - sigma * Y -B
            return Y-F.solve(R)
        else:
            return Y
    return mul_invA

def lanczos(mul_invA: Callable, n: int, b):
    """
    Extended Lanczos iterative method to approximate eigenvalues and eigenvectors of a symmetric matrix A,
    and construct Q_m, Q_{m+1}, T_m, and T_m tilde.

    Parameters:
    A (numpy.ndarray): Sparse symmetric matrix (m x m) over R
    n (int): Number of Lanczos iterations (size of the Krylov subspace)
    b (numpy.ndarray): Optional, initial vector for the first Lanczos vector (should be of size m)

    Returns:
    T_n (numpy.ndarray): Tridiagonal matrix of size (n x n) approximating A in the Krylov subspace
    T_n_tilde (numpy.ndarray): Extended tridiagonal matrix of size ((n+1) x n) approximating A in the Krylov subspace
    Q_n (numpy.ndarray): Matrix of Lanczos vectors (m x n)
    Q_n_plus_1 (numpy.ndarray): Matrix of Lanczos vectors (m x (n+1)) including the next Lanczos vector q_{n+1}
    """
    args = argument_parser()

    m = b.shape[0]
    if b is None:
        raise ValueError("b cannot be None!")
    
    # Initialize storage for alpha, beta, and Lanczos vectors
    alphas = np.zeros(n)
    betas = np.zeros(n)
    Q_n_plus_1 = np.zeros((m, n + 1))

    beta_prev = 0
    q_prev = np.zeros(m)
    q_curr = b / np.linalg.norm(b)

    # Perform Lanczos iterations
    for j in range(n + 1):
        Q_n_plus_1[:, j] = q_curr
        v = mul_invA(q_curr)
        if j < n:
            # Compute and store alpha (Rayleigh quotient)
            alpha = np.dot(q_curr, v)
            alphas[j] = alpha
            
            # Orthogonalize v against the current Lanczos vector
            v = v - (beta_prev * q_prev) - (alpha * q_curr)

            if args.reortho:
                # Reorthogonalization
                v -= Q_n_plus_1[:, :j+1] @ (Q_n_plus_1[:, :j+1].T @ v)

            # Compute beta
            beta = np.linalg.norm(v)
            betas[j] = beta

            if beta < 1e-16:
                break

            # Normalize the new Lanczos vector
            q_prev = q_curr
            q_curr = v / beta
            beta_prev = beta
        else:
            # For the last iteration, we don't compute alphas or betas, just store the Lanczos vector q_{n+1}
            break

    # Construct the tridiagonal matrix T_m of size (n x n)
    T_n = np.zeros((n, n))
    np.fill_diagonal(T_n, alphas)
    np.fill_diagonal(T_n[:-1, 1:], betas[:n - 1])
    np.fill_diagonal(T_n[1:, :-1], betas[:n - 1])

    # Construct the extended tridiagonal matrix T_n_tilde of size (n+1 x n)
    T_n_tilde = np.zeros((n + 1, n))
    np.fill_diagonal(T_n_tilde[:n, :], alphas)
    np.fill_diagonal(T_n_tilde[:n, 1:], betas[:n - 1])
    np.fill_diagonal(T_n_tilde[1:, :n - 1], betas[:n - 1])

    # Add the last beta_n for q_{n+1}
    T_n_tilde[n, n - 1] = betas[n - 1] if n > 1 else betas[0] 

    # Return the first n Lanczos vectors (Q_n), the first n+1 Lanczos vectors (Q_{n+1}),
    # the tridiagonal matrix T_n, and the extended tridiagonal matrix T_n tilde.
    Q_n = Q_n_plus_1[:, :n]

    return T_n, T_n_tilde, Q_n, Q_n_plus_1

def construct_A(l: int):
    """ Construct the matrix A """
    # form matrices
    v = np.ones(l**2)
    A1 = sp.spdiags([-v, 2*v, -v], [-1, 0, 1], l, l)
    I_l = sp.eye(l)
    A = sp.kron(I_l, A1, format='csc') + sp.kron(A1, I_l)
    return A

def count_multiplicities(eigenvalues):
    """ count eigenvalues with multiplicity 2"""
    counts = Counter(eigenvalues)
    mult_2 = [num for num, count in counts.items() if count == 2]
    return len(mult_2)

def argument_parser():
    """ Parse argument """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reortho", help="Option to include reorthogonalization", default=False, action="store_true")
    parser.add_argument("--refine", help="Option to include iterative refinement", default=False, action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argument_parser()
    is_reortho = "with" if args.reortho else "without"
    is_refine = "with" if args.refine else "without"
    print(f"Starting Lanczos algorithm {is_reortho} reorthogonalization and {is_refine} iterative refinement...")

    # parameters
    np.random.seed(256)
    l = 305
    m = l**2
    n = 100
    sigma = 6.001
    b = np.random.randn(m)

    # construct A
    A = construct_A(l)

    # perform lanczos
    mul_invA = make_mul_inv(A, sigma, m)
    T_n, _, Q_n, _ = lanczos(mul_invA=mul_invA, n=n, b=b)
    # compute eigenvalues and eigenvectors
    eigvals_Tn, eigvecs_Tn = np.linalg.eigh(T_n)
    eigvecs_A = Q_n @ eigvecs_Tn
    eigvals_A = sigma + 1/eigvals_Tn

    # check orthogonality of Ritz vectors
    I = np.eye(n)
    orth_err = np.linalg.norm(eigvecs_A.T @ eigvecs_A - I)
    print(f"\nOrthogonality Error: {orth_err}")

    # check condition number
    print(f"\nCondition number of Y: {np.linalg.cond(eigvecs_A)}")

    # count eigenvalues with multiplicity 2
    cnt = count_multiplicities(eigvals_A)
    print(f"\nNumber of eigenvalues with multiplicity 2: {cnt}")

    # check residual
    res = np.linalg.norm(A@eigvecs_A - eigvecs_A * eigvals_A, axis=0)
    #print(f"\nResiduals : {res}")
    
    # print eigenvalues
    #print(f"\nEigenvalues: {(eigvals_A)}")

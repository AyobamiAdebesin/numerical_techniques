#!/usr/bin/env python3
""" Cholesky method for solving a generalized eigenvalue problem """

import numpy as np
import scipy
import scipy.io as sio
import os
import sys

from typing import Sequence, List, Any, Dict, Tuple
from scipy.linalg import solve_triangular, cholesky
from scipy.linalg import eigvals

def is_positive_definite(A: np.ndarray) -> bool:
    """ Check if a matrix is positive definite using the eigenvalue property """
    if A is not None:
        return np.all(np.linalg.eigvals(A) > 0)


def cholesky_factorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factorize A using Cholesky Decomposition algorithm

    Args:
        A (np.ndarray): nxn upper triangular matrix over R

    Return:
        The lower triangular matrix L of the decomposition and its transpose L.T
    """
    # Input validation
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    # Check for positive definiteness of A
    # if is_positive_definite(A) == False:
    #     raise ValueError("A must be positive definite")

    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i+1):
            temp_sum = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - temp_sum)
            else:
                L[i][j] = (A[i][j] - temp_sum) / L[j][j]
    return (L, L.T)


def cholesky_solver(A: np.ndarray, B: np.ndarray):
    """ Compute the eigenvalue and eigenvectors of a generalized Eigenvalue problem using Cholesky factorization """
    try:
        L = cholesky(B, lower=True)
        L_t = L.T
        if L is None or L_t is None:
            print("Cholesky factorization unsuccessful!")
            sys.exit(1)
        Linv_A = solve_triangular(L, A, lower=True)
        A_hat = solve_triangular(L.T, Linv_A.T, lower=False).T
        eigenvalues = eigvals(A_hat)
        if eigenvalues is None:
            print("Eigenvalue computation unsuccessful!")
            sys.exit(1)
        #eigenvectors = solve_triangular(L.T, eigenvectors, lower=False)
        return eigenvalues
    except ValueError as e:
        print(f"Error occured: {e}")

def error_analysis(A: np.ndarray, B: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray):
    """ Compute the error and residual between the computed eigenvalues and the actual eigenvalues """
    actual_eigenvalues = eigvals(A, B)
    error = np.linalg.norm(eigenvalues - actual_eigenvalues, ord=np.inf)
    residuals = []
    for i in range(len(eigenvectors)):
        residual = np.linalg.norm(A @ eigenvectors[:, i] - B @ eigenvectors[:, i] * eigenvalues[i])
        residuals.append(residual)
    return error, residual

    


if __name__ == "__main__":
    print("Starting Cholesky Factorization method....")
    matrix_A = sio.mmread("./matrix_market/bcsstk11.mtx")
    matrix_B = sio.mmread("./matrix_market/bcsstm11.mtx")
    print("Done reading matrix data!")
    A = matrix_A.toarray()
    B = matrix_B.toarray()
    print(f"Matrix A: {A.shape}\nMatrix B: {B.shape}")
    
    eigenvalues = cholesky_solver(A, B)
    print(eigenvalues)
    error = error_analysis(A, B, eigenvalues)
    print(f"Error: {error}")
    
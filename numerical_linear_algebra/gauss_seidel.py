#!/usr/bin/python3
"""
This script contains the implementation of Cholesky Factorization of positive definite matrix,
Gauss-Seidel iteraive method for solving a system of linear equations. These implementations
follow from the pseudocode algorithms from the textbook.

Author: Ayobami Adebesin
Date: 3-29-2024

Usage:
    python3 main.py (./main.py on a unix system)

"""
import os
import time
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
from typing import Union, Sequence, List, Tuple


def backward_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Implement Backward substitution to solve a system of equations

    Args:
        A (np.ndarray): nxn upper triangular matrix over R
        b (np.ndarray): nx1 vector

    Return:
        The solution x to the system Ax=b
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b must be a ({A.shape[0]}, ) vector!")
    x = np.zeros((A.shape[0], ))
    n = A.shape[0] - 1
    x[n] = b[n]/A[n, n]
    for i in reversed((range(n))):
        extra_term = 0
        for j in range(i+1, n+1):
            extra_term += A[i][j]*x[j]
        x[i] = (b[i] - extra_term) / A[i][i]
    return x


def forward_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Implement forward substitution to solve a system of equations

    Args:
        A (np.ndarray): nxn lower triangular matrix over R
        b (np.ndarray): nx1 vector

    Return:
        The solution x to the system Ax=b
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b must be a ({A.shape[0]}, ) vector!")
    x = np.zeros((A.shape[0], 1))
    n = A.shape[0]
    x[0] = b[0]/A[0, 0]
    for i in range(1, n):
        extra_term = 0
        for j in range(0, i):
            extra_term += A[i][j]*x[j]
        x[i] = (b[i] - extra_term) / A[i][i]
    return x


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
    if is_positive_definite(A) == False:
        raise ValueError("A must be positive definite")

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


def cholesky_solver(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve a system of linear equation Ax=b using the Cholesky factorization

    Args:
        A (np.ndarray): nxn upper triangular matrix over R
        b (np.ndarray): nx1 vector b

    Return:
        x (np.ndarray): The solution to the system Ax=b
        r (float): The residual norm ||b-Ax||_inf
    """
    if A is not None and b is not None and A.shape[0] > 0 and b.shape[0] > 0:
        try:
            L, L_t = cholesky_factorization(A)
            y = forward_substitution(L, b)
            x = backward_substitution(L_t, y)
            r = compute_residual(A, x, b)
            return x, r
        except Exception as e:
            print(f"{e.__class__.__name__}: {e}")
            return
    else:
        return ValueError("A and b must not be None!")


def crout_factorization_tridiagonal(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Factorize a tridiagonal matrix using Crout factorization

    Args:
        A (np.ndarray): nxn tridiagonal matrix over R
        b (np.ndarray): nx1 vector b

    Return:
        The solution to the system Ax=b
    """
    # Input validation
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b must be a ({A.shape[0]}, ) vector!")
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    z = np.zeros((n, ))
    x = np.zeros((n, ))

    L[0][0] = A[0][0]
    U[0][1] = A[0][1]/L[0][0]
    z[0] = b[0] / L[0][0]

    for i in range(1, n-1):
        L[i][i-1] = A[i][i-1]
        L[i][i] = A[i][i] - (L[i][i-1] * U[i-1][i])
        U[i][i+1] = A[i][i+1] / L[i][i]
        z[i] = (b[i] - (L[i][i-1] * z[i-1])) / L[i][i]
    L[n-1][n-2] = A[n-1][n-2]
    L[n-1][n-1] = A[n-1][n-1] - (L[n-1][n-2] * U[n-2][n-1])
    z[n-1] = (b[n-1] - (L[n-1][n-2] * z[n-2]))/L[n-1][n-1]

    x[n-1] = z[n-1]
    for i in reversed(range(0, n-1)):
        x[i] = z[i] - (U[i][i+1] * x[i+1])
    return x


def create_tridiagonal_matrix(n: int) -> np.ndarray:
    """
    Generate the entries of a tridiagonal matrix A
        
    Args:
        n (int): size of the matrix

    Return:
        The matrix A
    """
    A = np.zeros((n, n))

    # Fill the diagonal with 2s
    np.fill_diagonal(A, 2)

    # Fill the off-diagonals with -1s
    np.fill_diagonal(A[1:], -1)  # Lower diagonal
    np.fill_diagonal(A[:, 1:], -1)  # Upper diagonal

    return A


def create_vector(n: int) -> np.ndarray:
    """
    Generate the entries of vector b
        
    Args:
        n (int): size of the vector

    Return:
        The vector b
    """
    b = np.zeros((n, ))
    for i in range(n):
        if i == 0:
            b[i] = 1 + (i+1)**2/(n+1)**4
        elif i == n-1:
            b[i] = 6 + (i+1)**2/(n+1)**4
        else:
            b[i] = (i+1)**2/(n+1)**4
    return b


def compute_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """ Compute the residual of the solution """
    r = b - np.dot(A, x)
    return norm(r, np.inf)


def gauss_seidel(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tol: float, N: int) -> np.ndarray:
    """ Compute the solution to the linear system Ax=b using Gauss Seidel algorithm """
    n = len(b)
    x = np.copy(x0)

    for k in range(N):
        x_old = np.copy(x)

        for i in range(n):

            # Without exploiting the band structure of A
            # sum1 = sum(A[i][j] * x[j] for j in range(i))
            # sum2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))
            
            sum1 = A[i][i-1] * x[i-1] if i > 0 else 0
            sum2 = A[i][i+1] * x_old[i+1] if i < n-1 else 0
            x[i] = (b[i] - sum1 - sum2) / A[i][i]

        # Check for convergence
        gauss_residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)
        if gauss_residual < tol:
            return x, gauss_residual, k

        # Prepare for next iteration
        x0 = np.copy(x)

    # If we reach this point, the maximum number of iterations was exceeded
    print('Maximum number of iterations exceeded')
    return None, None


def gauss_seidel_solver(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tol: float, N: int) -> Tuple[np.ndarray, float, float, int]:
    """ Gauss Seidel Solver """
    start_time = time.time()
    x_g, gauss_residual, k = gauss_seidel(A, b, x0=x0, tol=tol, N=N)
    end_time = time.time()
    print(f"Time taken to run Gauss Seidel: {end_time - start_time} seconds\n")
    x_c, cholesky_residual = cholesky_solver(A, b)
    error_norm = np.linalg.norm(x_g-x_c, ord=np.inf)
    return (x_g, x_c, gauss_residual, error_norm, k, cholesky_residual)


def compute_spectral_radius(A: np.ndarray) -> float:
    """Compute the spectral radius of the iteration matrix """
    n = len(A)
    iteration_matrix = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            if k >= 1:
                if j >= k-1:
                    iteration_matrix[j, k] = -1 / 2 ** (j - k + 2)
    # Calculate the spectral radius (the largest absolute eigenvalue of the iteration matrix)
    eigenvalues = np.linalg.eigvals(iteration_matrix)
    spectral_radius = max(abs(eigenvalues))

    return spectral_radius


if __name__ == "__main__":
    MATRIX_SIZES = [10, 50, 100, 200, 300, 400, 500]
    N = 50000
    tol = 1e-4
    df = pd.DataFrame(columns=['n', 'Gauss Residual', 'Error Norm',
                      'Cholesky Residual', 'k', 'spectral_radius'])
    results = []
    for n in MATRIX_SIZES:
        print(f"Running program with n = {n}...")
        A = create_tridiagonal_matrix(n)
        b = create_vector(n)
        x0 = np.zeros((n,))

        x_g, x_c, gauss_residual, error_norm, k, cholesky_residual = gauss_seidel_solver(
            A, b, x0=x0, tol=tol, N=N)
        spectral_radius = compute_spectral_radius(A)
        # Store results in DataFrame
        results.append({
            'n': n,
            'Gauss Residual': gauss_residual,
            'Error Norm': error_norm,
            'Cholesky Residual': cholesky_residual,
            'k': k,
            'spectral_radius': spectral_radius
        })
    df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
    print(df)
    print("Done!")

    # Solve using Crout factorization algorithm for triadiagonal matrices
    # print(f"Using Crout factorization: {crout_factorization_tridiagonal(A, b)}")

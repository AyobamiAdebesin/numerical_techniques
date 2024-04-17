#!/usr/bin/python3

""" Conjugate Gradient Descent """

import os
import time
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
from typing import Union, Sequence, List, Tuple
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def is_positive_definite(A):
    """ Check if a matrix is positive definite using the eigenvalue property """
    if A is not None:
        return np.all(eigsh(A, k=1, which='SA')[0] > 0)


# def create_tridiagonal_matrix(n: int, diag_elem: int) -> np.ndarray:
#     """
#     Generate the entries of a tridiagonal matrix A

#     Args:
#         n (int): size of the matrix
#         diag_elem (int): the value to fill the diagonal

#     Return:
#         The matrix A
#     """
#     A = np.zeros((n, n))

#     # Fill the diagonal with 2s
#     np.fill_diagonal(A, diag_elem)

#     # Fill the off-diagonals with -1s
#     np.fill_diagonal(A[1:], -1)  # Lower diagonal
#     np.fill_diagonal(A[:, 1:], -1)  # Upper diagonal

#     return A

def create_tridiagonal_matrix(n, diag_elem):
    """
    Create a tridiagonal matrix of size n x n, with diag_elem on the diagonal and -1 on the off-diagonals.
    This method was used instead of the previous one to generate a sparse matrix.

    Args:
        n (int): size of the matrix
        diag_elem (int): the value to fill the diagonal

    Return:
        The sparse matrix A
    """
    diagonals = [diag_elem, -1, -1]
    offsets = [0, -1, 1]  # 0 for main diagonal, -1 for lower diagonal, 1 for upper diagonal
    A = diags(diagonals, offsets, shape=(n, n))

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


def conjugate_gradient(A, b, x0, tolerance, max_iterations) -> Tuple[np.ndarray, int, List]:
    """
    Solve the linear system Ax = b using the Conjugate Gradient method

    Args:
        A (dia_matrix): the sparse tridiagonal matrix A
        b (np.ndarray): the vector b
        x0 (np.ndarray): the initial guess
        tolerance (float): the tolerance level
        max_iterations (int): the maximum number of iterations

    Return:
        The solution x, the number of iterations and the final residual

    """
    #NOTE I commented this check because I realized that it is time
    # consuming and makes the algorithm slower despite using a sparse matrix
    # and the matrix is already positive definite by nature of the problem.

    # Check for positive definiteness of A
    # if is_positive_definite(A.toarray()) == False:
    #     raise ValueError("A must be positive definite")

    n = b.shape[0]
    if x0 is None:
        x0 = np.zeros(n)
    r = b - A.dot(x0)
    r_norm = np.linalg.norm(r)
    if r_norm < tolerance:
        return x0, 0, r_norm
    p = r
    x = x0
    for k in range(1, max_iterations+1):
        Ap = A.dot(p)
        rTr = np.dot(r, r)
        alpha = rTr / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        r_norm = np.linalg.norm(r_new)
        if r_norm < tolerance:
            return x, k, r_norm
        beta = np.dot(r_new, r_new) / rTr
        p = r_new + beta * p
        r = r_new
    print("Solution did not converge within the maximum number of iterations")
    return x, k, r_norm


if __name__ == "__main__":
    MATRIX_SIZES = [10, 100, 1000, 2000]
    results = []
    results_prime = []
    tol = 1e-8
    max_iterations = 5000
    for n in MATRIX_SIZES:
        print(f"Running program for system A with matrix size n = {n} ...\n")

        # Initialize matrix and vectors
        A = create_tridiagonal_matrix(n=n, diag_elem=2)
        A_prime = create_tridiagonal_matrix(n=n, diag_elem=3)
        b = create_vector(n)
        x0 = np.zeros_like(b)
        y0 = np.zeros_like(b)

        # Solve the system
        x, k, r_norm = conjugate_gradient(
            A, b, x0=x0, tolerance=tol, max_iterations=max_iterations)
        y, k_prime, r_norm_prime = conjugate_gradient(
            A_prime, b, x0=y0, tolerance=tol, max_iterations=max_iterations)
        # Store the results
        results.append((n, k, r_norm))
        results_prime.append((n, k_prime, r_norm_prime))

    # Display the results
    df = pd.DataFrame(results, columns=[
                      "Matrix Size", "Iterations", "Residual"])
    df_prime = pd.DataFrame(results_prime, columns=[
                            "Matrix Size", "Iterations", "Residual"])
    print("Results for Ax=b\n====================================")
    print(df)
    print("\nResults for Ay=b\n====================================")
    print(df_prime)
    print("\nDone running the program for all matrix sizes!\n")


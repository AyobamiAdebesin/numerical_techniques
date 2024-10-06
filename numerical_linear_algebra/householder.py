#!/usr/bin/python3
"""
This script contains the implementation of the Householder QR factorization
algorithm for solving a least squares problem. This implementation follows
from the pseudocode algorithm from the lecture notebook.

Author: Ayobami Adebesin
Date: 06-10-2024

Usage:
    python3 householder.py (./householder.py on a unix system)
    
"""

import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
from typing import Sequence, Mapping, Tuple, Any, List, Dict



def householder(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the reduced QR factorization using Householder reflections.
        Returns Q (m x n) and R (n x n) where A is m x n and m >= n. """
    m, n = A.shape
    R = A.copy()
    # Store the Householder vectors only for the first n columns
    W = np.zeros((m, n))

    for k in range(n):
        # Extract the k-th column of R from row k onward
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)

        # Householder vector
        u = x - e
        u = u / np.linalg.norm(u)

        # Apply the Householder transformation to R (only affecting submatrix from k onward)
        R[k:, k:] -= 2 * np.outer(u, np.dot(u.T, R[k:, k:]))
        W[k:, k] = u  # Store the Householder vector in the matrix W

    # R is already upper triangular, but we return only the first n rows and n columns of R
    return W, np.triu(R[:n, :n])


def formQ(W: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Forms the reduced orthogonal matrix Q (m x n) from the Householder vectors stored in W.
    
    Args:
        W (np.ndarray): Householder vectors (m x n)
        m (int): Number of rows of the original matrix
        n (int): Number of columns of the original matrix
    
    Returns:
        Q (np.ndarray): Reduced orthogonal matrix Q (m x n)
    """
    Q = np.eye(m, n)  # Initialize Q as an identity matrix (size m x n)

    for k in range(n-1, -1, -1):
        u = W[k:, k]
        # Apply Householder transformation
        Q[k:, :] -= 2 * np.outer(u, np.dot(u.T, Q[k:, :]))
    return Q


def construct_A(x: np.ndarray) -> np.ndarray:
    """
    Construct the matrix A for the least squares problem

    Args:
        x (np.ndarray): The nodes x_j
    
    Returns:
        A (np.ndarray): The matrix A for the least squares problem
    
    """
    A = np.vander(x, N=9, increasing=True)
    return A


def test_QR(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test the QR factorization by computing the backward error and orthogonality error
    
    Args:
        A (np.ndarray): mxn matrix A

    Returns:
        Q (np.ndarray): mxn orthogonal matrix
        R (np.ndarray): nxn upper triangular matrix
    """
    W, R = householder(A)
    Q = formQ(W, A.shape[0], A.shape[1])
    assert Q.shape[0] == A.shape[0] and Q.shape[1] == A.shape[1], "Q must have the same dimensions as A"
    assert R.shape[0] == A.shape[1] and R.shape[1] == A.shape[1], "R must be square and have the same number of columns as A"
    backward_error = np.linalg.norm(Q @ R - A, ord=np.inf)
    orthogonality_error = np.linalg.norm(
        Q.T @ Q - np.eye(A.shape[1]), ord=np.inf)

    print(f"Backward Error: {backward_error}")
    print(f"Orthogonality Error: {orthogonality_error}")

    return Q, R


def backward_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Implement Backward substitution to solve a system of equations

    Args:
        A (np.ndarray): mxn upper triangular matrix over R
        b (np.ndarray): mx1 vector

    Return:
        The solution x to the system Ax=b
    """
    if A.shape[0] > A.shape[1]:
        raise ValueError("A must have more columns than rows (m <= n)")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b must be a ({A.shape[0]}, ) vector!")

    m, n = A.shape
    x = np.zeros((n, ))

    for i in reversed(range(m)):
        if A[i, i] == 0:
            raise ValueError("Matrix is singular or nearly singular")
        extra_term = 0
        for j in range(i+1, n):
            extra_term += A[i][j] * x[j]

        x[i] = (b[i] - extra_term) / A[i][i]

    return x


def solve_least_squares(A, b):
    """ Solve the least squares problem Ac = b using the QR factorization """
    Q, R = test_QR(A)

    # Solve the Q^Tb = Rc using backward substitution
    b_tilde = np.dot(Q.T, b)
    try:
        c = backward_substitution(R, b_tilde)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    return c


def evaluate_polynomial(c: np.ndarray, x: float) -> np.ndarray:
    """
    Evaluate he polynomial p(x) with coefficients c at point x
    This function uses Horner's method to evaluate the polynomial
    """
    return np.polyval(c[::-1], x)


def main():
    """ Main function """
    # Define the nodes
    h = 2/5
    xj = np.linspace(-4, 4, 21)

    # Construct the matrix A and the vector b
    A = construct_A(xj)
    b = np.sin(xj)
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")


    # Solve the least squares problem
    c = solve_least_squares(A, b)

    # Calculate the error at sin(pi)
    pi_error = np.abs(evaluate_polynomial(c, np.pi) - np.sin(np.pi))
    print(f"Error at pi: {pi_error}")

    # Plot the approximation
    x_vals = np.linspace(-4, 4, 21)
    p_vals = evaluate_polynomial(c, x_vals)
    plt.plot(x_vals, p_vals, label="Polynomial Approximation")
    plt.plot(x_vals, np.sin(x_vals), label="sin(x)", linestyle="dashed")
    plt.legend()
    plt.savefig("least_squares.png")


if __name__ == "__main__":
    main()

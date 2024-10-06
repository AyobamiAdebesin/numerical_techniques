#!/usr/bin/python3
""" Householder method for factorizing a matrix into its QR decomposition """

import numpy as np
import os
import time
import sys
from typing import Sequence, Mapping

def householder_reduced(A):
    """ Computes the reduced QR factorization using Householder reflections.
        Returns Q (m x n) and R (n x n) where A is m x n and m >= n. """
    m, n = A.shape
    R = A.copy()
    W = np.zeros((m, n))  # Store the Householder vectors only for the first n columns

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
    
    # Form the reduced Q using the Householder vectors
    Q = form_reduced_Q(W, m, n)
    
    # R is already upper triangular, but we return only the first n rows and n columns of R
    return Q, np.triu(R[:n, :n])

def form_reduced_Q(W, m, n):
    """ Forms the reduced orthogonal matrix Q (m x n) from the Householder vectors stored in W. """
    Q = np.eye(m, n)  # Initialize Q as an identity matrix (size m x n)
    
    for k in range(n-1, -1, -1):
        u = W[k:, k]
        Q[k:, :] -= 2 * np.outer(u, np.dot(u.T, Q[k:, :]))  # Apply Householder transformation
    
    return Q

def construct_A(x):
    """ Construct the matrix A for the least squares problem """
    A = np.vander(x, N=9, increasing=True)
    return A

def test_QR(A):
    """ Test the QR factorization by computing the backward error and orthogonality error """
    Q, R = householder_reduced(A)
    print(f"Shape of Q: {Q.shape}")
    print(f"Shape of R: {R.shape}")
    backward_error = np.linalg.norm(Q @ R - A)
    orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(A.shape[1]), ord=np.inf)

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
    
    # Start from the last row of A and move upwards
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

def evaluate_polynomial(c, x):
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

    # Solve the least squares problem
    c = solve_least_squares(A, b)

    # Calculate the error at sin(pi)
    pi_error = np.abs(evaluate_polynomial(c, np.pi) - np.sin(np.pi))
    print(f"Error at pi: {pi_error}")

    # Plot the approximation
    import matplotlib.pyplot as plt
    x_vals = np.linspace(-4, 4, 100)
    p_vals = evaluate_polynomial(c, x_vals)
    plt.plot(x_vals, p_vals, label="Polynomial Approximation")
    plt.plot(x_vals, np.sin(x_vals), label="sin(x)", linestyle="dashed")
    plt.legend()
    plt.show()
    plt.savefig("least_squares.png")

if __name__ == "__main__":
    main()
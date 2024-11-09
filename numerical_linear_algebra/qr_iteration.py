#!/usr/bin/python3
"""
This script contains the implementation of the explicitly shifted QR iteration
algorithm, using the Wilkinson shift, for solving an eigenvalue problem for a tridiagonal matrix.
The implementation follows from the provided pseudocode in the problem description

Author: Ayobami Adebesin
Date: 08-11-2024

Usage:
    python3 householder.py (./qr_iteration.py on a unix system)
    
"""
import numpy as np

def wilkinson_shift(a: float, b: float, c: float) -> float:
    """
    Calculate the wilkinson shift of the lower right 2 by 2 submatrix of A
        a = a_{m-1}
        b = b_{m-1}
        c = a_{m}
    """
    delta = (a - c) / 2
    sign_delta = np.sign(delta) if delta != 0 else 1
    mu = c - (sign_delta * b**2) / (np.abs(sign_delta) + np.sqrt(sign_delta**2 + b**2))
    return mu

def householder_vector(x: np.ndarray) -> np.ndarray:
    """ Compute the Householder vector v """
    if x is None:
        raise ValueError("x must not be none!")
    sign = np.sign(x[0]) if x[0] != 0 else 1
    e1 = np.array([1] + [0] * (len(x) - 1))
    v = x + sign * (np.linalg.norm(x, ord=2) * e1)
    v /= np.linalg.norm(v)
    return v

def create_tridiagonal_matrix(m: int, diag_value: float=2.0, off_diag_value: float=-1.0) -> np.ndarray:
    """ Creates an m x m tridiagonal matrix with `diag_value` on the diagonal
        and `off_diag_value` on the sub- and superdiagonals. """
    A = np.diag([diag_value] * m)
    for i in range(m - 1):
        A[i, i+1] = off_diag_value
        A[i+1, i] = off_diag_value
    return A

def check_diagonal(T:np.ndarray) -> bool:
    """ Check if T is diagonal """
    return np.all(np.diag(np.diag(T)) == T)

def tridiagonal_qr_iteration(T: np.ndarray, tol=1e-16):
    """ Compute the explicitly shifted qr iteration """
    m = T.shape[0]
    l = m - 1
    Q = np.eye(m)
    W = np.zeros((2, m-1))
    iter_cnt = 0
    while l > 0:
        iter_cnt += 1
        if np.abs(T[l, l-1]) < tol:
            # deflate and work on a smaller problem
            T[l, l-1] = 0
            T[l-1, l] = 0
            l -= 1
            continue
        # compute wilkinson shift
        sigma = wilkinson_shift(a = T[l-1, l-1], b = T[l, l-1], c= T[l, l])
        T -= sigma * np.eye(m)

        for j in range(l):
            # extract the column x from T
            x = T[j: j+2, j]      
            v = householder_vector(x)
            W[:, j] = v
            
            # apply householder to T from the left
            H = np.eye(2) - 2 *np.outer(v, v)
            T[j:j+2, j:j+3] = H @ T[j:j+2, j:j+3]
            T[j+1, j] = 0

            # apply householder to Q from the right
            Q[:, j:j+2] = Q[:, j:j+2] @ H
        for j in range(l):
            v = W[:, j]
            H = np.eye(2) - 2 * np.outer(v, v)
            T[max(0, j - 1):j + 2, j:j+2] = T[max(0, j - 1):j + 2, j:j+2] @ H
            if j - 1 >= 0:
                T[j - 1, j + 1] = 0

        T += sigma * np.eye(m)
    
    # check if T is diagonal
    if check_diagonal(T):
        return Q, np.diag(T), iter_cnt
    else:
        raise ValueError("Error occured! T is not diagonal.")

if __name__ == "__main__":
    # initialize parameters and run algorithm
    try:
        m = 10
        A = create_tridiagonal_matrix(m, diag_value= 2.0, off_diag_value=-1.0)
        Q, D, iter_cnt = tridiagonal_qr_iteration(A)

        exact_eigenvalues = np.array([2 - 2 * np.cos(np.pi * k / (m + 1)) for k in range(1, m + 1)])
        approx_eigenvalues = D

        # compute errors
        relative_backward_error = np.linalg.norm(Q @ D @ Q.T - A, ord=np.inf) / np.linalg.norm(A, ord=np.inf)
        orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(m), ord=np.inf)
        forward_errors = np.abs(approx_eigenvalues - exact_eigenvalues)

        # print results
        print("Relative Backward Error:", relative_backward_error)
        print("Orthogonality Error:", orthogonality_error)
        print("Forward Eigenvalue Errors:", forward_errors)
        print("Total Iterations:", iter_cnt)
    except ValueError as e:
        print(f"{e}")

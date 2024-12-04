#!/usr/bin/python3
import numpy as np
from scipy.linalg import qr
from standard_lanczos import generate_matrix, relative_ritz_pair_residual

def lanczos(A, m, b=None):
    np.random.seed(256)
    n = A.shape[0]
    if b is None:
        b = np.random.randn(n)
    
    alphas = np.zeros(m)
    betas = np.zeros(m)
    Q_m_plus_1 = np.zeros((n, m + 1))

    beta_prev = 0
    q_prev = np.zeros(n)
    q_curr = b / np.linalg.norm(b)

    for j in range(m + 1):
        Q_m_plus_1[:, j] = q_curr
        v = A @ q_curr

        if j < m:
            alpha = np.dot(q_curr, v)
            alphas[j] = alpha

            v = v - (beta_prev * q_prev) - (alpha * q_curr)

            v -= Q_m_plus_1[:, :j+1] @ (Q_m_plus_1[:, :j+1].T @ v)

            beta = np.linalg.norm(v)
            betas[j] = beta

            if beta < 1e-16:
                break

            q_prev = q_curr
            q_curr = v / beta
            beta_prev = beta
        else:
            break

    T_m = np.zeros((m, m))
    np.fill_diagonal(T_m, alphas)
    np.fill_diagonal(T_m[:-1, 1:], betas[:m - 1])
    np.fill_diagonal(T_m[1:, :-1], betas[:m - 1])

    T_m_tilde = np.zeros((m + 1, m))
    np.fill_diagonal(T_m_tilde[:m, :], alphas)
    np.fill_diagonal(T_m_tilde[:m, 1:], betas[:m - 1])
    np.fill_diagonal(T_m_tilde[1:, :m - 1], betas[:m - 1])

    T_m_tilde[m, m - 1] = betas[m - 1] if m > 1 else betas[0] 

    Q_m = Q_m_plus_1[:, :m]

    return T_m, T_m_tilde, Q_m, Q_m_plus_1


def irla(A, m, k, p, max_iter=100, tol=1e-10):
    """
    Implicitly Restarted Lanczos Algorithm (IRLA) for symmetric matrices.
    
    Parameters:
        A: ndarray
            Symmetric input square matrix of size (n, n).
        m: int
            The number of Lanczos iterations (dimension of Krylov subspace).
        k: int
            Desired number of eigenvalues/eigenvectors.
        p: int
            Number of shifts.
        max_iter: int
            Maximum number of iterations.
        tol: float
            Convergence tolerance for residuals.

    Returns:
        Vk: ndarray
            Matrix of k Ritz vectors (size n x k).
        Tk: ndarray
            Tridiagonal matrix of size (k, k).
        eigvecs: ndarray
            Eigenvectors corresponding to Ritz eigenvalues (size n x k).
    """
    n = A.shape[0]
    Vk = np.zeros((n, k))  # Store the final Ritz vectors
    T_k = np.zeros((k, k))  # Final k x k tridiagonal matrix
    Q_m_plus_1 = np.zeros((n, m + 1))  # Store m+1 Lanczos vectors
    f_m = np.random.randn(n)  # Initial residual vector

    for iter_num in range(max_iter):
        # Lanczos iteration to construct the Krylov subspace
        T_m, T_m_tilde, Q_m, Q_m_plus_1 = lanczos(A, m, b=f_m)

        # Compute eigenvalues and eigenvectors of the tridiagonal matrix T_m
        eigvals, eigvecs_T = np.linalg.eig(T_m)  # Ritz eigenvalues and eigenvectors

        # Sort eigenvalues and select the smallest p eigenvalues
        idx = np.argsort(np.abs(eigvals))[:p]
        shifts = eigvals[idx]

        # Perform QR shifts on the tridiagonal matrix
        Q = np.eye(m)
        for shift in shifts:
            Qi, Ri = qr(T_m - shift * np.eye(m))

            # Update T_m with the QR transformation
            T_m = Qi.T @ T_m @ Qi  
            Q = Q @ Qi  

        # Compute the residual and restart the Lanczos process
        # The residual is based on the last element of T_m
        beta_k = T_m[k-1, k-2]  
        f_k = Q_m[:, k-1] * beta_k + f_m * Q[m - 1, k - 1]

        # Restart the Lanczos process if the residual is small enough
        if np.abs(beta_k) < tol:
            break

        # Normalize the residual for next iteration
        f_m = f_k / np.linalg.norm(f_k)

        # Restart the process with the new vector f_m
        Q_m[:, 0] = f_m  

    # Mapping the Ritz vectors back to the original space
    eigvecs = Q_m @ eigvecs_T

    return Q_m, T_m, eigvecs, eigvals

if __name__ == "__main__":
    A = generate_matrix()
    A[0,0] = -10

    m = 20
    k = 10
    p = 10

    Q_m, T_m, eigvecs, eigvals = irla(A, m, k, p)

    print("Eigenvalues of reduced T_k:")
    print(eigvals)

    res = relative_ritz_pair_residual(A, eigenvalues=eigvals, eigenvectors=eigvecs)
    print(res)
    

    #print("Eigenvectors corresponding to Ritz eigenvalues:")
    # print(eigvecs)  # Ritz eigenvectors mapped to the original space
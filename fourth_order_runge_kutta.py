#!/usr/bin/python3
""" Runge Kutta Method """
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Sequence, Tuple


# Initialize mesh size list
MESH_SIZES = [32, 64, 128, 256, 512, 1024]


def runge_kutta(
        dv_dt: Callable,
        di_dt: Callable,
        interval: List[int],
        N: int,
        v0: float,
        i0: float) -> List[Tuple]:
    """
    Implement Fourth Order Runge Kutta Method for a system of ODEs

    Args:
        dv_dt (Callable): Rate of change of volatage wrt time. This is defined as a function: f1(t, v, i)
        di_dt (Callable): Rate of change of current wrt time. This is defined as a function f2(t, v, i)
        interval (list of integers): The time interval on which we are approximating the solution
        N (int): mesh size
        v0 (float): initial value of v at t=0
        i0 (float): initial value of i at t=0

    Returns:
        A list of tuples containing the the ordered pair (t, v, i)  for each time step in `interval` 
    """
    if len(interval) != 2 or not all(isinstance(x, int) for x in interval):
        raise ValueError("interval must be a pair of integers")
    if not interval[0] < interval[1]:
        raise ValueError("interval must be ordered")

    a = interval[0]
    b = interval[1]
    approx_sol = [(a, v0, i0)]
    h = np.abs((b - a) / N)
    t0 = a
    print(
        f"=========== About to start iteration with step size: {str(b-a)}/{str(N)} ================\n")

    for j in range(N):
        k1_1 = h*dv_dt(t0, v0, i0)
        k1_2 = h*di_dt(t0, v0, i0)

        k2_1 = h*dv_dt(t0 + h/2, v0 + k1_1/2, i0 + k1_2/2)
        k2_2 = h*di_dt(t0 + h/2, v0 + k1_1/2, i0 + k1_2/2)

        k3_1 = h*dv_dt(t0 + h/2, v0 + k2_1/2, i0 + k2_2/2)
        k3_2 = h*di_dt(t0 + h/2, v0 + k2_1/2, i0 + k2_2/2)

        k4_1 = h*dv_dt(t0 + h, v0 + k3_1, i0 + k3_2)
        k4_2 = h*di_dt(t0 + h, v0 + k3_1, i0 + k3_2)

        v0 = v0 + 1/6*(k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        i0 = i0 + 1/6*(k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
        t0 += h
        approx_sol.append((t0, v0, i0))
    assert len(approx_sol) == N+1, "solution out of range!"
    return approx_sol


def dv_dt(t, v, i):
    """ Define the first function """
    return -i


def di_dt(t, v, i):
    """ Define the second function """
    return v


def compute_vc_17(v_values: List, n: int) -> None:
    """ Compute vc(17) for different values of h """
    if v_values:
        print(f"vc(t_{n}) = vc(17): {v_values[-1]}")
        return
    return


def compute_errors(t_values: List, v_values: List, n: int):
    """ Compute the errors for each approximation """
    print(
        f"Error at step {n}: {np.abs(np.cos(t_values[-1]) - v_values[-1])}")
    return


if __name__ == "__main__":
    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(
        columns=['Step Size', 'Error', 'Rate of Decrease'])

    # Initialize the previous error to None
    prev_error = None

    for n in MESH_SIZES:
        # Get and unpack results
        result = runge_kutta(dv_dt, di_dt, interval=[0, 17], N=n, v0=1, i0=0)
        time_values = [t for t, _, _ in result]
        v_values = [v for _, v, _ in result]
        i_values = [i for _, _, i in result]

        # Print the values of vc(17) for each step
        compute_vc_17(v_values, n)

        # Print the errors
        compute_errors(time_values, v_values, n)
        print("\n")

        # Calculate the relative error
        error = np.abs(np.cos(time_values[-1]) - v_values[-1])

        # Calculate the rate of decrease
        rate_of_decrease = prev_error / error if prev_error is not None else None

        # Append the results to the DataFrame
        new_data = pd.DataFrame({
            'Step Size': [17/n],
            'Error': [error],
            'Rate of Decrease': [rate_of_decrease]
        })
        results_df = pd.concat([results_df, new_data], ignore_index=True)

        # Update the previous error
        prev_error = error
    print(results_df)

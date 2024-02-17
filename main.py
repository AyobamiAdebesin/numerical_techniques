#!/usr/bin/python3

"""
This script containsthe implementation of Euler's method,
Runge-Kutta method, and Adams-Bashforth method for solving a system of ODEs

Author: Ayobami Adebesin
Date: 2-16-2024

Usage:
    python3 main.py (./main.py on a unix system)

    If you are running this code on a jupyter notebook, you can uncomment
    lines 230-241 to visualize the results

"""
import numpy as np
import os
import shutil
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Sequence, List, Tuple, Mapping
from collections import deque

# Initialize mesh size list
MESH_SIZES = [32, 64, 128, 256, 512, 1024]


def euler_method(
        dv_dt: Callable,
        di_dt: Callable,
        interval: List[int],
        N: int,
        v0: float,
        i0: float) -> List[Tuple[float, float, float]]:
    """
    Implement Euler's method for a system of ODEs

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
    # Input validation
    if len(interval) != 2:
        raise ValueError("interval must be a pair of integers")
    if not interval[0] < interval[1]:
        raise ValueError("interval must be ordered")

    a = interval[0]
    b = interval[1]
    approx_sol = [(a, v0, i0)]
    h = np.abs((b - a) / N)
    t0 = a
    for j in range(N):
        # Keep track of the old value of v0
        v_old = v0
        v0 = v0 + (h * dv_dt(t0, v0, i0))
        i0 = i0 + (h * di_dt(t0, v_old, i0))
        t0 += h
        approx_sol.append((t0, v0, i0))
    assert len(approx_sol) == N+1, "solution out of range!"
    return approx_sol


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
    # Input validation
    if len(interval) != 2 or not all(isinstance(x, int) for x in interval):
        raise ValueError("interval must be a pair of integers")
    if not interval[0] < interval[1]:
        raise ValueError("interval must be ordered")

    a = interval[0]
    b = interval[1]
    approx_sol = [(a, v0, i0)]
    h = np.abs((b - a) / N)
    t0 = a
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


def adams_bashforth(
        dv_dt: Callable,
        di_dt: Callable,
        interval: List[int],
        N: int,
        initial_points: List[Tuple]
) -> List[Tuple]:
    """
    Implement Fourth Order Runge Kutta Method for a system of ODEs

    Args:
        dv_dt (Callable): Rate of change of volatage wrt time. This is defined as a function: f1(t, v, i)
        di_dt (Callable): Rate of change of current wrt time. This is defined as a function f2(t, v, i)
        interval (list of integers): The time interval on which we are approximating the solution
        N (int): mesh size
        initial_points: (list of tuples): This is the list of the 4 initial points obtained from Runge Kutta method

    Returns:
        A list of tuples containing the the ordered pair (t, v, i)  for each time step in `interval` 
    """
    # Input validation
    if len(interval) != 2 or not all(isinstance(x, int) for x in interval):
        raise TypeError("interval must be a pair of integers")
    if not interval[0] < interval[1]:
        raise ValueError("interval must be ordered")
    if len(initial_points) != 4 or not all(isinstance(x, tuple) for x in initial_points) or not all(len(x) == 3 for x in initial_points):
        raise ValueError(
            "initial points must contain a tuple (t, v, i) of 4 points")

    a = interval[0]
    b = interval[1]
    approx_sol = initial_points
    # A deque to keep track of the last 4 points for each iteration
    curr_point = deque(initial_points)
    h = np.abs((b - a) / N)
    t0 = a + 3*h
    for j in range(N - 3):
        v = curr_point[3][1] + h/24*(55*dv_dt(curr_point[3][0], curr_point[3][1], curr_point[3][2])
                                     - 59 *dv_dt(curr_point[2][0], curr_point[2][1], curr_point[2][2])
                                     + 37 *dv_dt(curr_point[1][0], curr_point[1][1], curr_point[1][2])
                                     - 9*dv_dt(curr_point[0][0], curr_point[0][1], curr_point[0][2]))
        i = curr_point[3][2] + h/24*(55*di_dt(curr_point[3][0], curr_point[3][1], curr_point[3][2])
                                     - 59 *di_dt(curr_point[2][0], curr_point[2][1], curr_point[2][2])
                                     + 37 *di_dt(curr_point[1][0], curr_point[1][1], curr_point[1][2])
                                     - 9*di_dt(curr_point[0][0], curr_point[0][1], curr_point[0][2]))
        curr_point.popleft()
        t0 += h
        curr_point.append((t0, v, i))
        approx_sol.append((t0, v, i))
    return approx_sol

# Define the system of equations


def dv_dt(t, v, i):
    """ Define the first function """
    return -i


def di_dt(t, v, i):
    """ Define the second function """
    return v


def rate_of_decrease(last_values: List[Tuple]):
    """ Compute the rate of decrease of error for the numerical methods """
    results_df = pd.DataFrame(
        columns=['Step Size', 'Error', 'Rate of Decrease'])
    prev_error = None
    for j in range(len(last_values)):
        error = np.abs(last_values[j][1] - np.cos(17))
        rate_of_decrease = prev_error/error if prev_error else None
        # Append the results to the DataFrame
        new_data = pd.DataFrame({
            'Step Size': [17/(2**(5+j))],
            'Error': [error],
            'Rate of Decrease': [rate_of_decrease]
        })
        results_df = pd.concat([results_df, new_data], ignore_index=True)
        prev_error = error
    return results_df


if __name__ == "__main__":
    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(
        columns=['Step Size', 'Error', 'Rate of Decrease'])

    last_values_adams_bashforth = []
    last_values_euler = []
    last_values_runge_kutta = []

    # Initialize the previous error to None
    prev_error = None
    for n in MESH_SIZES:
        print(
            f"=========== About to start iteration with step size: {str(17-0)}/{str(n)} ================\n")
        euler_result = euler_method(dv_dt, di_dt, interval=[
                                    0, 17], N=n, v0=1, i0=0)
        rk_result = runge_kutta(dv_dt, di_dt, interval=[
                                0, 17], N=n, v0=1, i0=0)

        # Use the first 4 points for the Runge Kutta method as  starting point for Adams Bashforth
        initial_points = rk_result[:4]
        ab_result = adams_bashforth(dv_dt, di_dt, interval=[
                                    0, 17], N=n, initial_points=initial_points)

        # Extract the time, v, and i values for each method
        time_values = [t for t, _, _ in euler_result]
        v_values_e = [v for _, v, _ in euler_result]
        i_values_e = [i for _, _, i in euler_result]
        v_values_r = [v for _, v, _ in rk_result]
        i_values_r = [i for _, _, i in rk_result]
        v_values_ab = [v for _, v, _ in ab_result]
        i_values_ab = [i for _, _, i in ab_result]

        v_values_map = {
            "Euler": (v_values_e, i_values_e),
            "Runge-Kutta": (v_values_r, i_values_r),
            "Adams-Bashforth": (v_values_ab, i_values_ab)
        }

        # Extract v(17) for each method
        v_euler = v_values_e[-1]
        v_runge_kutta = v_values_r[-1]
        v_adams_bashforth = v_values_ab[-1]

        last_values_adams_bashforth.append(ab_result[-1])
        last_values_euler.append(euler_result[-1])
        last_values_runge_kutta.append(rk_result[-1])

        # Compute cos(17)
        cos_17 = np.cos(17)

        # Compute the error for each method
        error_euler = np.abs(v_euler - cos_17)
        error_runge_kutta = np.abs(v_runge_kutta - cos_17)
        error_adams_bashforth = np.abs(v_adams_bashforth - cos_17)

        # for method in ["Euler", "Runge-Kutta", "Adams-Bashforth"]:
        #     plt.figure(figsize=(10,5))
        #     plt.plot(time_values, v_values_map[method][0], label='v(t)')
        #     plt.plot(time_values, v_values_map[method][1], label='i(t)')
        #     plt.plot(time_values, np.cos(time_values), 'x', label='True solution')
        #     plt.xlabel('Time')
        #     plt.ylabel('Solution')
        #     plt.title(f'Graphical solution of the system of IVP by {method} method for i = {n}')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f"result_{method}_{n}")
        #     plt.show()

        # Create a DataFrame to display the results
        df = pd.DataFrame({
            'Method': ['Euler', 'Runge-Kutta', 'Adams-Bashforth'],
            'v(17)': [v_euler, v_runge_kutta, v_adams_bashforth],
            'cos(17)': [cos_17, cos_17, cos_17],
            'Error': [error_euler, error_runge_kutta, error_adams_bashforth]
        })
        print(df)
        print("\n\n")
    print("Rate of decrease in error for Euler's method")
    print("============================================")
    print(rate_of_decrease(last_values_euler))
    print("\n\n")
    print("Rate of decrease in error for Adams Bashforth method")
    print("====================================================")
    print(rate_of_decrease(last_values_adams_bashforth))
    print("\n\n")
    print("Rate of decrease in error for Runge Kutta method")
    print("================================================")
    print(rate_of_decrease(last_values_runge_kutta))

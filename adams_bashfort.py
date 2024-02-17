#!/usr/bin/env python3
""" Adams Bashforth Method """
import numpy as np
import os
import time
from typing import Callable, List, Sequence, Tuple
from collections import deque
from fourth_order_runge_kutta import runge_kutta

# Initialize mesh size list
MESH_SIZES = [32, 64, 128, 256, 512, 1024]

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
        raise ValueError("initial points must contain 4 points")

    a = interval[0]
    b = interval[1]
    approx_sol = initial_points
    curr_point = deque(initial_points)
    h = np.abs((b - a) / N)
    t0 = a + 3*h
    print(
        f"=========== About to start iteration with step size: {str(b-a)}/{str(N)} ================\n")
    for j in range(N - 3):
        v = curr_point[3][1] + h/24*(55*dv_dt(curr_point[3][0], curr_point[3][1], curr_point[3][2])
                                     - 59*dv_dt(curr_point[2][0], curr_point[2][1], curr_point[2][2])
                                     + 37*dv_dt(curr_point[1][0], curr_point[1][1], curr_point[1][2])
                                     - 9*dv_dt(curr_point[0][0], curr_point[0][1], curr_point[0][2]))
        i = curr_point[3][2] + h/24*(55*di_dt(curr_point[3][0], curr_point[3][1], curr_point[3][2])
                                     - 59*di_dt(curr_point[2][0], curr_point[2][1], curr_point[2][2])
                                     + 37*di_dt(curr_point[1][0], curr_point[1][1], curr_point[1][2])
                                     - 9*di_dt(curr_point[0][0], curr_point[0][1], curr_point[0][2]))
        curr_point.popleft()
        t0 += h
        curr_point.append((t0, v, i))
        approx_sol.append((t0, v, i))
    return approx_sol


def dv_dt(t, v, i):
    """ Define the first function """
    return -i


def di_dt(t, v, i):
    """ Define the second function """
    return v

if __name__ == "__main__":
    rk_result = runge_kutta(dv_dt, di_dt, interval=[0, 17], N=1024, v0=1, i0=0)
    initial_points = rk_result[:4]
    ad_result = adams_bashforth(dv_dt, di_dt, interval=[0, 17], N=1024, initial_points=initial_points)
    print(f"Adams bashforth result: {ad_result[-1]}")
    print(f"Runge kutta result {rk_result[-1]}")

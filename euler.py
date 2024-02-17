#!/usr/bin/python3

""" Euler's method """
import numpy as np
import os
import shutil
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Sequence, List, Tuple

def euler_method(
        dx_dt: Callable,
        dy_dt: Callable,
        interval: List[int],
        N: int,
        v0: float,
        i0: float) -> List[Tuple[float, float, float]]:
    """ Implement Euler's method for a system of linear IVP's """
    if len(interval) != 2:
        raise ValueError("interval must be a pair of integers")
    if not interval[0] < interval[1]:
        raise ValueError("interval must be ordered")

    a = interval[0]
    b = interval[1]
    approx_sol = [(a, v0, i0)]
    h = np.abs((b - a) / N)
    print(
        f"=========== About to start iteration with step size: {h} ================\n")
    t0 = a
    for j in range(N):
        # Keep track of the old value of x0
        v_old = v0
        v0 = v0 + (h * dx_dt(t0, v0, i0))
        i0 = i0 + (h * dy_dt(t0, v_old, i0))
        t0 += h
        approx_sol.append((t0, v0, i0))
    assert len(approx_sol) == N+1, "solution out of range!"
    return approx_sol

# Define your system of equation


def dv_dt(t, v, i):
    """ Define dx_dt """
    return -i


def di_dt(t, v, i):
    """ Define dy_dt """
    return v


def plot_results(t_values, v_values, i_values):
    """ Plot the time values against the v_values and i_values """
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, v_values, label='v(t)')
    plt.plot(t_values, i_values, label='i(t)')
    # plt.plot(time_values, np.sin(time_values), label='True solution')
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.title('Solution of the system of ODEs by Euler method')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    result = euler_method(dx_dt=dv_dt, dy_dt=di_dt, interval=[
                          0, 17], N=1024, v0=1, i0=0)
    print(result[:10])
    time_values = [t for t, _, _ in result]
    v_values = [v for _, v, _ in result]
    i_values = [i for _, _, i in result]
    df = pd.DataFrame(result, columns=['time', 'v_values', 'i_values'])
    print(df)
    plot_results(t_values=time_values, v_values=v_values, i_values=i_values)

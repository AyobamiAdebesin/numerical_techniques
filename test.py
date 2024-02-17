#!/usr/bin/python3

def lc_circuit_derivatives(y, t):
    vC, iL = y
    dvC_dt = -iL  # d(vC)/dt = -iL/C, with C=1
    diL_dt = vC   # d(iL)/dt = vC/L, with L=1
    return [dvC_dt, diL_dt]

def euler_method(f, y0, t):
    y = [y0]
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        y.append(y[i-1] + h * f(y[i-1], t[i-1]))
    return y

def runge_kutta_method(f, y0, t):
    y = [y0]
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        k1 = f(y[i-1], t[i-1])
        k2 = f([y_i + 0.5*h*k1_i for y_i, k1_i in zip(y[i-1], k1)], t[i-1] + 0.5*h)
        k3 = f([y_i + 0.5*h*k2_i for y_i, k2_i in zip(y[i-1], k2)], t[i-1] + 0.5*h)
        k4 = f([y_i + h*k3_i for y_i, k3_i in zip(y[i-1], k3)], t[i])
        y.append([y_i + (h/6.0)*(k1_i + 2*k2_i + 2*k3_i + k4_i) for y_i, k1_i, k2_i, k3_i, k4_i in zip(y[i-1], k1, k2, k3, k4)])
    return y

def adams_bashforth_method(f, y0, t):
    # Start with Runge-Kutta to get the first four points
    y = runge_kutta_method(f, y0, t[:4])
    
    for i in range(4, len(t)):
        h = t[i] - t[i-1]
        y.append(y[i-1] + (h/24)*(55*f(y[i-1], t[i-1]) - 59*f(y[i-2], t[i-2]) + 37*f(y[i-3], t[i-3]) - 9*f(y[i-4], t[i-4])))
    return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Initial conditions
    y0 = [1, 0]  # vC(0) = 1, iL(0) = 0

    # Time points
    t = np.linspace(0, 17, 1000)

    # Solve using each method
    euler_sol = euler_method(lc_circuit_derivatives, y0, t)
    rk_sol = runge_kutta_method(lc_circuit_derivatives, y0, t)
    ab_sol = adams_bashforth_method(lc_circuit_derivatives, y0, t)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, [y[0] for y in euler_sol], label='Euler')
    plt.plot(t, [y[0] for y in rk_sol], label='Runge-Kutta')
    plt.plot(t, [y[0] for y in ab_sol], label='Adams-Bashforth')
    plt.plot(t, np.cos(t), label='Exact', linestyle='--')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('vC(t)')
    plt.title('LC Circuit Voltage Comparison')
    plt.show()


# def runge_kutta(
#         f: Callable,
#         interval: List[int],
#         N: int,
#         w: float):
#     """ Implement Fourth Order Runge Kutta Method """
#     if len(interval) != 2:
#         raise ValueError("interval must be a pair of integers")
#     if not interval[0] < interval[1]:
#         raise ValueError("interval must be ordered")

#     a = interval[0]
#     b = interval[1]
#     approx_sol = [(a, w)]
#     h = np.abs((b - a) / N)
#     print(f"h value: {h}")
#     t0 = a

#     for j in range(N):
#         k1 = h*f(t0, w)
#         k2 = h*f(t0 + h/2, w + k1/2)
#         k3 = h*f(t0 + h/2, w + k2/2)
#         k4 = h*f(t0 + h, w + k3)

#         w = w + 1/6*(k1 + 2*k2 + 2*k3 + k4)
#         t0 += h
#         approx_sol.append((t0, w))
#     assert len(approx_sol) == N+1, "solution out of range!"
#     return approx_sol

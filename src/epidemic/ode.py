"""SIR determinístico (Kermack–McKendrick) resolvido com scipy.integrate.solve_ivp."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def sir_rhs(t, y, beta, gamma):
    s, i, r = y
    return [-beta * s * i, beta * s * i - gamma * i, gamma * i]


def solve_sir(beta: float, gamma: float, s0: float, i0: float, r0: float,
              t_max: float, n_points: int = 2000):
    """Resolve o SIR em frações; retorna (t, s, i, r)."""
    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(sir_rhs, (0.0, t_max), [s0, i0, r0], args=(beta, gamma),
                    t_eval=t_eval, rtol=1e-9, atol=1e-12, method="DOP853")
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


def ode_peak(beta, gamma, s0, i0, r0, t_max):
    t, s, i, r = solve_sir(beta, gamma, s0, i0, r0, t_max, n_points=20001)
    k = int(np.argmax(i))
    return dict(peak_frac=float(i[k]), t_peak=float(t[k]), final_size_frac=float(s0 - s[-1] + i0))  # infectados ao longo da epidemia, incluindo i0

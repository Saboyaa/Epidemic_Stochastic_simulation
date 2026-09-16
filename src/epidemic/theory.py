"""Resultados analíticos para o SIR / processo de ramificação."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


def final_size_tau(R0: float, s0: float = 1.0, i0: float = 0.0) -> float:
    """Fração τ dos suscetíveis iniciais que é infectada.

    Equação (i0 = 0): 1 − τ = exp(−R0·s0·τ). Raiz não trivial via brentq, excluindo τ = 0.
    Com i0 > 0, usa-se a forma geral 1 − τ = exp(−R0·(s0·τ + i0)), que tem raiz
    única em (0,1) mesmo para R0·s0 ≤ 1.
    """
    def f(tau):
        return 1.0 - tau - math.exp(-R0 * (s0 * tau + i0))

    if i0 <= 0.0:
        if R0 * s0 <= 1.0:
            return 0.0
        # f(0) = 0 (raiz trivial); f'(0) = R0 s0 − 1 > 0 => f > 0 logo após 0; f(1) = −exp(...) < 0
        lo = 1e-12
        while f(lo) <= 0.0:
            lo *= 10.0
            if lo > 0.5:
                return 0.0
        return brentq(f, lo, 1.0 - 1e-15, xtol=1e-14)
    return brentq(f, 0.0, 1.0 - 1e-15, xtol=1e-14)


def final_size_fraction(R0: float, s0: float, i0: float = 0.0) -> float:
    """Tamanho final na população (fração de N) = s0·τ."""
    return s0 * final_size_tau(R0, s0, i0)


def prob_major_outbreak(R_eff: float, I0: int = 1) -> float:
    """Aproximação por processo de ramificação: 1 − (1/R_eff)^I0 se R_eff > 1, senão 0."""
    if R_eff <= 1.0:
        return 0.0
    return 1.0 - (1.0 / R_eff) ** I0


def expected_total_subcritical(R_eff: float, I0: int = 1) -> float:
    """Tamanho total esperado (incluindo os casos iniciais) a partir de I0 casos, R_eff < 1: I0/(1 − R_eff)."""
    if R_eff >= 1.0:
        return math.inf
    return I0 / (1.0 - R_eff)


def offspring_pmf(k: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Offspring do caso índice sem depleção: Geométrica, P(k) = (γ/(β+γ))·(β/(β+γ))^k, média β/γ = R0."""
    k = np.asarray(k, dtype=float)
    q = beta / (beta + gamma)
    return (1.0 - q) * q ** k


def infectious_period_cdf(t: np.ndarray, gamma: float) -> np.ndarray:
    """Duração infecciosa ~ Exp(γ): F(t) = 1 − e^{−γt}."""
    return 1.0 - np.exp(-gamma * np.asarray(t, dtype=float))


def herd_immunity_threshold(R0: float) -> float:
    return 1.0 - 1.0 / R0


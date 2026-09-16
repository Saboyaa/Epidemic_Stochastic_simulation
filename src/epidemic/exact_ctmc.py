"""Distribuição EXATA do tamanho final do SIR estocástico (N pequeno).

Cadeia de saltos embutida: a partir de (s, i) com i > 0,
    P(infecção)   = β·s·i/N / (β·s·i/N + γ·i) = β·s / (β·s + γ·N)  -> (s−1, i+1)
    P(recuperação) = γ·N / (β·s + γ·N)                              -> (s,   i−1)
Absorção em i = 0. O tamanho final é o número de indivíduos que foram infectados,
Z = (S0 − s_final) + I0.

Programação dinâmica: a soma s + i é constante na infecção e diminui de 1 na
recuperação; dentro de um nível s + i = L, a infecção diminui s. Percorrendo os
níveis em ordem decrescente e, em cada nível, s em ordem decrescente, todas as
transições vão para estados ainda não processados (ordem topológica).
"""
from __future__ import annotations

import numpy as np


def final_size_pmf(N: int, I0: int, beta: float, gamma: float, S0: int | None = None) -> np.ndarray:
    """PMF exata de Z (tamanho final, incluindo I0). Índice k = número total de infectados, k = I0..I0+S0."""
    if S0 is None:
        S0 = N - I0
    prob = np.zeros((S0 + 1, S0 + I0 + 1))  # prob[s, i]
    prob[S0, I0] = 1.0
    pmf = np.zeros(N + 1)
    for L in range(S0 + I0, -1, -1):
        for s in range(min(L, S0), -1, -1):
            i = L - s
            if i > S0 + I0 or i < 0:
                continue
            m = prob[s, i]
            if m == 0.0:
                continue
            if i == 0:
                pmf[S0 - s + I0] += m
                continue
            denom = beta * s + gamma * N
            p_inf = beta * s / denom if s > 0 else 0.0
            if s > 0:
                prob[s - 1, i + 1] += m * p_inf
            prob[s, i - 1] += m * (1.0 - p_inf)
    assert abs(pmf.sum() - 1.0) < 1e-12
    return pmf


def index_offspring_pmf(N: int, I0: int, beta: float, gamma: float, S0: int | None = None,
                        k_max: int = 60) -> np.ndarray:
    """PMF EXATA do número de infecções secundárias do caso índice (com depleção de suscetíveis).

    Cadeia embutida estendida enquanto o índice está infeccioso: estado (s, i, k), k = filhos
    do índice até agora. De (s, i, k), com a = β·s·i/N + γ·i:
        infecção pelo índice     (β·s·i/N)/a · (1/i)      -> (s−1, i+1, k+1)
        infecção por outro       (β·s·i/N)/a · (i−1)/i    -> (s−1, i+1, k)
        recuperação do índice    (γ·i)/a · (1/i)          -> absorve com K = k
        recuperação de outro     (γ·i)/a · (i−1)/i        -> (s, i−1, k)
    As probabilidades 1/i seguem do sorteio uniforme do indivíduo responsável (permutabilidade).
    Percorre-se os níveis L = s + i em ordem decrescente (a infecção conserva L; a recuperação
    reduz L em 1) e, em cada nível, s decrescente. Massa com k > k_max é acumulada em k_max.
    """
    if S0 is None:
        S0 = N - I0
    K = k_max
    # nível corrente: dicionário s -> vetor (K+1) de massa em (s, i = L - s, ·)
    L = S0 + I0
    cur = {S0: np.zeros(K + 1)}
    cur[S0][0] = 1.0
    pmf = np.zeros(K + 1)
    while L > 0 and cur:
        nxt: dict[int, np.ndarray] = {}
        for s in range(max(cur.keys()), -1, -1):   # s decrescente; chaves s-1 podem ser criadas durante o laço
            if s not in cur:
                continue
            i = L - s
            v = cur[s]
            if i <= 0 or not v.any():
                continue
            a_inf = beta * s * i / N
            a_rec = gamma * i
            a = a_inf + a_rec
            p_inf, p_rec = a_inf / a, a_rec / a
            # recuperação do índice: absorção
            pmf += v * (p_rec / i)
            # recuperação de outro -> (s, i-1) no nível L-1
            if i > 1:
                nxt.setdefault(s, np.zeros(K + 1))
                nxt[s] += v * (p_rec * (i - 1) / i)
            if s > 0 and p_inf > 0:
                # infecção por outro -> (s-1, i+1, k), mesmo nível
                w_other = v * (p_inf * (i - 1) / i)
                # infecção pelo índice -> (s-1, i+1, k+1)
                w_index = v * (p_inf / i)
                tgt = cur.setdefault(s - 1, np.zeros(K + 1))
                tgt += w_other
                tgt[1:] += w_index[:-1]
                tgt[-1] += w_index[-1]  # overflow acumulado em k_max
        cur = nxt
        L -= 1
    assert abs(pmf.sum() - 1.0) < 1e-9, pmf.sum()
    return pmf

"""Estatística descritiva, intervalos de confiança, classificação de surtos e testes."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import Z_975

METRICS = ["final_size_frac", "peak_I", "t_peak", "duration", "n_events", "index_offspring"]
METRIC_LABELS_PT = {
    "final_size_frac": "Tamanho final (fração de N)",
    "peak_I": "Pico de infectados",
    "t_peak": "Tempo até o pico (dias)",
    "duration": "Duração da epidemia (dias)",
    "n_events": "Número de eventos",
    "index_offspring": "Offspring do caso índice",
}


# ---------------------------------------------------------------- descritiva
def t_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (math.nan, math.nan)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(n)
    tq = sps.t.ppf(0.5 + conf / 2, df=n - 1)
    return (m - tq * se, m + tq * se)


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return dict(n=0, mean=math.nan, sd=math.nan, ci_low=math.nan, ci_high=math.nan,
                    median=math.nan, q1=math.nan, q3=math.nan, min=math.nan, max=math.nan)
    lo, hi = t_ci(x)
    return dict(n=n, mean=float(x.mean()), sd=float(x.std(ddof=1)) if n > 1 else 0.0,
                ci_low=lo, ci_high=hi, median=float(np.median(x)),
                q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)),
                min=float(x.min()), max=float(x.max()))


def wilson_ci(k: int, n: int, z: float = Z_975) -> tuple[float, float]:
    """Intervalo de Wilson para uma proporção."""
    if n == 0:
        return (math.nan, math.nan)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return (0.0 if lo < 1e-12 else lo, hi)


def sample_size(mean: float, sd: float, rel_err: float, z: float = Z_975) -> float:
    """n = (z·s/(e·x̄))²."""
    return (z * sd / (rel_err * mean)) ** 2


# ------------------------------------------------------ surto maior × menor
def valley_threshold(final_size: np.ndarray, N: int, bin_width: int | None = None) -> dict:
    """Limiar entre surto menor e maior pelo vale do histograma do tamanho final (contagens).

    Histograma com bins de largura `bin_width` (padrão: 1% de N). A moda maior é o bin
    mais frequente com centro acima de 10% de N; o vale é a região de menor contagem
    entre a origem e a moda maior. Em caso de empate (p.ex. bins vazios), o limiar
    é o ponto médio da faixa empatada.
    """
    fs = np.asarray(final_size, dtype=float)
    if bin_width is None:
        bin_width = max(1, N // 100)
    edges = np.arange(0, N + bin_width + 1, bin_width)
    counts, _ = np.histogram(fs, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    big = centers > 0.10 * N
    if not big.any() or counts[big].sum() == 0:
        return dict(threshold=0.10 * N, valley_found=False, major_mode=math.nan,
                    bin_width=bin_width, counts=counts, edges=edges)
    j_major = int(np.flatnonzero(big)[np.argmax(counts[big])])
    seg = counts[:j_major + 1]
    cmin = seg.min()
    idx = np.flatnonzero(seg == cmin)
    # faixa contígua de mínimos mais à esquerda que seja adjacente ao primeiro mínimo
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)  # maior platô de vale
    thr = 0.5 * (edges[run[0]] + edges[run[-1] + 1])
    return dict(threshold=float(thr), valley_found=True, major_mode=float(centers[j_major]),
                valley_count=int(cmin), bin_width=bin_width, counts=counts, edges=edges)


def classify(final_size: np.ndarray, threshold: float) -> np.ndarray:
    """True = surto maior."""
    return np.asarray(final_size) >= threshold


def summarize_scenario(df: pd.DataFrame, major: np.ndarray, scenario: str) -> pd.DataFrame:
    """Estatísticas incondicionais / condicionais a surto maior / menor por métrica (major: máscara booleana)."""
    major = np.asarray(major, bool)
    rows = []
    for cond, mask in [("incondicional", np.ones(len(df), bool)), ("surto maior", major), ("surto menor", ~major)]:
        for m in METRICS:
            d = describe(df.loc[mask, m].to_numpy())
            rows.append(dict(cenario=scenario, condicao=cond, metrica=m, **d))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- testes
def ks_exponential(x: np.ndarray, gamma: float) -> dict:
    res = sps.kstest(np.asarray(x, dtype=float), "expon", args=(0.0, 1.0 / gamma))
    return dict(teste="KS", estatistica=float(res.statistic), p_valor=float(res.pvalue), n=int(len(x)))


def chi2_gof(observed: np.ndarray, expected_probs: np.ndarray, min_expected: float = 5.0) -> dict:
    """Qui-quadrado de aderência agrupando, a partir da cauda direita, células com esperado < 5.

    `observed`: contagens por categoria k = 0..K; `expected_probs`: probabilidades
    correspondentes (devem somar ~1; a massa restante é acrescentada à última célula).
    """
    obs = np.asarray(observed, dtype=float)
    n = obs.sum()
    probs = np.asarray(expected_probs, dtype=float).copy()
    probs[-1] += max(0.0, 1.0 - probs.sum())
    exp = n * probs
    # agrupa a cauda direita
    o, e = list(obs), list(exp)
    # (atenção: `e[-2] += e.pop()` seria incorreto em Python — o índice -2 é reavaliado após o pop)
    while len(e) > 1 and e[-1] < min_expected:
        eo, oo = e.pop(), o.pop()
        e[-1] += eo; o[-1] += oo
    # agrupa a cauda esquerda, se necessário
    while len(e) > 1 and e[0] < min_expected:
        eo, oo = e.pop(0), o.pop(0)
        e[0] += eo; o[0] += oo
    o, e = np.array(o), np.array(e)
    e *= o.sum() / e.sum()
    stat = float(((o - e) ** 2 / e).sum())
    dof = len(o) - 1
    p = float(sps.chi2.sf(stat, dof))
    return dict(teste="qui-quadrado", estatistica=stat, gl=dof, p_valor=p, n_celulas=len(o), n=int(n),
                obs=o, esp=e)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def compare_theory(name: str, empirical: float, ci: tuple[float, float], theory: float, unit: str = "") -> dict:
    err = empirical - theory
    rel = err / theory if theory not in (0, 0.0) and not math.isnan(theory) else math.nan
    inside = (ci[0] <= theory <= ci[1]) if not any(math.isnan(v) for v in ci) else False
    return dict(quantidade=name, simulado=empirical, ic_low=ci[0], ic_high=ci[1], teorico=theory,
                erro_abs=err, erro_rel=rel, teorico_no_ic=bool(inside), unidade=unit)


def cumulative_mean_ci(x: np.ndarray) -> pd.DataFrame:
    """Média acumulada e IC 95% (t) em função do número de réplicas."""
    x = np.asarray(x, dtype=float)
    n = np.arange(1, len(x) + 1)
    csum = np.cumsum(x)
    mean = csum / n
    csq = np.cumsum(x * x)
    var = np.where(n > 1, (csq - n * mean ** 2) / np.maximum(n - 1, 1), np.nan)
    var = np.clip(var, 0, None)
    se = np.sqrt(var / n)
    tq = sps.t.ppf(0.975, df=np.maximum(n - 1, 1))
    return pd.DataFrame(dict(n=n, media=mean, ic_low=mean - tq * se, ic_high=mean + tq * se))

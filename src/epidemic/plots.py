"""Figuras (rótulos em português). Cada figura é salva em PNG (300 dpi) e PDF."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import FIG_DIR

# paleta categórica fixa (ordem nunca reciclada)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
C_THEORY = "#0b0b0b"
C_GRAY = "#8a8985"
SIR_COLORS = {"S": C_BLUE, "I": C_ORANGE, "R": C_AQUA}
SCEN_COLORS = {"C1": C_BLUE, "C2": C_ORANGE, "C3": C_AQUA}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6,
})


def save(fig, name: str, fig_dir: Path = FIG_DIR) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = fig_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p.relative_to(fig_dir.parents[1])))
    plt.close(fig)
    return paths


# ------------------------------------------------------------- trajetórias
def plot_trajectories(traj_df: pd.DataFrame, mean_df: pd.DataFrame | None, ode: dict, N: int,
                      title: str, name: str, mean_label: str = "média (surtos maiores)"):
    """~30 réplicas (linhas finas) + média + EDO, para S, I e R."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    reps = traj_df["rep"].unique()
    for comp in ("S", "I", "R"):
        for j, r in enumerate(reps):
            d = traj_df[traj_df.rep == r]
            ax.plot(d.t, d[comp] / N, color=SIR_COLORS[comp], alpha=0.18, lw=0.7,
                    label=f"{comp} — réplicas" if j == 0 else None)
    if mean_df is not None:
        for comp in ("S", "I", "R"):
            ax.plot(mean_df.t, mean_df[comp] / N, color=SIR_COLORS[comp], lw=2.2,
                    label=f"{comp} — {mean_label}")
    for comp, key in (("S", "s"), ("I", "i"), ("R", "r")):
        ax.plot(ode["t"], ode[key], color=C_THEORY, ls="--", lw=1.2,
                label="EDO (Kermack–McKendrick)" if comp == "S" else None)
    ax.set_xlabel("Tempo (dias)")
    ax.set_ylabel("Fração da população")
    ax.set_title(title)
    ax.set_xlim(0, traj_df.t.max())
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), framealpha=0.9)
    return save(fig, name)


# -------------------------------------------------------------- histograma
def plot_final_size_hist(final_size: np.ndarray, N: int, thresholds: dict, title: str, name: str,
                         theory_frac: float | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    bw = max(1, N // 100)
    edges = np.arange(0, N + bw + 1, bw)
    ax.hist(final_size, bins=edges, color=C_BLUE, alpha=0.85, edgecolor="white", lw=0.4,
            label="réplicas")
    styles = ["-", "--", ":"]
    for j, (lab, thr) in enumerate(thresholds.items()):
        ax.axvline(thr, color=C_ORANGE, ls=styles[j % 3], lw=1.4, label=f"limiar {lab}: {thr:.0f}")
    if theory_frac is not None:
        ax.axvline(theory_frac * N, color=C_THEORY, ls="-.", lw=1.2,
                   label=f"tamanho final teórico: {theory_frac * N:.0f}")
    ax.set_xlabel("Tamanho final (número de infectados, incluindo I₀)")
    ax.set_ylabel("Número de réplicas")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(framealpha=0.9)
    return save(fig, name)


# ---------------------------------------------------------------- boxplots
def plot_boxplots(dfs: dict[str, pd.DataFrame], masks: dict[str, np.ndarray], name: str, cond_label: str):
    metrics = [("peak_I", "Pico de infectados"), ("t_peak", "Tempo até o pico (dias)"),
               ("final_size_frac", "Tamanho final (fração de N)"), ("duration", "Duração (dias)")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6))
    for ax, (m, lab) in zip(axes, metrics):
        data, labels, colors = [], [], []
        for k, df in dfs.items():
            x = df.loc[masks[k], m].to_numpy()
            if len(x) == 0:
                x = np.array([np.nan])
            data.append(x); labels.append(k); colors.append(SCEN_COLORS[k])
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6, showfliers=True,
                        flierprops=dict(marker=".", markersize=2.5, alpha=0.4, markeredgecolor=C_GRAY),
                        medianprops=dict(color=C_THEORY, lw=1.2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7); patch.set_edgecolor(c)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Comparação C1 × C2 × C3 — {cond_label}", y=1.02)
    fig.tight_layout()
    return save(fig, name)


# ----------------------------------------------------------------- varredura
def plot_sweep(df: pd.DataFrame, name_prob: str, name_fs: str):
    paths = []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(df.R0, df.p_major, yerr=[np.maximum(df.p_major - df.p_major_low, 0), np.maximum(df.p_major_high - df.p_major, 0)],
                fmt="o", color=C_BLUE, ms=5, capsize=3, lw=1.2, label="simulação (IC 95% de Wilson)")
    ax.plot(df.R0, df.p_major_theory, color=C_THEORY, ls="--", label="teoria: 1 − 1/R₀ (ramificação)")
    ax.plot(df.R0, df.p_major_exact, color=C_ORANGE, ls=":", marker="x", ms=4, lw=1, label="CTMC exata (N = 1000, mesmo limiar)")
    ax.axvline(1.0, color=C_GRAY, ls=":", lw=1)
    ax.set_xlabel("R₀"); ax.set_ylabel("Probabilidade de surto maior")
    ax.set_title("Probabilidade de surto maior × R₀ (N = 1000, I₀ = 1)")
    ax.set_ylim(-0.02, 1.02); ax.legend()
    paths += save(fig, name_prob)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    d = df.dropna(subset=["fs_cond_mean"])
    ax.errorbar(d.R0, d.fs_cond_mean, yerr=[np.maximum(d.fs_cond_mean - d.fs_cond_low, 0), np.maximum(d.fs_cond_high - d.fs_cond_mean, 0)],
                fmt="o", color=C_BLUE, ms=5, capsize=3, lw=1.2, label="simulação, condicional a surto maior (IC 95%)")
    ax.plot(df.R0, df.fs_theory, color=C_THEORY, ls="--", label="equação do tamanho final: s₀·τ")
    ax.plot(d.R0, d.fs_cond_exact, color=C_ORANGE, ls=":", marker="x", ms=4, lw=1, label="CTMC exata (N = 1000, mesmo limiar)")
    ax.axvline(1.0, color=C_GRAY, ls=":", lw=1)
    ax.set_xlabel("R₀"); ax.set_ylabel("Tamanho final (fração de N)")
    ax.set_title("Tamanho final condicional × R₀ (N = 1000, I₀ = 1)")
    ax.set_ylim(-0.02, 1.02); ax.legend(loc="lower right")
    paths += save(fig, name_fs)
    return paths


# ---------------------------------------------------------------------- V2
def plot_mean_field(grid, mean_i, lo, hi, ode_t, ode_i, name: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(grid, lo, hi, color=C_BLUE, alpha=0.25, label="IC 95% da média (t)")
    ax.plot(grid, mean_i, color=C_BLUE, label="média das réplicas: I(t)/N")
    ax.plot(ode_t, ode_i, color=C_THEORY, ls="--", label="EDO: i(t)")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Fração infecciosa I(t)/N")
    ax.set_title(title); ax.legend()
    return save(fig, name)


# ------------------------------------------------------------ distribuições
def plot_duration_cdf_qq(dur: np.ndarray, gamma: float, name_cdf: str, name_qq: str):
    paths = []
    x = np.sort(dur)
    F = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    step = max(1, len(x) // 5000)
    ax.step(x[::step], F[::step], where="post", color=C_BLUE, label=f"CDF empírica (n = {len(x)})")
    tt = np.linspace(0, x.max(), 400)
    ax.plot(tt, 1 - np.exp(-gamma * tt), color=C_THEORY, ls="--", label=f"Exp(γ), γ = {gamma:g}/dia")
    ax.set_xlabel("Duração infecciosa (dias)"); ax.set_ylabel("F(t)")
    ax.set_title("Duração infecciosa — CDF empírica × Exp(γ) (C1, todas as réplicas)")
    ax.legend(loc="lower right")
    paths += save(fig, name_cdf)

    fig, ax = plt.subplots(figsize=(5, 5))
    probs = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    q_theory = -np.log(1 - probs) / gamma
    idx = np.unique(np.concatenate([np.arange(0, len(x), step), np.arange(max(0, len(x) - 300), len(x))]))
    ax.plot(q_theory[idx], x[idx], ".", ms=3, color=C_BLUE, label="quantis (amostra + 300 maiores)")
    m = max(q_theory.max(), x.max())
    ax.plot([0, m], [0, m], color=C_THEORY, ls="--", lw=1, label="y = x")
    ax.set_xlabel("Quantis teóricos Exp(γ) (dias)"); ax.set_ylabel("Quantis empíricos (dias)")
    ax.set_title("QQ-plot — duração infecciosa (C1)"); ax.legend()
    paths += save(fig, name_qq)
    return paths


def plot_pmf_compare(k, p_emp, p_theory, err, xlabel, title, name, label_theory, p_extra=None, label_extra=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.4
    ax.bar(np.asarray(k) - w / 2, p_emp, width=w, color=C_BLUE, label="simulação (IC 95% normal)", yerr=err,
           capsize=2, error_kw=dict(lw=0.8, color=C_GRAY))
    ax.bar(np.asarray(k) + w / 2, p_theory, width=w, color=C_THEORY, alpha=0.75, label=label_theory)
    if p_extra is not None:
        ax.plot(np.asarray(k), p_extra, "s", ms=4, color=C_ORANGE, label=label_extra)
    ax.set_xlabel(xlabel); ax.set_ylabel("Probabilidade")
    ax.set_title(title); ax.legend()
    ax.set_xticks(list(k))
    ax.grid(axis="x", visible=False)
    return save(fig, name)


# ------------------------------------------------------------- convergência
def plot_convergence(panels: list[tuple[pd.DataFrame, str, float | None]], name: str, title: str):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.8), squeeze=False)
    for ax, (df, lab, ref) in zip(axes[0], panels):
        ax.fill_between(df.n, df.ic_low, df.ic_high, color=C_BLUE, alpha=0.25, label="IC 95% (t)")
        ax.plot(df.n, df.media, color=C_BLUE, label="média acumulada")
        if ref is not None:
            ax.axhline(ref, color=C_THEORY, ls="--", lw=1, label="valor teórico")
        ax.set_xscale("log")
        ax.set_xlabel("Número de réplicas"); ax.set_ylabel("Tamanho final (fração de N)")
        ax.set_title(lab, fontsize=10); ax.legend(loc="upper right")
    fig.suptitle(title, y=1.02); fig.tight_layout()
    return save(fig, name)

"""Varredura em R0 (0.5 a 4.0, passo 0.25), p = 0, N = 1000, I0 = 1, 500 réplicas por ponto.
Curvas: probabilidade de surto maior (empírica × teórica) e tamanho final condicional (× equação)."""
import math

import numpy as np
import pandas as pd

from _common import log, DATA_RAW, DATA_PROC
from epidemic.config import GAMMA, EXP_KEY
from epidemic.gillespie import SIRParams
from epidemic.runner import run_experiment
from epidemic import stats as st
from epidemic import theory as th
from epidemic import plots
from epidemic.tables import write_table, save_json
from epidemic.exact_ctmc import final_size_pmf

N, I0, N_REPS = 1000, 1, 500
R0_GRID = np.round(np.arange(0.5, 4.0 + 1e-9, 0.25), 2)


def main():
    rows = []
    total = 0.0
    for j, R0 in enumerate(R0_GRID):
        params = SIRParams(N=N, I0=I0, R0=float(R0), gamma=GAMMA, p=0.0)
        name = f"sweep_R0_{R0:.2f}"
        df, _, _, elapsed = run_experiment(name, params, N_REPS, n_traj=0, exp_key=EXP_KEY["sweep"] * 1000 + j,
                                           out_dir=DATA_RAW / "sweep" / name)
        total += elapsed
        fs = df["final_size"].to_numpy()
        s0 = params.S0 / N
        vt = st.valley_threshold(fs, N)
        # limiar: vale do histograma quando R_eff > 1 e o vale existe; senão 10% de N
        thr = vt["threshold"] if (params.R_eff > 1 and vt["valley_found"]) else 0.10 * N
        major = st.classify(fs, thr)
        k = int(major.sum())
        lo, hi = st.wilson_ci(k, N_REPS)
        cond = st.describe(fs[major] / N) if k >= 2 else None
        # referência exata (CTMC, N = 1000) com o MESMO limiar
        pmf = final_size_pmf(N, I0, params.beta, GAMMA, S0=params.S0)
        kk = np.arange(N + 1)
        p_major_exact = float(pmf[kk >= thr].sum())
        fs_cond_exact = float((kk * pmf)[kk >= thr].sum() / p_major_exact / N) if p_major_exact > 0 else math.nan
        rows.append(dict(
            R0=float(R0), R_eff=params.R_eff, n_reps=N_REPS, limiar=thr, n_surto_maior=k,
            p_major=k / N_REPS, p_major_low=lo, p_major_high=hi,
            p_major_theory=th.prob_major_outbreak(params.R_eff, I0),
            fs_cond_mean=cond["mean"] if cond else math.nan,
            fs_cond_low=cond["ci_low"] if cond else math.nan,
            fs_cond_high=cond["ci_high"] if cond else math.nan,
            fs_theory=th.final_size_fraction(float(R0), s0),
            p_major_exact=p_major_exact, fs_cond_exact=fs_cond_exact,
            fs_uncond_mean=fs.mean() / N,
            fs_uncond_theory_branching=th.expected_total_subcritical(params.R_eff, I0) / N if params.R_eff < 1 else math.nan,
            tempo_s=elapsed,
        ))
        log(f"R0={R0:.2f}: P(maior)={k / N_REPS:.3f} [{lo:.3f},{hi:.3f}] teo={rows[-1]['p_major_theory']:.3f}  "
            f"FS|maior={rows[-1]['fs_cond_mean']:.4f} teo={rows[-1]['fs_theory']:.4f}  ({elapsed:.1f}s)")
    df = pd.DataFrame(rows)
    df["p_major_theory_in_ci"] = (df.p_major_low <= df.p_major_theory) & (df.p_major_theory <= df.p_major_high)
    df["fs_theory_in_ci"] = (df.fs_cond_low <= df.fs_theory) & (df.fs_theory <= df.fs_cond_high)
    df["p_major_exact_in_ci"] = (df.p_major_low <= df.p_major_exact) & (df.p_major_exact <= df.p_major_high)
    df["fs_exact_in_ci"] = (df.fs_cond_low <= df.fs_cond_exact) & (df.fs_cond_exact <= df.fs_cond_high)
    df["p_major_err_abs"] = df.p_major - df.p_major_theory
    df["fs_err_abs"] = df.fs_cond_mean - df.fs_theory
    df["fs_err_rel"] = df.fs_err_abs / df.fs_theory.where(df.fs_theory > 0)
    write_table(df, "varredura_R0")
    df.to_csv(DATA_PROC / "varredura_R0.csv", index=False)
    figs = plots.plot_sweep(df, "varredura_prob_surto_maior", "varredura_tamanho_final")
    save_json(dict(figuras=figs, tempo_total_s=total), "varredura_meta")
    with open(DATA_RAW / "sweep" / "runtime.json", "w") as f:
        import json
        json.dump(dict(experimento="sweep", tempo_execucao_s=total, n_replicacoes=int(N_REPS * len(R0_GRID))), f, indent=2)
    log(f"varredura concluída em {total:.1f}s")


if __name__ == "__main__":
    main()

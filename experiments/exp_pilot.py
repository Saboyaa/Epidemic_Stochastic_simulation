"""Piloto: 200 réplicas por cenário principal para dimensionar o número de réplicas.

n = (z·s/(e·x̄))², e = 2%, calculado sobre a média do tamanho final CONDICIONAL a
surto maior (C1, C2). Em C3 (R_eff < 1) não há surto maior; usa-se a média
incondicional do tamanho final. n_final = clip(ceil(n), N_MIN, N_CAP).
"""
import math

import numpy as np
import pandas as pd

from _common import log, SCENARIOS
from epidemic.config import EXP_KEY, N_PILOT, N_MIN, N_CAP, REL_ERR, Z_975, GAMMA
from epidemic.gillespie import SIRParams
from epidemic.runner import run_experiment
from epidemic.stats import valley_threshold, classify, sample_size
from epidemic.tables import write_table, save_json


def main():
    rows = []
    for key, sc in SCENARIOS.items():
        params = SIRParams(N=sc["N"], I0=sc["I0"], R0=sc["R0"], gamma=GAMMA, p=sc["p"])
        df, _, _, elapsed = run_experiment(f"pilot_{key}", params, N_PILOT, n_traj=0)
        fs = df["final_size"].to_numpy()
        if params.R_eff > 1:
            thr = valley_threshold(fs, params.N)["threshold"]
            x = fs[classify(fs, thr)] / params.N
            base = "condicional a surto maior"
        else:
            thr = math.nan
            x = fs / params.N
            base = "incondicional (R_eff < 1)"
        mean, sd = x.mean(), x.std(ddof=1)
        n_raw = sample_size(mean, sd, REL_ERR, Z_975)
        n_final = int(min(max(math.ceil(n_raw), N_MIN), N_CAP))
        rows.append(dict(cenario=key, n_piloto=N_PILOT, limiar_piloto=thr, base=base, n_base=len(x),
                         media=mean, desvio=sd, cv=sd / mean, z=Z_975, e=REL_ERR,
                         n_calculado=n_raw, n_escolhido=n_final, tempo_piloto_s=elapsed))
        log(f"{key}: x̄={mean:.4f} s={sd:.4f} n=({Z_975:.3f}·{sd:.4f}/({REL_ERR}·{mean:.4f}))²={n_raw:.1f} -> {n_final}")
    tab = pd.DataFrame(rows)
    write_table(tab, "tamanho_amostral")
    save_json({r["cenario"]: r["n_escolhido"] for r in rows}, "n_replicacoes")


if __name__ == "__main__":
    main()

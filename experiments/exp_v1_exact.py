"""V1 — validação contra a CTMC exata: N = 10, I0 = 1, R0 = 2.5, 100000 réplicas.
Compara a PMF simulada do tamanho final com a exata (qui-quadrado e distância de variação total)."""
import numpy as np
import pandas as pd

from _common import log, DATA_RAW, DATA_PROC
from epidemic.config import GAMMA
from epidemic.gillespie import SIRParams
from epidemic.runner import run_experiment
from epidemic.exact_ctmc import final_size_pmf
from epidemic import stats as st
from epidemic import plots
from epidemic.tables import write_table, save_json

N, I0, R0, N_REPS = 10, 1, 2.5, 100_000


def main():
    params = SIRParams(N=N, I0=I0, R0=R0, gamma=GAMMA, p=0.0)
    df, _, _, elapsed = run_experiment("V1", params, N_REPS, n_traj=0)
    log(f"V1: {N_REPS} réplicas em {elapsed:.1f}s")
    fs = df["final_size"].to_numpy()
    obs = np.bincount(fs, minlength=N + 1).astype(float)
    exact = final_size_pmf(N, I0, params.beta, GAMMA)
    np.save(DATA_RAW / "V1" / "final_size_pmf_exact.npy", exact)

    p_emp = obs / N_REPS
    tvd = st.total_variation(p_emp, exact)
    # qui-quadrado nas categorias k = 1..N (k = 0 é impossível; esperado 0)
    chi = st.chi2_gof(obs[1:], exact[1:], min_expected=5.0)
    tab = pd.DataFrame(dict(k=np.arange(N + 1), observado=obs.astype(int), p_simulada=p_emp, p_exata=exact,
                            esperado=N_REPS * exact, diferenca=p_emp - exact))
    tab["ic95_low"] = p_emp - 1.96 * np.sqrt(p_emp * (1 - p_emp) / N_REPS)
    tab["ic95_high"] = p_emp + 1.96 * np.sqrt(p_emp * (1 - p_emp) / N_REPS)
    tab["exata_no_ic"] = (tab.ic95_low <= tab.p_exata) & (tab.p_exata <= tab.ic95_high)
    write_table(tab, "V1_pmf_tamanho_final")
    tests = pd.DataFrame([
        dict(teste="qui-quadrado", comparacao="PMF simulada × CTMC exata (tamanho final)", n=N_REPS,
             estatistica=chi["estatistica"], gl=chi["gl"], p_valor=chi["p_valor"], n_celulas=chi["n_celulas"]),
        dict(teste="distância de variação total", comparacao="PMF simulada × CTMC exata", n=N_REPS,
             estatistica=tvd, gl=np.nan, p_valor=np.nan, n_celulas=N),
    ])
    write_table(tests, "V1_testes")
    k = np.arange(1, N + 1)
    err = 1.96 * np.sqrt(p_emp[1:] * (1 - p_emp[1:]) / N_REPS)
    figs = plots.plot_pmf_compare(k, p_emp[1:], exact[1:], err, "Tamanho final Z (número de infectados, incluindo I₀)",
                                  f"V1 — Tamanho final: PMF simulada ({N_REPS} réplicas) × CTMC exata (N = {N}, R₀ = {R0})",
                                  "V1_pmf_tamanho_final", "CTMC exata (programação dinâmica)")
    save_json(dict(tvd=tvd, chi2=chi["estatistica"], gl=chi["gl"], p=chi["p_valor"], media_sim=fs.mean(),
                   media_exata=float((np.arange(N + 1) * exact).sum()), figuras=figs, tempo_s=elapsed), "V1_resumo")
    log(f"V1: TVD={tvd:.5f}  chi2={chi['estatistica']:.2f} gl={chi['gl']} p={chi['p_valor']:.3f}")


if __name__ == "__main__":
    main()

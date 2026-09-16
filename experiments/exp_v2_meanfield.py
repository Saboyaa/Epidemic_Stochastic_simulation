"""V2 — limite de campo médio: N = 10000, I0 = 10, R0 = 2.5, 200 réplicas.
Trajetória média de I(t)/N com banda de IC 95% × EDO; RMSE e erro relativo no pico e no tempo do pico."""
import numpy as np
import pandas as pd

from _common import log, DATA_RAW, DATA_PROC
from epidemic.config import GAMMA
from epidemic.gillespie import SIRParams
from epidemic.ode import solve_sir
from epidemic.runner import run_experiment, save_trajectories
from epidemic import stats as st
from epidemic import plots
from epidemic.tables import write_table, save_json

N, I0, R0, N_REPS = 10_000, 10, 2.5, 200
T_MAX, N_GRID = 120.0, 601


def main():
    params = SIRParams(N=N, I0=I0, R0=R0, gamma=GAMMA, p=0.0)
    df, trajs, _, elapsed = run_experiment("V2", params, N_REPS, n_traj=N_REPS)
    log(f"V2: {N_REPS} réplicas em {elapsed:.1f}s; duração máx = {df.duration.max():.1f} d")
    grid = np.linspace(0.0, T_MAX, N_GRID)
    traj_df = save_trajectories(trajs, grid, DATA_RAW / "V2" / "trajectories.csv")
    I = traj_df.pivot(index="rep", columns="t", values="I").to_numpy() / N   # (reps, grid)
    mean_i = I.mean(axis=0)
    se = I.std(axis=0, ddof=1) / np.sqrt(N_REPS)
    from scipy import stats as sps
    tq = sps.t.ppf(0.975, N_REPS - 1)
    lo, hi = mean_i - tq * se, mean_i + tq * se

    s0, i0 = params.S0 / N, I0 / N
    ot, os_, oi, orr = solve_sir(params.beta, GAMMA, s0, i0, 0.0, T_MAX, n_points=N_GRID)
    rmse = float(np.sqrt(np.mean((mean_i - oi) ** 2)))
    k_sim, k_ode = int(np.argmax(mean_i)), int(np.argmax(oi))
    peak_mean_traj, t_peak_mean_traj = float(mean_i[k_sim]), float(grid[k_sim])
    peak_ode, t_peak_ode = float(oi[k_ode]), float(ot[k_ode])
    # também: média dos picos individuais (por réplica) — distingue "pico da média" de "média dos picos"
    peaks = df["peak_I"].to_numpy() / N
    tpeaks = df["t_peak"].to_numpy()
    n_major = int((df["final_size_frac"] > 0.1).sum())

    rows = [
        dict(quantidade="RMSE de I(t)/N (média das réplicas × EDO), 0–120 d", simulado=rmse, teorico=np.nan, erro_abs=np.nan, erro_rel=np.nan),
        dict(quantidade="pico da trajetória MÉDIA I(t)/N", simulado=peak_mean_traj, teorico=peak_ode,
             erro_abs=peak_mean_traj - peak_ode, erro_rel=(peak_mean_traj - peak_ode) / peak_ode),
        dict(quantidade="tempo do pico da trajetória MÉDIA (dias)", simulado=t_peak_mean_traj, teorico=t_peak_ode,
             erro_abs=t_peak_mean_traj - t_peak_ode, erro_rel=(t_peak_mean_traj - t_peak_ode) / t_peak_ode),
        dict(quantidade="média dos picos por réplica I_max/N", simulado=peaks.mean(), teorico=peak_ode,
             erro_abs=peaks.mean() - peak_ode, erro_rel=(peaks.mean() - peak_ode) / peak_ode),
        dict(quantidade="média dos tempos até o pico por réplica (dias)", simulado=tpeaks.mean(), teorico=t_peak_ode,
             erro_abs=tpeaks.mean() - t_peak_ode, erro_rel=(tpeaks.mean() - t_peak_ode) / t_peak_ode),
        dict(quantidade="tamanho final médio (fração de N)", simulado=df.final_size_frac.mean(), teorico=float(s0 - os_[-1] + i0),
             erro_abs=df.final_size_frac.mean() - float(s0 - os_[-1] + i0),
             erro_rel=(df.final_size_frac.mean() - float(s0 - os_[-1] + i0)) / float(s0 - os_[-1] + i0)),
    ]
    tab = pd.DataFrame(rows)
    ci_peaks = st.t_ci(peaks); ci_tp = st.t_ci(tpeaks)
    tab["ic_low"] = [np.nan, np.nan, np.nan, ci_peaks[0], ci_tp[0], st.t_ci(df.final_size_frac.to_numpy())[0]]
    tab["ic_high"] = [np.nan, np.nan, np.nan, ci_peaks[1], ci_tp[1], st.t_ci(df.final_size_frac.to_numpy())[1]]
    write_table(tab, "V2_campo_medio")
    pd.DataFrame(dict(t=grid, media_I_N=mean_i, ic_low=lo, ic_high=hi, edo_i=oi)).to_csv(DATA_PROC / "V2_media_vs_edo.csv", index=False)
    figs = plots.plot_mean_field(grid, mean_i, lo, hi, ot, oi, "V2_campo_medio",
                                 f"V2 — I(t)/N médio ({N_REPS} réplicas, N = {N}, I₀ = {I0}, R₀ = {R0}) × EDO")
    save_json(dict(rmse=rmse, peak_mean_traj=peak_mean_traj, t_peak_mean_traj=t_peak_mean_traj, peak_ode=peak_ode,
                   t_peak_ode=t_peak_ode, mean_of_peaks=float(peaks.mean()), mean_of_tpeaks=float(tpeaks.mean()),
                   n_major=n_major, figuras=figs, tempo_s=elapsed), "V2_resumo")
    log(f"V2: RMSE={rmse:.5f}; pico média {peak_mean_traj:.4f} vs EDO {peak_ode:.4f}; t_pico {t_peak_mean_traj:.1f} vs {t_peak_ode:.1f}")


if __name__ == "__main__":
    main()

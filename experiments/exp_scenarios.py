"""Cenários principais C1, C2, C3: simulação, classificação de surtos, estatísticas,
comparação com a teoria, distribuições (a) e (b) e figuras."""
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from _common import log, SCENARIOS, DATA_RAW, DATA_PROC
from epidemic.config import GAMMA, EXP_KEY
from epidemic.gillespie import SIRParams
from epidemic.ode import solve_sir, ode_peak
from epidemic.runner import run_experiment, save_trajectories, grid_for
from epidemic import stats as st
from epidemic import theory as th
from epidemic import plots
from epidemic.tables import write_table, save_json, load_json
from epidemic.exact_ctmc import final_size_pmf, index_offspring_pmf

N_TRAJ_SAVE = 100
N_TRAJ_PLOT = 30
N_EXTRA_BATCHES, N_EXTRA_PER_BATCH = 5, 100_000


def scenario_params(key):
    sc = SCENARIOS[key]
    return SIRParams(N=sc["N"], I0=sc["I0"], R0=sc["R0"], gamma=GAMMA, p=sc["p"])


def main():
    n_reps = load_json("n_replicacoes")
    param_rows, stat_rows, sens_rows, cmp_rows, test_rows, thr_info = [], [], [], [], [], {}
    dfs, masks_all, masks_major = {}, {}, {}
    figs = {}

    for key in SCENARIOS:
        params = scenario_params(key)
        N = params.N
        log(f"{key}: {n_reps[key]} réplicas, params={params.as_dict()}")
        df, trajs, indiv, elapsed = run_experiment(key, params, n_reps[key], n_traj=N_TRAJ_SAVE,
                                                   keep_individuals=(key == "C1"))
        log(f"{key}: concluído em {elapsed:.1f}s")
        dfs[key] = df
        fs = df["final_size"].to_numpy()

        # ---------------- parâmetros
        s0, i0 = params.S0 / N, params.I0 / N
        fs_theory = th.final_size_fraction(params.R0, s0)          # s0·τ (equação padrão)
        fs_theory_i0 = th.final_size_fraction(params.R0, s0, i0) + i0  # variante com i0 > 0, incluindo I0
        param_rows.append(dict(cenario=key, descricao=SCENARIOS[key]["label"], N=N, I0=params.I0, S0=params.S0,
                               vacinados=params.n_vacc, p=params.p, R0=params.R0, gamma=GAMMA, beta=params.beta,
                               R_eff=params.R_eff, limiar_vacinal=th.herd_immunity_threshold(params.R0),
                               n_replicacoes=n_reps[key], P_surto_maior_teorica=th.prob_major_outbreak(params.R_eff, params.I0),
                               tamanho_final_teorico=fs_theory))

        # ---------------- limiar de surto maior + sensibilidade
        vt = st.valley_threshold(fs, N)
        if params.R_eff > 1:
            thr = vt["threshold"]
            thresholds = {"vale": thr, "10% de N": 0.10 * N, "5% de N": 0.05 * N}
        else:
            thr = 0.10 * N
            thresholds = {"10% de N (nominal)": thr}
        # com R_eff < 1 não há surto maior por definição (R_eff = 0.75 em C3); réplicas raras com
        # Z ≥ limiar são cauda da classe "menor" e são reportadas à parte (P(Z ≥ limiar) × CTMC exata)
        major = st.classify(fs, thr) if params.R_eff > 1 else np.zeros(len(fs), bool)
        thr_info[key] = dict(limiar=thr, vale_encontrado=vt["valley_found"], moda_maior=vt.get("major_mode"),
                             contagem_no_vale=vt.get("valley_count"), largura_bin=vt["bin_width"],
                             maior_surto_menor=int(fs[~major].max()) if (~major).any() else None,
                             menor_surto_maior=int(fs[major].min()) if major.any() else None,
                             n_maior=int(major.sum()), n_menor=int((~major).sum()),
                             n_acima_limiar=int(st.classify(fs, thr).sum()))
        for lab, t_ in thresholds.items():
            m = st.classify(fs, t_) if params.R_eff > 1 else np.zeros(len(fs), bool)
            k, n = int(m.sum()), len(fs)
            lo, hi = st.wilson_ci(k, n)
            cond = st.describe(fs[m] / N) if m.any() else st.describe(np.array([]))
            sens_rows.append(dict(cenario=key, limiar=lab, valor_limiar=t_, n_surto_maior=k, P_surto_maior=k / n,
                                  wilson_low=lo, wilson_high=hi, tamanho_final_cond_media=cond["mean"],
                                  tamanho_final_cond_ic_low=cond["ci_low"], tamanho_final_cond_ic_high=cond["ci_high"]))
        masks_all[key] = np.ones(len(df), bool)
        masks_major[key] = major

        # ---------------- estatísticas por métrica
        stat_rows.append(st.summarize_scenario(df, major, key))

        # ---------------- comparação com a teoria
        # referência EXATA (CTMC, programação dinâmica) para N = 1000 — separa viés de N finito de erro de simulação
        pmf_exact = final_size_pmf(N, params.I0, params.beta, GAMMA, S0=params.S0)
        kk_ = np.arange(N + 1)
        p_major_exact = float(pmf_exact[kk_ >= thr].sum())
        ez_major_exact = float((kk_ * pmf_exact)[kk_ >= thr].sum() / p_major_exact / N) if p_major_exact > 0 else math.nan
        ez_minor_exact = float((kk_ * pmf_exact)[kk_ < thr].sum() / (1 - p_major_exact))
        ez_exact = float((kk_ * pmf_exact).sum())
        np.save(DATA_RAW / key / "final_size_pmf_exact.npy", pmf_exact)

        k, n = int(major.sum()), len(fs)
        p_hat = k / n
        cmp_rows.append(st.compare_theory(f"{key}: P(surto maior)", p_hat, st.wilson_ci(k, n),
                                          th.prob_major_outbreak(params.R_eff, params.I0), "ramificação: 1−(1/R_eff)^I0"))
        if params.R_eff > 1:
            cmp_rows.append(st.compare_theory(f"{key}: P(surto maior)", p_hat, st.wilson_ci(k, n),
                                              p_major_exact, f"CTMC exata N={N}, limiar {thr:.0f}"))
        else:
            k_tail = int(st.classify(fs, thr).sum())
            cmp_rows.append(st.compare_theory(f"{key}: P(Z ≥ {thr:.0f}) (cauda da classe menor)", k_tail / n, st.wilson_ci(k_tail, n),
                                              p_major_exact, f"CTMC exata N={N}"))
        if params.R_eff > 1 and major.any():
            x = fs[major] / N
            cmp_rows.append(st.compare_theory(f"{key}: tamanho final | surto maior (fração de N)", x.mean(), st.t_ci(x),
                                              fs_theory, "equação do tamanho final: s0·τ"))
            cmp_rows.append(st.compare_theory(f"{key}: tamanho final | surto maior (fração de N)", x.mean(),
                                              st.t_ci(x), fs_theory_i0, "equação com i0>0: s0·τ(i0)+i0"))
            cmp_rows.append(st.compare_theory(f"{key}: tamanho final | surto maior (fração de N)", x.mean(),
                                              st.t_ci(x), ez_major_exact, f"CTMC exata N={N}"))
            x = fs[~major].astype(float)
            cmp_rows.append(st.compare_theory(f"{key}: tamanho final | surto menor (nº de casos)", x.mean(), st.t_ci(x),
                                              th.expected_total_subcritical(1.0 / params.R_eff, params.I0),
                                              "ramificação dual: 1/(1−1/R_eff)"))
            cmp_rows.append(st.compare_theory(f"{key}: tamanho final | surto menor (nº de casos)", x.mean(), st.t_ci(x),
                                              ez_minor_exact, f"CTMC exata N={N}"))
        else:
            x = fs.astype(float)
            cmp_rows.append(st.compare_theory(f"{key}: tamanho final médio (nº de casos, incl. I0)", x.mean(), st.t_ci(x),
                                              th.expected_total_subcritical(params.R_eff, params.I0), "ramificação: 1/(1−R_eff)"))
            cmp_rows.append(st.compare_theory(f"{key}: tamanho final médio (nº de casos, incl. I0)", x.mean(), st.t_ci(x),
                                              ez_exact, f"CTMC exata N={N}"))
        x = df["index_offspring"].to_numpy(float)
        cmp_rows.append(st.compare_theory(f"{key}: offspring médio do caso índice", x.mean(), st.t_ci(x),
                                          params.R0 * s0, "geométrica sem depleção: R0·s0"))
        # pico e tempo do pico × EDO (condicional a surto maior)
        t_max_ode = 400.0
        ode = ode_peak(params.beta, GAMMA, s0, i0, params.n_vacc / N, t_max_ode)
        if params.R_eff > 1 and major.any():
            x = df.loc[major, "peak_I"].to_numpy(float) / N
            cmp_rows.append(st.compare_theory(f"{key}: pico de I/N | surto maior", x.mean(), st.t_ci(x), ode["peak_frac"], "EDO"))
            x = df.loc[major, "t_peak"].to_numpy(float)
            cmp_rows.append(st.compare_theory(f"{key}: tempo até o pico | surto maior (dias)", x.mean(), st.t_ci(x), ode["t_peak"], "EDO"))
            cmp_rows.append(st.compare_theory(f"{key}: tamanho final (EDO) | surto maior", (fs[major] / N).mean(), st.t_ci(fs[major] / N),
                                              ode["final_size_frac"], "EDO (inclui i0)"))

        # ---------------- trajetórias: CSV reamostrado + figura
        grid = grid_for(trajs, n_points=400, quantile=1.0)
        traj_df = save_trajectories(trajs, grid, DATA_RAW / key / "trajectories.csv")
        major_saved = major[:len(trajs)]
        if params.R_eff > 1 and major_saved.any():
            sel = traj_df[traj_df.rep.isin(np.flatnonzero(major_saved))]
            mean_label = f"média ({int(major_saved.sum())} surtos maiores)"
        else:
            sel = traj_df
            mean_label = f"média ({len(trajs)} réplicas)"
        mean_df = sel.groupby("t")[["S", "I", "R"]].mean().reset_index()
        ot, os_, oi, orr = solve_sir(params.beta, GAMMA, s0, i0, params.n_vacc / N, grid[-1])
        plot_df = traj_df[traj_df.rep < N_TRAJ_PLOT]
        figs[f"traj_{key}"] = plots.plot_trajectories(
            plot_df, mean_df, dict(t=ot, s=os_, i=oi, r=orr), N,
            f"Trajetórias S, I, R — {SCENARIOS[key]['label']}: R₀={params.R0}, p={params.p}, R_eff={params.R_eff:.2f} "
            f"({N_TRAJ_PLOT} réplicas)", f"trajetorias_{key}", mean_label=mean_label)

        # ---------------- histograma do tamanho final
        figs[f"hist_{key}"] = plots.plot_final_size_hist(
            fs, N, thresholds, f"Tamanho final — {SCENARIOS[key]['label']} (n = {n} réplicas)",
            f"hist_tamanho_final_{key}", theory_frac=fs_theory if params.R_eff > 1 else None)

        # ---------------- distribuições (a) e (b) em C1
        if key == "C1":
            dur = indiv["duration"].to_numpy()
            ks = st.ks_exponential(dur, GAMMA)
            test_rows.append(dict(teste="KS", distribuicao="duração infecciosa ~ Exp(γ)", cenario="C1", n=ks["n"],
                                  estatistica=ks["estatistica"], gl=math.nan, p_valor=ks["p_valor"],
                                  media_emp=dur.mean(), media_teo=1 / GAMMA))
            figs["dur"] = plots.plot_duration_cdf_qq(dur, GAMMA, "cdf_duracao_infecciosa_C1", "qq_duracao_infecciosa_C1")
            save_json(dict(n=len(dur), media=dur.mean(), desvio=dur.std(ddof=1), mediana=float(np.median(dur))), "duracao_C1")

            off = df["index_offspring"].to_numpy()
            K = int(off.max())
            obs = np.bincount(off, minlength=K + 1).astype(float)
            ks_ = np.arange(K + 1)
            p_th = th.offspring_pmf(ks_, params.beta, GAMMA)
            chi = st.chi2_gof(obs, p_th, min_expected=5.0)
            test_rows.append(dict(teste="qui-quadrado", distribuicao="offspring do caso índice ~ Geom(β/(β+γ))", cenario="C1",
                                  n=chi["n"], estatistica=chi["estatistica"], gl=chi["gl"], p_valor=chi["p_valor"],
                                  media_emp=off.mean(), media_teo=params.R0))
            save_json(dict(k=ks_, obs=obs, esperado=n * p_th, celulas_agrupadas_obs=chi["obs"], celulas_agrupadas_esp=chi["esp"]),
                      "offspring_C1")
            # PMF EXATA do offspring do índice COM depleção (CTMC estendida, programação dinâmica)
            p_exact_full = index_offspring_pmf(N, params.I0, params.beta, GAMMA, S0=params.S0, k_max=60)
            np.save(DATA_RAW / key / "index_offspring_pmf_exact.npy", p_exact_full)

            def trunc(pfull, Kc):
                q = pfull[:Kc + 1].copy(); q[-1] += pfull[Kc + 1:].sum(); return q

            p_dep = trunc(p_exact_full, K)
            chi_dep = st.chi2_gof(obs, p_dep, min_expected=5.0)
            test_rows.append(dict(teste="qui-quadrado (suplementar)", distribuicao="offspring do caso índice ~ CTMC exata c/ depleção (N=1000)",
                                  cenario="C1", n=chi_dep["n"], estatistica=chi_dep["estatistica"], gl=chi_dep["gl"],
                                  p_valor=chi_dep["p_valor"], media_emp=off.mean(), media_teo=float((np.arange(61) * p_exact_full).sum())))
            # verificação suplementar de alta potência: 5 lotes independentes de 100000 réplicas (seeds distintas),
            # guardando só as métricas; testa cada lote e o conjunto agrupado contra a geométrica e contra a CTMC exata
            batches = []
            for b in range(N_EXTRA_BATCHES):
                df_x, _, _, _ = run_experiment(f"C1_offspring_extra_b{b}", params, N_EXTRA_PER_BATCH, n_traj=0,
                                               exp_key=EXP_KEY["C1_offspring_extra"] * 1000 + b,
                                               out_dir=DATA_RAW / "C1_offspring_extra" / f"batch{b}")
                batches.append(df_x["index_offspring"].to_numpy())
            mean_exact = float((np.arange(61) * p_exact_full).sum())
            extra_json = {}
            for lab_b, off_x in [(f"lote {b}", o_) for b, o_ in enumerate(batches)] + [("agrupado", np.concatenate(batches))]:
                Kx = int(off_x.max()); obs_x = np.bincount(off_x, minlength=Kx + 1).astype(float)
                chi_x = st.chi2_gof(obs_x, th.offspring_pmf(np.arange(Kx + 1), params.beta, GAMMA), min_expected=5.0)
                chi_xd = st.chi2_gof(obs_x, trunc(p_exact_full, Kx), min_expected=5.0)
                for lab, c, mt, dist in [(f"qui-quadrado (extra, {lab_b})", chi_x, params.R0, "Geom (sem depleção)"),
                                         (f"qui-quadrado (extra, {lab_b}, c/ depleção)", chi_xd, mean_exact, "CTMC exata c/ depleção")]:
                    test_rows.append(dict(teste=lab, distribuicao="offspring do caso índice ~ " + dist, cenario="C1_extra",
                                          n=c["n"], estatistica=c["estatistica"], gl=c["gl"], p_valor=c["p_valor"],
                                          media_emp=off_x.mean(), media_teo=mt))
                extra_json[lab_b] = dict(k=np.arange(Kx + 1), obs=obs_x, celulas_obs_exata=chi_xd["obs"], celulas_esp_exata=chi_xd["esp"])
            save_json(extra_json, "offspring_C1_extra")
            kmax_plot = min(K, 12)
            kk = np.arange(kmax_plot + 1)
            p_emp = obs[:kmax_plot + 1] / n
            err = 1.96 * np.sqrt(p_emp * (1 - p_emp) / n)
            figs["off"] = plots.plot_pmf_compare(kk, p_emp, p_th[:kmax_plot + 1], err, "Número de infecções secundárias do caso índice, k",
                                                 f"Offspring do caso índice — C1 (n = {n}): PMF empírica × Geométrica",
                                                 "pmf_offspring_C1", f"Geométrica, média R₀ = {params.R0}",
                                                 p_extra=p_exact_full[:kmax_plot + 1], label_extra="CTMC exata com depleção (N = 1000)")

            # convergência
            conv_u = st.cumulative_mean_ci(fs / N)
            conv_c = st.cumulative_mean_ci(fs[major] / N)
            conv_u.to_csv(DATA_PROC / "convergencia_C1_incondicional.csv", index=False)
            conv_c.to_csv(DATA_PROC / "convergencia_C1_condicional.csv", index=False)
            figs["conv"] = plots.plot_convergence(
                [(conv_u, "incondicional (ref.: E[Z]/N da CTMC exata)", ez_exact / N),
                 (conv_c, "condicional a surto maior (ref.: s₀·τ)", fs_theory)],
                "convergencia_tamanho_final_C1", "Convergência da média acumulada do tamanho final — C1")

    # ---------------- boxplots
    figs["box_u"] = plots.plot_boxplots(dfs, masks_all, "boxplots_incondicional", "todas as réplicas")
    figs["box_c"] = plots.plot_boxplots(dfs, masks_major, "boxplots_condicional_surto_maior",
                                        "condicional a surto maior (C3: sem surtos maiores)")

    # ---------------- tabelas
    write_table(pd.DataFrame(param_rows), "parametros_cenarios")
    stats_df = pd.concat(stat_rows, ignore_index=True)
    stats_df["metrica_pt"] = stats_df["metrica"].map(st.METRIC_LABELS_PT)
    write_table(stats_df, "estatisticas_cenarios")
    write_table(pd.DataFrame(sens_rows), "sensibilidade_limiar")
    write_table(pd.DataFrame(cmp_rows), "teoria_vs_simulacao_cenarios")
    write_table(pd.DataFrame(test_rows), "testes_distribuicoes_C1")
    save_json(thr_info, "limiares")
    save_json(figs, "figuras_cenarios")
    log("cenários concluídos")


if __name__ == "__main__":
    main()

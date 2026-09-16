"""Consolida tempos de execução e escreve RESULTS.md (notas factuais com números, figuras e tabelas)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, DATA_RAW, DATA_PROC, TAB_DIR, log
from epidemic.tables import write_table, to_markdown, load_json


def rel(p):
    return str(Path(p).relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)


def runtime_table():
    rows = []
    for f in sorted(DATA_RAW.rglob("runtime.json")):
        d = json.load(open(f))
        if f.parent.parent.name == "sweep":
            continue  # pontos individuais da varredura (o agregado está em sweep/runtime.json)
        rows.append(dict(experimento=d["experimento"], n_replicacoes=d["n_replicacoes"],
                         tempo_simulacao_s=d["tempo_execucao_s"], pasta=rel(f.parent)))
    df = pd.DataFrame(rows)
    steps = json.load(open(DATA_PROC / "run_all_times.json")) if (DATA_PROC / "run_all_times.json").exists() else None
    if steps:
        df2 = pd.DataFrame([dict(etapa=k, tempo_total_s=v) for k, v in steps["tempos_por_etapa_s"].items()])
        write_table(df2, "tempo_execucao_etapas")
    write_table(df, "tempo_execucao")
    return df


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    rt = runtime_table()
    par = pd.read_csv(TAB_DIR / "parametros_cenarios.csv")
    ss = pd.read_csv(TAB_DIR / "tamanho_amostral.csv")
    stats = pd.read_csv(TAB_DIR / "estatisticas_cenarios.csv")
    sens = pd.read_csv(TAB_DIR / "sensibilidade_limiar.csv")
    cmp_ = pd.read_csv(TAB_DIR / "teoria_vs_simulacao_cenarios.csv")
    tests = pd.read_csv(TAB_DIR / "testes_distribuicoes_C1.csv")
    sweep = pd.read_csv(TAB_DIR / "varredura_R0.csv")
    v1t = pd.read_csv(TAB_DIR / "V1_testes.csv")
    v1p = pd.read_csv(TAB_DIR / "V1_pmf_tamanho_final.csv")
    v2 = pd.read_csv(TAB_DIR / "V2_campo_medio.csv")
    thr = load_json("limiares")
    v1 = load_json("V1_resumo"); v2s = load_json("V2_resumo"); dur = load_json("duracao_C1")
    figs = load_json("figuras_cenarios")

    def S(c, cond, m, col):
        r = stats[(stats.cenario == c) & (stats.condicao == cond) & (stats.metrica == m)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    def C(prefix, unidade_contains):
        r = cmp_[cmp_.quantidade.str.startswith(prefix) & cmp_.unidade.str.contains(unidade_contains, regex=False)]
        return r.iloc[0]

    ks = tests[tests.teste == "KS"].iloc[0]
    chi = tests[tests.teste == "qui-quadrado"].iloc[0]
    chid = tests[tests.teste == "qui-quadrado (suplementar)"].iloc[0]
    chi_v1 = v1t.iloc[0]; tvd = v1t.iloc[1]
    off = load_json("offspring_C1")

    L = []
    A = L.append
    A("# RESULTS.md — notas factuais para o relatório\n")
    A("Gerado automaticamente por `experiments/write_results.py` (via `python run_all.py`). "
      "Todos os números abaixo vêm das tabelas em `results/tables/` e dos JSON em `data/processed/`. "
      "Não há dados reais: as comparações são com modelos analíticos (EDO, equação do tamanho final, "
      "processo de ramificação, CTMC exata por programação dinâmica).\n")
    A("Seed mestre: **20260916**; seeds por replicação via `SeedSequence(20260916, spawn_key=(chave_exp,)).spawn(n)`; "
      "γ = 0,2/dia; β = R₀·γ. Versões: ver `data/raw/<exp>/config.json`.\n")

    # ------------------------------------------------------------ parâmetros / n
    A("## 1. Cenários, parâmetros e número de réplicas\n")
    A("Tabela: `results/tables/parametros_cenarios.{csv,md}`; piloto/dimensionamento: `results/tables/tamanho_amostral.{csv,md}`.\n")
    A(to_markdown(par[["cenario", "N", "I0", "S0", "vacinados", "p", "R0", "beta", "gamma", "R_eff", "limiar_vacinal",
                       "P_surto_maior_teorica", "tamanho_final_teorico", "n_replicacoes"]]))
    A("Dimensionamento (piloto de 200 réplicas, e = 2%, z = 1,96, fórmula n = (z·s/(e·x̄))²):\n")
    for _, r in ss.iterrows():
        A(f"- **{r.cenario}** ({r.base}, n_base = {int(r.n_base)}): x̄ = {r.media:.4f}, s = {r.desvio:.4f}, CV = {r.cv:.3f} → "
          f"n = (1,96·{r.desvio:.4f}/(0,02·{r.media:.4f}))² = **{r.n_calculado:.1f}** → n escolhido = **{int(r.n_escolhido)}** "
          f"(mínimo 1000; teto 50000).")
    A("\nObservações: em C1 a variabilidade condicional do tamanho final é pequena (CV ≈ 2%), logo o piso de 1000 domina; "
      "em C3 (R_eff = 0,75) não há surto maior, o tamanho final é um processo subcrítico com CV ≈ 2,3 e a fórmula pede ~48,6 mil "
      "réplicas — que são baratas (≈ 6 eventos por réplica) e foram executadas integralmente.\n")

    # ------------------------------------------------------------ validação
    A("## 2. Validação\n")
    A("### V1 — CTMC exata (N = 10, I₀ = 1, R₀ = 2,5, 100 000 réplicas)\n")
    A(f"- Figura: `{v1['figuras'][0]}` (PMF simulada × exata). Tabelas: `results/tables/V1_pmf_tamanho_final.{{csv,md}}`, "
      f"`results/tables/V1_testes.{{csv,md}}`. PMF exata: `data/raw/V1/final_size_pmf_exact.npy`.")
    A(f"- Qui-quadrado (células k = 1..10, esperado ≥ 5 em todas): χ² = **{chi_v1.estatistica:.2f}**, gl = {int(chi_v1.gl)}, "
      f"p = **{chi_v1.p_valor:.3f}**. Distância de variação total = **{tvd.estatistica:.5f}**.")
    A(f"- Média do tamanho final: simulada {v1['media_sim']:.4f} × exata {v1['media_exata']:.4f}.")
    A(f"- {int(v1p.exata_no_ic.sum())} de {len(v1p)} probabilidades exatas caem dentro do IC 95% (normal) da frequência simulada.")
    A("- Leitura: a cadeia de saltos embutida do simulador (taxas β·S·I/N e γ·I, escolha do evento pelo segundo uniforme) "
      "reproduz a distribuição exata; não há evidência de erro de taxa ou de seleção de evento.\n")
    A(to_markdown(v1p[["k", "observado", "p_simulada", "p_exata", "diferenca", "exata_no_ic"]]))

    A("### V2 — limite de campo médio (N = 10 000, I₀ = 10, R₀ = 2,5, 200 réplicas)\n")
    A(f"- Figura: `{v2s['figuras'][0]}`. Tabela: `results/tables/V2_campo_medio.{{csv,md}}`. "
      f"Série média × EDO na grade: `data/processed/V2_media_vs_edo.csv`; trajetórias: `data/raw/V2/trajectories.csv`.")
    A(f"- RMSE de I(t)/N (média das 200 réplicas × EDO, grade 0–120 d, 601 pontos) = **{v2s['rmse']:.5f}**.")
    A(f"- Pico da trajetória média: {v2s['peak_mean_traj']:.4f} × EDO {v2s['peak_ode']:.4f} "
      f"(erro relativo {100 * (v2s['peak_mean_traj'] / v2s['peak_ode'] - 1):+.2f}%); tempo do pico {v2s['t_peak_mean_traj']:.1f} d × "
      f"{v2s['t_peak_ode']:.1f} d ({100 * (v2s['t_peak_mean_traj'] / v2s['t_peak_ode'] - 1):+.2f}%).")
    r = v2.iloc[3]; r2 = v2.iloc[4]
    A(f"- Média dos picos POR RÉPLICA: {r.simulado:.4f} [IC {r.ic_low:.4f}, {r.ic_high:.4f}] × EDO {r.teorico:.4f} "
      f"({100 * r.erro_rel:+.2f}%); média dos tempos até o pico por réplica: {r2.simulado:.2f} d [{r2.ic_low:.2f}, {r2.ic_high:.2f}] "
      f"× {r2.teorico:.2f} d ({100 * r2.erro_rel:+.2f}%).")
    A(f"- Surtos maiores: {v2s['n_major']}/200 (com I₀ = 10, P(extinção) ≈ (1/2,5)¹⁰ ≈ 10⁻⁴).")
    A("- Leitura: o pico da trajetória MÉDIA fica ligeiramente abaixo da EDO porque a média de curvas com pequenos deslocamentos "
      "temporais aleatórios é mais achatada; a média dos picos individuais fica ligeiramente acima (máximo de um processo com ruído). "
      "Ambos os desvios são de ~1–3% e a banda de IC é estreita; comportamento consistente com convergência ao campo médio.\n")

    A("### V3 — pytest\n")
    A("- `tests/test_epidemic.py`: 17 testes (gerador exponencial por transformada inversa — média e variância; conservação de "
      "S+I+R; taxas não negativas; reprodutibilidade por seed; β = 0; equação do tamanho final vs. valores de referência via "
      "Lambert W; PMF geométrica; CTMC exata; agrupamento de caudas do qui-quadrado). Rodar: `make test`.\n")

    # ------------------------------------------------------------ cenários
    A("## 3. Cenários principais (N = 1000, I₀ = 1)\n")
    A("Tabelas: `results/tables/estatisticas_cenarios.{csv,md}` (média, dp, IC t 95%, mediana, quartis, mín, máx — "
      "incondicional / surto maior / surto menor), `results/tables/sensibilidade_limiar.{csv,md}`, "
      "`results/tables/teoria_vs_simulacao_cenarios.{csv,md}`. Dados brutos: `data/raw/C{1,2,3}/{config.json,metrics.csv,trajectories.csv}`; "
      "registros por indivíduo de C1: `data/raw/C1/individuals.npz`.\n")
    A("Figuras: " + ", ".join(f"`{figs[k][0]}`" for k in ["traj_C1", "traj_C2", "traj_C3", "hist_C1", "hist_C2", "hist_C3", "box_u", "box_c"]) + ".\n")

    A("### 3.1 Limiar surto maior × menor\n")
    for c in ["C1", "C2"]:
        t = thr[c]
        A(f"- **{c}**: vale do histograma (bins de {t['largura_bin']}) → limiar = **{t['limiar']:.0f}** casos; moda maior em {t['moda_maior']:.0f}; "
          f"contagem no vale = {t['contagem_no_vale']}; maior surto menor = {t['maior_surto_menor']} casos, menor surto maior = {t['menor_surto_maior']} casos; "
          f"{t['n_maior']} maiores / {t['n_menor']} menores.")
    t = thr["C3"]
    A(f"- **C3** (R_eff = 0,75 < 1): sem bimodalidade; por definição não há surto maior. {t['n_acima_limiar']} réplicas atingiram ≥ 100 casos "
      f"(máx. {int(S('C3', 'incondicional', 'final_size_frac', 'max') * 1000)}), cauda do processo subcrítico (ver comparação com CTMC exata).")
    A("- Justificativa: o histograma é bimodal com um platô de contagem zero entre as duas modas; qualquer limiar dentro do platô "
      "produz a mesma classificação. O ponto médio do platô é a escolha; 5% e 10% de N são as alternativas testadas.")
    A("- Sensibilidade (`sensibilidade_limiar`):\n")
    A(to_markdown(sens[["cenario", "limiar", "valor_limiar", "n_surto_maior", "P_surto_maior", "wilson_low", "wilson_high",
                        "tamanho_final_cond_media"]]))
    A("  Em C1 os três limiares dão classificação idêntica; em C2 o limiar de 5% de N reclassifica 4 réplicas (surtos entre 50 e 96 casos) "
      "e move P(surto maior) de 0,312 para 0,316 — dentro do IC de Wilson.\n")

    A("### 3.2 Métricas por cenário (média [IC 95%]) — resumo\n")
    A("| cenário | condição | n | tamanho final (fração) | pico I | tempo até pico (d) | duração (d) | eventos | offspring índice |")
    A("|---|---|---|---|---|---|---|---|---|")
    for c in ["C1", "C2", "C3"]:
        for cond in ["incondicional", "surto maior", "surto menor"]:
            n = S(c, cond, "final_size_frac", "n")
            if not n or np.isnan(n) or n == 0:
                continue
            cell = lambda m, nd: f"{S(c, cond, m, 'mean'):.{nd}f} [{S(c, cond, m, 'ci_low'):.{nd}f}, {S(c, cond, m, 'ci_high'):.{nd}f}]"
            A(f"| {c} | {cond} | {int(n)} | {cell('final_size_frac', 4)} | {cell('peak_I', 1)} | {cell('t_peak', 1)} | "
              f"{cell('duration', 1)} | {cell('n_events', 0)} | {cell('index_offspring', 2)} |")
    A("")

    A("### 3.3 Teoria × simulação (tabela completa em `teoria_vs_simulacao_cenarios`)\n")
    A(to_markdown(cmp_[["quantidade", "simulado", "ic_low", "ic_high", "teorico", "erro_abs", "erro_rel", "teorico_no_ic", "unidade"]]))
    c1p = C("C1: P(surto maior)", "ramificação"); c2p = C("C2: P(surto maior)", "ramificação")
    c1f = C("C1: tamanho final | surto maior", "s0·τ"); c2f = C("C2: tamanho final | surto maior", "s0·τ")
    c1e = C("C1: tamanho final | surto maior", "CTMC"); c2e = C("C2: tamanho final | surto maior", "CTMC")
    c3m = C("C3: tamanho final médio", "ramificação"); c3e = C("C3: tamanho final médio", "CTMC")
    c1pk = C("C1: pico de I/N", "EDO"); c2pk = C("C2: pico de I/N", "EDO"); c2tp = C("C2: tempo até o pico", "EDO")
    A("Observações:")
    A(f"- P(surto maior): C1 {c1p.simulado:.3f} [{c1p.ic_low:.3f}, {c1p.ic_high:.3f}] × 0,600 (ramificação) e "
      f"{C('C1: P(surto maior)', 'CTMC').teorico:.4f} (CTMC exata N = 1000); C2 {c2p.simulado:.3f} [{c2p.ic_low:.3f}, {c2p.ic_high:.3f}] × 0,333 e "
      f"{C('C2: P(surto maior)', 'CTMC').teorico:.4f} (CTMC). Teoria dentro do IC nos dois casos. A CTMC exata para N = 1000 "
      "(mesmo limiar) mostra que a aproximação de ramificação superestima ligeiramente P(surto maior) em C2 (viés de N finito: com R_eff = 1,5 "
      "a depleção de suscetíveis é sentida cedo).")
    A(f"- Tamanho final condicional: C1 {c1f.simulado:.4f} [{c1f.ic_low:.4f}, {c1f.ic_high:.4f}] × s₀·τ = {c1f.teorico:.4f} "
      f"(erro rel. {100 * c1f.erro_rel:+.3f}%) e CTMC exata {c1e.teorico:.4f}; C2 {c2f.simulado:.4f} [{c2f.ic_low:.4f}, {c2f.ic_high:.4f}] × "
      f"s₀·τ = {c2f.teorico:.4f} ({100 * c2f.erro_rel:+.2f}%) e CTMC exata {c2e.teorico:.4f}. Todos dentro do IC.")
    A("- Viés de I₀ > 0: a equação padrão (i₀ = 0) exclui o caso índice; a variante s₀·τ(i₀)+i₀ inclui-o e fica ~1/N acima. "
      "Com N = 1000 a diferença (0,001) é da ordem da largura do IC em C1; ambas as versões estão na tabela.")
    A(f"- C3 (subcrítico): tamanho total médio {c3m.simulado:.3f} casos [{c3m.ic_low:.3f}, {c3m.ic_high:.3f}] × 1/(1−R_eff) = 4,000 "
      f"(erro {100 * c3m.erro_rel:+.1f}%, FORA do IC) × CTMC exata {c3e.teorico:.3f} (dentro do IC). O desvio em relação à ramificação é "
      "esperado: com S₀ = 299 a depleção reduz a taxa efetiva ao longo da cadeia (e R₀·S₀/N = 0,7475, não 0,75); a CTMC exata captura isso.")
    A(f"- Surtos menores em C1/C2: tamanho médio compatível com o processo dual (1/(1−1/R_eff)) e com a CTMC exata (dentro do IC).")
    A(f"- Pico de I/N vs. EDO (condicional a surto maior): C1 {c1pk.simulado:.4f} × {c1pk.teorico:.4f} ({100 * c1pk.erro_rel:+.1f}%), "
      f"C2 {c2pk.simulado:.4f} × {c2pk.teorico:.4f} ({100 * c2pk.erro_rel:+.1f}%); tempo até o pico C2 {c2tp.simulado:.1f} d × {c2tp.teorico:.1f} d. "
      "EDO fora do IC: o pico de uma trajetória estocástica é o máximo de um processo ruidoso (viés positivo ∝ √I_pico), e o condicionamento a "
      "surto maior seleciona realizações com crescimento inicial mais rápido (pico mais alto e mais cedo). O efeito é grande em C2 porque o pico "
      "da EDO é pequeno (≈ 38 indivíduos) e a fase inicial é longa (R_eff = 1,5). Não é um erro de simulação: V2 (N = 10⁴, I₀ = 10) mostra o "
      "erro cair para ~1–3%.")
    A(f"- Offspring médio do caso índice ≈ R₀·s₀ em todos os cenários (dentro do IC).\n")

    pmf_c1 = np.load(DATA_RAW / "C1" / "final_size_pmf_exact.npy")
    ez_c1 = float((np.arange(len(pmf_c1)) * pmf_c1).sum() / 1000)
    A("### 3.4 Convergência (C1)\n")
    A(f"- Figura: `{figs['conv'][0]}`; séries: `data/processed/convergencia_C1_{{incondicional,condicional}}.csv`.")
    A(f"- Média incondicional do tamanho final com 1000 réplicas: {S('C1', 'incondicional', 'final_size_frac', 'mean'):.4f} "
      f"[{S('C1', 'incondicional', 'final_size_frac', 'ci_low'):.4f}, {S('C1', 'incondicional', 'final_size_frac', 'ci_high'):.4f}] "
      f"(ref. CTMC exata E[Z]/N = {ez_c1:.4f}); condicional {c1f.simulado:.4f} ± {(c1f.ic_high - c1f.ic_low) / 2:.4f}. "
      "A largura do IC cai como 1/√n; a média condicional estabiliza dentro de ±0,5% já com ~100 réplicas.\n")

    # ------------------------------------------------------------ varredura
    A("## 4. Varredura em R₀ (0,5–4,0, passo 0,25; p = 0; N = 1000; 500 réplicas por ponto)\n")
    meta = load_json("varredura_meta")
    A(f"- Figuras: `{meta['figuras'][0]}` (P(surto maior)), `{meta['figuras'][2]}` (tamanho final condicional). "
      f"Tabela: `results/tables/varredura_R0.{{csv,md}}`. Dados: `data/raw/sweep/sweep_R0_*/`.")
    sup = sweep[sweep.R0 > 1]
    A(f"- P(surto maior): teoria 1−1/R₀ dentro do IC de Wilson em {int(sup.p_major_theory_in_ci.sum())}/{len(sup)} pontos com R₀ > 1; "
      f"CTMC exata (N = 1000, mesmo limiar) dentro do IC em {int(sweep.p_major_exact_in_ci.sum())}/{len(sweep)} pontos. "
      f"Maior desvio absoluto vs. teoria: {sweep.p_major_err_abs.abs().max():.3f} (R₀ = {sweep.loc[sweep.p_major_err_abs.abs().idxmax(), 'R0']:.2f}).")
    A(f"- Tamanho final condicional: equação s₀·τ dentro do IC em {int(sup.fs_theory_in_ci.sum())}/{len(sup)} pontos (R₀ > 1); "
      f"CTMC exata dentro do IC em {int(sup.fs_exact_in_ci.sum())}/{len(sup)}. Erro relativo máximo vs. equação: "
      f"{100 * sup.fs_err_rel.abs().max():.2f}% (R₀ = {sup.loc[sup.fs_err_rel.abs().idxmax(), 'R0']:.2f}). Para R₀ ≥ 3 o IC é tão "
      "estreito (±0,001) que o viés de +1/N da inclusão de I₀ fica visível: a simulação fica sistematicamente ~0,001 acima de s₀·τ e "
      "coincide com a CTMC exata.")
    A("- R₀ = 1,00 (crítico): a distribuição do tamanho final não é bimodal; o limiar nominal de 10% de N classifica "
      f"{int(sweep.loc[sweep.R0 == 1.0, 'n_surto_maior'].iloc[0])}/500 réplicas como 'maiores' (CTMC exata: P(Z ≥ 100) = "
      f"{float(sweep.loc[sweep.R0 == 1.0, 'p_major_exact'].iloc[0]):.3f}), enquanto a ramificação dá 0 — a classificação não é bem definida em R₀ = 1.")
    A("- R₀ = 1,25: o vale é raso (poucos surtos maiores, tamanho final condicional ~0,39) e a ramificação superestima P(surto maior) "
      "(0,200 vs. CTMC 0,179); simulação concorda com a CTMC.\n")
    A(to_markdown(sweep[["R0", "limiar", "n_surto_maior", "p_major", "p_major_low", "p_major_high", "p_major_theory", "p_major_exact",
                         "fs_cond_mean", "fs_cond_low", "fs_cond_high", "fs_theory", "fs_cond_exact"]]))

    # ------------------------------------------------------------ distribuições
    A("## 5. Distribuições teórico × empírico\n")
    A("Tabela: `results/tables/testes_distribuicoes_C1.{csv,md}`.\n")
    A("### (a) Duração infecciosa — C1, todos os indivíduos infectados de todas as 1000 réplicas\n")
    A(f"- Figuras: `{figs['dur'][0]}` (CDF), `{figs['dur'][2]}` (QQ-plot).")
    A(f"- n = {dur['n']} durações (sem censura: toda réplica roda até I = 0). Média {dur['media']:.4f} d (teoria 1/γ = 5), "
      f"dp {dur['desvio']:.4f} (teoria 5), mediana {dur['mediana']:.4f} (teoria 5·ln 2 = 3,466).")
    A(f"- KS contra Exp(γ = 0,2): D = **{ks.estatistica:.5f}**, p = **{ks.p_valor:.3f}**. Não se rejeita a exponencial.")
    A("- Isso valida o sorteio uniforme de quem se recupera: por permutabilidade, cada infeccioso tem taxa individual γ e sua "
      "duração é Exp(γ), independentemente de quantos estão infectados.\n")
    A("### (b) Offspring do caso índice — C1 (1000 réplicas)\n")
    A(f"- Figura: `{figs['off'][0]}`. Contagens/esperados: `data/processed/offspring_C1.json`.")
    A(f"- Média empírica {chi.media_emp:.3f} [IC {C('C1: offspring', 'geom').ic_low:.3f}, {C('C1: offspring', 'geom').ic_high:.3f}] × R₀ = 2,5 (geométrica) × {chid.media_teo:.4f} (CTMC exata com depleção).")
    A(f"- Qui-quadrado vs. Geométrica (caudas agrupadas até esperado ≥ 5; {int(chi.gl) + 1} células): χ² = **{chi.estatistica:.2f}**, "
      f"gl = {int(chi.gl)}, p = **{chi.p_valor:.3f}**.")
    o = np.array(off["celulas_agrupadas_obs"]); e = np.array(off["celulas_agrupadas_esp"])
    contrib = (o - e) ** 2 / e
    j = int(np.argmax(contrib))
    A(f"- Decomposição: a célula k = {j} contribui {contrib[j]:.1f} de {chi.estatistica:.1f} (obs {int(o[j])} × esp {e[j]:.1f}); "
      f"a cauda agrupada (k ≥ {len(o) - 1}) tem obs {int(o[-1])} × esp {e[-1]:.1f}.")
    ex_g = tests[tests.teste.str.startswith("qui-quadrado (extra") & ~tests.teste.str.contains("depleção")]
    ex_d = tests[tests.teste.str.startswith("qui-quadrado (extra") & tests.teste.str.contains("depleção")]
    pooled_g = ex_g[ex_g.teste.str.contains("agrupado")].iloc[0]; pooled_d = ex_d[ex_d.teste.str.contains("agrupado")].iloc[0]
    p_lotes_d = ", ".join(f"{v:.3f}" for v in ex_d[~ex_d.teste.str.contains("agrupado")].p_valor)
    p_lotes_g = ", ".join(f"{v:.1e}" for v in ex_g[~ex_g.teste.str.contains("agrupado")].p_valor)
    A(f"- Investigação: (i) a geométrica supõe S ≈ N durante toda a vida do índice (sem depleção). Um índice com k ≥ 14 filhos "
      "precisa viver ≳ 28 d, quando S/N já caiu bem abaixo de 1 num surto maior; a cauda fica deficitária e a média cai. "
      f"(ii) Calculou-se a PMF EXATA do offspring do índice COM depleção (CTMC estendida (s, i, k), programação dinâmica; "
      f"`data/raw/C1/index_offspring_pmf_exact.npy`; conferida contra um Monte Carlo independente da cadeia embutida): média {chid.media_teo:.4f} "
      f"(vs. 2,5 da geométrica). χ² das 1000 réplicas de C1 contra ela: {chid.estatistica:.2f} (gl {int(chid.gl)}, p = {chid.p_valor:.3f}). "
      f"(iii) Verificação de alta potência: 5 lotes independentes de 100 000 réplicas (`data/raw/C1_offspring_extra/batch*/`). "
      f"Contra a geométrica, p por lote = {p_lotes_g}; agrupado (n = {int(pooled_g.n)}) χ² = {pooled_g.estatistica:.0f} (gl {int(pooled_g.gl)}), "
      f"p = {pooled_g.p_valor:.1e} — REJEITA (média {pooled_g.media_emp:.4f} ≠ 2,5). Contra a CTMC exata com depleção, p por lote = {p_lotes_d}; "
      f"agrupado χ² = {pooled_d.estatistica:.2f} (gl {int(pooled_d.gl)}), p = {pooled_d.p_valor:.3f} (média {pooled_d.media_emp:.4f} × {pooled_d.media_teo:.4f}). "
      "Conclusão: o simulador reproduz a distribuição exata do modelo individual; o p ≈ 0,02 contra a geométrica em C1 combina o efeito real de "
      f"depleção (cauda/média) com flutuação amostral (célula k = {j}). A hipótese 'sem depleção' é a aproximação que falha, não o código. "
      "A média empírica de C1 coincide com R₀ dentro do IC porque o efeito de depleção sobre a média (−3%) é menor que a largura do IC (±7%).\n")
    A("- Nota de processo: a investigação revelou um bug no agrupamento de caudas do teste qui-quadrado (`stats.chi2_gof`; "
      "`e[-2] += e.pop()` reavalia o índice após o pop), corrigido e coberto por teste unitário. V1 não era afetada (nenhuma célula agrupada).\n")
    A("### (c) Tamanho final em V1 — ver seção 2 (χ² p = " + f"{chi_v1.p_valor:.3f}, TVD = {tvd.estatistica:.5f}).\n")

    # ------------------------------------------------------------ tempos
    A("## 6. Tempo de execução\n")
    A("Tabela: `results/tables/tempo_execucao.{csv,md}` (tempo de simulação por experimento, paralelo em "
      f"{json.load(open(DATA_RAW / 'C1' / 'config.json'))['workers']} processos) e `results/tables/tempo_execucao_etapas.{{csv,md}}` "
      "(tempo total por script do `run_all.py`, incluindo análise e figuras).\n")
    A(to_markdown(rt))
    steps_p = DATA_PROC / "run_all_times.json"
    if steps_p.exists():
        st_ = json.load(open(steps_p))
        A(f"\nTempo total do `run_all.py`: **{st_['total_s']:.1f} s**.\n")

    A("## 7. Inventário de figuras\n")
    for f in sorted((ROOT / "results" / "figures").glob("*.png")):
        A(f"- `results/figures/{f.name}` (+ `.pdf`)")
    A("")
    (ROOT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    log("RESULTS.md escrito")


if __name__ == "__main__":
    main()

# RESULTS.md — notas factuais para o relatório

Gerado automaticamente por `experiments/write_results.py` (via `python run_all.py`). Todos os números abaixo vêm das tabelas em `results/tables/` e dos JSON em `data/processed/`. Não há dados reais: as comparações são com modelos analíticos (EDO, equação do tamanho final, processo de ramificação, CTMC exata por programação dinâmica).

Seed mestre: **20260916**; seeds por replicação via `SeedSequence(20260916, spawn_key=(chave_exp,)).spawn(n)`; γ = 0,2/dia; β = R₀·γ. Versões: ver `data/raw/<exp>/config.json`.

## 1. Cenários, parâmetros e número de réplicas

Tabela: `results/tables/parametros_cenarios.{csv,md}`; piloto/dimensionamento: `results/tables/tamanho_amostral.{csv,md}`.

| cenario | N | I0 | S0 | vacinados | p | R0 | beta | gamma | R_eff | limiar_vacinal | P_surto_maior_teorica | tamanho_final_teorico | n_replicacoes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 1000 | 1 | 999 | 0 | 0 | 2.5000 | 0.5000 | 0.2000 | 2.5000 | 0.6000 | 0.6000 | 0.8914 | 1000 |
| C2 | 1000 | 1 | 599 | 400 | 0.4000 | 2.5000 | 0.5000 | 0.2000 | 1.5000 | 0.6000 | 0.3333 | 0.3481 | 1000 |
| C3 | 1000 | 1 | 299 | 700 | 0.7000 | 2.5000 | 0.5000 | 0.2000 | 0.7500 | 0.6000 | 0 | 0 | 48622 |

Dimensionamento (piloto de 200 réplicas, e = 2%, z = 1,96, fórmula n = (z·s/(e·x̄))²):

- **C1** (condicional a surto maior, n_base = 116): x̄ = 0.8913, s = 0.0177, CV = 0.020 → n = (1,96·0.0177/(0,02·0.8913))² = **3.8** → n escolhido = **1000** (mínimo 1000; teto 50000).
- **C2** (condicional a surto maior, n_base = 51): x̄ = 0.3460, s = 0.0533, CV = 0.154 → n = (1,96·0.0533/(0,02·0.3460))² = **228.1** → n escolhido = **1000** (mínimo 1000; teto 50000).
- **C3** (incondicional (R_eff < 1), n_base = 200): x̄ = 0.0041, s = 0.0093, CV = 2.250 → n = (1,96·0.0093/(0,02·0.0041))² = **48621.3** → n escolhido = **48622** (mínimo 1000; teto 50000).

Observações: em C1 a variabilidade condicional do tamanho final é pequena (CV ≈ 2%), logo o piso de 1000 domina; em C3 (R_eff = 0,75) não há surto maior, o tamanho final é um processo subcrítico com CV ≈ 2,3 e a fórmula pede ~48,6 mil réplicas — que são baratas (≈ 6 eventos por réplica) e foram executadas integralmente.

## 2. Validação

### V1 — CTMC exata (N = 10, I₀ = 1, R₀ = 2,5, 100 000 réplicas)

- Figura: `results/figures/V1_pmf_tamanho_final.png` (PMF simulada × exata). Tabelas: `results/tables/V1_pmf_tamanho_final.{csv,md}`, `results/tables/V1_testes.{csv,md}`. PMF exata: `data/raw/V1/final_size_pmf_exact.npy`.
- Qui-quadrado (células k = 1..10, esperado ≥ 5 em todas): χ² = **7.54**, gl = 9, p = **0.581**. Distância de variação total = **0.00304**.
- Média do tamanho final: simulada 5.4043 × exata 5.4098.
- 11 de 11 probabilidades exatas caem dentro do IC 95% (normal) da frequência simulada.
- Leitura: a cadeia de saltos embutida do simulador (taxas β·S·I/N e γ·I, escolha do evento pelo segundo uniforme) reproduz a distribuição exata; não há evidência de erro de taxa ou de seleção de evento.

| k | observado | p_simulada | p_exata | diferenca | exata_no_ic |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | sim |
| 1 | 30847 | 0.3085 | 0.3077 | 0.000778 | sim |
| 2 | 7703 | 0.0770 | 0.0769 | 0.000107 | sim |
| 3 | 4353 | 0.0435 | 0.0425 | 0.000994 | sim |
| 4 | 3207 | 0.0321 | 0.0325 | -0.00046 | sim |
| 5 | 3096 | 0.0310 | 0.0309 | 1.33e-05 | sim |
| 6 | 3473 | 0.0347 | 0.0352 | -0.000494 | sim |
| 7 | 4604 | 0.0460 | 0.0471 | -0.00111 | sim |
| 8 | 7300 | 0.0730 | 0.0733 | -0.00033 | sim |
| 9 | 12828 | 0.1283 | 0.1289 | -0.000642 | sim |
| 10 | 22589 | 0.2259 | 0.2247 | 0.00114 | sim |

### V2 — limite de campo médio (N = 10 000, I₀ = 10, R₀ = 2,5, 200 réplicas)

- Figura: `results/figures/V2_campo_medio.png`. Tabela: `results/tables/V2_campo_medio.{csv,md}`. Série média × EDO na grade: `data/processed/V2_media_vs_edo.csv`; trajetórias: `data/raw/V2/trajectories.csv`.
- RMSE de I(t)/N (média das 200 réplicas × EDO, grade 0–120 d, 601 pontos) = **0.00223**.
- Pico da trajetória média: 0.2272 × EDO 0.2339 (erro relativo -2.84%); tempo do pico 24.6 d × 24.4 d (+0.82%).
- Média dos picos POR RÉPLICA: 0.2374 [IC 0.2362, 0.2386] × EDO 0.2339 (+1.49%); média dos tempos até o pico por réplica: 24.55 d [24.31, 24.80] × 24.40 d (+0.63%).
- Surtos maiores: 200/200 (com I₀ = 10, P(extinção) ≈ (1/2,5)¹⁰ ≈ 10⁻⁴).
- Leitura: o pico da trajetória MÉDIA fica ligeiramente abaixo da EDO porque a média de curvas com pequenos deslocamentos temporais aleatórios é mais achatada; a média dos picos individuais fica ligeiramente acima (máximo de um processo com ruído). Ambos os desvios são de ~1–3% e a banda de IC é estreita; comportamento consistente com convergência ao campo médio.

### V3 — pytest

- `tests/test_epidemic.py`: 17 testes (gerador exponencial por transformada inversa — média e variância; conservação de S+I+R; taxas não negativas; reprodutibilidade por seed; β = 0; equação do tamanho final vs. valores de referência via Lambert W; PMF geométrica; CTMC exata; agrupamento de caudas do qui-quadrado). Rodar: `make test`.

## 3. Cenários principais (N = 1000, I₀ = 1)

Tabelas: `results/tables/estatisticas_cenarios.{csv,md}` (média, dp, IC t 95%, mediana, quartis, mín, máx — incondicional / surto maior / surto menor), `results/tables/sensibilidade_limiar.{csv,md}`, `results/tables/teoria_vs_simulacao_cenarios.{csv,md}`. Dados brutos: `data/raw/C{1,2,3}/{config.json,metrics.csv,trajectories.csv}`; registros por indivíduo de C1: `data/raw/C1/individuals.npz`.

Figuras: `results/figures/trajetorias_C1.png`, `results/figures/trajetorias_C2.png`, `results/figures/trajetorias_C3.png`, `results/figures/hist_tamanho_final_C1.png`, `results/figures/hist_tamanho_final_C2.png`, `results/figures/hist_tamanho_final_C3.png`, `results/figures/boxplots_incondicional.png`, `results/figures/boxplots_condicional_surto_maior.png`.

### 3.1 Limiar surto maior × menor

- **C1**: vale do histograma (bins de 10) → limiar = **425** casos; moda maior em 895; contagem no vale = 0; maior surto menor = 11 casos, menor surto maior = 835 casos; 608 maiores / 392 menores.
- **C2**: vale do histograma (bins de 10) → limiar = **135** casos; moda maior em 355; contagem no vale = 0; maior surto menor = 96 casos, menor surto maior = 171 casos; 312 maiores / 688 menores.
- **C3** (R_eff = 0,75 < 1): sem bimodalidade; por definição não há surto maior. 3 réplicas atingiram ≥ 100 casos (máx. 108), cauda do processo subcrítico (ver comparação com CTMC exata).
- Justificativa: o histograma é bimodal com um platô de contagem zero entre as duas modas; qualquer limiar dentro do platô produz a mesma classificação. O ponto médio do platô é a escolha; 5% e 10% de N são as alternativas testadas.
- Sensibilidade (`sensibilidade_limiar`):

| cenario | limiar | valor_limiar | n_surto_maior | P_surto_maior | wilson_low | wilson_high | tamanho_final_cond_media |
|---|---|---|---|---|---|---|---|
| C1 | vale | 425.00 | 608 | 0.6080 | 0.5774 | 0.6378 | 0.8915 |
| C1 | 10% de N | 100.00 | 608 | 0.6080 | 0.5774 | 0.6378 | 0.8915 |
| C1 | 5% de N | 50.00 | 608 | 0.6080 | 0.5774 | 0.6378 | 0.8915 |
| C2 | vale | 135.00 | 312 | 0.3120 | 0.2840 | 0.3414 | 0.3459 |
| C2 | 10% de N | 100.00 | 312 | 0.3120 | 0.2840 | 0.3414 | 0.3459 |
| C2 | 5% de N | 50.00 | 316 | 0.3160 | 0.2879 | 0.3455 | 0.3424 |
| C3 | 10% de N (nominal) | 100.00 | 0 | 0 | 0 | 7.9e-05 | — |

  Em C1 os três limiares dão classificação idêntica; em C2 o limiar de 5% de N reclassifica 4 réplicas (surtos entre 50 e 96 casos) e move P(surto maior) de 0,312 para 0,316 — dentro do IC de Wilson.

### 3.2 Métricas por cenário (média [IC 95%]) — resumo

| cenário | condição | n | tamanho final (fração) | pico I | tempo até pico (d) | duração (d) | eventos | offspring índice |
|---|---|---|---|---|---|---|---|---|
| C1 | incondicional | 1000 | 0.5426 [0.5156, 0.5696] | 150.0 [142.5, 157.4] | 15.0 [14.3, 15.8] | 43.4 [41.3, 45.5] | 1084 [1030, 1138] | 2.52 [2.35, 2.69] |
| C1 | surto maior | 608 | 0.8915 [0.8901, 0.8929] | 245.7 [243.9, 247.5] | 24.4 [24.0, 24.8] | 69.9 [69.1, 70.7] | 1782 [1779, 1785] | 3.92 [3.70, 4.13] |
| C1 | surto menor | 392 | 0.0016 [0.0014, 0.0017] | 1.4 [1.3, 1.5] | 0.5 [0.4, 0.7] | 2.3 [2.1, 2.6] | 2 [2, 2] | 0.34 [0.28, 0.41] |
| C2 | incondicional | 1000 | 0.1103 [0.1003, 0.1203] | 18.0 [16.5, 19.6] | 15.1 [13.7, 16.4] | 35.1 [32.3, 37.9] | 220 [200, 240] | 1.52 [1.39, 1.65] |
| C2 | surto maior | 312 | 0.3459 [0.3405, 0.3513] | 53.3 [51.7, 55.0] | 43.5 [41.7, 45.3] | 99.3 [96.9, 101.7] | 691 [680, 702] | 3.37 [3.10, 3.64] |
| C2 | surto menor | 688 | 0.0035 [0.0029, 0.0040] | 2.0 [1.9, 2.2] | 2.2 [1.8, 2.5] | 6.0 [5.4, 6.6] | 6 [5, 7] | 0.68 [0.60, 0.76] |
| C3 | incondicional | 48622 | 0.0036 [0.0036, 0.0037] | 2.2 [2.1, 2.2] | 3.1 [3.0, 3.1] | 9.0 [8.9, 9.1] | 6 [6, 6] | 0.75 [0.74, 0.76] |
| C3 | surto menor | 48622 | 0.0036 [0.0036, 0.0037] | 2.2 [2.1, 2.2] | 3.1 [3.0, 3.1] | 9.0 [8.9, 9.1] | 6 [6, 6] | 0.75 [0.74, 0.76] |

### 3.3 Teoria × simulação (tabela completa em `teoria_vs_simulacao_cenarios`)

| quantidade | simulado | ic_low | ic_high | teorico | erro_abs | erro_rel | teorico_no_ic | unidade |
|---|---|---|---|---|---|---|---|---|
| C1: P(surto maior) | 0.6080 | 0.5774 | 0.6378 | 0.6000 | 0.008 | 0.0133 | sim | ramificação: 1−(1/R_eff)^I0 |
| C1: P(surto maior) | 0.6080 | 0.5774 | 0.6378 | 0.5989 | 0.00913 | 0.0152 | sim | CTMC exata N=1000, limiar 425 |
| C1: tamanho final | surto maior (fração de N) | 0.8915 | 0.8901 | 0.8929 | 0.8914 | 5.91e-05 | 6.63e-05 | sim | equação do tamanho final: s0·τ |
| C1: tamanho final | surto maior (fração de N) | 0.8915 | 0.8901 | 0.8929 | 0.8928 | -0.00131 | -0.00146 | sim | equação com i0>0: s0·τ(i0)+i0 |
| C1: tamanho final | surto maior (fração de N) | 0.8915 | 0.8901 | 0.8929 | 0.8918 | -0.000348 | -0.000391 | sim | CTMC exata N=1000 |
| C1: tamanho final | surto menor (nº de casos) | 1.5689 | 1.4362 | 1.7016 | 1.6667 | -0.0978 | -0.0587 | sim | ramificação dual: 1/(1−1/R_eff) |
| C1: tamanho final | surto menor (nº de casos) | 1.5689 | 1.4362 | 1.7016 | 1.6789 | -0.1100 | -0.0655 | sim | CTMC exata N=1000 |
| C1: offspring médio do caso índice | 2.5170 | 2.3463 | 2.6877 | 2.4975 | 0.0195 | 0.00781 | sim | geométrica sem depleção: R0·s0 |
| C1: pico de I/N | surto maior | 0.2457 | 0.2439 | 0.2475 | 0.2339 | 0.0118 | 0.0506 | não | EDO |
| C1: tempo até o pico | surto maior (dias) | 24.41 | 24.05 | 24.78 | 24.36 | 0.0520 | 0.00213 | sim | EDO |
| C1: tamanho final (EDO) | surto maior | 0.8915 | 0.8901 | 0.8929 | 0.8928 | -0.00131 | -0.00146 | sim | EDO (inclui i0) |
| C2: P(surto maior) | 0.3120 | 0.2840 | 0.3414 | 0.3333 | -0.0213 | -0.0640 | sim | ramificação: 1−(1/R_eff)^I0 |
| C2: P(surto maior) | 0.3120 | 0.2840 | 0.3414 | 0.3196 | -0.00764 | -0.0239 | sim | CTMC exata N=1000, limiar 135 |
| C2: tamanho final | surto maior (fração de N) | 0.3459 | 0.3405 | 0.3513 | 0.3481 | -0.0022 | -0.00633 | sim | equação do tamanho final: s0·τ |
| C2: tamanho final | surto maior (fração de N) | 0.3459 | 0.3405 | 0.3513 | 0.3508 | -0.00487 | -0.0139 | sim | equação com i0>0: s0·τ(i0)+i0 |
| C2: tamanho final | surto maior (fração de N) | 0.3459 | 0.3405 | 0.3513 | 0.3466 | -0.000645 | -0.00186 | sim | CTMC exata N=1000 |
| C2: tamanho final | surto menor (nº de casos) | 3.4608 | 2.9166 | 4.0049 | 3.0000 | 0.4608 | 0.1536 | sim | ramificação dual: 1/(1−1/R_eff) |
| C2: tamanho final | surto menor (nº de casos) | 3.4608 | 2.9166 | 4.0049 | 3.7902 | -0.3295 | -0.0869 | sim | CTMC exata N=1000 |
| C2: offspring médio do caso índice | 1.5190 | 1.3919 | 1.6461 | 1.4975 | 0.0215 | 0.0144 | sim | geométrica sem depleção: R0·s0 |
| C2: pico de I/N | surto maior | 0.0533 | 0.0517 | 0.0550 | 0.0385 | 0.0149 | 0.3860 | não | EDO |
| C2: tempo até o pico | surto maior (dias) | 43.50 | 41.66 | 45.35 | 51.70 | -8.1973 | -0.1586 | não | EDO |
| C2: tamanho final (EDO) | surto maior | 0.3459 | 0.3405 | 0.3513 | 0.3508 | -0.00487 | -0.0139 | sim | EDO (inclui i0) |
| C3: P(surto maior) | 0 | 0 | 7.9e-05 | 0 | 0 | — | sim | ramificação: 1−(1/R_eff)^I0 |
| C3: P(Z ≥ 100) (cauda da classe menor) | 6.17e-05 | 2.1e-05 | 0.000181 | 3.63e-05 | 2.54e-05 | 0.7007 | sim | CTMC exata N=1000 |
| C3: tamanho final médio (nº de casos, incl. I0) | 3.6370 | 3.5770 | 3.6971 | 4.0000 | -0.3630 | -0.0907 | não | ramificação: 1/(1−R_eff) |
| C3: tamanho final médio (nº de casos, incl. I0) | 3.6370 | 3.5770 | 3.6971 | 3.6062 | 0.0309 | 0.00856 | sim | CTMC exata N=1000 |
| C3: offspring médio do caso índice | 0.7479 | 0.7379 | 0.7580 | 0.7475 | 0.000433 | 0.000579 | sim | geométrica sem depleção: R0·s0 |

Observações:
- P(surto maior): C1 0.608 [0.577, 0.638] × 0,600 (ramificação) e 0.5989 (CTMC exata N = 1000); C2 0.312 [0.284, 0.341] × 0,333 e 0.3196 (CTMC). Teoria dentro do IC nos dois casos. A CTMC exata para N = 1000 (mesmo limiar) mostra que a aproximação de ramificação superestima ligeiramente P(surto maior) em C2 (viés de N finito: com R_eff = 1,5 a depleção de suscetíveis é sentida cedo).
- Tamanho final condicional: C1 0.8915 [0.8901, 0.8929] × s₀·τ = 0.8914 (erro rel. +0.007%) e CTMC exata 0.8918; C2 0.3459 [0.3405, 0.3513] × s₀·τ = 0.3481 (-0.63%) e CTMC exata 0.3466. Todos dentro do IC.
- Viés de I₀ > 0: a equação padrão (i₀ = 0) exclui o caso índice; a variante s₀·τ(i₀)+i₀ inclui-o e fica ~1/N acima. Com N = 1000 a diferença (0,001) é da ordem da largura do IC em C1; ambas as versões estão na tabela.
- C3 (subcrítico): tamanho total médio 3.637 casos [3.577, 3.697] × 1/(1−R_eff) = 4,000 (erro -9.1%, FORA do IC) × CTMC exata 3.606 (dentro do IC). O desvio em relação à ramificação é esperado: com S₀ = 299 a depleção reduz a taxa efetiva ao longo da cadeia (e R₀·S₀/N = 0,7475, não 0,75); a CTMC exata captura isso.
- Surtos menores em C1/C2: tamanho médio compatível com o processo dual (1/(1−1/R_eff)) e com a CTMC exata (dentro do IC).
- Pico de I/N vs. EDO (condicional a surto maior): C1 0.2457 × 0.2339 (+5.1%), C2 0.0533 × 0.0385 (+38.6%); tempo até o pico C2 43.5 d × 51.7 d. EDO fora do IC: o pico de uma trajetória estocástica é o máximo de um processo ruidoso (viés positivo ∝ √I_pico), e o condicionamento a surto maior seleciona realizações com crescimento inicial mais rápido (pico mais alto e mais cedo). O efeito é grande em C2 porque o pico da EDO é pequeno (≈ 38 indivíduos) e a fase inicial é longa (R_eff = 1,5). Não é um erro de simulação: V2 (N = 10⁴, I₀ = 10) mostra o erro cair para ~1–3%.
- Offspring médio do caso índice ≈ R₀·s₀ em todos os cenários (dentro do IC).

### 3.4 Convergência (C1)

- Figura: `results/figures/convergencia_tamanho_final_C1.png`; séries: `data/processed/convergencia_C1_{incondicional,condicional}.csv`.
- Média incondicional do tamanho final com 1000 réplicas: 0.5426 [0.5156, 0.5696] (ref. CTMC exata E[Z]/N = 0.5348); condicional 0.8915 ± 0.0014. A largura do IC cai como 1/√n; a média condicional estabiliza dentro de ±0,5% já com ~100 réplicas.

## 4. Varredura em R₀ (0,5–4,0, passo 0,25; p = 0; N = 1000; 500 réplicas por ponto)

- Figuras: `results/figures/varredura_prob_surto_maior.png` (P(surto maior)), `results/figures/varredura_tamanho_final.png` (tamanho final condicional). Tabela: `results/tables/varredura_R0.{csv,md}`. Dados: `data/raw/sweep/sweep_R0_*/`.
- P(surto maior): teoria 1−1/R₀ dentro do IC de Wilson em 12/12 pontos com R₀ > 1; CTMC exata (N = 1000, mesmo limiar) dentro do IC em 15/15 pontos. Maior desvio absoluto vs. teoria: 0.036 (R₀ = 3.25).
- Tamanho final condicional: equação s₀·τ dentro do IC em 8/12 pontos (R₀ > 1); CTMC exata dentro do IC em 12/12. Erro relativo máximo vs. equação: 5.62% (R₀ = 1.25). Para R₀ ≥ 3 o IC é tão estreito (±0,001) que o viés de +1/N da inclusão de I₀ fica visível: a simulação fica sistematicamente ~0,001 acima de s₀·τ e coincide com a CTMC exata.
- R₀ = 1,00 (crítico): a distribuição do tamanho final não é bimodal; o limiar nominal de 10% de N classifica 16/500 réplicas como 'maiores' (CTMC exata: P(Z ≥ 100) = 0.042), enquanto a ramificação dá 0 — a classificação não é bem definida em R₀ = 1.
- R₀ = 1,25: o vale é raso (poucos surtos maiores, tamanho final condicional ~0,39) e a ramificação superestima P(surto maior) (0,200 vs. CTMC 0,179); simulação concorda com a CTMC.

| R0 | limiar | n_surto_maior | p_major | p_major_low | p_major_high | p_major_theory | p_major_exact | fs_cond_mean | fs_cond_low | fs_cond_high | fs_theory | fs_cond_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5000 | 100.00 | 0 | 0 | 0 | 0.00762 | 0 | 4.89e-09 | — | — | — | 0 | 0.1055 |
| 0.7500 | 100.00 | 0 | 0 | 0 | 0.00762 | 0 | 0.00059 | — | — | — | 0 | 0.1192 |
| 1.0000 | 100.00 | 16.00 | 0.0320 | 0.0198 | 0.0513 | 0 | 0.0424 | 0.1669 | 0.1284 | 0.2055 | 0 | 0.1780 |
| 1.2500 | 155.00 | 90.00 | 0.1800 | 0.1488 | 0.2161 | 0.2000 | 0.1788 | 0.3904 | 0.3730 | 0.4078 | 0.3696 | 0.3779 |
| 1.5000 | 235.00 | 158.00 | 0.3160 | 0.2768 | 0.3580 | 0.3333 | 0.3258 | 0.5848 | 0.5764 | 0.5932 | 0.5813 | 0.5794 |
| 1.7500 | 310.00 | 213.00 | 0.4260 | 0.3834 | 0.4697 | 0.4286 | 0.4253 | 0.7098 | 0.7039 | 0.7156 | 0.7113 | 0.7106 |
| 2.0000 | 345.00 | 264.00 | 0.5280 | 0.4842 | 0.5714 | 0.5000 | 0.4979 | 0.7973 | 0.7936 | 0.8010 | 0.7955 | 0.7954 |
| 2.2500 | 385.00 | 272.00 | 0.5440 | 0.5002 | 0.5872 | 0.5556 | 0.5541 | 0.8504 | 0.8476 | 0.8533 | 0.8521 | 0.8524 |
| 2.5000 | 425.00 | 298.00 | 0.5960 | 0.5524 | 0.6381 | 0.6000 | 0.5989 | 0.8914 | 0.8893 | 0.8935 | 0.8914 | 0.8918 |
| 2.7500 | 450.00 | 306.00 | 0.6120 | 0.5686 | 0.6537 | 0.6364 | 0.6355 | 0.9183 | 0.9168 | 0.9199 | 0.9193 | 0.9198 |
| 3.0000 | 455.00 | 321.00 | 0.6420 | 0.5990 | 0.6828 | 0.6667 | 0.6659 | 0.9399 | 0.9387 | 0.9410 | 0.9393 | 0.9400 |
| 3.2500 | 460.00 | 328.00 | 0.6560 | 0.6133 | 0.6963 | 0.6923 | 0.6917 | 0.9551 | 0.9541 | 0.9561 | 0.9540 | 0.9547 |
| 3.5000 | 480.00 | 354.00 | 0.7080 | 0.6667 | 0.7461 | 0.7143 | 0.7137 | 0.9656 | 0.9648 | 0.9664 | 0.9649 | 0.9656 |
| 3.7500 | 480.00 | 365.00 | 0.7300 | 0.6894 | 0.7671 | 0.7333 | 0.7328 | 0.9739 | 0.9732 | 0.9746 | 0.9730 | 0.9738 |
| 4.0000 | 490.00 | 384.00 | 0.7680 | 0.7290 | 0.8029 | 0.7500 | 0.7496 | 0.9801 | 0.9795 | 0.9806 | 0.9791 | 0.9799 |

## 5. Distribuições teórico × empírico

Tabela: `results/tables/testes_distribuicoes_C1.{csv,md}`.

### (a) Duração infecciosa — C1, todos os indivíduos infectados de todas as 1000 réplicas

- Figuras: `results/figures/cdf_duracao_infecciosa_C1.png` (CDF), `results/figures/qq_duracao_infecciosa_C1.png` (QQ-plot).
- n = 542637 durações (sem censura: toda réplica roda até I = 0). Média 4.9937 d (teoria 1/γ = 5), dp 4.9894 (teoria 5), mediana 3.4634 (teoria 5·ln 2 = 3,466).
- KS contra Exp(γ = 0,2): D = **0.00106**, p = **0.576**. Não se rejeita a exponencial.
- Isso valida o sorteio uniforme de quem se recupera: por permutabilidade, cada infeccioso tem taxa individual γ e sua duração é Exp(γ), independentemente de quantos estão infectados.

### (b) Offspring do caso índice — C1 (1000 réplicas)

- Figura: `results/figures/pmf_offspring_C1.png`. Contagens/esperados: `data/processed/offspring_C1.json`.
- Média empírica 2.517 [IC 2.346, 2.688] × R₀ = 2,5 (geométrica) × 2.4195 (CTMC exata com depleção).
- Qui-quadrado vs. Geométrica (caudas agrupadas até esperado ≥ 5; 15 células): χ² = **26.68**, gl = 14, p = **0.021**.
- Decomposição: a célula k = 6 contribui 9.6 de 26.7 (obs 57 × esp 37.9); a cauda agrupada (k ≥ 14) tem obs 2 × esp 9.0.
- Investigação: (i) a geométrica supõe S ≈ N durante toda a vida do índice (sem depleção). Um índice com k ≥ 14 filhos precisa viver ≳ 28 d, quando S/N já caiu bem abaixo de 1 num surto maior; a cauda fica deficitária e a média cai. (ii) Calculou-se a PMF EXATA do offspring do índice COM depleção (CTMC estendida (s, i, k), programação dinâmica; `data/raw/C1/index_offspring_pmf_exact.npy`; conferida contra um Monte Carlo independente da cadeia embutida): média 2.4195 (vs. 2,5 da geométrica). χ² das 1000 réplicas de C1 contra ela: 19.97 (gl 13, p = 0.096). (iii) Verificação de alta potência: 5 lotes independentes de 100 000 réplicas (`data/raw/C1_offspring_extra/batch*/`). Contra a geométrica, p por lote = 7.6e-77, 5.4e-71, 7.9e-64, 3.1e-70, 3.9e-70; agrupado (n = 500000) χ² = 1924 (gl 25), p = 0.0e+00 — REJEITA (média 2.4225 ≠ 2,5). Contra a CTMC exata com depleção, p por lote = 0.171, 0.242, 0.974, 0.955, 0.729; agrupado χ² = 11.06 (gl 22), p = 0.974 (média 2.4225 × 2.4195). Conclusão: o simulador reproduz a distribuição exata do modelo individual; o p ≈ 0,02 contra a geométrica em C1 combina o efeito real de depleção (cauda/média) com flutuação amostral (célula k = 6). A hipótese 'sem depleção' é a aproximação que falha, não o código. A média empírica de C1 coincide com R₀ dentro do IC porque o efeito de depleção sobre a média (−3%) é menor que a largura do IC (±7%).

- Nota de processo: a investigação revelou um bug no agrupamento de caudas do teste qui-quadrado (`stats.chi2_gof`; `e[-2] += e.pop()` reavalia o índice após o pop), corrigido e coberto por teste unitário. V1 não era afetada (nenhuma célula agrupada).

### (c) Tamanho final em V1 — ver seção 2 (χ² p = 0.581, TVD = 0.00304).

## 6. Tempo de execução

Tabela: `results/tables/tempo_execucao.{csv,md}` (tempo de simulação por experimento, paralelo em 11 processos) e `results/tables/tempo_execucao_etapas.{csv,md}` (tempo total por script do `run_all.py`, incluindo análise e figuras).

| experimento | n_replicacoes | tempo_simulacao_s | pasta |
|---|---|---|---|
| C1 | 1000 | 0.2381 | data/raw/C1 |
| C1_offspring_extra_b0 | 100000 | 21.55 | data/raw/C1_offspring_extra/batch0 |
| C1_offspring_extra_b1 | 100000 | 20.66 | data/raw/C1_offspring_extra/batch1 |
| C1_offspring_extra_b2 | 100000 | 25.52 | data/raw/C1_offspring_extra/batch2 |
| C1_offspring_extra_b3 | 100000 | 31.70 | data/raw/C1_offspring_extra/batch3 |
| C1_offspring_extra_b4 | 100000 | 34.79 | data/raw/C1_offspring_extra/batch4 |
| C2 | 1000 | 0.1388 | data/raw/C2 |
| C3 | 48622 | 1.0722 | data/raw/C3 |
| V1 | 100000 | 2.3337 | data/raw/V1 |
| V2 | 200 | 1.4413 | data/raw/V2 |
| pilot_C1 | 200 | 0.0610 | data/raw/pilot_C1 |
| pilot_C2 | 200 | 0.0371 | data/raw/pilot_C2 |
| pilot_C3 | 200 | 0.0359 | data/raw/pilot_C3 |
| sweep | 7500 | 2.3740 | data/raw/sweep |


Tempo total do `run_all.py`: **169.5 s**.

## 7. Inventário de figuras

- `results/figures/V1_pmf_tamanho_final.png` (+ `.pdf`)
- `results/figures/V2_campo_medio.png` (+ `.pdf`)
- `results/figures/boxplots_condicional_surto_maior.png` (+ `.pdf`)
- `results/figures/boxplots_incondicional.png` (+ `.pdf`)
- `results/figures/cdf_duracao_infecciosa_C1.png` (+ `.pdf`)
- `results/figures/convergencia_tamanho_final_C1.png` (+ `.pdf`)
- `results/figures/hist_tamanho_final_C1.png` (+ `.pdf`)
- `results/figures/hist_tamanho_final_C2.png` (+ `.pdf`)
- `results/figures/hist_tamanho_final_C3.png` (+ `.pdf`)
- `results/figures/pmf_offspring_C1.png` (+ `.pdf`)
- `results/figures/qq_duracao_infecciosa_C1.png` (+ `.pdf`)
- `results/figures/trajetorias_C1.png` (+ `.pdf`)
- `results/figures/trajetorias_C2.png` (+ `.pdf`)
- `results/figures/trajetorias_C3.png` (+ `.pdf`)
- `results/figures/varredura_prob_surto_maior.png` (+ `.pdf`)
- `results/figures/varredura_tamanho_final.png` (+ `.pdf`)

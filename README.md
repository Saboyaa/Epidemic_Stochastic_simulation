# Simulador estocástico de epidemia SIR (CTMC / Gillespie)

Trabalho de Simulação e Análise de Desempenho (IME). Núcleo de simulação em código próprio
(sem bibliotecas de epidemiologia); `scipy` é usado apenas para EDOs (`solve_ivp`), raízes
(`brentq`) e testes estatísticos (`kstest`, `chi2`, `t`).

## Como rodar

```bash
# Python 3.11+ (testado em 3.12.14). Com uv:
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# ou: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

make test        # pytest (V3)
make all         # = python run_all.py: testes + todos os experimentos + figuras + tabelas + RESULTS.md
make clean       # apaga data/, results/ e RESULTS.md
```

`python run_all.py` regenera tudo do zero (~3 min em 12 núcleos; as replicações rodam em paralelo com
`ProcessPoolExecutor`, o que não afeta os resultados porque cada replicação tem sua própria seed).
Os scripts em `experiments/` também podem ser rodados individualmente (a partir de `experiments/`),
na ordem `exp_pilot.py → exp_scenarios.py → exp_sweep.py → exp_v1_exact.py → exp_v2_meanfield.py → write_results.py`.

## Reprodutibilidade

* Seed mestre: `MASTER_SEED = 20260916` em `src/epidemic/config.py`.
* Cada experimento tem uma chave (`EXP_KEY`), e cada replicação `r` recebe
  `numpy.random.SeedSequence(MASTER_SEED, spawn_key=(chave,)).spawn(n)[r]` → `PCG64`.
  Replicações são independentes e o resultado não depende do número de processos.
* Por experimento, em `data/raw/<exp>/`: `config.json` (parâmetros, seed mestre, chave de spawn, regra de
  derivação das seeds, nº de replicações, workers, tempo, versões de Python/numpy/scipy/pandas/matplotlib),
  `metrics.csv` (uma linha por replicação, com a chave de seed), `trajectories.csv` (S, I, R de uma
  amostra de replicações reamostrados numa grade de tempo), `runtime.json`. Em C1 também
  `individuals.npz` (registros por indivíduo de todas as réplicas) e as PMFs exatas (`*.npy`).

## Estrutura

```
src/epidemic/
  config.py      seed mestre, chaves por experimento, γ, cenários, caminhos
  rng.py         SeedSequence.spawn por replicação; fluxo de uniformes; transformada inversa da exponencial
  gillespie.py   método direto de Gillespie em nível individual (núcleo) + reamostragem em grade
  ode.py         SIR determinístico (Kermack–McKendrick) via solve_ivp
  theory.py      equação do tamanho final (brentq), P(surto maior), tamanho esperado subcrítico,
                 geométrica do offspring, CDF exponencial, limiar vacinal
  exact_ctmc.py  programação dinâmica na cadeia embutida: PMF exata do tamanho final e do offspring do índice
  stats.py       descritivas, IC t, IC de Wilson, n por erro relativo, limiar pelo vale, KS, qui-quadrado, TVD
  plots.py       figuras (português, PNG 300 dpi + PDF)
  runner.py      execução paralela das replicações e persistência (config.json, metrics.csv, trajetórias)
  tables.py      tabelas CSV + Markdown
experiments/     exp_pilot.py, exp_scenarios.py, exp_sweep.py, exp_v1_exact.py, exp_v2_meanfield.py, write_results.py
tests/           test_epidemic.py (V3)
data/raw/        entradas e saídas brutas por experimento      data/processed/  agregados (JSON/CSV)
results/figures/ PNG + PDF                                     results/tables/  CSV + MD
run_all.py  Makefile  requirements.txt  README.md  RESULTS.md
```

## Modelo

CTMC com estado (S, I, R), N = S + I + R constante:

* infecção (S, I) → (S−1, I+1) com taxa β·S·I/N;
* recuperação (I, R) → (I−1, R+1) com taxa γ·I;
* γ = 1/5 por dia, β = R₀·γ; vacinação prévia de uma fração p: `round(p·N)` indivíduos começam em R,
  S₀ = N − I₀ − round(p·N), R_eff = R₀·(1 − p).

## Lógica do Gillespie (método direto), `gillespie.simulate`

Enquanto I > 0:

1. a_inf = β·S·I/N, a_rec = γ·I, a₀ = a_inf + a_rec.
2. **Tempo até o próximo evento por transformada inversa**: Δt = −ln(U₁)/a₀, U₁ ~ U(0,1)
   (`rng.exp_inverse`). Usa-se U = 1 − numpy.random() ∈ (0, 1] para que ln(U) seja finito.
3. **Escolha do evento** com um segundo uniforme: infecção se U₂·a₀ < a_inf, senão recuperação.
4. **Nível individual**: mantém-se a lista de ids infecciosos e a lista de ids suscetíveis.
   Na infecção, o infectante é sorteado uniformemente entre os infecciosos (3º uniforme) e o infectado
   uniformemente entre os suscetíveis (4º uniforme); registra-se quem infectou quem. Na recuperação,
   quem se recupera é sorteado uniformemente entre os infecciosos. Remoções em O(1) por troca com o último.
5. Registro por indivíduo: instante de infecção, de recuperação, duração infecciosa e nº de infecções
   secundárias. A simulação corre **sempre até I = 0** (sem horizonte), logo não há censura.

Métricas por replicação: tamanho final (fração de N infectada, incluindo I₀), pico de I, tempo até o
pico, duração (tempo até I = 0), nº de eventos, offspring do caso índice (indivíduo 0).

Os uniformes vêm de um único fluxo PCG64 por replicação, consumido em blocos (`rng.UniformStream`)
apenas por desempenho; o algoritmo é o método direto padrão. Não foi necessário `numba`
(~1 µs por evento; C1 com 1000 réplicas leva < 1 s em paralelo).

### Por que o sorteio individual uniforme é equivalente ao modelo agregado

No modelo agregado, a taxa total de infecção é β·S·I/N e a de recuperação γ·I. No modelo em nível
individual, cada um dos I infecciosos tem taxa própria de recuperação γ e taxa própria de gerar uma
infecção β·S/N (contatos com cada um dos S suscetíveis à taxa β/N). Os indivíduos são
**permutáveis**: todos os infecciosos têm as mesmas taxas e todos os suscetíveis têm a mesma
suscetibilidade. Pela superposição de processos de Poisson, a taxa total de recuperação é I·γ e a de
infecção é I·(β·S/N) = β·S·I/N — exatamente as taxas agregadas — e, condicionado ao tipo de evento,
o indivíduo responsável (ou atingido) é uniforme entre os elegíveis, pois cada um contribui com uma
fração igual (1/I ou 1/S) da taxa total. Portanto sortear o evento pelo agregado e depois sortear o
indivíduo uniformemente gera a mesma lei de probabilidade que simular os I·(S+1) relógios individuais.
Consequências verificadas empiricamente: a duração infecciosa de cada indivíduo é Exp(γ) (KS, seção 5a
de RESULTS.md) e o offspring do caso índice segue a PMF exata da CTMC estendida (seção 5b).

## Modelos analíticos

* **EDO** (`ode.py`): ṡ = −β·s·i, i̇ = β·s·i − γ·i, ṙ = γ·i, com as mesmas condições iniciais
  (s₀ = S₀/N, i₀ = I₀/N, r₀ = vacinados/N), `solve_ivp` (DOP853, rtol 1e-9).
* **Equação do tamanho final** (`theory.final_size_tau`): 1 − τ = exp(−R₀·s₀·τ), raiz não trivial
  por `brentq` num intervalo que exclui τ = 0; tamanho final na população = s₀·τ. Há a variante com
  i₀ > 0 (1 − τ = exp(−R₀(s₀τ + i₀))), usada para quantificar o viés de I₀ > 0.
* **P(surto maior)**: 1 − (1/R_eff)^I₀ se R_eff > 1, senão 0. **Subcrítico**: E[tamanho] = 1/(1 − R_eff).
* **Offspring do índice sem depleção**: Geom, P(k) = (γ/(β+γ))·(β/(β+γ))^k, média R₀.
* **Duração infecciosa**: Exp(γ).
* **CTMC exata** (`exact_ctmc.final_size_pmf`): na cadeia de saltos embutida, de (s, i) vai-se a
  (s−1, i+1) com probabilidade β·s/(β·s + γ·N) e a (s, i−1) com a complementar; absorção em i = 0.
  A DP percorre os níveis s + i em ordem decrescente (a infecção conserva s + i e a recuperação o reduz),
  o que dá uma ordem topológica; custo O(S₀·N). Além do N = 10 da V1, a mesma DP é avaliada para
  N = 1000 nos cenários e na varredura, servindo de referência exata que separa "viés de N finito" de
  "erro de simulação". `exact_ctmc.index_offspring_pmf` estende o estado com k (filhos do índice) e
  usa as probabilidades 1/i do sorteio uniforme para obter a PMF exata do offspring do índice com depleção.

## Validação

* **V1** (N = 10, 10⁵ réplicas): PMF simulada do tamanho final × exata — qui-quadrado e distância de variação total.
* **V2** (N = 10⁴, I₀ = 10, 200 réplicas): I(t)/N médio com IC 95% × EDO — RMSE, erro no pico e no tempo do pico.
* **V3** (`pytest`): média/variância do gerador exponencial por transformada inversa; conservação de
  S + I + R; taxas não negativas; reprodutibilidade por seed; β = 0; equação do tamanho final contra
  valores de referência (Lambert W); geométrica; CTMC exata; agrupamento de caudas do qui-quadrado.
* **Cenários e varredura**: P(surto maior) e tamanho final condicional × ramificação/equação do tamanho
  final e × CTMC exata (N = 1000); pico e tempo do pico × EDO; surtos menores × processo dual.
* **Distribuições**: (a) duração infecciosa × Exp(γ) (KS, CDF, QQ); (b) offspring do índice × Geom
  (qui-quadrado com caudas agrupadas) e × PMF exata com depleção; (c) tamanho final V1 × exata.

## Estatística

* Surto maior × menor: limiar pelo vale do histograma do tamanho final (bins de 1% de N; ponto médio do
  platô de contagem mínima entre a origem e a moda maior); sensibilidade com 5% e 10% de N.
  Com R_eff < 1 não há surto maior por definição.
* Por métrica e cenário: média, dp, IC 95% (t), mediana, quartis, mín, máx — incondicional e condicional.
* Proporções: IC de Wilson. Nº de réplicas: piloto de 200 e n = (z·s/(e·x̄))² com e = 2%, mínimo 1000.
* Comparação com teoria: erro absoluto/relativo e se o valor teórico cai no IC.

Todos os números-chave, caminhos de figuras/tabelas e observações estão em `RESULTS.md`.

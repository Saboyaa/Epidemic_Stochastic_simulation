# Notas explicativas completas — simulador estocástico SIR (CTMC / Gillespie)

Este texto explica, do começo ao fim, o que foi construído, por que cada escolha foi feita, como cada
resultado foi obtido e o que os números mostram. Tudo o que está aqui foi gerado pelo código do
repositório (`python run_all.py`), é reprodutível a partir da seed mestre 20260916, e não usa nenhum
dado real: todas as comparações são contra modelos analíticos (EDO de Kermack–McKendrick, equação do
tamanho final, processo de ramificação, e a cadeia de Markov exata resolvida por programação dinâmica).

---

## 1. O problema e o modelo

O objeto de estudo é a epidemia SIR estocástica numa população fechada de N indivíduos, formulada como
uma cadeia de Markov de tempo contínuo (CTMC). O estado é (S, I, R): suscetíveis, infecciosos e
removidos (recuperados ou vacinados), com S + I + R = N constante. Há apenas dois tipos de evento:

- **Infecção**: (S, I) → (S−1, I+1), com taxa β·S·I/N. É a lei de ação de massas com contato
  dependente da frequência: cada infeccioso encontra cada suscetível à taxa β/N.
- **Recuperação**: (I, R) → (I−1, R+1), com taxa γ·I. Cada infeccioso se recupera à taxa γ,
  independentemente dos demais.

Parâmetros: γ = 1/5 por dia (período infeccioso médio de 5 dias, distribuído como Exp(γ)), R₀ = β/γ,
logo β = R₀·γ. Com R₀ = 2,5, β = 0,5/dia. Vacinação prévia de uma fração p da população coloca
round(p·N) indivíduos diretamente em R no instante zero; eles nunca participam da dinâmica. O número
de reprodução efetivo é R_eff = R₀·(1 − p), e o limiar de imunidade de rebanho é p* = 1 − 1/R₀ = 0,6
para R₀ = 2,5. A condição inicial é I₀ infecciosos, S₀ = N − I₀ − round(p·N) suscetíveis e
round(p·N) removidos.

O processo termina quando I = 0 (estado absorvente). Não há horizonte de tempo: a simulação sempre roda
até a absorção, o que garante que todas as durações infecciosas registradas por indivíduo são
completas (sem censura), e que o tamanho final e a duração da epidemia são observados exatamente.

---

## 2. O simulador: algoritmo de Gillespie (método direto) em nível individual

O núcleo é código próprio (`src/epidemic/gillespie.py`), sem biblioteca de epidemiologia. É o método
direto de Gillespie (também chamado SSA), que gera trajetórias estatisticamente exatas da CTMC. Enquanto
I > 0, a cada passo:

1. **Propensões.** a_inf = β·S·I/N, a_rec = γ·I, a₀ = a_inf + a_rec. As duas são não negativas por
   construção (S, I ≥ 0), e a_rec > 0 enquanto I > 0, então a₀ > 0 e o passo é bem definido.
2. **Tempo até o próximo evento por transformada inversa.** O tempo de espera numa CTMC é Exp(a₀).
   A CDF é F(t) = 1 − e^(−a₀t), cuja inversa é F⁻¹(v) = −ln(1−v)/a₀. Como 1 − V também é U(0,1),
   usa-se Δt = −ln(U)/a₀ com U ~ U(0,1). Isso está implementado explicitamente em
   `rng.exp_inverse` (não se usa `numpy.random.exponential`). Um detalhe: `numpy` devolve uniformes
   em [0, 1), então usa-se U = 1 − u ∈ (0, 1] para que ln(U) seja sempre finito.
3. **Escolha do evento** com um segundo uniforme independente: infecção se U₂·a₀ < a_inf, senão
   recuperação. Isso escolhe o evento com probabilidade proporcional à sua taxa.
4. **Nível individual.** O simulador mantém a lista de ids dos infecciosos e a lista de ids dos
   suscetíveis. Numa infecção, sorteia-se o infectante uniformemente entre os I infecciosos (terceiro
   uniforme) e o infectado uniformemente entre os S suscetíveis (quarto uniforme); registra-se quem
   infectou quem (árvore de transmissão). Numa recuperação, sorteia-se uniformemente entre os
   infecciosos quem se recupera. As remoções das listas são O(1) (troca com o último elemento e pop).
   O índice sorteado é min(int(U·n), n−1) para blindar o caso U = 1.
5. **Registros por indivíduo**: instante de infecção (0 para os casos iniciais), instante de
   recuperação, duração infecciosa (diferença) e número de infecções secundárias causadas.
6. **Trajetória**: (t, S, I, R) após cada evento é armazenado (o número máximo de eventos é
   2·S₀ + I₀, então os arrays são pré-alocados).

Métricas por replicação: tamanho final (número/fração de N de indivíduos que foram infectados em algum
momento, incluindo os I₀ iniciais), pico de infectados (máximo de I), tempo até o pico (primeiro
instante em que o máximo é atingido), duração da epidemia (instante em que I chega a 0), número de
eventos, e offspring do caso índice (número de infecções secundárias do indivíduo de id 0).

Os uniformes vêm de um único gerador PCG64 por replicação, consumidos em blocos de 8192 por um buffer
(`rng.UniformStream`), apenas para reduzir o custo de chamadas Python. O desempenho ficou em cerca de
1 µs por evento em Python puro (3.12), o que dispensou `numba`: os cenários principais com 1000
réplicas levam menos de 1 s em paralelo, e o experimento mais pesado (5 lotes de 100 000 réplicas com
N = 1000, ver seção 8) leva cerca de 2 minutos.

### 2.1 Por que sortear o indivíduo uniformemente é equivalente ao modelo agregado

Este ponto é o que justifica simular em nível individual usando as taxas agregadas. No modelo
individual, cada um dos I infecciosos carrega um relógio exponencial de recuperação de taxa γ e um
relógio de infecção de taxa β·S/N (soma dos contatos com cada um dos S suscetíveis à taxa β/N). Como
todos os infecciosos são idênticos (mesmas taxas) e todos os suscetíveis são idênticos (mesma
suscetibilidade), os indivíduos são **permutáveis**. Pela propriedade de superposição de processos de
Poisson, a taxa total de recuperação é I·γ e a de infecção é I·(β·S/N) = β·S·I/N — exatamente as
taxas do modelo agregado. E, condicionado a ter ocorrido um evento de um dado tipo, o indivíduo
responsável é aquele cujo relógio disparou primeiro entre relógios iid, o que é uniforme entre os
elegíveis, porque cada um contribui com fração igual (1/I para infecciosos, 1/S para suscetíveis) da
taxa total. Portanto "sortear o evento pelo agregado e depois sortear o indivíduo uniformemente"
gera exatamente a mesma lei de probabilidade que simular todos os I·(S+1) relógios individuais, com
custo O(1) por evento em vez de O(N).

Duas consequências testáveis dessa equivalência, ambas verificadas empiricamente (seção 8): a duração
infecciosa de cada indivíduo é Exp(γ) independentemente de quantos estão infectados ao mesmo tempo;
e o número de infecções causadas por um indivíduo específico (o caso índice) segue exatamente a
distribuição que se obtém da cadeia de Markov estendida com probabilidades 1/I.

---

## 3. Reprodutibilidade e armazenamento

- Seed mestre: `MASTER_SEED = 20260916` em `src/epidemic/config.py`.
- Cada experimento tem uma chave inteira (`EXP_KEY`: C1 = 1, C2 = 2, C3 = 3, C1_offspring_extra = 4,
  varredura = 10, V1 = 20, V2 = 30, pilotos = 101–103). A replicação r do experimento com chave k usa
  `numpy.random.SeedSequence(20260916, spawn_key=(k,)).spawn(n)[r]`, que alimenta um `PCG64`. Isso dá
  fluxos estatisticamente independentes entre replicações e entre experimentos, e o resultado não
  depende do número de processos nem da ordem de execução (as replicações rodam num
  `ProcessPoolExecutor` com 11 workers).
- Por experimento, em `data/raw/<exp>/`: `config.json` (parâmetros completos, seed mestre, chave de
  spawn, regra de derivação das seeds, nº de replicações, nº de workers, tempo de execução e versões de
  Python 3.12.14, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2), `metrics.csv` (uma linha
  por replicação com todas as métricas e a chave de seed), `trajectories.csv` (S, I, R de uma amostra
  de replicações reamostrados numa grade de tempo de 400 pontos, como função em escada — o valor
  vigente em cada instante), `runtime.json`. Em C1, além disso, `individuals.npz` (registros de todos
  os 542 637 indivíduos infectados em todas as réplicas), `final_size_pmf_exact.npy` e
  `index_offspring_pmf_exact.npy` (as distribuições exatas usadas como referência).
- `data/processed/` guarda agregados (JSON e CSV: limiares, resumos de V1/V2, séries de convergência,
  varredura). `results/tables/` tem todas as tabelas em CSV e Markdown; `results/figures/` tem todas as
  figuras em PNG (300 dpi) e PDF, com rótulos em português.
- `python run_all.py` (ou `make all`) roda pytest e depois, em ordem: piloto, cenários, varredura, V1,
  V2 e a consolidação (tempos + `RESULTS.md`). Tempo total ≈ 170 s em 12 núcleos.

Arquitetura do código (`src/epidemic/`): `config.py` (constantes e cenários), `rng.py` (seeds,
uniformes, transformada inversa), `gillespie.py` (núcleo), `ode.py` (EDO via `solve_ivp`, DOP853,
rtol 1e-9), `theory.py` (fórmulas analíticas), `exact_ctmc.py` (programação dinâmica), `stats.py`
(descritivas, ICs, dimensionamento, limiar, testes), `plots.py`, `runner.py` (paralelismo e
persistência), `tables.py`. Os experimentos são scripts em `experiments/`; os testes em
`tests/test_epidemic.py`.

---

## 4. Modelos analíticos usados como referência

1. **SIR determinístico (Kermack–McKendrick)**: ṡ = −β·s·i, i̇ = β·s·i − γ·i, ṙ = γ·i, em frações,
   com as mesmas condições iniciais da simulação (s₀ = S₀/N, i₀ = I₀/N, r₀ = vacinados/N).
   Resolvido com `scipy.integrate.solve_ivp`. Dele saem o pico i_max, o tempo do pico e o tamanho
   final determinístico s₀ − s(∞) + i₀.
2. **Equação do tamanho final**: com s₀ a fração suscetível inicial e τ a fração dos suscetíveis que
   acaba infectada, 1 − τ = exp(−R₀·s₀·τ). Tem a raiz trivial τ = 0 e, se R₀·s₀ > 1, uma raiz não
   trivial em (0, 1), encontrada com `brentq` num intervalo [ε, 1) que exclui o zero. O tamanho final
   na população é s₀·τ. Valores de referência: R₀ = 2,5, s₀ = 0,999 → s₀·τ = 0,8914; R₀ = 2,5,
   s₀ = 0,599 (R_eff = 1,5) → 0,3481. Também foi implementada a variante com i₀ > 0,
   1 − τ = exp(−R₀(s₀τ + i₀)), usada apenas para quantificar o viés de I₀ > 0 (ela inclui o caso
   índice e fica ≈ 1/N acima).
3. **Probabilidade de surto maior** (processo de ramificação): 1 − (1/R_eff)^I₀ se R_eff > 1, e 0
   caso contrário. Para C1 (R_eff = 2,5, I₀ = 1): 0,6. Para C2 (R_eff = 1,5): 1/3. Para C3: 0.
4. **Caso subcrítico**: tamanho total esperado a partir de 1 caso = 1/(1 − R_eff). Para C3
   (R_eff = 0,75): 4 casos. O mesmo resultado se aplica aos surtos menores quando R_eff > 1, via o
   processo dual com R = 1/R_eff: tamanho esperado de um surto menor = 1/(1 − 1/R_eff), que dá 1,667
   em C1 e 3,0 em C2.
5. **Offspring do caso índice sem depleção**: enquanto S ≈ N, o índice infecta a taxa β durante um
   tempo Exp(γ), o que dá uma Geométrica em k = 0, 1, 2, …: P(k) = (γ/(β+γ))·(β/(β+γ))^k, com média
   β/γ = R₀. Para β = 0,5, γ = 0,2: q = 5/7, P(0) = 2/7 ≈ 0,286.
6. **Duração infecciosa**: Exp(γ), CDF 1 − e^(−γt), média 5 d, dp 5 d, mediana 5·ln 2 ≈ 3,466 d.
7. **CTMC exata por programação dinâmica** (`exact_ctmc.final_size_pmf`). Usa-se a cadeia de saltos
   embutida: a partir de (s, i) com i > 0, a próxima transição é infecção com probabilidade
   (β·s·i/N)/(β·s·i/N + γ·i) = β·s/(β·s + γ·N), indo para (s−1, i+1), e recuperação com a
   probabilidade complementar, indo para (s, i−1); absorção em i = 0. Observa-se que a infecção
   conserva s + i e a recuperação reduz s + i em 1; dentro de um nível s + i = L, a infecção reduz s.
   Percorrendo os níveis em ordem decrescente e, em cada nível, s em ordem decrescente, toda transição
   vai para um estado ainda não processado (ordem topológica), então uma única passagem propaga toda
   a massa de probabilidade. O tamanho final é Z = (S₀ − s_final) + I₀. Custo O(S₀·N), o que
   permite calcular a distribuição exata não só para N = 10 (validação V1) mas também para N = 1000:
   isso foi feito para C1, C2, C3 e para cada ponto da varredura, e é a referência mais forte do
   trabalho, porque separa "viés de N finito da aproximação analítica" de "erro do simulador".
8. **PMF exata do offspring do caso índice com depleção** (`exact_ctmc.index_offspring_pmf`).
   Estende-se o estado para (s, i, k), k = filhos do índice até agora, enquanto o índice está
   infeccioso. De (s, i, k): infecção pelo índice com prob p_inf·(1/i) → (s−1, i+1, k+1); infecção
   por outro com p_inf·(i−1)/i → (s−1, i+1, k); recuperação do índice com p_rec·(1/i) → absorção
   com K = k; recuperação de outro com p_rec·(i−1)/i → (s, i−1, k). Os fatores 1/i são exatamente as
   probabilidades do sorteio uniforme. A DP percorre os níveis como acima, guardando um vetor em k
   por estado. Isso foi conferido contra um Monte Carlo independente da cadeia embutida (escrito à
   parte, com `random.Random` do Python, 200 000 réplicas): χ² = 18,5 (gl 19), p = 0,49, médias
   2,4192 × 2,4195. Para N = 1000, β = 0,5, γ = 0,2, a média exata do offspring do índice é 2,4195
   (contra 2,5 da geométrica sem depleção) e P(k ≥ 14) = 0,0038 (contra 0,0090 na geométrica).

---

## 5. Desenho experimental

### 5.1 Cenários principais (N = 1000, I₀ = 1, γ = 0,2, β = 0,5)

| cenário | p | vacinados | S₀ | R_eff | P(surto maior) teórica | tamanho final teórico s₀·τ |
|---|---|---|---|---|---|---|
| C1 (base) | 0 | 0 | 999 | 2,50 | 0,600 | 0,8914 |
| C2 (vacinação parcial) | 0,4 | 400 | 599 | 1,50 | 0,333 | 0,3481 |
| C3 (acima do limiar; p* = 0,6) | 0,7 | 700 | 299 | 0,75 | 0 | 0 |

### 5.2 Dimensionamento do número de réplicas

Piloto de 200 réplicas por cenário (chaves 101–103, independentes das rodadas principais). A fórmula é
n = (z·s/(e·x̄))² com z = 1,96 e erro relativo alvo e = 2% sobre a média do tamanho final condicional a
surto maior (em C1 e C2, com o limiar do vale calculado no próprio piloto). Em C3 não existe surto
maior (R_eff < 1), então a fórmula foi aplicada à média incondicional do tamanho final. Regras: mínimo
1000, teto de segurança 50 000.

- C1: x̄ = 0,8913, s = 0,0177 (CV 2,0%), n_base = 116 surtos maiores → n = 3,8 → **1000** (o piso domina).
- C2: x̄ = 0,3460, s = 0,0533 (CV 15%), n_base = 51 → n = 228,1 → **1000**.
- C3: x̄ = 0,00414, s = 0,00932 (CV 2,25), n_base = 200 → n = 48 621,3 → **48 622** (rodadas
  integralmente; são baratas, ≈ 6 eventos por réplica).

### 5.3 Varredura em R₀

R₀ de 0,5 a 4,0 em passos de 0,25 (15 pontos), p = 0, N = 1000, I₀ = 1, 500 réplicas por ponto, cada
ponto com sua própria chave de seed. Para cada ponto: P(surto maior) com IC de Wilson, tamanho final
condicional com IC t, e as referências (ramificação, equação do tamanho final, CTMC exata com o mesmo
limiar).

### 5.4 Validações

- **V1**: N = 10, I₀ = 1, R₀ = 2,5, 100 000 réplicas; PMF simulada do tamanho final × PMF exata (DP);
  qui-quadrado e distância de variação total.
- **V2**: N = 10 000, I₀ = 10, R₀ = 2,5, 200 réplicas, todas as trajetórias salvas; I(t)/N médio com
  banda de IC 95% (t, ponto a ponto na grade de 601 instantes em 0–120 d) × EDO; RMSE, erro no pico e
  no tempo do pico.
- **V3**: 17 testes `pytest`: média e variância do gerador exponencial por transformada inversa
  (400 000 amostras, tolerância de 4 erros-padrão); valor pontual da inversa; conservação de S+I+R,
  monotonicidade do tempo e não negatividade ao longo de toda a trajetória (com e sem vacinação);
  consistência dos registros individuais (todo infectado tem recuperação posterior à infecção, nº de
  infectados = I₀ + soma das infecções secundárias, e quem infectou estava infeccioso naquele
  instante); taxas não negativas em toda a grade de estados e erro quando I₀ + vacinados > N;
  reprodutibilidade (mesma seed → trajetória idêntica; seed diferente → diferente); seeds derivadas
  distintas e determinísticas; caso trivial β = 0 (só recuperações, tamanho final = I₀, exatamente I₀
  eventos); equação do tamanho final contra valores de referência calculados independentemente via
  função W de Lambert (τ = 1 + W(−R₀e^(−R₀))/R₀) para R₀ ∈ {1,5; 2; 2,5; 3} e casos subcríticos e com
  vacinação; P(surto maior) e geométrica (soma 1, média R₀); CTMC exata (soma 1, P(Z=1) =
  γN/(βS₀+γN), β = 0 → Z = I₀); agrupamento de caudas do qui-quadrado.

### 5.5 Estatística

- **Classificação surto maior × menor.** Com R_eff > 1 a distribuição do tamanho final é bimodal. O
  limiar é escolhido pelo vale do histograma do tamanho final em contagens (bins de 1% de N = 10
  casos): identifica-se a moda maior (bin mais frequente com centro acima de 10% de N) e, entre a
  origem e ela, a faixa contígua de bins com contagem mínima (o platô do vale); o limiar é o ponto
  médio dessa faixa. Sensibilidade: repetiu-se a classificação com 10% de N (100 casos) e 5% de N
  (50 casos). Com R_eff < 1 (C3) não há surto maior por definição; réplicas raras que passam de 100
  casos são tratadas como cauda da classe "menor" e comparadas à CTMC exata.
- **Por métrica e cenário**: média, desvio padrão, IC 95% pela t de Student, mediana, quartis, mínimo e
  máximo — incondicionais, condicionais a surto maior e condicionais a surto menor.
- **Proporções**: IC de Wilson a 95%.
- **Comparação com a teoria**: erro absoluto, erro relativo, e indicador de o valor teórico cair
  dentro do IC.
- **Testes**: KS (`scipy.stats.kstest`) contra Exp(γ); qui-quadrado de aderência com agrupamento das
  caudas até que toda célula tenha esperado ≥ 5 (sem parâmetros estimados, então gl = células − 1);
  distância de variação total ½Σ|p̂ − p|.

---

## 6. Resultados de validação

### V1 — CTMC exata (N = 10)

Qui-quadrado nas 10 células k = 1..10 (todas com esperado ≥ 5, nenhum agrupamento): χ² = 7,54,
gl = 9, p = 0,581. TVD = 0,00304. Média do tamanho final: 5,4043 simulada × 5,4098 exata. As 11
probabilidades exatas (k = 0..10) caem dentro do IC 95% da frequência simulada. A distribuição para
N = 10 é fortemente bimodal já em escala tão pequena: P(Z = 1) = 0,3077 (o índice se recupera antes de
infectar alguém: 2/(0,5·9 + 2) = 2/6,5), mínimo em Z = 5 (0,0309) e P(Z = 10) = 0,2247. A
concordância mostra que as taxas e a seleção do evento pelo segundo uniforme estão corretas — a
cadeia embutida do simulador é a cadeia exata.

### V2 — limite de campo médio (N = 10 000, I₀ = 10)

Todas as 200 réplicas foram surtos maiores (com I₀ = 10, P(extinção) ≈ 0,4¹⁰ ≈ 10⁻⁴). RMSE entre a
trajetória média de I(t)/N e a EDO em 0–120 d: 0,00223. O pico da trajetória MÉDIA é 0,2272 contra
0,2339 da EDO (−2,8%), e seu instante 24,6 d contra 24,4 d (+0,8%). A média dos picos individuais
(cada réplica no seu próprio instante) é 0,2374 [IC 0,2362; 0,2386] (+1,5% sobre a EDO), e a média dos
tempos até o pico é 24,55 d [24,31; 24,80] (+0,6%). Tamanho final médio 0,8936 [0,8928; 0,8944] contra
0,8928 da EDO (+0,09%). Interpretação: a média de curvas com pequenos deslocamentos temporais
aleatórios é mais achatada que cada curva, por isso o pico da média fica ligeiramente abaixo; já o
pico de cada réplica é o máximo de um processo com ruído, por isso fica ligeiramente acima. Ambos os
desvios são de 1–3%, a banda de IC é estreita e a curva média praticamente coincide com a EDO em todo
o intervalo: convergência ao campo médio com N = 10⁴.

### V3 — 17 testes pytest passando.

---

## 7. Resultados dos cenários principais

### 7.1 Limiar e bimodalidade

- **C1**: histograma claramente bimodal; a moda maior está em ≈ 895 casos; entre 11 casos (maior
  surto menor) e 835 casos (menor surto maior) não há nenhuma réplica — o vale é um platô de contagem
  zero de largura ~820 casos, cujo ponto médio é 425. 608 surtos maiores e 392 menores. Os três
  limiares (425, 100, 50) dão classificação idêntica.
- **C2**: moda maior em ≈ 355; maior surto menor = 96 casos, menor surto maior = 171 casos; vale de
  contagem zero com ponto médio 135. 312 maiores e 688 menores. Com 100 casos, idêntico; com 50 casos,
  4 réplicas com tamanhos entre 50 e 96 são reclassificadas como maiores e P(surto maior) vai de 0,312
  para 0,316 (dentro do IC de Wilson [0,284; 0,341]); o tamanho final condicional cai de 0,3459 para
  0,3424. Ou seja, a classificação é robusta ao limiar desde que ele fique dentro do platô, e o limiar
  de 5% de N já começa a invadir a cauda dos surtos menores em R_eff = 1,5, onde essa cauda é mais
  longa (processo dual com R = 0,667).
- **C3**: distribuição unimodal, decaindo aproximadamente geometricamente a partir de 1 caso; máximo
  observado 108 casos em 48 622 réplicas; 3 réplicas ≥ 100 casos (6,2·10⁻⁵ [2,1·10⁻⁵; 1,8·10⁻⁴]),
  contra 3,6·10⁻⁵ na CTMC exata — a cauda subcrítica pesada é real, não são "surtos maiores".

### 7.2 Estatísticas por cenário (média [IC 95%]; tabela completa com dp, mediana, quartis, mín, máx em `estatisticas_cenarios`)

| cenário | condição | n | tamanho final (fração) | pico I | tempo até pico (d) | duração (d) | eventos | offspring índice |
|---|---|---|---|---|---|---|---|---|
| C1 | incondicional | 1000 | 0,5426 [0,5156; 0,5696] | 150,0 [142,5; 157,4] | 15,0 [14,3; 15,8] | 43,4 [41,3; 45,5] | 1084 [1030; 1138] | 2,52 [2,35; 2,69] |
| C1 | surto maior | 608 | 0,8915 [0,8901; 0,8929] | 245,7 [243,9; 247,5] | 24,4 [24,0; 24,8] | 69,9 [69,1; 70,7] | 1782 [1779; 1785] | 3,92 [3,70; 4,13] |
| C1 | surto menor | 392 | 0,0016 [0,0014; 0,0017] | 1,4 [1,3; 1,5] | 0,5 [0,4; 0,7] | 2,3 [2,1; 2,6] | 2,1 [1,9; 2,4] | 0,34 [0,28; 0,41] |
| C2 | incondicional | 1000 | 0,1103 [0,1003; 0,1203] | 18,0 [16,5; 19,6] | 15,1 [13,7; 16,4] | 35,1 [32,3; 37,9] | 220 [200; 240] | 1,52 [1,39; 1,65] |
| C2 | surto maior | 312 | 0,3459 [0,3405; 0,3513] | 53,3 [51,7; 55,0] | 43,5 [41,7; 45,3] | 99,3 [96,9; 101,7] | 691 [680; 702] | 3,37 [3,10; 3,64] |
| C2 | surto menor | 688 | 0,0035 [0,0029; 0,0040] | 2,0 [1,9; 2,2] | 2,2 [1,8; 2,5] | 6,0 [5,4; 6,6] | 5,9 [4,8; 7,0] | 0,68 [0,60; 0,76] |
| C3 | incondicional (= menor) | 48622 | 0,00364 [0,00358; 0,00370] | 2,15 [2,13; 2,17] | 3,09 [3,03; 3,14] | 9,02 [8,91; 9,13] | 6,27 [6,15; 6,39] | 0,748 [0,738; 0,758] |

Alguns detalhes descritivos: em C1 o tamanho final condicional tem dp 0,018, mediana 0,894, quartis
0,879–0,903, mínimo 0,835 e máximo 0,941; o pico condicional tem dp 22,7 (mín 163, máx 327); a duração
condicional vai de 50,8 a 119,8 d. Em C2 o tamanho final condicional tem dp 0,049 (mín 0,171, máx
0,439); o tempo até o pico condicional é muito disperso (dp 16,6 d, de 12,5 a 142 d), refletindo a fase
inicial longa e aleatória com R_eff = 1,5. Em C3, mediana do tamanho final = 1 caso (só o índice), Q3 =
3 casos; duração mediana 4,5 d. Nas estatísticas incondicionais de C1 e C2 a média é uma mistura das
duas modas (a média incondicional 0,5426 em C1 é ≈ 0,608·0,8915 + 0,392·0,0016) e por isso a mediana
(0,874) fica longe da média — o que justifica reportar condicionalmente. As distribuições de pico e
duração dos surtos maiores são aproximadamente simétricas; as dos surtos menores são fortemente
assimétricas à direita.

Efeito da vacinação (C1 → C2 → C3): o tamanho final condicional cai de 0,89 para 0,35 e desaparece;
o pico cai de 246 para 53 e para ~2; a epidemia fica mais lenta e mais longa em C2 (pico em 43,5 d e
duração de 99 d contra 24,4 d e 70 d em C1) porque o crescimento inicial com R_eff = 1,5 é mais lento;
em C3 (p = 0,7 > p* = 0,6) toda introdução se extingue rapidamente, com 3,6 casos em média.

### 7.3 Teoria × simulação

| quantidade | simulado [IC 95%] | referência | valor | erro rel. | no IC? |
|---|---|---|---|---|---|
| C1 P(surto maior) | 0,608 [0,577; 0,638] | ramificação 1 − 1/2,5 | 0,600 | +1,3% | sim |
| C1 P(surto maior) | idem | CTMC exata N=1000, limiar 425 | 0,5989 | +1,5% | sim |
| C1 tamanho final ∣ maior | 0,8915 [0,8901; 0,8929] | equação s₀·τ | 0,8914 | +0,007% | sim |
| C1 tamanho final ∣ maior | idem | equação com i₀ (inclui I₀) | 0,8928 | −0,15% | sim |
| C1 tamanho final ∣ maior | idem | CTMC exata | 0,8918 | −0,04% | sim |
| C1 tamanho ∣ menor (casos) | 1,569 [1,436; 1,702] | dual 1/(1 − 1/2,5) | 1,667 | −5,9% | sim |
| C1 tamanho ∣ menor (casos) | idem | CTMC exata | 1,679 | −6,6% | sim |
| C1 offspring médio do índice | 2,517 [2,346; 2,688] | R₀·s₀ (geométrica) | 2,4975 | +0,8% | sim |
| C1 pico I/N ∣ maior | 0,2457 [0,2439; 0,2475] | EDO | 0,2339 | +5,1% | não |
| C1 tempo até pico ∣ maior | 24,41 d [24,05; 24,78] | EDO | 24,36 d | +0,2% | sim |
| C2 P(surto maior) | 0,312 [0,284; 0,341] | ramificação 1 − 1/1,5 | 0,3333 | −6,4% | sim |
| C2 P(surto maior) | idem | CTMC exata N=1000, limiar 135 | 0,3196 | −2,4% | sim |
| C2 tamanho final ∣ maior | 0,3459 [0,3405; 0,3513] | equação s₀·τ | 0,3481 | −0,6% | sim |
| C2 tamanho final ∣ maior | idem | CTMC exata | 0,3466 | −0,2% | sim |
| C2 tamanho ∣ menor (casos) | 3,46 [2,92; 4,00] | dual 1/(1 − 1/1,5) | 3,0 | +15% | sim |
| C2 tamanho ∣ menor (casos) | idem | CTMC exata | 3,79 | −8,7% | sim |
| C2 offspring médio do índice | 1,519 [1,392; 1,646] | R₀·s₀ | 1,4975 | +1,4% | sim |
| C2 pico I/N ∣ maior | 0,0533 [0,0517; 0,0550] | EDO | 0,0385 | +39% | não |
| C2 tempo até pico ∣ maior | 43,5 d [41,7; 45,3] | EDO | 51,7 d | −16% | não |
| C3 P(surto maior) | 0 [0; 7,9·10⁻⁵] | ramificação | 0 | — | sim |
| C3 P(Z ≥ 100) | 6,2·10⁻⁵ [2,1·10⁻⁵; 1,8·10⁻⁴] | CTMC exata | 3,6·10⁻⁵ | +70% | sim |
| C3 tamanho médio (casos) | 3,637 [3,577; 3,697] | ramificação 1/(1 − 0,75) | 4,000 | −9,1% | **não** |
| C3 tamanho médio (casos) | idem | CTMC exata | 3,606 | +0,9% | sim |
| C3 offspring médio do índice | 0,748 [0,738; 0,758] | R₀·s₀ = 0,7475 | 0,7475 | +0,06% | sim |

Leitura dos pontos que merecem explicação:

- **P(surto maior) e tamanho final condicional** concordam com a ramificação e com a equação do
  tamanho final dentro do IC nos dois cenários supercríticos, e concordam ainda melhor com a CTMC
  exata para N = 1000. Em C2 a ramificação superestima P(surto maior) (0,333 vs 0,320 exato): com
  R_eff = 1,5 a fase inicial é longa e a depleção de suscetíveis já é sentida antes de a epidemia
  "decolar", o que aumenta um pouco a chance de extinção precoce. A simulação (0,312) fica entre as
  duas e dentro do IC de ambas.
- **Viés de I₀ > 0**: a equação padrão (i₀ = 0) mede apenas os suscetíveis infectados; a métrica
  simulada inclui o caso índice, logo é ≈ 1/N = 0,001 maior. Com N = 1000 essa diferença é da ordem
  da largura do IC em C1 (±0,0014), então a equação padrão e a variante com i₀ ficam ambas dentro do
  IC. Na varredura, com ICs mais estreitos, o viés fica visível (ver seção 9).
- **C3**: o tamanho médio simulado (3,637) fica fora do IC em relação a 1/(1 − R_eff) = 4, mas dentro em
  relação à CTMC exata (3,606). A ramificação ignora a depleção: com apenas S₀ = 299 suscetíveis, cada
  infecção reduz a taxa efetiva da cadeia (e, a rigor, R₀·S₀/N = 0,7475, não 0,75). A média do processo
  subcrítico é dominada pela cauda (cadeias longas), justamente onde a depleção atua. É um viés de N
  finito esperado, não um erro.
- **Surtos menores em C1 e C2** concordam com o processo dual e com a CTMC exata (ICs largos porque a
  variável é muito assimétrica).
- **Pico e tempo do pico versus EDO (condicional a surto maior)**: em C1 o pico simulado é 5% maior
  que o da EDO (fora do IC) e o tempo do pico coincide; em C2 o pico é 39% maior e o tempo do pico
  16% menor. Isso não é erro de simulação, é a combinação de dois efeitos esperados: (i) o pico de uma
  trajetória estocástica é o máximo de um processo ruidoso, e o máximo tem viés positivo da ordem do
  desvio padrão local (≈ √I_pico: ~15 indivíduos sobre 234 em C1, ~6 sobre 38 em C2 — relativamente
  muito maior em C2); (ii) condicionar a surto maior seleciona as realizações em que o crescimento
  inicial foi rápido (o índice infectou cedo e muito), o que equivale a um "adiantamento" aleatório
  em relação à EDO que parte deterministicamente de 1 infectado — por isso o pico vem mais cedo, e
  numa população em que a depleção ainda não avançou, mais alto. Em C2 o efeito é grande porque a fase
  de crescimento com R_eff = 1,5 é longa (a EDO leva 51,7 d para chegar ao pico) e o pico é pequeno. A
  figura de trajetórias de C2 mostra as curvas médias dos surtos maiores nitidamente à frente da EDO.
  A validação V2 (N = 10⁴, I₀ = 10, sem condicionamento relevante) mostra os mesmos erros caindo para
  1–3%, confirmando que se trata de efeito de N finito e de condicionamento.
- **Offspring médio do índice** coincide com R₀·s₀ em todos os cenários — mas a distribuição não é a
  geométrica (seção 8).

### 7.4 Convergência (C1)

A média acumulada do tamanho final (incondicional e condicional a surto maior) em função do número de
réplicas, com IC 95% t, é mostrada em `convergencia_tamanho_final_C1`. A média incondicional
converge para 0,5426 [0,5156; 0,5696] com 1000 réplicas, contra E[Z]/N = 0,5348 da CTMC exata; a
condicional estabiliza em 0,8915 ± 0,0014, contra 0,8914 (equação) e 0,8918 (exata). A largura do IC
decai como 1/√n; a média condicional já está dentro de ±0,5% do valor final com ~100 réplicas, enquanto
a incondicional precisa de várias centenas por causa da bimodalidade (dp 0,43).

---

## 8. Distribuições teórico × empírico e a investigação do offspring

### (a) Duração infecciosa (C1, todos os 542 637 indivíduos infectados em todas as 1000 réplicas)

Média 4,9937 d (teoria 5), dp 4,9894 (teoria 5), mediana 3,4634 d (teoria 3,466). KS contra Exp(0,2):
D = 0,00106, p = 0,576. A CDF empírica e a teórica são indistinguíveis no gráfico, e o QQ-plot está
sobre a reta y = x até os quantis extremos (máximo observado ≈ 47 d). Isso confirma que o sorteio
uniforme de quem se recupera preserva a lei individual Exp(γ), e que não há censura (a simulação roda
até I = 0).

### (b) Offspring do caso índice (C1, 1000 réplicas)

Média empírica 2,517 [2,346; 2,688] contra 2,5 da geométrica. Qui-quadrado contra a Geométrica com
caudas agrupadas (15 células, k = 0..13 e k ≥ 14): χ² = 26,68, gl = 14, p = 0,021 — rejeição
marginal a 5%. A decomposição mostra que a célula k = 6 contribui 9,6 dos 26,7 (57 observados contra
37,9 esperados) e que a cauda k ≥ 14 tem 2 observados contra 9,0 esperados.

Como a instrução era investigar qualquer divergência além do explicável, a investigação foi levada até
o fim e teve três partes:

1. **Mecanismo**: a geométrica supõe que o índice vê S ≈ N durante toda a sua vida infecciosa. Mas um
   índice com k ≥ 14 filhos precisa viver da ordem de 28 dias (β = 0,5/dia), e nesse tempo, num surto
   maior com N = 1000, S/N já caiu para bem menos de 1 (o pico ocorre em ~24 d). Portanto a cauda da
   distribuição real tem que ser mais leve que a geométrica e a média tem que ser um pouco menor que
   R₀. Isso não some com N grande tão rápido quanto se imagina, porque o tempo até o pico cresce só
   logaritmicamente com N.
2. **Referência exata**: em vez de uma aproximação, calculou-se a PMF exata do offspring do índice com
   depleção pela CTMC estendida (s, i, k) descrita na seção 4, item 8 (conferida contra um Monte Carlo
   independente). Para N = 1000: média 2,4195, P(k ≥ 14) = 0,0038 (metade da geométrica). As 1000
   réplicas de C1 contra ela: χ² = 19,97, gl = 13, p = 0,096 (não rejeita).
3. **Verificação de alta potência**: 5 lotes independentes de 100 000 réplicas cada (chaves de seed
   4000–4004, `data/raw/C1_offspring_extra/batch*/`), guardando só as métricas. Contra a geométrica,
   os p-valores por lote são 7,6·10⁻⁷⁷, 5,4·10⁻⁷¹, 7,9·10⁻⁶⁴, 3,1·10⁻⁷⁰, 3,9·10⁻⁷⁰; agrupado
   (n = 500 000) χ² = 1924 com gl 25 (p numericamente 0), média 2,4225 ≠ 2,5. Contra a CTMC exata com
   depleção: p por lote = 0,171, 0,242, 0,974, 0,955, 0,729; agrupado χ² = 11,06, gl = 22, p = 0,974,
   média 2,4225 contra 2,4195 exata (diferença de 0,003, dentro de 1 erro-padrão de 0,004).

Conclusão: o simulador reproduz exatamente a distribuição do modelo individual; a hipótese "sem
depleção" da geométrica é a aproximação que falha para N = 1000, e o p = 0,021 em C1 mistura esse
efeito real (cauda e média) com flutuação amostral na célula k = 6 (um desvio de ~3 dp entre 15
células). A média empírica de C1 (2,517) coincide com R₀ dentro do IC porque o efeito da depleção
sobre a média (−3%) é menor que a meia-largura do IC (±7% com 1000 réplicas); com 500 000 réplicas a
diferença fica evidente. A figura `pmf_offspring_C1` mostra as três coisas sobrepostas: barras
empíricas com IC, geométrica, e a PMF exata com depleção.

**Nota de processo relevante**: durante essa investigação apareceu um bug no agrupamento de caudas da
função de qui-quadrado (`stats.chi2_gof`): a linha `e[-2] += e.pop()` em Python reavalia o índice −2
depois do `pop`, então a massa agrupada ia para a célula errada e algumas células desapareciam. O bug
foi detectado porque uma verificação com N = 100 e 400 000 réplicas mostrou z-scores de +7 nas células
finais enquanto as contagens brutas coincidiam com a PMF exata. Foi corrigido (variáveis temporárias),
coberto por um teste unitário, e todos os experimentos foram reexecutados do zero. O teste V1 não era
afetado (nenhuma célula precisou ser agrupada), e os p-valores contra a geométrica em C1 não mudaram
(26,68); o que mudou foram os testes suplementares contra a PMF exata. Todos os números deste texto
são posteriores à correção.

### (c) Tamanho final em V1

Já descrito na seção 6: χ² = 7,54 (gl 9), p = 0,581, TVD = 0,00304, figura `V1_pmf_tamanho_final`.

---

## 9. Varredura em R₀ (N = 1000, I₀ = 1, p = 0, 500 réplicas por ponto)

| R₀ | limiar | P(maior) sim [Wilson] | ramificação | CTMC exata | tam. final ∣ maior [IC] | equação s₀·τ | CTMC exata |
|---|---|---|---|---|---|---|---|
| 0,50 | 100 | 0 [0; 0,008] | 0 | 5·10⁻⁹ | — | 0 | — |
| 0,75 | 100 | 0 [0; 0,008] | 0 | 6·10⁻⁴ | — | 0 | — |
| 1,00 | 100 | 0,032 [0,020; 0,051] | 0 | 0,042 | 0,167 [0,128; 0,206] | 0 | 0,178 |
| 1,25 | 155 | 0,180 [0,149; 0,216] | 0,200 | 0,179 | 0,390 [0,373; 0,408] | 0,370 | 0,378 |
| 1,50 | 235 | 0,316 [0,277; 0,358] | 0,333 | 0,326 | 0,585 [0,576; 0,593] | 0,581 | 0,579 |
| 1,75 | 310 | 0,426 [0,383; 0,470] | 0,429 | 0,425 | 0,710 [0,704; 0,716] | 0,711 | 0,711 |
| 2,00 | 345 | 0,528 [0,484; 0,571] | 0,500 | 0,498 | 0,797 [0,794; 0,801] | 0,795 | 0,795 |
| 2,25 | 385 | 0,544 [0,500; 0,587] | 0,556 | 0,554 | 0,850 [0,848; 0,853] | 0,852 | 0,852 |
| 2,50 | 425 | 0,596 [0,552; 0,638] | 0,600 | 0,599 | 0,8914 [0,8893; 0,8935] | 0,8914 | 0,8918 |
| 2,75 | 450 | 0,612 [0,569; 0,654] | 0,636 | 0,635 | 0,918 [0,917; 0,920] | 0,919 | 0,920 |
| 3,00 | 455 | 0,642 [0,599; 0,683] | 0,667 | 0,666 | 0,940 [0,939; 0,941] | 0,939 | 0,940 |
| 3,25 | 460 | 0,656 [0,613; 0,696] | 0,692 | 0,692 | 0,955 [0,954; 0,956] | 0,954 | 0,955 |
| 3,50 | 480 | 0,708 [0,667; 0,746] | 0,714 | 0,714 | 0,966 [0,965; 0,966] | 0,965 | 0,966 |
| 3,75 | 480 | 0,730 [0,689; 0,767] | 0,733 | 0,733 | 0,974 [0,973; 0,975] | 0,973 | 0,974 |
| 4,00 | 490 | 0,768 [0,729; 0,803] | 0,750 | 0,750 | 0,980 [0,980; 0,981] | 0,979 | 0,980 |

- **P(surto maior)**: a curva 1 − 1/R₀ cai dentro do IC de Wilson em todos os 12 pontos com R₀ > 1;
  a CTMC exata cai dentro do IC em 15/15. O maior desvio absoluto em relação à ramificação é 0,036
  (R₀ = 3,25), compatível com o erro-padrão de uma proporção com 500 réplicas (≈ 0,021). A transição
  em R₀ = 1 é nítida: zero surtos maiores abaixo, crescimento côncavo acima.
- **Tamanho final condicional**: a equação s₀·τ cai dentro do IC em 8/12 pontos com R₀ > 1; a CTMC
  exata em 12/12. Nos quatro pontos em que a equação fica fora (R₀ = 3,25; 3,75; 4,0 e 1,25), o
  motivo em três deles é o viés de I₀: a partir de R₀ ≈ 3 o IC é tão estreito (±0,001) que a
  diferença de ≈ 1/N entre "incluir o índice" (simulação, CTMC exata) e "não incluir" (equação com
  i₀ = 0) fica estatisticamente visível; a simulação fica sistematicamente ~0,001 acima da equação e
  coincide com a CTMC exata. Em R₀ = 1,25, o erro relativo é o maior da varredura (+5,6% contra a
  equação, +3,3% contra a exata) porque perto do limiar a distribuição condicional é muito larga (o
  vale é raso, os surtos "maiores" variam de ~155 a ~500 casos) e a fronteira entre as duas classes é
  menos nítida.
- **R₀ = 1,00 (crítico)**: a distribuição do tamanho final não é bimodal; a classificação não é bem
  definida. Com o limiar nominal de 100 casos, 16/500 réplicas ficam acima (a CTMC exata dá P(Z ≥ 100)
  = 0,042 — concordante), enquanto a ramificação dá 0. Não é uma discrepância: é o comportamento
  crítico, em que o tamanho final tem cauda pesada (decaimento em lei de potência) sem uma segunda
  moda.
- **R₀ = 1,25**: a ramificação (0,200) superestima ligeiramente P(surto maior) em relação à exata
  (0,179), pelo mesmo motivo discutido em C2; a simulação (0,180) coincide com a exata.
- Para R₀ ≤ 0,75, tamanho médio incondicional 2,0 e 4,0 casos (×1/(1 − R₀) = 2,0 e 4,0), e nenhuma
  réplica ultrapassou 100 casos.

---

## 10. Figuras e tabelas geradas

Figuras (todas em PNG 300 dpi e PDF, rótulos em português, em `results/figures/`):
`trajetorias_C1/C2/C3` (30 réplicas finas de S, I, R em cores fixas, a média sobre os surtos maiores
das 100 réplicas salvas — ou de todas em C3 — em traço grosso, e a EDO tracejada);
`hist_tamanho_final_C1/C2/C3` (escala log, bins de 10 casos, os limiares testados e o tamanho final
teórico); `boxplots_incondicional` e `boxplots_condicional_surto_maior` (pico, tempo até o pico,
tamanho final e duração, C1 × C2 × C3; C3 não aparece no condicional porque não tem surtos maiores);
`varredura_prob_surto_maior` e `varredura_tamanho_final` (pontos com barras de IC, curva teórica
tracejada, CTMC exata pontilhada); `V2_campo_medio` (média com banda de IC e EDO);
`cdf_duracao_infecciosa_C1`, `qq_duracao_infecciosa_C1`, `pmf_offspring_C1`,
`V1_pmf_tamanho_final`; `convergencia_tamanho_final_C1` (dois painéis, eixo x logarítmico).

Tabelas (CSV + Markdown em `results/tables/`): `parametros_cenarios`, `tamanho_amostral`,
`estatisticas_cenarios`, `sensibilidade_limiar`, `teoria_vs_simulacao_cenarios`,
`testes_distribuicoes_C1`, `varredura_R0`, `V1_pmf_tamanho_final`, `V1_testes`, `V2_campo_medio`,
`tempo_execucao` (tempo de simulação por experimento) e `tempo_execucao_etapas` (tempo por script).

Tempos de execução (12 núcleos, 11 workers): C1 1000 réplicas em 0,24 s; C2 0,14 s; C3 48 622
réplicas em 1,1 s; V1 100 000 réplicas em 2,3 s; V2 200 réplicas com N = 10⁴ em 1,4 s; varredura
(7500 réplicas) 2,4 s; cada lote de 100 000 réplicas do offspring 21–35 s. Por script:
`exp_scenarios.py` 150 s (dominado pelos 5 lotes extras e pelas DPs exatas para N = 1000),
`exp_sweep.py` 9,5 s, `exp_v1_exact.py` 4,8 s, `exp_v2_meanfield.py` 3,2 s, piloto 0,9 s, pytest
0,8 s; `run_all.py` completo ≈ 170 s.

---

## 11. Limitações e observações honestas

- N = 1000 é pequeno o suficiente para que efeitos de N finito sejam visíveis em quantidades
  sensíveis (pico condicional, tamanho de surtos menores, P(surto maior) perto de R_eff = 1). Todos
  esses desvios foram quantificados e explicados com a CTMC exata para o mesmo N, que é a referência
  correta; a ramificação e a equação do tamanho final são limites N → ∞.
- O tamanho final simulado inclui o caso índice; a equação padrão não. A diferença é 1/N e foi
  reportada nas duas formas.
- O tempo até o pico é o primeiro instante em que I atinge seu máximo; para surtos menores isso é
  frequentemente t = 0 (o pico é o próprio I₀ = 1), o que faz a média incondicional dessa métrica
  ter pouco significado — daí a ênfase nas versões condicionais.
- A comparação da trajetória média com a EDO (V2 e figuras de trajetórias) envolve dois efeitos que
  agem em sentidos opostos (achatamento da média por dispersão temporal e viés positivo do máximo por
  réplica), ambos pequenos em N = 10⁴ e ambos reportados separadamente.
- O teste qui-quadrado do offspring contra a geométrica rejeita para N = 1000 por uma razão física
  (depleção), não numérica; a distribuição exata correspondente foi derivada e a simulação concorda
  com ela com p = 0,97 em 500 000 réplicas.
- Não há dados reais no trabalho; todas as "verdades" são analíticas ou exatas por programação
  dinâmica.

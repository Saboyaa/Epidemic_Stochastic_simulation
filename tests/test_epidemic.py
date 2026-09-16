"""V3 — testes unitários (pytest)."""
import math

import numpy as np
import pytest

from epidemic.exact_ctmc import final_size_pmf
from epidemic.gillespie import SIRParams, simulate
from epidemic.rng import exp_inverse, make_generator, sample_exponential, spawn_seeds
from epidemic.theory import final_size_tau, final_size_fraction, prob_major_outbreak, offspring_pmf


# ---------------------------------------------------------------- gerador
def test_exponential_inverse_transform_mean_and_var():
    gen = make_generator(np.random.SeedSequence(12345))
    rate = 0.2
    x = sample_exponential(gen, rate, 400_000)
    assert x.min() > 0
    # média 1/λ e variância 1/λ² com tolerância ~4 erros-padrão
    se_mean = (1 / rate) / math.sqrt(len(x))
    assert abs(x.mean() - 1 / rate) < 4 * se_mean
    assert abs(x.var(ddof=1) - 1 / rate**2) < 0.02 * (1 / rate**2)


def test_exp_inverse_scalar_matches_formula():
    assert exp_inverse(math.exp(-1.0), 2.0) == pytest.approx(0.5)
    assert exp_inverse(1.0, 3.0) == 0.0
    assert exp_inverse(0.5, 0.0) == math.inf


# ------------------------------------------------------------- conservação
@pytest.mark.parametrize("p", [0.0, 0.4])
def test_conservation_S_I_R(p):
    params = SIRParams(N=300, I0=2, R0=2.0, p=p)
    res = simulate(params, np.random.SeedSequence(7))
    assert np.all(res.S + res.I + res.R == params.N)
    assert res.I[-1] == 0
    assert np.all(np.diff(res.t) >= 0)
    assert np.all(res.S >= 0) and np.all(res.I >= 0) and np.all(res.R >= 0)


def test_individual_records_consistent():
    params = SIRParams(N=500, I0=1, R0=3.0)
    res = simulate(params, np.random.SeedSequence(99))
    inf = ~np.isnan(res.t_inf)
    # todo infectado se recuperou (sem censura) e a duração é positiva
    assert np.all(~np.isnan(res.t_rec[inf]))
    assert np.all(res.t_rec[inf] - res.t_inf[inf] > 0)
    # número de infectados = I0 + soma das infecções secundárias
    assert inf.sum() == params.I0 + res.n_secondary.sum()
    assert res.metrics["final_size"] == inf.sum()
    # quem infectou estava infeccioso naquele instante
    ids = np.flatnonzero(res.infector >= 0)
    for j in ids:
        i = res.infector[j]
        assert res.t_inf[i] < res.t_inf[j] < res.t_rec[i]


# --------------------------------------------------------- taxas não negativas
def test_rates_nonnegative():
    params = SIRParams(N=100, I0=5, R0=1.5, p=0.3)
    beta, gamma = params.beta, params.gamma
    for S in range(0, params.S0 + 1):
        for I in range(1, params.N + 1 - S):
            assert beta * S * I / params.N >= 0
            assert gamma * I > 0
    with pytest.raises(ValueError):
        SIRParams(N=10, I0=5, R0=2.0, p=0.9).S0


# ------------------------------------------------------------ reprodutibilidade
def test_reproducibility_same_seed():
    params = SIRParams(N=400, I0=1, R0=2.5)
    a = simulate(params, np.random.SeedSequence(2024))
    b = simulate(params, np.random.SeedSequence(2024))
    assert np.array_equal(a.t, b.t) and np.array_equal(a.I, b.I)
    assert a.metrics == b.metrics
    c = simulate(params, np.random.SeedSequence(2025))
    assert not (np.array_equal(a.t, c.t))


def test_spawned_seeds_are_distinct_and_deterministic():
    s1 = spawn_seeds(1, 5, 10)
    s2 = spawn_seeds(1, 5, 10)
    assert [x.spawn_key for x in s1] == [x.spawn_key for x in s2]
    states = {tuple(x.generate_state(4)) for x in s1}
    assert len(states) == 10


# ----------------------------------------------------------------- β = 0
def test_trivial_beta_zero():
    params = SIRParams(N=200, I0=7, R0=0.0)
    res = simulate(params, np.random.SeedSequence(1))
    assert res.metrics["final_size"] == 7
    assert res.metrics["n_events"] == 7            # apenas recuperações
    assert res.S[-1] == params.S0 and res.R[-1] == 7
    assert res.n_secondary.sum() == 0
    # duração = máximo de 7 exponenciais; apenas checa positividade
    assert res.metrics["duration"] > 0


# ------------------------------------------------------ equação do tamanho final
@pytest.mark.parametrize("R0, expected", [
    (2.0, 0.7968121300200202),   # valores de referência via Lambert W: τ = 1 + W(−R0 e^{−R0})/R0
    (2.5, 0.8926447536092110),
    (1.5, 0.5828116438658117),
    (3.0, 0.9404797907073597),
])
def test_final_size_equation_known_values(R0, expected):
    tau = final_size_tau(R0, s0=1.0)
    assert tau == pytest.approx(expected, rel=1e-9)
    assert 1 - tau == pytest.approx(math.exp(-R0 * tau), abs=1e-12)


def test_final_size_subcritical_is_zero_and_with_vaccination():
    assert final_size_tau(0.8, s0=1.0) == 0.0
    assert final_size_tau(2.5, s0=0.3) == 0.0            # R0·s0 = 0.75 < 1
    tau = final_size_tau(2.5, s0=0.6)                     # R_eff = 1.5
    assert tau == pytest.approx(final_size_tau(1.5, s0=1.0), rel=1e-9)
    assert final_size_fraction(2.5, 0.6) == pytest.approx(0.6 * tau)


def test_prob_major_and_offspring():
    assert prob_major_outbreak(2.5, 1) == pytest.approx(0.6)
    assert prob_major_outbreak(2.5, 2) == pytest.approx(1 - 0.16)
    assert prob_major_outbreak(0.75, 1) == 0.0
    k = np.arange(0, 2000)
    pk = offspring_pmf(k, beta=0.5, gamma=0.2)
    assert pk.sum() == pytest.approx(1.0, abs=1e-12)
    assert (k * pk).sum() == pytest.approx(2.5, abs=1e-9)


# -------------------------------------------------------------- CTMC exata
def test_exact_ctmc_pmf_sums_to_one_and_matches_small_case():
    pmf = final_size_pmf(N=10, I0=1, beta=0.5, gamma=0.2)
    assert pmf.sum() == pytest.approx(1.0)
    # P(Z=1) = P(primeira transição é recuperação) = γN/(β S0 + γN)
    assert pmf[1] == pytest.approx(0.2 * 10 / (0.5 * 9 + 0.2 * 10))
    # β = 0 => Z = I0 com prob. 1
    pmf0 = final_size_pmf(N=10, I0=1, beta=0.0, gamma=0.2)
    assert pmf0[1] == pytest.approx(1.0)


# ---------------------------------------------------------- qui-quadrado
def test_chi2_gof_merges_tail_correctly():
    from epidemic.stats import chi2_gof
    probs = np.array([0.5, 0.3, 0.15, 0.03, 0.015, 0.005])
    obs = np.array([50, 30, 15, 3, 1, 1], dtype=float)  # n = 100 -> esperados 50,30,15,3,1.5,0.5
    res = chi2_gof(obs, probs, min_expected=5.0)
    # cauda agrupada: {3,4,5} -> esperado 5, observado 5
    assert res["n_celulas"] == 4 and res["gl"] == 3
    assert np.allclose(res["esp"], [50, 30, 15, 5]) and np.allclose(res["obs"], [50, 30, 15, 5])
    assert res["estatistica"] == pytest.approx(0.0)
    assert res["obs"].sum() == obs.sum()

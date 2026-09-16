"""Algoritmo de Gillespie (método direto) para o SIR estocástico em nível individual.

Estado agregado: (S, I, R), N = S + I + R constante.
Eventos:
  infecção   (S, I) -> (S-1, I+1)  taxa a_inf = β·S·I/N
  recuperação (I, R) -> (I-1, R+1)  taxa a_rec = γ·I

Passo do método direto:
  1. a0 = a_inf + a_rec;  Δt = −ln(U1)/a0   (transformada inversa, U1 ~ U(0,1))
  2. evento = infecção se U2·a0 < a_inf, senão recuperação (U2 ~ U(0,1))
  3. nível individual: o infectante é sorteado uniformemente entre os I infecciosos,
     o infectado uniformemente entre os S suscetíveis e, na recuperação, quem se
     recupera é sorteado uniformemente entre os infecciosos.

O sorteio uniforme individual é equivalente ao modelo agregado por
permutabilidade: cada infeccioso contribui com taxa β·S/N de infecção e γ de
recuperação, iguais para todos, de modo que, condicionado ao tipo de evento, o
indivíduo responsável é uniforme entre os elegíveis (ver README).

A simulação corre sempre até I = 0 (sem horizonte de tempo), evitando censura das
durações infecciosas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .rng import UniformStream, exp_inverse, make_generator


@dataclass
class SIRParams:
    N: int
    I0: int
    R0: float
    gamma: float = 0.2
    p: float = 0.0  # fração vacinada previamente (começa em R)

    @property
    def beta(self) -> float:
        return self.R0 * self.gamma

    @property
    def n_vacc(self) -> int:
        return int(round(self.p * self.N))

    @property
    def S0(self) -> int:
        s0 = self.N - self.I0 - self.n_vacc
        if s0 < 0:
            raise ValueError("I0 + vacinados excede N")
        return s0

    @property
    def R_eff(self) -> float:
        return self.R0 * (1.0 - self.p)

    def as_dict_init(self) -> dict:
        return dict(N=self.N, I0=self.I0, R0=self.R0, gamma=self.gamma, p=self.p)

    def as_dict(self) -> dict:
        return dict(N=self.N, I0=self.I0, R0=self.R0, gamma=self.gamma, beta=self.beta,
                    p=self.p, n_vacc=self.n_vacc, S0=self.S0, R_eff=self.R_eff)


@dataclass
class SIRResult:
    """Saída de uma replicação."""
    # séries temporais por evento (t[0] = 0 é o estado inicial)
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    # registros por indivíduo (nan / -1 quando não se aplica)
    t_inf: np.ndarray       # instante de infecção (0 para casos iniciais; nan se nunca infectado)
    t_rec: np.ndarray       # instante de recuperação (nan se nunca infectado)
    infector: np.ndarray    # id de quem infectou (-1: caso inicial; -2: nunca infectado; -3: vacinado)
    n_secondary: np.ndarray  # número de infecções secundárias causadas
    n_events: int
    params: SIRParams
    metrics: dict = field(default_factory=dict)


def simulate(params: SIRParams, seed, record_trajectory: bool = True) -> SIRResult:
    """Uma replicação do SIR estocástico (Gillespie, método direto), até I = 0."""
    N, I0 = params.N, params.I0
    beta, gamma = params.beta, params.gamma
    S0 = params.S0
    nV = params.n_vacc

    gen = make_generator(seed)
    U = UniformStream(gen)

    # ids: 0..I0-1 infecciosos iniciais; I0..I0+S0-1 suscetíveis; restantes vacinados (R)
    t_inf = np.full(N, np.nan)
    t_rec = np.full(N, np.nan)
    infector = np.full(N, -2, dtype=np.int64)
    n_secondary = np.zeros(N, dtype=np.int64)
    t_inf[:I0] = 0.0
    infector[:I0] = -1
    if nV > 0:
        infector[I0 + S0:] = -3

    infectious = list(range(I0))          # ids dos infecciosos
    susceptible = list(range(I0, I0 + S0))  # ids dos suscetíveis

    S, I, R = S0, I0, nV
    t = 0.0
    beta_over_N = beta / N

    # capacidade máxima de eventos: cada suscetível pode ser infectado uma vez e
    # cada infectado se recupera uma vez -> 2·S0 + I0 eventos no máximo
    max_events = 2 * S0 + I0 + 1
    if record_trajectory:
        tt = np.empty(max_events); SS = np.empty(max_events, dtype=np.int64)
        II = np.empty(max_events, dtype=np.int64); RR = np.empty(max_events, dtype=np.int64)
        tt[0] = 0.0; SS[0] = S; II[0] = I; RR[0] = R
    k = 0
    peak_I, t_peak = I, 0.0

    while I > 0:
        a_inf = beta_over_N * S * I
        a_rec = gamma * I
        a0 = a_inf + a_rec
        assert a_inf >= 0.0 and a_rec > 0.0

        # 1) tempo até o próximo evento: transformada inversa
        u1 = U.next()
        t += exp_inverse(u1, a0)

        # 2) escolha do evento com o segundo uniforme
        u2 = U.next()
        if u2 * a0 < a_inf:
            # infecção: infectante uniforme entre os infecciosos, infectado uniforme entre os suscetíveis
            j_inf = min(int(U.next() * I), I - 1)   # índice uniforme na lista de infecciosos
            j_sus = min(int(U.next() * S), S - 1)   # índice uniforme na lista de suscetíveis
            infector_id = infectious[j_inf]
            # remoção O(1) por troca com o último
            last = susceptible[-1]
            new_id = susceptible[j_sus]
            susceptible[j_sus] = last
            susceptible.pop()
            infectious.append(new_id)
            t_inf[new_id] = t
            infector[new_id] = infector_id
            n_secondary[infector_id] += 1
            S -= 1; I += 1
            if I > peak_I:
                peak_I, t_peak = I, t
        else:
            # recuperação: quem se recupera é uniforme entre os infecciosos
            j = min(int(U.next() * I), I - 1)   # índice uniforme na lista de infecciosos
            last = infectious[-1]
            rec_id = infectious[j]
            infectious[j] = last
            infectious.pop()
            t_rec[rec_id] = t
            I -= 1; R += 1

        k += 1
        if record_trajectory:
            tt[k] = t; SS[k] = S; II[k] = I; RR[k] = R

    assert S + I + R == N
    if record_trajectory:
        traj = (tt[:k + 1], SS[:k + 1], II[:k + 1], RR[:k + 1])
    else:
        traj = (np.array([0.0, t]), np.array([S0, S]), np.array([I0, 0]), np.array([nV, R]))

    ever_infected = N - S - nV  # inclui I0
    metrics = dict(
        final_size=ever_infected,
        final_size_frac=ever_infected / N,
        peak_I=int(peak_I),
        t_peak=float(t_peak),
        duration=float(t),
        n_events=int(k),
        index_offspring=int(n_secondary[0]),
        mean_initial_offspring=float(n_secondary[:I0].mean()),
    )
    return SIRResult(t=traj[0], S=traj[1], I=traj[2], R=traj[3], t_inf=t_inf, t_rec=t_rec,
                     infector=infector, n_secondary=n_secondary, n_events=k, params=params,
                     metrics=metrics)


def resample_on_grid(t: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Amostra a trajetória em escada (t, x) nos instantes `grid` (valor vigente)."""
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    return x[idx]

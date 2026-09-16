"""Geração de números aleatórios: seeds por replicação e transformada inversa.

* `spawn_seeds`: deriva uma SeedSequence independente por replicação a partir da
  seed mestre (numpy.random.SeedSequence.spawn).
* `UniformStream`: fluxo de uniformes U(0,1) em blocos (buffer) para reduzir o
  custo de chamadas Python no laço do Gillespie.
* `exp_inverse`: transformada inversa da exponencial, escrita explicitamente:
  Δt = −ln(U)/a0.
"""
from __future__ import annotations

import math

import numpy as np


def spawn_seeds(master_seed: int, exp_key: int, n: int) -> list[np.random.SeedSequence]:
    """Uma SeedSequence filha por replicação: SeedSequence(master, spawn_key=(exp_key,)).spawn(n)."""
    parent = np.random.SeedSequence(master_seed, spawn_key=(exp_key,))
    return parent.spawn(n)


def make_generator(seed) -> np.random.Generator:
    """Generator PCG64 a partir de uma SeedSequence (ou inteiro)."""
    return np.random.Generator(np.random.PCG64(seed))


class UniformStream:
    """Fornece U ~ U(0,1) estritamente em (0,1), um por vez, com buffer interno.

    numpy devolve U em [0,1); usamos 1 − U ∈ (0,1] para que ln(U) seja finito.
    """

    __slots__ = ("_gen", "_buf", "_i", "_size")

    def __init__(self, gen: np.random.Generator, block: int = 8192):
        self._gen = gen
        self._size = block
        self._buf = 1.0 - gen.random(block)
        self._i = 0

    def next(self) -> float:
        if self._i >= self._size:
            self._buf = 1.0 - self._gen.random(self._size)
            self._i = 0
        u = self._buf[self._i]
        self._i += 1
        return float(u)


def exp_inverse(u: float, rate: float) -> float:
    """Transformada inversa: se U ~ U(0,1], então −ln(U)/rate ~ Exp(rate).

    A CDF é F(t) = 1 − e^{−rate·t}; F^{-1}(v) = −ln(1−v)/rate, e 1−V ~ U(0,1)
    também, logo −ln(U)/rate tem a mesma distribuição.
    """
    if rate <= 0.0:
        return math.inf
    return -math.log(u) / rate


def sample_exponential(gen: np.random.Generator, rate: float, size: int) -> np.ndarray:
    """Amostra vetorial de Exp(rate) por transformada inversa (usado nos testes)."""
    u = 1.0 - gen.random(size)
    return -np.log(u) / rate

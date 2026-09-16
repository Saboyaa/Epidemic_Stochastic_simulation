"""Execução de replicações independentes (em paralelo) e persistência de entradas/saídas."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import MASTER_SEED, EXP_KEY, DATA_RAW
from .gillespie import SIRParams, simulate, resample_on_grid
from .rng import spawn_seeds


def lib_versions() -> dict:
    import matplotlib, scipy
    return dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, matplotlib=matplotlib.__version__, epidemic=__version__,
                platform=platform.platform())


def _worker(args):
    params_dict, seed_state, rep, keep_traj, keep_indiv = args
    params = SIRParams(**params_dict)
    seed = np.random.SeedSequence(**seed_state)
    res = simulate(params, seed, record_trajectory=keep_traj)
    out = dict(rep=rep, **res.metrics)
    traj = (res.t, res.S, res.I, res.R) if keep_traj else None
    indiv = None
    if keep_indiv:
        mask = ~np.isnan(res.t_inf)
        ids = np.flatnonzero(mask)
        indiv = dict(id=ids, t_inf=res.t_inf[mask], t_rec=res.t_rec[mask],
                     infector=res.infector[mask], n_secondary=res.n_secondary[mask])
    return out, traj, indiv


def _seed_state(ss: np.random.SeedSequence) -> dict:
    return dict(entropy=int(ss.entropy), spawn_key=[int(k) for k in ss.spawn_key],
                pool_size=int(ss.pool_size), n_children_spawned=0)


def run_experiment(name: str, params: SIRParams, n_reps: int, n_traj: int = 30,
                   keep_individuals: bool = False, workers: int | None = None,
                   exp_key: int | None = None, extra_config: dict | None = None,
                   out_dir: Path | None = None):
    """Roda n_reps replicações independentes; grava config.json, metrics.csv e trajetórias.

    Retorna (df_metrics, trajs, indiv, elapsed) onde trajs é uma lista com as
    (t,S,I,R) das primeiras n_traj replicações.
    """
    exp_key = EXP_KEY[name] if exp_key is None else exp_key
    out_dir = (DATA_RAW / name) if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = spawn_seeds(MASTER_SEED, exp_key, n_reps)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    t0 = time.perf_counter()
    jobs = [(params.as_dict_init(), _seed_state(seeds[r]), r, r < n_traj, keep_individuals)
            for r in range(n_reps)]
    rows, trajs, indivs = [], [], []
    chunk = max(1, n_reps // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out, traj, indiv in ex.map(_worker, jobs, chunksize=chunk):
            rows.append(out)
            if traj is not None:
                trajs.append(traj)
            if indiv is not None:
                indivs.append(indiv)
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    df.insert(1, "seed_spawn_key", [f"{exp_key},{r}" for r in range(n_reps)])
    df.to_csv(out_dir / "metrics.csv", index=False)

    config = dict(
        experimento=name, parametros=params.as_dict(), n_replicacoes=n_reps,
        seed_mestre=MASTER_SEED, exp_spawn_key=exp_key,
        derivacao_seeds="SeedSequence(seed_mestre, spawn_key=(exp_spawn_key,)).spawn(n_replicacoes)[rep]",
        n_trajetorias_salvas=min(n_traj, n_reps), workers=workers,
        tempo_execucao_s=elapsed, versoes=lib_versions(),
    )
    if extra_config:
        config.update(extra_config)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(out_dir / "runtime.json", "w") as f:
        json.dump(dict(experimento=name, tempo_execucao_s=elapsed, n_replicacoes=n_reps), f, indent=2)

    indiv_df = None
    if indivs:
        parts = []
        for r, d in enumerate(indivs):
            part = pd.DataFrame(d)
            part.insert(0, "rep", r)
            parts.append(part)
        indiv_df = pd.concat(parts, ignore_index=True)
        indiv_df["duration"] = indiv_df["t_rec"] - indiv_df["t_inf"]
        np.savez_compressed(out_dir / "individuals.npz", **{c: indiv_df[c].to_numpy() for c in indiv_df.columns})
    return df, trajs, indiv_df, elapsed


def save_trajectories(trajs, grid: np.ndarray, out_path: Path) -> pd.DataFrame:
    """Reamostra as trajetórias em grade de tempo e salva em CSV longo (rep, t, S, I, R)."""
    parts = []
    for r, (t, S, I, R) in enumerate(trajs):
        parts.append(pd.DataFrame(dict(rep=r, t=grid, S=resample_on_grid(t, S, grid),
                                       I=resample_on_grid(t, I, grid), R=resample_on_grid(t, R, grid))))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out_path, index=False)
    return df


def grid_for(trajs, n_points: int = 400, quantile: float = 1.0) -> np.ndarray:
    ends = np.array([t[-1] for t, *_ in trajs])
    t_max = float(np.quantile(ends, quantile))
    return np.linspace(0.0, max(t_max, 1.0), n_points)

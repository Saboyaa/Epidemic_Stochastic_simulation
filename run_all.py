#!/usr/bin/env python
"""Regenera TUDO do zero: testes, experimentos, tabelas, figuras e RESULTS.md.

Uso: python run_all.py [--skip-tests]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STEPS = [
    ("exp_pilot.py", "piloto (dimensionamento de réplicas)"),
    ("exp_scenarios.py", "cenários C1, C2, C3 + distribuições (a), (b) + convergência"),
    ("exp_sweep.py", "varredura em R0"),
    ("exp_v1_exact.py", "V1: CTMC exata (N = 10)"),
    ("exp_v2_meanfield.py", "V2: limite de campo médio (N = 10000)"),
    ("write_results.py", "tabelas finais e RESULTS.md"),
]


def run(cmd, cwd):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"falhou: {' '.join(map(str, cmd))}")
    return time.perf_counter() - t0


def main():
    t_start = time.perf_counter()
    times = {}
    if "--skip-tests" not in sys.argv:
        print("== pytest ==", flush=True)
        times["pytest"] = run([PY, "-m", "pytest", "-q"], ROOT)
    for script, desc in STEPS:
        print(f"\n== {desc} ({script}) ==", flush=True)
        times[script] = run([PY, script], ROOT / "experiments")
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data" / "processed" / "run_all_times.json", "w") as f:
        json.dump(dict(tempos_por_etapa_s=times, total_s=time.perf_counter() - t_start), f, indent=2)
    print(f"\nconcluído em {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

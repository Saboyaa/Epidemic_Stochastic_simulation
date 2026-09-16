"""Configuração global: seed mestre, caminhos e parâmetros fixos."""
from pathlib import Path

# Seed mestre. TODAS as seeds de todos os experimentos derivam dela via
# numpy.random.SeedSequence(MASTER_SEED, spawn_key=(EXP_KEY[nome],)).spawn(n).
MASTER_SEED = 20260916

# Chave de spawn por experimento (garante fluxos independentes entre experimentos).
EXP_KEY = {
    "pilot_C1": 101, "pilot_C2": 102, "pilot_C3": 103,
    "C1": 1, "C2": 2, "C3": 3, "C1_offspring_extra": 4,
    "sweep": 10,
    "V1": 20,
    "V2": 30,
}

GAMMA = 1.0 / 5.0        # taxa de recuperação (1/dia); período infeccioso médio de 5 dias
Z_975 = 1.959963984540054
REL_ERR = 0.02           # erro relativo alvo para o dimensionamento de réplicas
N_PILOT = 200
N_MIN = 1000
N_CAP = 50000            # teto de segurança para o n calculado

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"

SCENARIOS = {
    "C1": dict(label="C1 (base)", N=1000, I0=1, R0=2.5, p=0.0),
    "C2": dict(label="C2 (vacinação parcial)", N=1000, I0=1, R0=2.5, p=0.4),
    "C3": dict(label="C3 (vacinação acima do limiar)", N=1000, I0=1, R0=2.5, p=0.7),
}

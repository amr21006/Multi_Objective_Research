from __future__ import annotations
"""
Case-study datasets for the fuzzy multi-algorithm tournament framework.

The original case-study data accompanying the study was not recoverable, so
the cases below are *re-specified transparently* to match every parameter stated
in the study (20 activities, three execution modes, 120-day deadline,
trapezoidal fuzzy durations, working + idle carbon, renewable resource demand).
All values are disclosed here; nothing is hidden in a binary blob.

A Case bundles:
  - activities : ordered list of activity names
  - preds      : {activity_index: [(pred_index, lag_days), ...]}  (FS relations,
                 positive lag = mandatory wait, negative lag = lead/overlap)
  - For each activity i and mode m in {0,1,2}:
        fuzzy duration (a,b,c,d)  [trapezoidal, right-skewed risk]
        direct_cost   [currency units, lump sum for the activity]
        crew          [renewable workers/day -> resource leveling + histogram]
        work_co2_day  [working emissions, kgCO2 per active day]
  - indirect_rate  : currency/day (time-dependent overhead)
  - deadline       : contract duration (days)
  - omega          : quadratic late-penalty coefficient (currency/day^2)
  - standby_co2_day: site idle/standby emissions, kgCO2 per calendar day

Mode 0 is the slow / low-intensity / low-emission baseline mode; modes 1 and 2
are progressively faster but more resource- and cost-intensive. Faster modes
raise the *daily* emission rate but compress duration, so the working-emission
product and the duration-proportional idle term move in opposite directions;
this is the mechanism behind the small net carbon change discussed in the study.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class Mode:
    fuzzy: Tuple[float, float, float, float]  # (a,b,c,d) trapezoidal duration
    direct_cost: float
    crew: int
    work_co2_day: float


@dataclass
class Activity:
    name: str
    preds: List[Tuple[int, float]]  # (predecessor_index, lag_days)
    modes: List[Mode]


@dataclass
class Case:
    name: str
    activities: List[Activity]
    indirect_rate: float
    deadline: float
    omega: float
    standby_co2_day: float

    @property
    def n(self) -> int:
        return len(self.activities)


# ---------------------------------------------------------------------------
# Mode-generation helpers (deterministic; fully disclosed multipliers)
# ---------------------------------------------------------------------------
# duration scale, direct-cost scale, crew scale, daily-CO2 scale per mode.
# Regime: time-dependent indirect overhead dominates, so the all-Mode-0
# conservative baseline is both the slowest and the costliest schedule, while
# faster modes add crew (worse leveling) and a higher daily emission rate
# (working-emission pushback). This yields a genuine four-way conflict.
DUR_SCALE = (1.00, 0.82, 0.68)     # faster modes compress duration
COST_SCALE = (1.00, 1.06, 1.15)    # modest direct-cost premium for speed
CREW_SCALE = (1.00, 1.10, 1.20)    # faster modes need modestly larger crews
CO2_SCALE = (1.00, 1.30, 1.62)     # faster modes raise daily emission rate
DIRECT_COST_SCALE = 0.45           # scales base direct costs (indirect-dominated)

# trapezoidal spread around the per-mode most-likely value (right-skewed risk)
A_FACTOR = 0.86   # optimistic boundary
B_FACTOR = 0.96   # most-likely minimum
C_FACTOR = 1.06   # most-likely maximum
D_FACTOR = 1.32   # pessimistic boundary (longer right tail)


def _modes(b0: float, cost0: float, crew0: int, co2day0: float) -> List[Mode]:
    modes = []
    for m in range(3):
        ml = b0 * DUR_SCALE[m]
        fuzzy = (round(ml * A_FACTOR, 2), round(ml * B_FACTOR, 2),
                 round(ml * C_FACTOR, 2), round(ml * D_FACTOR, 2))
        modes.append(Mode(
            fuzzy=fuzzy,
            direct_cost=round(cost0 * DIRECT_COST_SCALE * COST_SCALE[m], 2),
            crew=max(1, int(round(crew0 * CREW_SCALE[m]))),
            work_co2_day=round(co2day0 * CO2_SCALE[m], 3),
        ))
    return modes


# ---------------------------------------------------------------------------
# CASE A : primary 20-activity infrastructure / building network
# Activity names corresponding to the primary experimental network.
# Columns: name, preds[(idx,lag)], dur0(days), cost0($), crew0, co2day0(kgCO2/day)
# ---------------------------------------------------------------------------
_A = [
    ("Site Preparation",            [],                         5,  18000,  3,  90),
    ("Excavation",                  [(0, 0)],                   7,  26000,  4, 165),
    ("Underground Utilities",       [(1, 0)],                   6,  31000,  3, 120),
    ("Foundation Installation",     [(1, 0)],                   7,  42000,  4, 150),
    ("Concrete Foundation Pouring", [(3, 0), (2, 0)],           5,  48000,  5, 175),
    ("Foundation Curing",           [(4, 2)],                   7,   6000,  1,  20),
    ("Steel Frame Assembly",        [(5, 0)],                   9,  72000,  5, 220),
    ("Roof Structure",              [(6, 0)],                   7,  46000,  4, 140),
    ("Exterior Wall Installation",  [(7, 0)],                   8,  53000,  4, 130),
    ("Interior Framing",            [(7, 0)],                   6,  34000,  3,  95),
    ("Plumbing Rough-In",           [(9, 0)],                   5,  28000,  3,  70),
    ("Electrical Rough-In",         [(9, 0)],                   5,  27000,  3,  68),
    ("HVAC Installation",           [(9, 0)],                   6,  39000,  3,  88),
    ("Insulation",                  [(8, 0), (10, 0), (11, 0), (12, 0)], 4, 19000, 3, 45),
    ("Drywall Installation",        [(13, 0)],                  6,  31000,  4,  60),
    ("Interior Finishing",          [(14, 0)],                  7,  44000,  4,  72),
    ("Flooring",                    [(15, 0)],                  5,  29000,  3,  55),
    ("Final Plumbing",              [(15, 0)],                  4,  16000,  2,  40),
    ("Final Electrical",            [(15, 0)],                  4,  17000,  2,  42),
    ("Final Inspection",            [(16, 0), (17, 0), (18, 0)], 3,  9000,  2,  25),
]


def case_A() -> Case:
    acts = [Activity(name=row[0], preds=row[1], modes=_modes(row[2], row[3], row[4], row[5]))
            for row in _A]
    return Case(name="Case A - 20-activity infrastructure",
                activities=acts, indirect_rate=5200.0, deadline=120.0,
                omega=900.0, standby_co2_day=58.0)


# ---------------------------------------------------------------------------
# CASE B : smaller 10-activity building fit-out (different scale)
# ---------------------------------------------------------------------------
_B = [
    ("Mobilization",          [],                 3, 12000, 2,  55),
    ("Demolition",            [(0, 0)],           5, 22000, 4, 130),
    ("Structural Repair",     [(1, 0)],           7, 48000, 4, 180),
    ("MEP First Fix",         [(2, 0)],           6, 33000, 3,  85),
    ("Partition Walls",       [(2, 0)],           5, 24000, 3,  60),
    ("Plastering",            [(3, 0), (4, 0)],   4, 17000, 3,  48),
    ("MEP Second Fix",        [(5, 0)],           5, 26000, 3,  70),
    ("Finishes",              [(5, 0)],           6, 38000, 4,  64),
    ("Testing & Commissioning", [(6, 0), (7, 0)], 4, 21000, 2,  52),
    ("Handover",              [(8, 0)],           2,  7000, 2,  20),
]


def case_B() -> Case:
    acts = [Activity(name=row[0], preds=row[1], modes=_modes(row[2], row[3], row[4], row[5]))
            for row in _B]
    return Case(name="Case B - 10-activity building fit-out",
                activities=acts, indirect_rate=3200.0, deadline=70.0,
                omega=700.0, standby_co2_day=34.0)


# ---------------------------------------------------------------------------
# CASE C : larger 32-activity linear utility / pipeline network (scale-up)
# Built as four repeated segments of an 8-task pipeline crew chain so the
# network has long parallel/sequential structure suitable for stress-testing.
# ---------------------------------------------------------------------------
def case_C() -> Case:
    seg_template = [
        ("Survey",         3, 14000, 2,  60),
        ("Trenching",      6, 30000, 4, 170),
        ("Bedding",        4, 16000, 3,  70),
        ("Pipe Laying",    7, 52000, 4, 150),
        ("Jointing/Weld",  5, 33000, 3, 110),
        ("Backfill",       4, 19000, 3,  95),
        ("Compaction",     3, 13000, 3,  80),
        ("Reinstatement",  5, 27000, 3,  88),
    ]
    acts: List[Activity] = []
    n_seg = 4
    seglen = len(seg_template)
    for s in range(n_seg):
        for j, (nm, d0, c0, cr0, co0) in enumerate(seg_template):
            idx = s * seglen + j
            preds: List[Tuple[int, float]] = []
            if j > 0:
                preds.append((idx - 1, 0))            # in-segment chain
            if s > 0 and j == 0:
                preds.append(((s - 1) * seglen + 1, 0))  # segment start trails prior trenching
            lag = 1 if nm == "Jointing/Weld" else 0      # curing/weld inspection lag
            preds = [(p, lag if (j > 0 and nm == "Jointing/Weld") else l) for (p, l) in preds]
            acts.append(Activity(name=f"S{s+1}-{nm}", preds=preds,
                                 modes=_modes(d0, c0, cr0, co0)))
    return Case(name="Case C - 32-activity linear utility network",
                activities=acts, indirect_rate=6000.0, deadline=160.0,
                omega=1100.0, standby_co2_day=72.0)


CASES = {"A": case_A, "B": case_B, "C": case_C}


if __name__ == "__main__":
    for k, fn in CASES.items():
        c = fn()
        print(f"Case {k}: {c.name} | activities={c.n} | deadline={c.deadline}")


"""
Core library for the fuzzy multi-algorithm tournament + consensus + SHAP
framework.

Implements the following equations:
  Eq.1  centroid defuzzification of trapezoidal fuzzy durations
  Eq.2-3 modified CPM forward pass with lag/lead
  Eq.4  project duration
  Eq.5  total cost = direct + indirect + quadratic late penalty
  Eq.6  resource leveling = sum of squared deviations from mean daily demand
  Eq.7  total carbon = working emissions + idle emissions
  Eq.8  CRITIC weights
  Eq.9  MEREC weights
  Eq.10 hybrid weights + sustainability bias
  Eq.11 TOPSIS closeness coefficient
  Eq.12 EDAS appraisal score
  Eq.13 WASPAS joint measure
  Eq.14 consensus score
  Eq.15 SHAP values (via the shap library on a Random Forest surrogate)

Everything is deterministic given a seed; no value is hand-set to a target.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from data import Case

EPS = 1e-12

# ===========================================================================
# 1. Fuzzy logic (Eq. 1)
# ===========================================================================
def defuzz_centroid(fuzzy: Tuple[float, float, float, float]) -> float:
    """Centroid defuzzification of a trapezoidal fuzzy number (a,b,c,d):
    d = (a + 2b + 2c + d) / 6  -- Eq.(1)."""
    a, b, c, d = fuzzy
    return (a + 2.0 * b + 2.0 * c + d) / 6.0


def crisp_durations(case: Case, modes: np.ndarray) -> np.ndarray:
    """Defuzzified crisp duration for each activity under the chosen modes."""
    return np.array([defuzz_centroid(case.activities[i].modes[int(modes[i])].fuzzy)
                     for i in range(case.n)], dtype=float)


# ===========================================================================
# 2. Modified CPM forward pass (Eq. 2-3) + resource profile
# ===========================================================================
def forward_pass(case: Case, dur: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (ES, FT, project_duration). Topological order assumed by index
    (every predecessor index < successor index in the case definitions)."""
    n = case.n
    ES = np.zeros(n)
    FT = np.zeros(n)
    for i in range(n):
        preds = case.activities[i].preds
        if not preds:
            ES[i] = 0.0
        else:
            ES[i] = max(FT[p] + lag for (p, lag) in preds)
            ES[i] = max(ES[i], 0.0)
        FT[i] = ES[i] + dur[i]
    return ES, FT, float(FT.max())


def daily_resource_profile(case: Case, modes: np.ndarray, ES: np.ndarray,
                           FT: np.ndarray) -> np.ndarray:
    """Integer-day labor demand profile R_t (total crew active each day)."""
    horizon = int(np.ceil(FT.max())) if FT.max() > 0 else 1
    R = np.zeros(horizon)
    for i in range(case.n):
        crew = case.activities[i].modes[int(modes[i])].crew
        start = int(np.floor(ES[i]))
        end = int(np.ceil(FT[i]))
        end = min(end, horizon)
        if end <= start:
            end = min(start + 1, horizon)
        R[start:end] += crew
    return R


# ===========================================================================
# 3. Objective functions (Eq. 4-7)
# ===========================================================================
def evaluate_modes(case: Case, modes: np.ndarray) -> Dict[str, float]:
    """Compute the four objectives + diagnostics for one mode vector."""
    modes = np.asarray(modes, dtype=int)
    dur = crisp_durations(case, modes)
    ES, FT, duration = forward_pass(case, dur)

    # F1: duration (Eq.4)
    f1 = duration

    # F2: cost (Eq.5)
    direct = sum(case.activities[i].modes[modes[i]].direct_cost for i in range(case.n))
    indirect = case.indirect_rate * duration
    penalty = case.omega * (max(0.0, duration - case.deadline) ** 2)
    f2 = direct + indirect + penalty

    # F3: resource leveling (Eq.6) -- sum of squared deviations from mean daily demand
    R = daily_resource_profile(case, modes, ES, FT)
    mean_R = R.mean() if R.size else 0.0
    f3 = float(np.sum((R - mean_R) ** 2))

    # F4: carbon (Eq.7) -- working + idle
    work_co2 = sum(case.activities[i].modes[modes[i]].work_co2_day * dur[i]
                   for i in range(case.n))
    # idle: site standby over calendar duration + equipment integer-rounding standby
    idle_round = sum(0.30 * case.activities[i].modes[modes[i]].work_co2_day *
                     (np.ceil(dur[i]) - dur[i]) for i in range(case.n))
    idle_co2 = case.standby_co2_day * duration + idle_round
    f4 = float(work_co2 + idle_co2)

    return dict(time=f1, cost=f2, resource=f3, carbon=f4,
                direct=direct, indirect=indirect, penalty=penalty,
                work_co2=float(work_co2), idle_co2=float(idle_co2),
                peak_labor=float(R.max() if R.size else 0.0),
                std_labor=float(R.std() if R.size else 0.0),
                duration=duration, ES=ES, FT=FT, profile=R)


OBJ_KEYS = ["time", "cost", "resource", "carbon"]
OBJ_LABELS = ["Time (days)", "Cost ($)", "Resource Leveling", "Carbon (kgCO2)"]


def objective_vector(case: Case, modes: np.ndarray) -> np.ndarray:
    r = evaluate_modes(case, modes)
    return np.array([r[k] for k in OBJ_KEYS], dtype=float)


# ===========================================================================
# 4. pymoo problem
# ===========================================================================
from pymoo.core.problem import ElementwiseProblem


class ScheduleProblem(ElementwiseProblem):
    def __init__(self, case: Case):
        self.case = case
        super().__init__(n_var=case.n, n_obj=4, n_ieq_constr=0,
                         xl=np.zeros(case.n), xu=np.full(case.n, 2), vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = objective_vector(self.case, x)


# ===========================================================================
# 5. MCDM weighting (Eq. 8-10)
# ===========================================================================
def _minmax_benefit(F: np.ndarray) -> np.ndarray:
    """Map a cost matrix (minimize) to benefit-normalized [0,1] columns."""
    mn = F.min(axis=0)
    mx = F.max(axis=0)
    rng = np.where((mx - mn) < EPS, 1.0, (mx - mn))
    return (mx - F) / rng  # smaller cost -> closer to 1 (better)


def critic_weights(F: np.ndarray) -> np.ndarray:
    """CRITIC weights (Eq.8): std * sum(1 - corr)."""
    X = _minmax_benefit(F)
    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma < EPS, EPS, sigma)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    C = sigma * np.sum(1.0 - corr, axis=1)
    C = np.where(C < 0, 0.0, C)
    if C.sum() < EPS:
        return np.full(F.shape[1], 1.0 / F.shape[1])
    return C / C.sum()


def merec_weights(F: np.ndarray) -> np.ndarray:
    """MEREC weights (Eq.9): removal effect of each criterion.
    All four objectives are non-beneficial (cost), normalized n_ij = x_ij/max_j."""
    m, k = F.shape
    mx = F.max(axis=0)
    mx = np.where(mx < EPS, EPS, mx)
    N = F / mx                      # non-beneficial normalization, in (0,1]
    N = np.clip(N, EPS, None)
    lnN = np.abs(np.log(N))         # |ln n_ij|
    S = np.log(1.0 + lnN.mean(axis=1))                 # overall performance
    E = np.zeros(k)
    for j in range(k):
        keep = [c for c in range(k) if c != j]
        Sj = np.log(1.0 + lnN[:, keep].mean(axis=1))   # performance without crit j
        E[j] = np.sum(np.abs(Sj - S))
    if E.sum() < EPS:
        return np.full(k, 1.0 / k)
    return E / E.sum()


def hybrid_weights(F: np.ndarray, carbon_bias: float = 1.0,
                   carbon_index: int = 3) -> Dict[str, np.ndarray]:
    """Average CRITIC and MEREC (Eq.10) then apply an optional sustainability
    bias multiplier on the carbon weight. carbon_bias=1.0 -> neutral."""
    wc = critic_weights(F)
    wm = merec_weights(F)
    w = 0.5 * (wc + wm)
    if abs(carbon_bias - 1.0) > EPS:
        w = w.copy()
        w[carbon_index] *= carbon_bias
    w = w / w.sum()
    return dict(critic=wc, merec=wm, hybrid=w)


# ===========================================================================
# 6. MCDM ranking methods (Eq. 11-14)
# ===========================================================================
def topsis(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    """TOPSIS closeness coefficient (Eq.11); all criteria are cost."""
    norm = np.sqrt((F ** 2).sum(axis=0))
    norm = np.where(norm < EPS, EPS, norm)
    V = (F / norm) * w
    ideal_best = V.min(axis=0)    # cost -> best is min
    ideal_worst = V.max(axis=0)
    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    cc = d_worst / (d_best + d_worst + EPS)
    return cc


def edas(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    """EDAS appraisal score (Eq.12); all criteria are cost."""
    AV = F.mean(axis=0)
    AV = np.where(np.abs(AV) < EPS, EPS, AV)
    PDA = np.maximum(0.0, (AV - F)) / AV     # cost: below-average is positive
    NDA = np.maximum(0.0, (F - AV)) / AV
    SP = (PDA * w).sum(axis=1)
    SN = (NDA * w).sum(axis=1)
    NSP = SP / (SP.max() + EPS)
    NSN = 1.0 - SN / (SN.max() + EPS)
    return 0.5 * (NSP + NSN)


def waspas(F: np.ndarray, w: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """WASPAS joint measure (Eq.13), lambda=0.5; all criteria are cost."""
    mn = F.min(axis=0)
    N = mn / np.where(F < EPS, EPS, F)       # cost normalization -> (0,1]
    N = np.clip(N, EPS, None)
    wsm = (N * w).sum(axis=1)
    wpm = np.prod(N ** w, axis=1)
    return lam * wsm + (1.0 - lam) * wpm


def _norm01(s: np.ndarray) -> np.ndarray:
    mn, mx = s.min(), s.max()
    if (mx - mn) < EPS:
        return np.ones_like(s)
    return (s - mn) / (mx - mn)


def consensus_rank(F: np.ndarray, w: np.ndarray) -> Dict[str, np.ndarray]:
    """Multi-MCDM consensus (Eq.14).

    Each base method already returns an intrinsically normalized score in [0,1]:
    TOPSIS closeness coefficient CC_i, EDAS appraisal score AS_i, and WASPAS
    joint measure Q_i. The consensus score of solution i is the arithmetic mean
    of these three scores, S_i = (CC_i + AS_i + Q_i)/3, reported in [0,1]
    (x100 for percent). It measures average closeness-to-best across the three
    decision logics; it is NOT a ranking-agreement percentage. The selected
    schedule is argmax_i S_i. (Diagnostic min-max-normalized variants are also
    returned for the sensitivity/figure code.)"""
    t = topsis(F, w)
    e = edas(F, w)
    wa = waspas(F, w)
    cons = (t + e + wa) / 3.0
    return dict(topsis=t, edas=e, waspas=wa,
                topsis_n=_norm01(t), edas_n=_norm01(e), waspas_n=_norm01(wa),
                consensus=cons)


# ===========================================================================
# 7. Tournament: run solvers, build combined non-dominated front
# ===========================================================================
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ALGO_NAMES = ["NSGA-II", "NSGA-III", "MOEA/D", "RVEA", "UNSGA-III"]


def _int_operators(n_var: int, sbx_prob: float = 0.9):
    return dict(
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=sbx_prob, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0 / n_var, eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )


def make_algorithms(n_var: int, pop: int, ref_dirs):
    ops = _int_operators(n_var)
    return {
        "NSGA-II": NSGA2(pop_size=pop, **ops),
        "NSGA-III": NSGA3(ref_dirs=ref_dirs, pop_size=pop, **ops),
        "MOEA/D": MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7,
                        sampling=ops["sampling"], crossover=ops["crossover"],
                        mutation=ops["mutation"]),
        "RVEA": RVEA(ref_dirs=ref_dirs, pop_size=pop, **ops),
        "UNSGA-III": UNSGA3(ref_dirs=ref_dirs, pop_size=pop, **ops),
    }


def run_solver(case: Case, algo, n_gen: int, seed: int):
    res = minimize(ScheduleProblem(case), algo, ("n_gen", n_gen),
                   seed=seed, verbose=False)
    X = np.atleast_2d(res.X).astype(int)
    F = np.atleast_2d(res.F).astype(float)
    return X, F


def run_tournament(case: Case, pop: int = 200, n_gen: int = 500, seed: int = 1,
                   ref_dirs=None, verbose: bool = False):
    """Run all five solvers; aggregate; extract the combined approximate
    combined approximate non-dominated front."""
    if ref_dirs is None:
        ref_dirs = get_reference_directions("energy", 4, pop, seed=1)
    algos = make_algorithms(case.n, pop, ref_dirs)

    all_X, all_F, source = [], [], []
    raw_counts = {}
    for k, name in enumerate(ALGO_NAMES):
        X, F = run_solver(case, algos[name], n_gen, seed + k)
        raw_counts[name] = len(X)
        all_X.append(X)
        all_F.append(F)
        source.append(np.full(len(X), k))
        if verbose:
            print(f"    {name}: {len(X)} solutions")

    X = np.vstack(all_X)
    F = np.vstack(all_F)
    source = np.concatenate(source)

    # de-duplicate identical mode vectors
    _, uniq = np.unique(X, axis=0, return_index=True)
    uniq = np.sort(uniq)
    X, F, source = X[uniq], F[uniq], source[uniq]

    # combined non-dominated sorting (Rank-1 = combined approximate ND front)
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    Xf, Ff, src_f = X[nd], F[nd], source[nd]

    contrib = {ALGO_NAMES[k]: int(np.sum(src_f == k)) for k in range(5)}
    total = len(Ff)
    table = pd.DataFrame({
        "Algorithm": ALGO_NAMES,
        "Raw Solutions": [raw_counts[a] for a in ALGO_NAMES],
        "Front Contribution": [contrib[a] for a in ALGO_NAMES],
        "Success Rate": [f"{100*contrib[a]/total:.1f}%" if total else "0%"
                         for a in ALGO_NAMES],
    })
    return dict(X=Xf, F=Ff, source=src_f, table=table,
                raw_counts=raw_counts, contrib=contrib, n_front=total,
                X_all=X, F_all=F, source_all=source, ref_dirs=ref_dirs)


# ===========================================================================
# 8. Decision pipeline: weights -> consensus -> best solution
# ===========================================================================
def decide(F: np.ndarray, carbon_bias: float = 1.0) -> Dict:
    w = hybrid_weights(F, carbon_bias=carbon_bias)
    rk = consensus_rank(F, w["hybrid"])
    best = int(np.argmax(rk["consensus"]))
    return dict(weights=w, ranks=rk, best=best,
                consensus_best=float(rk["consensus"][best]))


# ===========================================================================
# 9. Baseline (all Mode 0, conservative deterministic schedule)
# ===========================================================================
def baseline(case: Case) -> Dict[str, float]:
    return evaluate_modes(case, np.zeros(case.n, dtype=int))


def improvement_table(case: Case, best_modes: np.ndarray) -> pd.DataFrame:
    b = baseline(case)
    o = evaluate_modes(case, best_modes)
    rows = []
    for key, lab in [("time", "Time"), ("cost", "Cost"),
                     ("resource", "Resource Leveling"), ("carbon", "Emissions")]:
        base, opt = b[key], o[key]
        diff = base - opt
        pct = 100.0 * diff / base if base else 0.0
        rows.append([lab, round(base, 2), round(opt, 2), round(diff, 2),
                     f"{pct:.2f}%"])
    return pd.DataFrame(rows, columns=["Objective", "Baseline (Mode 0)",
                                       "Optimized", "Diff", "% Improvement"])


# ===========================================================================
# 10. SHAP surrogate (Eq. 15) -- Random Forest + cross-validated performance
# ===========================================================================
def shap_surrogate(case: Case, X: np.ndarray, consensus: np.ndarray,
                   seed: int = 1, n_splits: int = 5):
    """Train a Random Forest surrogate mapping mode vectors -> consensus score;
    report k-fold CV performance and SHAP values with cross-fold stability."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error
    import shap

    y = consensus.astype(float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
    y_pred = cross_val_predict(base, X, y, cv=kf, n_jobs=-1)
    r2 = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

    # fit on full data for global SHAP
    model = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X, check_additivity=False)
    mean_abs = np.abs(sv).mean(axis=0)

    # SHAP stability: refit on bootstraps, correlate feature importances
    rng = np.random.default_rng(seed)
    imps = []
    for b in range(10):
        idx = rng.integers(0, len(X), len(X))
        mb = RandomForestRegressor(n_estimators=200, random_state=seed + b, n_jobs=-1)
        mb.fit(X[idx], y[idx])
        sb = shap.TreeExplainer(mb).shap_values(X, check_additivity=False)
        imps.append(np.abs(sb).mean(axis=0))
    imps = np.array(imps)
    # rank correlation of bootstrap importances vs full-data importances
    from scipy.stats import spearmanr
    stab = np.mean([spearmanr(mean_abs, imps[b]).correlation for b in range(len(imps))])

    names = [case.activities[i].name for i in range(case.n)]
    order = np.argsort(mean_abs)[::-1]
    imp_df = pd.DataFrame({"Activity": [names[i] for i in order],
                           "Mean |SHAP|": mean_abs[order]})
    return dict(r2=float(r2), rmse=rmse, mean_abs=mean_abs, shap_values=sv,
                importance=imp_df, stability=float(stab), model=model,
                feature_names=names)


# ===========================================================================
# 11. Quality indicators for algorithm/baseline comparison
# ===========================================================================
def hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    from pymoo.indicators.hv import HV
    return float(HV(ref_point=ref)(F))


def igd(F: np.ndarray, pf: np.ndarray) -> float:
    from pymoo.indicators.igd import IGD
    return float(IGD(pf)(F))


"""
Master reproducible experiment script (addresses R3.8: one workflow that
generates the analysis).

Runs the optimization:
  [1] Primary Case A tournament -> Tables 1-3, headline improvements   (R-core)
  [2] SHAP surrogate: RF, k-fold R2/RMSE, stability                    (R3.6)
  [3] 30-run statistics + hypervolume + Friedman test                  (R1.3)
  [4] Single-algorithm vs tournament comparison                        (R1.4,R3.7)
  [5] Ablation: solver / weighting / MCDM / bias layers                (R3.2)
  [6] Weight & method sensitivity                                      (R2.3,R3.5)
  [7] Objective correlation analysis                                   (R2.5,R1.5)
  [8] Carbon decomposition + green-solution trade-off                  (R1.5,R1.6)
  [9] Duration-uncertainty robustness (fuzzy Monte Carlo)              (R3.3)
  [10] Additional case studies B and C                                 (R1.2,R2.1)
  [11] All figures regenerated

Usage:  python run_all.py            (full protocol; writes results/ and figures/)
        python run_all.py --quick    (reduced settings for a fast dry run)
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats




HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ---- experiment configuration --------------
QUICK = "--quick" in sys.argv
CFG = dict(
    primary_pop=200, primary_gen=500, primary_seed=1,
    multirun_n=30, multirun_pop=100, multirun_gen=200,
    case_pop=150, case_gen=300,
    ablation_pop=120, ablation_gen=250,
    mc_samples=5000,
)
if QUICK:
    CFG.update(primary_pop=60, primary_gen=80, multirun_n=6, multirun_pop=50,
               multirun_gen=60, case_pop=50, case_gen=60, ablation_pop=50,
               ablation_gen=60, mc_samples=1000)

SUMMARY = {}
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def savecsv(df, name): df.to_csv(os.path.join(RES, name), index=False)


# =========================================================================
# [1] PRIMARY CASE A
# =========================================================================
def experiment_primary():
    log("=== [1] Primary Case A tournament ===")
    case = case_A()
    t0 = time.time()
    tour = run_tournament(case, pop=CFG["primary_pop"], n_gen=CFG["primary_gen"],
                          seed=CFG["primary_seed"], verbose=True)
    F = tour["F"]; X = tour["X"]
    d = decide(F, carbon_bias=1.0)
    best_modes = X[d["best"]]
    o = evaluate_modes(case, best_modes)
    b = baseline(case)
    imp = improvement_table(case, best_modes)

    # Table 1 (algorithm performance)
    savecsv(tour["table"], "table1_algorithm_performance.csv")
    # Table 2 (statistical summary of the front)
    rows = []
    for j, k in enumerate(OBJ_KEYS):
        col = F[:, j]
        rows.append([OBJ_LABELS[j], round(o[k], 2), round(col.min(), 2),
                     round(col.max(), 2), round(col.mean(), 2), round(col.std(ddof=1), 2)])
    t2 = pd.DataFrame(rows, columns=["Objective", "Best Solution", "Pareto Min",
                                     "Pareto Max", "Pareto Mean", "Pareto Std"])
    savecsv(t2, "table2_statistical_summary.csv")
    savecsv(imp, "table3_improvement.csv")

    # weights table
    wt = pd.DataFrame({"Objective": OBJ_LABELS,
                       "CRITIC": np.round(d["weights"]["critic"], 4),
                       "MEREC": np.round(d["weights"]["merec"], 4),
                       "Hybrid": np.round(d["weights"]["hybrid"], 4)})
    savecsv(wt, "table_weights.csv")

    SUMMARY["primary"] = dict(
        front_size=int(tour["n_front"]),
        contributions=tour["contrib"], raw_counts=tour["raw_counts"],
        consensus_best=round(d["consensus_best"], 4),
        best_modes=best_modes.tolist(),
        baseline={k: round(b[k], 2) for k in OBJ_KEYS},
        optimized={k: round(o[k], 2) for k in OBJ_KEYS},
        improvement={r[0]: r[4] for r in imp.values.tolist()},
        peak_labor_opt=o["peak_labor"], std_labor_opt=round(o["std_labor"], 3),
        peak_labor_base=b["peak_labor"], std_labor_base=round(b["std_labor"], 3),
        carbon_work_opt=round(o["work_co2"], 1), carbon_idle_opt=round(o["idle_co2"], 1),
        runtime_s=round(time.time() - t0, 1),
        weights={"critic": np.round(d["weights"]["critic"], 4).tolist(),
                 "merec": np.round(d["weights"]["merec"], 4).tolist(),
                 "hybrid": np.round(d["weights"]["hybrid"], 4).tolist()},
    )
    log(f"    front={tour['n_front']}  consensus={d['consensus_best']:.4f}  "
        f"impr={SUMMARY['primary']['improvement']}  ({SUMMARY['primary']['runtime_s']}s)")

    # figures that depend on the primary result
    fig_pareto_by_algorithm(case, tour, d["best"])
    fig_3d(F, d["best"])
    fig_top20(F, d)
    fig_resource_hist(case, best_modes)
    fig_correlation(F)
    return case, tour, d, best_modes


# =========================================================================
# [2] SHAP
# =========================================================================
def experiment_shap(case, tour, d):
    log("=== [2] SHAP surrogate ===")
    X = tour["X"]; F = tour["F"]
    cons = d["ranks"]["consensus"]
    s = shap_surrogate(case, X, cons, seed=1, n_splits=5 if not QUICK else 3)
    savecsv(s["importance"], "table_shap_importance.csv")
    SUMMARY["shap"] = dict(r2=round(s["r2"], 4), rmse=round(s["rmse"], 5),
                           stability=round(s["stability"], 4),
                           top5=s["importance"].head(5).values.tolist())
    log(f"    RF CV R2={s['r2']:.4f}  RMSE={s['rmse']:.5f}  stability={s['stability']:.3f}")
    fig_shap(case, s, X)
    return s


# =========================================================================
# [3] 30-run statistics + Friedman
# =========================================================================
def experiment_multirun():
    log("=== [3] Multi-run statistics ===")
    case = case_A()
    n = CFG["multirun_n"]
    obj_runs, hv_algo, hv_comb, cons_runs = [], {a: [] for a in ALGO_NAMES}, [], []
    # normalization bounds + reference point from a pilot (objectives -> [0,1],
    # 0=best so hypervolume is comparable and bounded)
    pilot = run_tournament(case, pop=CFG["multirun_pop"], n_gen=CFG["multirun_gen"], seed=999)
    lo = pilot["F_all"].min(axis=0)
    hi = pilot["F_all"].max(axis=0)
    span = np.where((hi - lo) < 1e-9, 1.0, (hi - lo))
    ref = np.full(4, 1.10)
    norm = lambda M: (np.atleast_2d(M) - lo) / span
    from xmoo_lib import NonDominatedSorting
    for r in range(n):
        tour = run_tournament(case, pop=CFG["multirun_pop"], n_gen=CFG["multirun_gen"],
                              seed=100 + r)
        F = tour["F"]
        d = decide(F)
        o = evaluate_modes(case, tour["X"][d["best"]])
        obj_runs.append([o[k] for k in OBJ_KEYS])
        cons_runs.append(d["consensus_best"])
        hv_comb.append(hypervolume(np.clip(norm(F), None, ref), ref))
        # per-algorithm own non-dominated front hypervolume (same normalization)
        for k, a in enumerate(ALGO_NAMES):
            mask = tour["source_all"] == k
            if mask.sum() == 0:
                hv_algo[a].append(0.0); continue
            Fa = tour["F_all"][mask]
            nd = NonDominatedSorting().do(Fa, only_non_dominated_front=True)
            hv_algo[a].append(hypervolume(np.clip(norm(Fa[nd]), None, ref), ref))
        if (r + 1) % 5 == 0 or QUICK:
            log(f"    run {r+1}/{n}")
    obj_runs = np.array(obj_runs)
    stat = []
    for j, k in enumerate(OBJ_KEYS):
        col = obj_runs[:, j]
        ci = stats.t.interval(0.95, len(col) - 1, loc=col.mean(),
                              scale=stats.sem(col)) if len(col) > 1 else (col.mean(), col.mean())
        stat.append([OBJ_LABELS[j], round(col.mean(), 2), round(col.std(ddof=1), 2),
                     round(ci[0], 2), round(ci[1], 2),
                     round(100 * col.std(ddof=1) / col.mean(), 2)])
    stat_df = pd.DataFrame(stat, columns=["Objective", "Mean", "Std", "CI95_low",
                                          "CI95_high", "CoV_%"])
    savecsv(stat_df, "table_multirun_stats.csv")

    # hypervolume comparison + Friedman test
    hv_df = pd.DataFrame({a: hv_algo[a] for a in ALGO_NAMES})
    hv_df["Tournament"] = hv_comb
    savecsv(hv_df, "table_hypervolume_runs.csv")
    # Friedman across the five solvers
    fr = stats.friedmanchisquare(*[hv_df[a].values for a in ALGO_NAMES])
    # Wilcoxon: tournament vs best single solver per run
    best_single = hv_df[ALGO_NAMES].max(axis=1)
    wil = stats.wilcoxon(hv_df["Tournament"], best_single, alternative="greater")
    SUMMARY["multirun"] = dict(
        n_runs=n, settings=f"pop={CFG['multirun_pop']}, gen={CFG['multirun_gen']}",
        consensus_mean=round(float(np.mean(cons_runs)), 4),
        consensus_std=round(float(np.std(cons_runs, ddof=1)), 4),
        hv_mean={a: round(float(np.mean(hv_algo[a])), 4) for a in ALGO_NAMES},
        hv_tournament=round(float(np.mean(hv_comb)), 4),
        friedman_chi2=round(float(fr.statistic), 3), friedman_p=float(fr.pvalue),
        wilcoxon_p=float(wil.pvalue),
        cov={r[0]: r[5] for r in stat},
    )
    log(f"    Friedman chi2={fr.statistic:.2f} p={fr.pvalue:.2e}; "
        f"Wilcoxon(tour>best single) p={wil.pvalue:.2e}")
    fig_hv_box(hv_df)
    return stat_df, hv_df


# =========================================================================
# [4] single-algorithm vs tournament  (uses multirun hv_df already)
# =========================================================================
def experiment_single_vs_tournament(hv_df):
    log("=== [4] Single-algorithm vs tournament ===")
    rows = []
    for a in ALGO_NAMES + ["Tournament"]:
        col = hv_df[a].values
        rows.append([a, round(col.mean(), 4), round(col.std(ddof=1), 4)])
    df = pd.DataFrame(rows, columns=["Method", "Mean Hypervolume", "Std"])
    df = df.sort_values("Mean Hypervolume", ascending=False)
    savecsv(df, "table_single_vs_tournament.csv")
    SUMMARY["single_vs_tournament"] = df.values.tolist()
    log("    " + "; ".join(f"{r[0]}={r[1]}" for r in df.values.tolist()))
    return df


# =========================================================================
# [5] Ablation
# =========================================================================
def _norm_hv(F, lo, span, ref):
    from xmoo_lib import NonDominatedSorting
    nd = NonDominatedSorting().do(np.atleast_2d(F), only_non_dominated_front=True)
    Fn = (np.atleast_2d(F)[nd] - lo) / span
    return hypervolume(np.clip(Fn, None, ref), ref), len(nd)

def experiment_ablation(case, tour, d, best_modes):
    log("=== [5] Ablation (each layer's contribution) ===")
    F = tour["F"]; X = tour["X"]
    F_all = tour["F_all"]; src = tour["source_all"]
    lo = F_all.min(axis=0); span = np.where((F_all.max(axis=0) - lo) < 1e-9, 1.0,
                                             F_all.max(axis=0) - lo)
    ref = np.full(4, 1.10)

    # (a) tournament layer: leave-one-algorithm-out hypervolume vs full front
    full_hv, full_n = _norm_hv(F, lo, span, ref)
    loo_rows = [["Full tournament (5 solvers)", full_n, round(full_hv, 4), "-"]]
    for k, a in enumerate(ALGO_NAMES):
        keep = src != k
        hv_k, n_k = _norm_hv(F_all[keep], lo, span, ref)
        loo_rows.append([f"Without {a}", n_k, round(hv_k, 4),
                         f"{100*(full_hv-hv_k)/full_hv:+.2f}%"])
    loo = pd.DataFrame(loo_rows, columns=["Configuration", "Front size",
                                          "Hypervolume", "HV change vs full"])
    savecsv(loo, "table_ablation_tournament.csv")

    # (b) weighting layer: weight vectors differ across CRITIC / MEREC / hybrid
    wc, wm = critic_weights(F), merec_weights(F)
    wh = hybrid_weights(F)["hybrid"]
    wtab = pd.DataFrame({"Objective": OBJ_LABELS,
                         "CRITIC": np.round(wc, 4), "MEREC": np.round(wm, 4),
                         "Hybrid": np.round(wh, 4)})
    savecsv(wtab, "table_ablation_weights.csv")
    sel_w = {lab: int(np.argmax(consensus_rank(F, w)["consensus"]))
             for lab, w in [("CRITIC", wc), ("MEREC", wm), ("Hybrid", wh)]}

    # (c) MCDM layer: inter-method ranking agreement (Spearman on full front)
    rk = consensus_rank(F, wh)
    rt, re, rw = (stats.rankdata(-rk["topsis"]), stats.rankdata(-rk["edas"]),
                  stats.rankdata(-rk["waspas"]))
    mcdm_corr = {
        "TOPSIS_vs_EDAS": round(float(stats.spearmanr(rt, re).correlation), 4),
        "TOPSIS_vs_WASPAS": round(float(stats.spearmanr(rt, rw).correlation), 4),
        "EDAS_vs_WASPAS": round(float(stats.spearmanr(re, rw).correlation), 4),
    }
    sel_m = {m: int(np.argmax(rk[m])) for m in ["topsis", "edas", "waspas"]}
    sel_m["consensus"] = int(np.argmax(rk["consensus"]))

    # (d) sustainability-bias sweep: carbon weight + selected carbon value
    bias_rows = []
    for bias in [1.0, 1.25, 1.5, 2.0, 3.0]:
        w = hybrid_weights(F, carbon_bias=bias)["hybrid"]
        bi = int(np.argmax(consensus_rank(F, w)["consensus"]))
        o = evaluate_modes(case, X[bi])
        bias_rows.append([bias, round(w[3], 4), bi, round(o["carbon"], 1),
                          round(o["time"], 2)])
    bias_df = pd.DataFrame(bias_rows, columns=["carbon_bias", "carbon_weight",
                                               "selected_idx", "carbon", "time"])
    savecsv(bias_df, "table_ablation_bias.csv")

    SUMMARY["ablation"] = dict(
        leave_one_out=loo.values.tolist(),
        weights={"critic": np.round(wc, 4).tolist(), "merec": np.round(wm, 4).tolist(),
                 "hybrid": np.round(wh, 4).tolist()},
        weighting_selected=sel_w,
        mcdm_rank_correlation=mcdm_corr, mcdm_selected=sel_m,
        bias_sweep=bias_rows,
    )
    log(f"    LOO HV change: " +
        ", ".join(f"{r[0].replace('Without ','-')}:{r[3]}" for r in loo_rows[1:]))
    log(f"    MCDM rank corr (full front): {mcdm_corr}")
    log(f"    bias sweep carbon_weight {bias_df['carbon_weight'].tolist()}")
    return loo


# =========================================================================
# [6] Sensitivity of weights and methods
# =========================================================================
def experiment_sensitivity(case, tour, d):
    log("=== [6] Sensitivity ===")
    F = tour["F"]; X = tour["X"]
    w0 = hybrid_weights(F)["hybrid"]
    base_sel = d["best"]
    base_rank = stats.rankdata(-d["ranks"]["consensus"])
    rng = np.random.default_rng(7)
    same, spear = 0, []
    n_trials = 500 if not QUICK else 50
    for _ in range(n_trials):
        pert = w0 * (1.0 + rng.uniform(-0.15, 0.15, size=4))
        pert = pert / pert.sum()
        rk = consensus_rank(F, pert)
        sel = int(np.argmax(rk["consensus"]))
        same += int(sel == base_sel)
        spear.append(stats.spearmanr(base_rank, stats.rankdata(-rk["consensus"])).correlation)
    # method-drop stability (rank correlation when dropping one MCDM method)
    drops = {}
    for drop in ["topsis", "edas", "waspas"]:
        keep = [m for m in ["topsis", "edas", "waspas"] if m != drop]
        sc = np.mean([d["ranks"][m] for m in keep], axis=0)
        drops[f"drop_{drop}"] = round(float(
            stats.spearmanr(base_rank, stats.rankdata(-sc)).correlation), 4)
    SUMMARY["sensitivity"] = dict(
        weight_perturbation="±15%", trials=n_trials,
        same_selection_rate=round(same / n_trials, 4),
        spearman_mean=round(float(np.nanmean(spear)), 4),
        spearman_min=round(float(np.nanmin(spear)), 4),
        method_drop=drops,
    )
    log(f"    same selection under +/-15%: {same/n_trials:.1%}; "
        f"mean rank-corr {np.nanmean(spear):.3f}; method-drop {drops}")
    fig_sensitivity(spear)


# =========================================================================
# [7] correlation (returned via figure) ; [8] carbon ; [9] robustness
# =========================================================================
def experiment_correlation(tour):
    log("=== [7] Objective correlation ===")
    F = tour["F"]
    pear = np.corrcoef(F, rowvar=False)
    sp = stats.spearmanr(F).correlation
    cdf = pd.DataFrame(pear, columns=OBJ_LABELS, index=OBJ_LABELS).round(3)
    cdf.to_csv(os.path.join(RES, "table_correlation_pearson.csv"))
    SUMMARY["correlation_pearson"] = pear.round(3).tolist()
    SUMMARY["correlation_spearman"] = np.round(sp, 3).tolist()
    log(f"    Pearson(Time,Cost)={pear[0,1]:.2f}  (Time,Carbon)={pear[0,3]:.2f}  "
        f"(Resource,Time)={pear[2,0]:.2f}")


def experiment_carbon(case, tour, d, best_modes):
    log("=== [8] Carbon decomposition + green solution ===")
    F = tour["F"]; X = tour["X"]
    o = evaluate_modes(case, best_modes)
    b = baseline(case)
    # green solution = min carbon on front
    gi = int(np.argmin(F[:, 3]))
    g = evaluate_modes(case, X[gi])
    rows = [
        ["Baseline (Mode 0)", round(b["work_co2"], 1), round(b["idle_co2"], 1),
         round(b["carbon"], 1), round(b["time"], 2), round(b["cost"], 0)],
        ["Consensus solution", round(o["work_co2"], 1), round(o["idle_co2"], 1),
         round(o["carbon"], 1), round(o["time"], 2), round(o["cost"], 0)],
        ["Greenest on front", round(g["work_co2"], 1), round(g["idle_co2"], 1),
         round(g["carbon"], 1), round(g["time"], 2), round(g["cost"], 0)],
    ]
    cdf = pd.DataFrame(rows, columns=["Schedule", "Working kgCO2", "Idle kgCO2",
                                      "Total kgCO2", "Time (d)", "Cost ($)"])
    savecsv(cdf, "table_carbon_decomposition.csv")
    green_vs_cons = dict(
        carbon_cut_pct=round(100 * (o["carbon"] - g["carbon"]) / o["carbon"], 2),
        time_penalty_pct=round(100 * (g["time"] - o["time"]) / o["time"], 2),
        cost_penalty_pct=round(100 * (g["cost"] - o["cost"]) / o["cost"], 2),
    )
    SUMMARY["carbon"] = dict(table=rows, green_vs_consensus=green_vs_cons)
    log(f"    green vs consensus: carbon -{green_vs_cons['carbon_cut_pct']}% "
        f"costs +{green_vs_cons['time_penalty_pct']}% time / "
        f"+{green_vs_cons['cost_penalty_pct']}% cost")


def experiment_robustness(case, best_modes):
    log("=== [9] Duration-uncertainty robustness (fuzzy Monte Carlo) ===")
    n = CFG["mc_samples"]
    rng = np.random.default_rng(2024)
    modes = np.asarray(best_modes, dtype=int)
    # sample each activity duration from its trapezoidal possibility distribution
    durs = np.zeros((n, case.n)) if False else None
    samples_time, samples_cost = [], []
    # precompute trapezoid params
    traps = [case.activities[i].modes[modes[i]].fuzzy for i in range(case.n)]
    for s in range(n):
        dur = np.empty(case.n)
        for i, (a, b, c, dd) in enumerate(traps):
            span = max(dd - a, 1e-9)
            cfrac, dfrac = (b - a) / span, (c - a) / span
            dur[i] = stats.trapezoid.rvs(cfrac, dfrac, loc=a, scale=span,
                                         random_state=rng)
        _, FT, dur_proj = forward_pass(case, dur)
        direct = sum(case.activities[i].modes[modes[i]].direct_cost for i in range(case.n))
        cost = direct + case.indirect_rate * dur_proj + \
            case.omega * (max(0.0, dur_proj - case.deadline) ** 2)
        samples_time.append(dur_proj); samples_cost.append(cost)
    samples_time = np.array(samples_time); samples_cost = np.array(samples_cost)
    det = evaluate_modes(case, modes)
    pct_time = float((samples_time <= det["time"]).mean() * 100)
    SUMMARY["robustness"] = dict(
        mc_samples=n,
        deterministic_time=round(det["time"], 2),
        mc_time_mean=round(float(samples_time.mean()), 2),
        mc_time_std=round(float(samples_time.std(ddof=1)), 2),
        mc_time_P90=round(float(np.percentile(samples_time, 90)), 2),
        deterministic_in_percentile=round(pct_time, 1),
        mc_cost_mean=round(float(samples_cost.mean()), 0),
        mc_cost_P90=round(float(np.percentile(samples_cost, 90)), 0),
    )
    log(f"    deterministic time {det['time']:.1f}d at P{pct_time:.0f}; "
        f"MC mean {samples_time.mean():.1f}d std {samples_time.std():.1f}")
    fig_robustness(samples_time, det["time"])


# =========================================================================
# [10] additional cases
# =========================================================================
def experiment_cases():
    log("=== [10] Additional case studies B and C ===")
    rows = []
    for key in ["A", "B", "C"]:
        case = CASES[key]()
        tour = run_tournament(case, pop=CFG["case_pop"], n_gen=CFG["case_gen"], seed=11)
        d = decide(tour["F"])
        best_modes = tour["X"][d["best"]]
        imp = improvement_table(case, best_modes)
        impd = {r[0]: r[4] for r in imp.values.tolist()}
        rows.append([case.name, case.n, tour["n_front"], round(d["consensus_best"], 4),
                     impd["Time"], impd["Cost"], impd["Resource Leveling"], impd["Emissions"]])
        log(f"    {case.name}: front={tour['n_front']} impr={impd}")
    df = pd.DataFrame(rows, columns=["Case", "Activities", "Front size",
                                     "Consensus", "Time%", "Cost%", "Resource%", "Emissions%"])
    savecsv(df, "table_cases.csv")
    SUMMARY["cases"] = df.values.tolist()
    return df


# =========================================================================
# FIGURES
# =========================================================================
ALGO_COLORS = {"NSGA-II": "#ff7f0e", "NSGA-III": "#2ca02c", "MOEA/D": "#1f77b4",
               "RVEA": "#d62728", "UNSGA-III": "#9467bd"}

def fig_pareto_by_algorithm(case, tour, best_idx):
    F = tour["F"]; src = tour["source"]
    pairs = [(0, 1, "Time vs Cost"), (0, 3, "Time vs Emissions"),
             (1, 3, "Cost vs Emissions"), (2, 3, "Res. Fluct. vs Emissions")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (a, b, title) in zip(axes.ravel(), pairs):
        for k, name in enumerate(ALGO_NAMES):
            m = src == k
            if m.sum():
                ax.scatter(F[m, a], F[m, b], s=18, c=ALGO_COLORS[name],
                           label=name, alpha=0.75, edgecolors="none")
        ax.scatter(F[best_idx, a], F[best_idx, b], marker="*", s=320, c="black",
                   label="Best Consensus", zorder=5)
        ax.set_xlabel(OBJ_LABELS[a]); ax.set_ylabel(OBJ_LABELS[b])
        ax.set_title(f"Combined non-dominated front: {title}", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_pareto_by_algorithm.png"),
                                    dpi=150, bbox_inches="tight"); plt.close(fig)

def fig_3d(F, best_idx):
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(F[:, 0], F[:, 1], F[:, 3], c=F[:, 2], cmap="viridis", s=20)
    ax.scatter(F[best_idx, 0], F[best_idx, 1], F[best_idx, 3], marker="*", s=300, c="red")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Cost ($)"); ax.set_zlabel("Carbon (kgCO2)")
    fig.colorbar(p, label="Resource Leveling", shrink=0.6)
    ax.set_title("3D Pareto front (colour = resource leveling)")
    fig.savefig(os.path.join(FIG, "fig3_pareto_3d.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

def fig_top20(F, d):
    order = np.argsort(-d["ranks"]["consensus"])[:20]
    norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(11, 6))
    for r, idx in enumerate(order):
        ax.plot(range(4), norm[idx], marker="o", alpha=0.5,
                color="crimson" if r == 0 else "steelblue",
                lw=2.4 if r == 0 else 1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(OBJ_LABELS, rotation=20)
    ax.set_ylabel("Normalized objective (0=best, 1=worst)")
    ax.set_title("Top-20 consensus solutions (red = Rank 1)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_top20.png"), dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

def fig_resource_hist(case, best_modes):
    b = baseline(case); o = evaluate_modes(case, best_modes)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, prof, title, col in [(axes[0], b["profile"], "Baseline (Mode 0)", "#999999"),
                                 (axes[1], o["profile"], "Optimized consensus", "#2c7fb8")]:
        ax.bar(range(len(prof)), prof, color=col)
        ax.axhline(prof.mean(), color="red", ls="--", label=f"mean={prof.mean():.1f}")
        ax.set_title(f"{title}\nstd={prof.std():.2f}, peak={prof.max():.0f}")
        ax.set_xlabel("Project day"); ax.legend()
    axes[0].set_ylabel("Labor demand (workers)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_resource_hist.png"), dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

def fig_correlation(F):
    pear = np.corrcoef(F, rowvar=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pear, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(OBJ_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(OBJ_LABELS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{pear[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(pear[i, j]) > 0.5 else "black")
    fig.colorbar(im, label="Pearson correlation"); ax.set_title("Objective correlation (front)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_correlation.png"), dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

def fig_shap(case, s, X):
    try:
        import shap
        plt.figure()
        shap.summary_plot(s["shap_values"], X, feature_names=s["feature_names"],
                          show=False, max_display=20)
        plt.title("SHAP summary: activity mode impact on consensus score")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig6_shap_summary.png"),
                                        dpi=150, bbox_inches="tight"); plt.close()
    except Exception as e:
        log(f"    SHAP plot fallback (bar): {e}")
        imp = s["importance"].head(20)[::-1]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(imp["Activity"], imp["Mean |SHAP|"], color="#1f77b4")
        ax.set_title("SHAP feature importance"); fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig6_shap_summary.png"), dpi=150,
                    bbox_inches="tight"); plt.close(fig)

def fig_hv_box(hv_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = ALGO_NAMES + ["Tournament"]
    ax.boxplot([hv_df[c] for c in cols], labels=cols)
    ax.set_ylabel("Hypervolume"); ax.set_title("Hypervolume across runs (higher = better)")
    plt.xticks(rotation=20); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_hypervolume_box.png"), dpi=150,
                bbox_inches="tight"); plt.close(fig)

def fig_sensitivity(spear):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(spear, bins=25, color="#5ab4ac", edgecolor="white")
    ax.set_xlabel("Spearman rank correlation vs nominal weights")
    ax.set_ylabel("Frequency")
    ax.set_title("Consensus-ranking stability under +/-15% weight perturbation")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_sensitivity.png"), dpi=150,
                                    bbox_inches="tight"); plt.close(fig)

def fig_robustness(samples_time, det_time):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.sort(samples_time); ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color="#2b8cbe")
    ax.axvline(det_time, color="red", ls="--", label=f"deterministic = {det_time:.1f} d")
    ax.set_xlabel("Project duration (days)"); ax.set_ylabel("Cumulative probability")
    ax.set_title("Fuzzy Monte-Carlo duration robustness of the selected schedule")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_robustness.png"), dpi=150,
                bbox_inches="tight"); plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    t_start = time.time()
    log(f"START full protocol (QUICK={QUICK})")
    case, tour, d, best_modes = experiment_primary()
    experiment_shap(case, tour, d)
    stat_df, hv_df = experiment_multirun()
    experiment_single_vs_tournament(hv_df)
    experiment_ablation(case, tour, d, best_modes)
    experiment_sensitivity(case, tour, d)
    experiment_correlation(tour)
    experiment_carbon(case, tour, d, best_modes)
    experiment_robustness(case, best_modes)
    experiment_cases()
    SUMMARY["_meta"] = dict(quick=QUICK, config=CFG,
                            total_runtime_min=round((time.time() - t_start) / 60, 1))
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(SUMMARY, f, indent=2)
    log(f"DONE in {(time.time()-t_start)/60:.1f} min. Results -> {RES}, figures -> {FIG}")

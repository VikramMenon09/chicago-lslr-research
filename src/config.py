"""
config.py
Single registry of every parameter used in the pipeline, with its source.
'EPA EA' = U.S. EPA (2024) Economic Analysis for the Final LCRI.
'Assumption' = modeling choice varied in the probabilistic sensitivity analysis.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
PAPER_DIR = ROOT / "paper"
SEED = 20261021


@dataclass(frozen=True)
class Inventory:
    n_zones: int = 160            # k-means corridors of action lines (~2,600 lines each)
    base_year: int = 2027         # LCRI compliance date (Nov. 1, 2027)
    horizon: int = 20             # Chicago's deferred LCRI schedule (~20 years)
    max_years_sq: int = 40        # simulation cap for schedules that miss the deadline


@dataclass(frozen=True)
class Costs:
    # Reported Chicago averages (Inside Climate News/WBEZ/Grist, June 2026):
    # about $31k overall, $34k block-level, $39k one-off.
    c_block_marginal: float = 31_000.0   # Assumption, calibrated: c_B + F/500 = $34k
    c_oneoff: float = 39_000.0           # reported
    mobilization: float = 1_500_000.0    # Assumption: $ per zone-year block mobilization
    min_block_batch: int = 250           # Assumption
    epa_national: float = 4_700.0        # EPA per-line estimate (reported)


@dataclass(frozen=True)
class Health:
    # Children: IQ = -beta * ln(BLL); Lanphear et al. (2005; erratum 2019), as in EPA EA
    iq_beta: float = 3.25
    iq_beta_low: float = 2.70            # original 2005 pooled coefficient
    child_bll0: float = 1.0              # ug/dL background without LSL (Assumption)
    child_dbll: float = 0.8              # ug/dL increment from LSL water (Assumption)
    window_yrs: float = 6.0              # ages 0-5
    value_per_iq: float = 42_226.0       # 2022$, EPA EA Exhibit 5-25 (2% rate, to age 7)
    value_per_iq_low: float = 38_404.0
    value_per_iq_high: float = 46_204.0
    addon_adhd_sped: float = 0.05        # ADHD + special education, share of IQ value (EPA EA ratio ~3%)
    # Adults: CVD premature mortality HIF, EPA EA Eq. 16 / Exhibit 5-31
    cvd_beta_low: float = 0.36           # Aoki et al. (2016), log10 scale
    cvd_beta_high: float = 0.96          # Lanphear et al. (2018), log10 scale
    adult_bll0: float = 1.0              # ug/dL (Assumption, NHANES order of magnitude)
    adult_dbll: float = 0.2              # ug/dL increment from LSL water (Assumption)
    cvd_mort_rate: float = 0.0033        # annual CVD deaths per adult (NCHS, ~2022)
    vsl: float = 12_980_000.0            # 2022$, EPA EA
    benefit_life_yrs: int = 30


@dataclass(frozen=True)
class Finance:
    social_rates: tuple = (0.02, 0.03, 0.07)   # OMB A-4 (2023: 2%; 2003: 3%, 7%)
    dwsrf_rate: float = 0.015            # Assumption: subsidized SRF loan
    muni_bond_rate: float = 0.050        # Assumption: water revenue bond
    loan_term: int = 30
    base_rate: float = 0.03


@dataclass(frozen=True)
class Operations:
    capacity: tuple = (15_000, 19_000, 22_000) + (24_000,) * 37   # first years = city targets
    max_block_zones_per_year: int = 34
    budget_start_musd: float = 450.0
    budget_step_musd: float = 40.0
    budget_cap_musd: float = 820.0
    pace_slack: float = 0.06
    ej_svi_quantile: float = 0.60


@dataclass(frozen=True)
class Equity:
    lambda_equity: float = 4_000.0       # $ per SVI-weighted line-year of excess wait
    convex_power: float = 3.0            # SVI^p weights for the convex variant
    eta: float = 1.0                     # inequality aversion for distributional weights


INV, COST, HEALTH, FIN, OPS, EQ = Inventory(), Costs(), Health(), Finance(), Operations(), Equity()

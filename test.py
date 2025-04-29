#!/usr/bin/env python3
"""
test.py  —  Balance-check for the Canadian election simulator
(updated after first tuning pass).

Quick usage
-----------
# Analytical expectation only
python test.py

# 1 000 Monte-Carlo simulations
python test.py --runs 1000

# Experiment with a sharper preference curve
python test.py --beta 4
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

# --------------------------------------------------------------------------
#  CONSTANTS  (mirror main app)
# --------------------------------------------------------------------------
SHAPEFILE_PATH = Path("data") / "FED_CA_2021_EN.shp"

# Sharper party preference curves
BETA = 3.0          # default; may override with --beta

SIZE_SCALE = 0.25   # bigger urban-rural separation

# Province baselines (mean will be re-zeroed after size modifier)
PROV_LEAN = {
    10: -0.10, 11: -0.05, 12: -0.05, 13: -0.05,
    24: -0.20,             35: -0.02,
    46:  0.20, 47:  0.30,  48:  0.40, 59:  0.10,
    60: -0.05, 61: -0.05,  62: -0.05,
}

# Global slider defaults (must sum to 1.0)
DEFAULT_GLOBAL = {
    "Liberal":      0.30,
    "Conservative": 0.30,
    "NDP":          0.188,
    "Bloc":         0.08,
    "Green":        0.05,
    "PPC":          0.08,
    "Independent":  0.002,
}

PARTY_IDEAL = {
    "NDP": -0.9, "Green": -0.8, "Liberal": -0.365, "Bloc": -0.25,
    "Independent": 0.0, "Conservative": 0.44, "PPC": 1.0,
}

# Bloc multipliers
BQ_BOOST_QC     = 3.0
BQ_SUPPRESS_ROC = 0.2

# Independent cap (as fraction of residual probability)
IND_RESIDUAL_CAP = 0.15


# --------------------------------------------------------------------------
#  HELPERS
# --------------------------------------------------------------------------
def load_ridings() -> gpd.GeoDataFrame:
    """Load shapefile and compute LEAN (province + size)."""
    gdf = gpd.read_file(SHAPEFILE_PATH).to_crs(4326)
    gdf = gdf.rename(columns={"ED_NAMEE": "FEDNAME"})[["FED_NUM", "FEDNAME", "geometry"]]
    gdf["PRCODE"] = (gdf["FED_NUM"] // 1000).astype(int)

    # Urban/rural modifier
    proj = gdf.to_crs(3347)  # StatsCan Lambert
    z = (np.log(proj.geometry.area) - np.log(proj.geometry.area).mean()) / np.log(proj.geometry.area).std()
    size_mod = -SIZE_SCALE * z

    gdf["LEAN"] = gdf["PRCODE"].map(PROV_LEAN) + size_mod
    gdf["LEAN"] = gdf["LEAN"] - gdf["LEAN"].mean()  # centre to 0

    return gdf


def make_weights(ridings: gpd.GeoDataFrame, globals_: dict[str, float], beta: float) -> tuple[np.ndarray, list[str]]:
    """Return N×P probability matrix incorporating Bloc/Independent tweaks."""
    parties = list(globals_.keys())
    lean = ridings["LEAN"].values.reshape(-1, 1)
    party_ideal = np.array([PARTY_IDEAL[p] for p in parties])
    base = np.array([globals_[p] for p in parties])

    w = base * np.exp(-beta * (lean - party_ideal) ** 2)

    # Bloc boost/suppression
    if "Bloc" in parties:
        idx_bq = parties.index("Bloc")
        in_qc = ridings["PRCODE"] == 24
        w[in_qc, idx_bq] *= BQ_BOOST_QC
        w[~in_qc, idx_bq] *= BQ_SUPPRESS_ROC

    # Independent damp: limit to fraction of residual
    if "Independent" in parties:
        idx_ind = parties.index("Independent")
        residual = w.sum(axis=1) - w[:, idx_ind]
        w[:, idx_ind] = np.minimum(w[:, idx_ind], residual * IND_RESIDUAL_CAP)

    # Row-normalise
    w = w / w.sum(axis=1, keepdims=True)
    return w, parties


def analytical_expectation(w: np.ndarray, parties: list[str]) -> pd.Series:
    return pd.Series(w.sum(axis=0), index=parties)


def monte_carlo(w: np.ndarray, parties: list[str], runs: int, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    winners = rng.choice(len(parties), size=(runs, w.shape[0]), p=None)
    # Replace vectorised choice with row-wise due to varying probs
    winners = np.vstack([rng.choice(len(parties), p=row) for row in w] for _ in range(runs))
    seat_totals = np.apply_along_axis(lambda r: np.bincount(r, minlength=len(parties)), axis=1, arr=winners)
    return pd.DataFrame(seat_totals, columns=parties)


# --------------------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=0, help="Monte-Carlo simulations to run (0 = expectation only)")
    parser.add_argument("--beta", type=float, default=BETA, help="Override BETA preference-width")
    args = parser.parse_args()

    ridings = load_ridings()
    weights, parties = make_weights(ridings, DEFAULT_GLOBAL, beta=args.beta)

    print("\n=== Analytical expected seats ===")
    exp = analytical_expectation(weights, parties).round(2)
    print(exp.sort_values(ascending=False))

    seat_mat = pd.DataFrame(weights, columns=parties)
    seat_mat["prov"] = ridings["PRCODE"].values
    prov = seat_mat.groupby("prov")[parties].sum().round(2)

    print("\n=== Expected seats by province code ===")
    print(prov)

    if args.runs:
        print(f"\nRunning {args.runs:,} Monte-Carlo simulations …")
        rng = np.random.default_rng()
        seat_totals = np.zeros((args.runs, len(parties)), dtype=int)
        for i in range(args.runs):
            winners = [rng.choice(len(parties), p=row) for row in weights]
            seat_totals[i] = np.bincount(winners, minlength=len(parties))
        df = pd.DataFrame(seat_totals, columns=parties)
        summary = pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=0)}).round(2)
        print("\n=== Simulated seats (mean ± σ) ===")
        print(summary.sort_values("mean", ascending=False))


if __name__ == "__main__":
    main()

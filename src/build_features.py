"""Compute the 158 Alpha158 features for universe members -> <market data>/alpha158.parquet

    python build_features.py                  # China
    python build_features.py --market india   # India
"""
import markets
from data import load_market
from features import build_matrix

if __name__ == "__main__":
    cfg = markets.from_argv()
    out = cfg.proc / "alpha158.parquet"
    X = build_matrix(load_market())
    X.to_parquet(out)
    print(f"    {X.shape[0]:,} rows x {X.shape[1]} features -> {out}")

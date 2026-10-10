"""Compute the 158 Alpha158 features for CSI300 members -> data/processed/alpha158.parquet"""
from pathlib import Path

from data import load_market
from features import build_matrix

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "alpha158.parquet"

if __name__ == "__main__":
    X = build_matrix(load_market())
    X.to_parquet(OUT)
    print(f"    {X.shape[0]:,} rows x {X.shape[1]} features -> {OUT}")

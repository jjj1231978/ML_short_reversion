"""
S&P 500 Sample Statistics — Data Pipeline Validation
=====================================================
Run: python notebooks/01_sample_statistics.py
Or use in Jupyter/IPython for interactive exploration.

Fetches price + fundamental data, computes all factors,
and produces diagnostic plots of the feature distributions.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import RAW_DIR, load_config
from src.data.fetch import fetch_price_data, fetch_fundamentals_simfin
from src.data.universe import get_sp500_constituents, filter_by_gics_sector
from src.features.build import build_feature_matrix, build_target, resample_to_wednesday
from src.features import factors

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
cfg = load_config()
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# 1. Universe
# --------------------------------------------------------------------------
print("=" * 60)
print("1. UNIVERSE")
print("=" * 60)

constituents = get_sp500_constituents()
sector_map = dict(zip(constituents["Symbol"], constituents["GICS Sector"]))
all_tickers = constituents["Symbol"].tolist()
tickers = filter_by_gics_sector(all_tickers, sector_map, exclude_sectors=["Financials"])

print(f"Total S&P 500 constituents: {len(all_tickers)}")
print(f"After excluding Financials: {len(tickers)}")
print(f"\nSector breakdown:")
print(constituents["GICS Sector"].value_counts().to_string())

# Sector distribution chart
fig, ax = plt.subplots(figsize=(10, 5))
sector_counts = constituents["GICS Sector"].value_counts()
colors = ["salmon" if s == "Financials" else "steelblue" for s in sector_counts.index]
sector_counts.plot.barh(ax=ax, color=colors)
ax.set_xlabel("Number of Stocks")
ax.set_title("S&P 500 Sector Distribution (red = excluded Financials)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sector_distribution.png", dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'sector_distribution.png'}")

# --------------------------------------------------------------------------
# 2. Price Data
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. PRICE DATA")
print("=" * 60)

price_data = fetch_price_data(
    tickers,
    start=cfg["data"]["start_date"],
    end=cfg["data"]["end_date"],
    dataset=cfg["data"]["databento_dataset"],
)
close = price_data["Close"]
volume = price_data["Volume"]

print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")
print(f"Trading days: {close.shape[0]}")
print(f"Tickers with data: {close.shape[1]}")
print(f"Missing data %: {close.isna().mean().mean() * 100:.1f}%")

# Data coverage over time
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

non_null_pct = close.notna().sum(axis=1) / close.shape[1] * 100
axes[0].plot(close.index, non_null_pct, color="steelblue", linewidth=0.8)
axes[0].set_ylabel("% Tickers with Data")
axes[0].set_title("Data Coverage Over Time")
axes[0].set_ylim(0, 105)

# Average daily dollar volume
avg_dv = (close * volume).mean(axis=1) / 1e6
axes[1].plot(close.index, avg_dv.rolling(21).mean(), color="darkgreen", linewidth=0.8)
axes[1].set_ylabel("Avg Dollar Volume ($M)")
axes[1].set_title("Average Daily Dollar Volume (21-day MA)")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "data_coverage.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'data_coverage.png'}")

# --------------------------------------------------------------------------
# 3. Price Factor Distributions
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. PRICE-BASED FACTOR DISTRIBUTIONS")
print("=" * 60)

# Compute factors on daily data, sample at Wednesdays
factor_data = {
    "R1W": factors.r1w(close).pipe(resample_to_wednesday),
    "RSI5D": factors.rsi(close, window=5).pipe(resample_to_wednesday),
    "RSI14": factors.rsi(close, window=14).pipe(resample_to_wednesday),
    "R3M1M": factors.momentum(close, lookback=63, skip=21).pipe(resample_to_wednesday),
    "R12M1M": factors.momentum(close, lookback=252, skip=21).pipe(resample_to_wednesday),
    "VOL6M": factors.rolling_volatility(close, window=126).pipe(resample_to_wednesday),
}

# Summary statistics table
stats_rows = []
for name, df in factor_data.items():
    flat = df.stack().dropna()
    stats_rows.append({
        "Factor": name,
        "Count": len(flat),
        "Mean": flat.mean(),
        "Std": flat.std(),
        "Min": flat.min(),
        "5%": flat.quantile(0.05),
        "25%": flat.quantile(0.25),
        "Median": flat.median(),
        "75%": flat.quantile(0.75),
        "95%": flat.quantile(0.95),
        "Max": flat.max(),
    })

stats_df = pd.DataFrame(stats_rows).set_index("Factor")
print("\nWeekly Cross-Sectional Factor Statistics:")
print(stats_df.to_string(float_format=lambda x: f"{x:.4f}"))

# Histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (name, df) in zip(axes.flat, factor_data.items()):
    flat = df.stack().dropna()
    # Clip extreme tails for visualization
    lo, hi = flat.quantile(0.01), flat.quantile(0.99)
    clipped = flat.clip(lo, hi)
    ax.hist(clipped, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(flat.median(), color="red", linestyle="--", linewidth=1, label=f"median={flat.median():.3f}")
    ax.set_title(name)
    ax.legend(fontsize=8)

plt.suptitle("Price Factor Distributions (Weekly, Cross-Sectional)", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_factor_distributions.png", dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'price_factor_distributions.png'}")

# Time-series of cross-sectional means
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
for ax, (name, df) in zip(axes.flat, factor_data.items()):
    cs_mean = df.mean(axis=1)
    cs_std = df.std(axis=1)
    ax.plot(cs_mean.index, cs_mean, color="steelblue", linewidth=0.8, label="mean")
    ax.fill_between(cs_mean.index, cs_mean - cs_std, cs_mean + cs_std, alpha=0.2, color="steelblue")
    ax.set_title(name)
    ax.legend(fontsize=8)

plt.suptitle("Cross-Sectional Mean +/- 1 Std Over Time", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_factor_timeseries.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'price_factor_timeseries.png'}")

# --------------------------------------------------------------------------
# 4. Cross-Sectional Correlations
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. FACTOR CORRELATIONS")
print("=" * 60)

# Build a single date snapshot for correlation (use most recent complete week)
latest_date = None
for d in reversed(factor_data["R1W"].index):
    non_null = sum(1 for name in factor_data if factor_data[name].loc[d].notna().sum() > 50)
    if non_null == len(factor_data):
        latest_date = d
        break

if latest_date is not None:
    snapshot = pd.DataFrame({name: df.loc[latest_date] for name, df in factor_data.items()}).dropna()
    corr = snapshot.corr()
    print(f"\nCross-sectional correlation (snapshot {latest_date.date()}):")
    print(corr.to_string(float_format=lambda x: f"{x:.2f}"))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                vmin=-1, vmax=1, square=True)
    ax.set_title(f"Factor Correlation Matrix ({latest_date.date()})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "factor_correlation.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'factor_correlation.png'}")

# Average rolling correlation over time
print("\nComputing average rolling cross-sectional correlations...")
stacked_factors = pd.concat(
    {name: df.stack().rename(name) for name, df in factor_data.items()},
    axis=1,
)
stacked_factors.index.names = ["date", "ticker"]

# Sample 52 evenly spaced weeks for rolling correlation
dates = factor_data["R1W"].dropna(how="all").index
sample_dates = dates[::max(1, len(dates) // 52)]
corr_over_time = []
for d in sample_dates:
    snap = pd.DataFrame({name: df.loc[d] for name, df in factor_data.items()}).dropna()
    if len(snap) > 30:
        c = snap.corr()
        # Extract upper triangle
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                corr_over_time.append({
                    "date": d,
                    "pair": f"{c.index[i]} / {c.columns[j]}",
                    "corr": c.iloc[i, j],
                })

if corr_over_time:
    corr_ts = pd.DataFrame(corr_over_time)
    fig, ax = plt.subplots(figsize=(12, 6))
    for pair, grp in corr_ts.groupby("pair"):
        ax.plot(grp["date"], grp["corr"], linewidth=0.8, alpha=0.7, label=pair)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cross-Sectional Correlation")
    ax.set_title("Pairwise Factor Correlations Over Time")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "factor_correlation_timeseries.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'factor_correlation_timeseries.png'}")

# --------------------------------------------------------------------------
# 5. Target Variable
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. TARGET VARIABLE (Forward 1-Week Return)")
print("=" * 60)

target = build_target(close)
target_flat = target["target"].dropna()

print(f"Observations: {len(target_flat):,}")
print(f"Mean: {target_flat.mean():.4f}")
print(f"Std:  {target_flat.std():.4f}")
print(f"Skew: {target_flat.skew():.4f}")
print(f"Kurt: {target_flat.kurtosis():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Distribution
axes[0].hist(target_flat.clip(-4, 4), bins=100, color="steelblue", alpha=0.7, edgecolor="none")
axes[0].set_xlabel("Z-scored Forward 1W Return")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Target Distribution (clipped at +/-4)")

# Time series of cross-sectional dispersion
weekly_std = target["target"].groupby(level="date").std()
axes[1].plot(weekly_std.index, weekly_std, color="darkgreen", linewidth=0.8)
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Cross-Sectional Std")
axes[1].set_title("Weekly Return Dispersion Over Time")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'target_distribution.png'}")

# --------------------------------------------------------------------------
# 6. Fundamental Data (if available)
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. FUNDAMENTAL DATA")
print("=" * 60)

try:
    fundamentals = fetch_fundamentals_simfin(
        tickers=tickers,
        pub_lag_days=cfg["data"]["fundamental_pub_lag_days"],
    )

    if fundamentals:
        fund_stats = []
        for name, df in fundamentals.items():
            flat = df.stack().dropna()
            fund_stats.append({
                "Factor": name,
                "Count": len(flat),
                "Tickers": df.shape[1],
                "Mean": flat.mean(),
                "Std": flat.std(),
                "5%": flat.quantile(0.05),
                "Median": flat.median(),
                "95%": flat.quantile(0.95),
            })

        fund_stats_df = pd.DataFrame(fund_stats).set_index("Factor")
        print("\nFundamental Factor Statistics:")
        print(fund_stats_df.to_string(float_format=lambda x: f"{x:.4f}"))

        # Overlap check
        price_tickers = set(close.columns)
        for name, df in fundamentals.items():
            fund_tickers = set(df.columns)
            overlap = price_tickers & fund_tickers
            only_price = price_tickers - fund_tickers
            print(f"\n{name}: {len(overlap)} tickers overlap with price data, {len(only_price)} price-only")

        # Fundamental factor distributions
        n_funds = len(fundamentals)
        if n_funds > 0:
            fig, axes = plt.subplots(1, n_funds, figsize=(5 * n_funds, 5))
            if n_funds == 1:
                axes = [axes]
            for ax, (name, df) in zip(axes, fundamentals.items()):
                flat = df.stack().dropna()
                lo, hi = flat.quantile(0.02), flat.quantile(0.98)
                clipped = flat.clip(lo, hi)
                ax.hist(clipped, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
                ax.axvline(flat.median(), color="red", linestyle="--", linewidth=1)
                ax.set_title(f"{name} (median={flat.median():.2f})")

            plt.suptitle("Fundamental Factor Distributions (2%/98% winsorized)", fontsize=13)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "fundamental_distributions.png", dpi=150)
            plt.close()
            print(f"\nSaved: {OUTPUT_DIR / 'fundamental_distributions.png'}")
    else:
        print("No fundamental data returned — check SimFin API key and connectivity.")

except Exception as e:
    print(f"Fundamental data fetch failed: {e}")
    print("Skipping fundamental diagnostics. Pipeline will still work with price-only factors.")

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Universe: {len(tickers)} tickers (S&P 500 ex-Financials)")
print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")
print(f"Price factors: {list(factor_data.keys())}")
print(f"Fundamental factors: {list(fundamentals.keys()) if 'fundamentals' in dir() and fundamentals else 'N/A'}")
print(f"Target obs: {len(target_flat):,}")
print(f"\nAll figures saved to: {OUTPUT_DIR}")

"""
KNN Outlier Scoring — Ablation Test Pipeline
=============================================
Tests the contribution of each feature group to the KNN outlier scores
produced from the slim transaction feature set.

Design summary
--------------
KNN outlier score for transaction i = mean distance to its k nearest
neighbours in the feature space (higher score = more anomalous).

Ablation strategy
-----------------
Three complementary lenses, all unlabelled-friendly:

  1. Group-leave-one-out  — drop one feature GROUP at a time, compare
                            score ranking via Spearman correlation and
                            top-N overlap against the full-feature baseline.

  2. Window sensitivity   — for windowed groups (amount, receiver, portfolio)
                            test each lookback window individually to find
                            which horizon (7d / 30d / 90d) drives the signal.

  3. Synthetic anomaly    — inject known anomalies, measure detection-rate
                            drop when each group is removed (precision@K proxy).

Feature groups (matching slim pipeline output)
----------------------------------------------
  AMOUNT      — amt_*_zscore, amt_*_vs_median, amt_*_pct_rank, is_round_amount
  TEMPORAL    — is_weekend, is_off_hours, seconds_since_last_tx,
                gap_zscore_30d, tx_count_24h, velocity_7d_vs_30d
  RECEIVER    — recv_*_is_new, recv_*_tx_count, recv_*_amt_zscore,
                recv_*_unique, recv_*_herfindahl
  CURRENCY    — ccy_is_new_*, currency_switched
  PORTFOLIO   — port_*_vol_zscore, port_*_tx_vol_share,
                tx_share_of_daily_vol, vol_7d_vs_30d_ratio
  COMPOSITE   — flag_large_amt_new_recv, flag_new_ccy_new_recv,
                flag_burst_new_recv
  TX_COUNT    — amt_*_tx_count  (activity volume — tested separately because
                it is present in both amount and receiver groups)

Usage
-----
    from knn_ablation import AblationPipeline

    pipeline = AblationPipeline(features_df, k=10, top_n=20, n_synthetic=30)
    report   = pipeline.run()
    pipeline.print_report(report)
    pipeline.plot_report(report)          # requires matplotlib
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Feature group definitions
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_groups(columns: List[str]) -> Dict[str, List[str]]:
    """
    Dynamically assign every feature column to exactly one group,
    based on the slim-pipeline naming convention.
    Returns {group_name: [col, ...]} keeping only columns present in data.
    """
    def pick(patterns):
        return [c for c in columns if any(c.startswith(p) or p in c for p in patterns)]

    # Order matters — more specific patterns first
    groups = {
        "AMOUNT":    pick(["amt_", "is_round_amount"]),
        "RECEIVER":  pick(["recv_"]),
        "PORTFOLIO": pick(["port_", "tx_share_of_daily_vol", "vol_7d_vs_30d"]),
        "TEMPORAL":  pick(["is_weekend", "is_off_hours", "seconds_since_last_tx",
                           "gap_zscore", "tx_count_24h", "velocity_7d_vs_30d"]),
        "CURRENCY":  pick(["ccy_", "currency_switched"]),
        "COMPOSITE": pick(["flag_"]),
    }

    # Remove tx_count columns from AMOUNT (they measure activity, not amount anomaly)
    # and expose as their own group so we can test them independently
    tx_count_cols = [c for c in groups["AMOUNT"] if c.endswith("_tx_count")]
    groups["AMOUNT"]   = [c for c in groups["AMOUNT"] if c not in tx_count_cols]
    groups["TX_COUNT"] = tx_count_cols

    # Deduplicate: if a column matches multiple groups keep the first match
    seen = set()
    for name in list(groups):
        groups[name] = [c for c in groups[name] if c not in seen]
        seen.update(groups[name])
        if not groups[name]:
            del groups[name]

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# KNN scorer
# ─────────────────────────────────────────────────────────────────────────────

def _knn_outlier_scores(X: np.ndarray, k: int) -> np.ndarray:
    """
    Mean distance to k nearest neighbours (excluding self).
    Higher score = more isolated = more anomalous.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    return distances[:, 1:].mean(axis=1)   # drop col 0 (self, distance=0)


def _prepare_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Scale selected feature columns; impute NaNs with column median."""
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    return RobustScaler().fit_transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic anomaly injection
# ─────────────────────────────────────────────────────────────────────────────

def _inject_synthetic_anomalies(
    df: pd.DataFrame,
    feature_cols: List[str],
    n: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create n synthetic anomalies by perturbing real transactions.

    Perturbation strategy (combines multiple signals so anomalies are
    detectable regardless of which group is ablated):
      - amount features   → multiply by 10–20×
      - receiver features → set is_new flags to 1, zero tx_counts
      - currency features → set is_new flags to 1
      - temporal features → set is_off_hours=1, is_weekend=1
      - portfolio features→ set vol_share to 0.99 (dominates the day)
      - composite flags   → set all to 1
    """
    base_idx   = rng.choice(len(df), size=n, replace=False)
    synth      = df.iloc[base_idx][feature_cols].copy().reset_index(drop=True)

    for col in synth.columns:
        if "amt_" in col and "zscore" in col:
            synth[col] = synth[col].fillna(0) + rng.uniform(8, 15, n)
        elif "amt_" in col and "vs_median" in col:
            synth[col] = rng.uniform(10, 20, n)
        elif "amt_" in col and "pct_rank" in col:
            synth[col] = rng.uniform(0.97, 1.0, n)
        elif col == "is_round_amount":
            synth[col] = 1
        elif "recv_" in col and "is_new" in col:
            synth[col] = 1
        elif "recv_" in col and "tx_count" in col:
            synth[col] = 0
        elif "recv_" in col and "amt_zscore" in col:
            synth[col] = synth[col].fillna(0) + rng.uniform(8, 15, n)
        elif "recv_" in col and "unique" in col:
            synth[col] = synth[col].fillna(1) * rng.uniform(3, 6, n)
        elif "ccy_" in col and "is_new" in col:
            synth[col] = 1
        elif col == "currency_switched":
            synth[col] = 1
        elif col in ("is_off_hours", "is_weekend"):
            synth[col] = 1
        elif "vol_zscore" in col:
            synth[col] = synth[col].fillna(0) + rng.uniform(8, 15, n)
        elif "vol_share" in col or "tx_share" in col:
            synth[col] = rng.uniform(0.85, 0.99, n)
        elif col.startswith("flag_"):
            synth[col] = 1

    combined       = pd.concat([df[feature_cols].reset_index(drop=True), synth],
                                ignore_index=True)
    is_synthetic   = np.zeros(len(combined), dtype=bool)
    is_synthetic[len(df):] = True
    return combined, is_synthetic


# ─────────────────────────────────────────────────────────────────────────────
# Ablation pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AblationPipeline:
    """
    Parameters
    ----------
    features_df   : output of TransactionFeaturePipeline.fit_transform()
    k             : number of nearest neighbours for KNN scoring
    top_n         : number of top-scored transactions to track for overlap metric
    n_synthetic   : synthetic anomalies to inject for detection-rate metric
    seed          : random seed (for synthetic injection reproducibility)
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        k: int = 10,
        top_n: int = 20,
        n_synthetic: int = 30,
        seed: int = 42,
    ):
        self.df          = features_df
        self.k           = k
        self.top_n       = top_n
        self.n_synthetic = n_synthetic
        self.rng         = np.random.default_rng(seed)

        # Identify feature columns (everything that isn't raw input)
        raw_cols = {"transaction_id", "date", "timestamp",
                    "debit_amount", "receiver_id", "currency"}
        self.feature_cols = [c for c in features_df.columns if c not in raw_cols]

        self.groups = _resolve_groups(self.feature_cols)
        self._validate()

    def _validate(self):
        ungrouped = [c for c in self.feature_cols
                     if not any(c in g for g in self.groups.values())]
        if ungrouped:
            print(f"[ablation] Warning: {len(ungrouped)} columns not assigned to any "
                  f"group — they will always be included:\n  {ungrouped}")

    # ── public ────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Run all three ablation lenses.
        Returns a nested dict consumed by print_report() and plot_report().
        """
        print(f"[ablation] Feature groups:")
        for g, cols in self.groups.items():
            print(f"  {g:12s} ({len(cols):2d} cols)")
        print(f"[ablation] k={self.k}, top_n={self.top_n}, "
              f"n_synthetic={self.n_synthetic}\n")

        # Baseline scores on real transactions
        X_full        = _prepare_matrix(self.df, self.feature_cols)
        baseline_scores = _knn_outlier_scores(X_full, self.k)

        results = {
            "baseline_scores":  baseline_scores,
            "groups":           self.groups,
            "group_ablation":   self._run_group_ablation(baseline_scores),
            "window_sensitivity": self._run_window_sensitivity(baseline_scores),
            "synthetic":        self._run_synthetic_ablation(),
        }
        return results

    # ── Lens 1: group leave-one-out ──────────────────────────────────────────

    def _run_group_ablation(self, baseline_scores: np.ndarray) -> pd.DataFrame:
        """Drop one group at a time; measure score-rank shift."""
        print("[ablation] Running group leave-one-out …")
        baseline_top = set(np.argsort(baseline_scores)[-self.top_n:])
        rows = []

        for group_name, drop_cols in self.groups.items():
            kept = [c for c in self.feature_cols if c not in drop_cols]
            if len(kept) == 0:
                continue

            X_ablated = _prepare_matrix(self.df, kept)
            scores    = _knn_outlier_scores(X_ablated, self.k)

            spearman_r, _   = spearmanr(baseline_scores, scores)
            ablated_top      = set(np.argsort(scores)[-self.top_n:])
            top_n_overlap    = len(baseline_top & ablated_top) / self.top_n
            mean_score_delta = float(np.mean(np.abs(scores - baseline_scores)))

            rows.append({
                "group":             group_name,
                "n_features_dropped": len(drop_cols),
                "spearman_r":        round(spearman_r, 4),
                "top_n_overlap":     round(top_n_overlap, 4),
                "mean_score_delta":  round(mean_score_delta, 6),
                "importance":        round((1 - spearman_r) + (1 - top_n_overlap), 4),
            })

        df_out = pd.DataFrame(rows).sort_values("importance", ascending=False)
        df_out["importance_rank"] = range(1, len(df_out) + 1)
        return df_out

    # ── Lens 2: window sensitivity ───────────────────────────────────────────

    def _run_window_sensitivity(self, baseline_scores: np.ndarray) -> pd.DataFrame:
        """
        For each windowed group, test dropping each window individually.
        Windowed groups: AMOUNT, RECEIVER, PORTFOLIO.
        """
        print("[ablation] Running window sensitivity …")
        windowed_groups = {
            g: cols for g, cols in self.groups.items()
            if any(f"_{w}d_" in c or f"_{w}d" in c
                   for c in cols for w in [7, 30, 90])
        }

        rows = []
        baseline_top = set(np.argsort(baseline_scores)[-self.top_n:])

        for group_name, all_cols in windowed_groups.items():
            for window in [7, 30, 90]:
                window_cols = [c for c in all_cols
                               if f"_{window}d_" in c or c.endswith(f"_{window}d")]
                if not window_cols:
                    continue

                kept      = [c for c in self.feature_cols if c not in window_cols]
                X_ablated = _prepare_matrix(self.df, kept)
                scores    = _knn_outlier_scores(X_ablated, self.k)

                spearman_r, _ = spearmanr(baseline_scores, scores)
                ablated_top   = set(np.argsort(scores)[-self.top_n:])
                top_n_overlap = len(baseline_top & ablated_top) / self.top_n

                rows.append({
                    "group":          group_name,
                    "window_days":    window,
                    "n_cols_dropped": len(window_cols),
                    "spearman_r":     round(spearman_r, 4),
                    "top_n_overlap":  round(top_n_overlap, 4),
                    "importance":     round((1 - spearman_r) + (1 - top_n_overlap), 4),
                })

        return pd.DataFrame(rows).sort_values(
            ["group", "importance"], ascending=[True, False]
        )

    # ── Lens 3: synthetic anomaly detection rate ─────────────────────────────

    def _run_synthetic_ablation(self) -> pd.DataFrame:
        """
        Inject synthetic anomalies, measure how detection rate drops
        when each feature group is removed.
        Metric: recall@top_n = fraction of synthetics in top-N scored transactions.
        """
        print("[ablation] Running synthetic anomaly ablation …")

        # Baseline with all features
        combined_full, is_synth = _inject_synthetic_anomalies(
            self.df, self.feature_cols, self.n_synthetic, self.rng
        )
        X_full    = _prepare_matrix(combined_full, self.feature_cols)
        scores_full = _knn_outlier_scores(X_full, self.k)
        baseline_recall = _recall_at_k(scores_full, is_synth, self.top_n)

        rows = [{"group": "BASELINE (all features)",
                 "recall_at_top_n":  round(baseline_recall, 4),
                 "recall_drop":      0.0,
                 "recall_drop_pct":  0.0}]

        for group_name, drop_cols in self.groups.items():
            kept = [c for c in self.feature_cols if c not in drop_cols]
            if not kept:
                continue

            combined_abl, is_synth_abl = _inject_synthetic_anomalies(
                self.df, kept, self.n_synthetic, self.rng
            )
            X_abl   = _prepare_matrix(combined_abl, kept)
            scores  = _knn_outlier_scores(X_abl, self.k)
            recall  = _recall_at_k(scores, is_synth_abl, self.top_n)

            drop     = baseline_recall - recall
            drop_pct = drop / baseline_recall * 100 if baseline_recall > 0 else 0.0

            rows.append({
                "group":            group_name,
                "recall_at_top_n":  round(recall, 4),
                "recall_drop":      round(drop, 4),
                "recall_drop_pct":  round(drop_pct, 1),
            })

        return (pd.DataFrame(rows)
                .sort_values("recall_drop", ascending=False)
                .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _recall_at_k(scores: np.ndarray, is_synthetic: np.ndarray, k: int) -> float:
    """Fraction of synthetic anomalies that fall in the top-k scored rows."""
    top_k_idx      = set(np.argsort(scores)[-k:])
    synth_idx      = set(np.where(is_synthetic)[0])
    hits           = top_k_idx & synth_idx
    return len(hits) / len(synth_idx) if len(synth_idx) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: Dict):
    """Pretty-print all three ablation lenses to stdout."""
    sep = "─" * 70

    print(f"\n{'KNN ABLATION REPORT':^70}")
    print(sep)

    # ── Lens 1 ──
    print("\n📊 LENS 1 — Group leave-one-out")
    print("   Metrics when each group is removed from the full feature set.")
    print("   spearman_r    : rank correlation with baseline scores (1 = identical)")
    print("   top_n_overlap : fraction of top-N flagged txs that survive removal")
    print("   importance    : composite score = (1-spearman) + (1-overlap); higher = more important\n")
    print(report["group_ablation"].to_string(index=False))

    # ── Lens 2 ──
    print(f"\n{sep}")
    print("\n🔍 LENS 2 — Window sensitivity (within windowed groups)")
    print("   Shows which lookback horizon (7d / 30d / 90d) carries the signal.\n")
    print(report["window_sensitivity"].to_string(index=False))

    # ── Lens 3 ──
    print(f"\n{sep}")
    print(f"\n🎯 LENS 3 — Synthetic anomaly detection (top_n={report['synthetic'].at[0,'recall_at_top_n']:.0%} baseline)")
    print("   recall_at_top_n : fraction of injected anomalies in top-N scored txs")
    print("   recall_drop     : how much recall falls when this group is removed\n")
    print(report["synthetic"].to_string(index=False))

    # ── Summary ──
    print(f"\n{sep}")
    print("\n✅ RECOMMENDED PRUNING CANDIDATES")
    print("   Groups that are LOW importance AND LOW recall_drop:\n")

    grp_abl  = report["group_ablation"].set_index("group")
    synth    = report["synthetic"].set_index("group")

    for group in grp_abl.index:
        imp    = grp_abl.at[group, "importance"]
        recall = synth.at[group, "recall_drop"] if group in synth.index else np.nan
        if imp < 0.15 and (np.isnan(recall) or recall < 0.05):
            n_feat = grp_abl.at[group, "n_features_dropped"]
            print(f"  → {group} (importance={imp:.3f}, recall_drop={recall:.3f}, "
                  f"cols={n_feat})")

    print()


def plot_report(report: Dict, figsize=(14, 10)):
    """
    Four-panel summary plot.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[plot] matplotlib not available — skipping plot.")
        return

    grp_abl   = report["group_ablation"]
    win_sens  = report["window_sensitivity"]
    synth     = report["synthetic"]
    baseline  = report["baseline_scores"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("KNN Ablation Report", fontsize=14, fontweight="bold")

    # ── Panel A: Spearman r by group ─────────────────────────────────────────
    ax = axes[0, 0]
    colors = ["#d62728" if r < 0.95 else "#2ca02c"
              for r in grp_abl["spearman_r"]]
    ax.barh(grp_abl["group"], grp_abl["spearman_r"], color=colors)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0.95, color="orange", linestyle=":", linewidth=0.8, label="threshold 0.95")
    ax.set_xlim(max(0, grp_abl["spearman_r"].min() - 0.05), 1.02)
    ax.set_xlabel("Spearman r (higher = less important)")
    ax.set_title("A — Score-rank correlation after group removal")
    ax.legend(fontsize=8)

    # ── Panel B: top-N overlap by group ──────────────────────────────────────
    ax = axes[0, 1]
    colors = ["#d62728" if o < 0.80 else "#2ca02c"
              for o in grp_abl["top_n_overlap"]]
    ax.barh(grp_abl["group"], grp_abl["top_n_overlap"], color=colors)
    ax.axvline(0.80, color="orange", linestyle=":", linewidth=0.8, label="threshold 0.80")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"Overlap (fraction of top-{report['group_ablation']['n_features_dropped'].sum()} txs surviving)")
    ax.set_title("B — Top-N flagged tx overlap after group removal")
    ax.legend(fontsize=8)

    # ── Panel C: window sensitivity heatmap ──────────────────────────────────
    ax = axes[1, 0]
    pivot = win_sens.pivot(index="group", columns="window_days", values="importance")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    im    = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}d" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax, label="Importance")
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            ax.text(c, r, f"{val:.2f}" if not np.isnan(val) else "–",
                    ha="center", va="center", fontsize=8,
                    color="white" if val > 0.3 else "black")
    ax.set_title("C — Window sensitivity (importance by group × horizon)")

    # ── Panel D: synthetic recall drop ───────────────────────────────────────
    ax = axes[1, 1]
    synth_plot = synth[synth["group"] != "BASELINE (all features)"].copy()
    colors = ["#d62728" if d > 0.10 else "#2ca02c"
              for d in synth_plot["recall_drop"]]
    ax.barh(synth_plot["group"], synth_plot["recall_drop_pct"], color=colors)
    ax.axvline(10, color="orange", linestyle=":", linewidth=0.8, label="10% threshold")
    ax.set_xlabel("Recall drop (%) when group removed")
    ax.set_title("D — Synthetic anomaly detection drop")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = "/mnt/user-data/outputs/knn_ablation_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved to {out_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Generate data using the slim feature pipeline ─────────────────────────
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from transaction_features import TransactionFeaturePipeline, _generate_sample_data

    print("Building features …")
    raw_df      = _generate_sample_data(n=300, seed=42)
    feat_pipe   = TransactionFeaturePipeline(lookback_windows=[7, 30, 90])
    features_df = feat_pipe.fit_transform(raw_df)

    # ── Run ablation ──────────────────────────────────────────────────────────
    ablation = AblationPipeline(
        features_df,
        k=10,
        top_n=20,
        n_synthetic=30,
        seed=42,
    )
    report = ablation.run()

    print_report(report)
    plot_report(report)

    # Save tabular results
    report["group_ablation"].to_csv(
        "/mnt/user-data/outputs/ablation_group.csv", index=False)
    report["window_sensitivity"].to_csv(
        "/mnt/user-data/outputs/ablation_window.csv", index=False)
    report["synthetic"].to_csv(
        "/mnt/user-data/outputs/ablation_synthetic.csv", index=False)
    print("CSV results saved.")

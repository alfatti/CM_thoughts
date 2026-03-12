"""
Outlier Scoring — Multi-Scorer Ablation Pipeline
=================================================
Scorer-agnostic ablation for transaction outlier detection.
Plug in any scorer via the registry or a custom callable.

Bundled scorers
---------------
  KNN             Mean distance to k nearest neighbours.
                  Higher score = more isolated = more anomalous.

  Que             Quantile-of-distance scorer. For each point, computes
                  the empirical quantile of its distance-to-kNN within the
                  global distance distribution. Captures relative rarity
                  rather than absolute distance magnitude.

  naive_spectral  Laplacian-based spectral anomaly score. Builds a kNN
                  affinity graph, computes the graph Laplacian, projects
                  onto its bottom eigenvectors, and scores each point by
                  its reconstruction error from those smooth eigenvectors.
                  High score = point lies in a sparse, disconnected region.

Ablation lenses (all unlabelled-friendly)
-----------------------------------------
  1. Group leave-one-out   Drop one feature group at a time; compare score
                           rankings via Spearman r and top-N overlap.

  2. Window sensitivity    Within windowed groups (amount, receiver,
                           portfolio), test each lookback horizon separately.

  3. Synthetic anomaly     Inject known anomalies; measure recall@top-N drop
                           when each group is removed.

  4. Cross-scorer agreement  Run lenses 1–3 for every registered scorer and
                             surface where they agree / disagree on importance.

Feature groups (slim pipeline naming convention)
-------------------------------------------------
  AMOUNT    amt_*_zscore, amt_*_vs_median, amt_*_pct_rank, is_round_amount
  RECEIVER  recv_*_is_new, recv_*_tx_count, recv_*_amt_zscore,
            recv_*_unique, recv_*_herfindahl
  PORTFOLIO port_*_vol_zscore, port_*_tx_vol_share,
            tx_share_of_daily_vol, vol_7d_vs_30d_ratio
  TEMPORAL  is_weekend, is_off_hours, seconds_since_last_tx,
            gap_zscore_30d, tx_count_24h, velocity_7d_vs_30d
  CURRENCY  ccy_is_new_*, currency_switched
  COMPOSITE flag_*
  TX_COUNT  amt_*_tx_count  (separated from AMOUNT intentionally)

Usage
-----
    from ablation import AblationPipeline, SCORER_REGISTRY

    # Run with a single scorer
    pipeline = AblationPipeline(features_df, scorer="KNN")
    report   = pipeline.run()
    pipeline.print_report(report)
    pipeline.plot_report(report)

    # Run with all registered scorers and compare
    pipeline = AblationPipeline(features_df, scorer="all")
    report   = pipeline.run()          # includes cross-scorer comparison
    pipeline.plot_report(report)

    # Plug in your own scorer — any callable (X: np.ndarray) -> np.ndarray
    def my_scorer(X):
        ...
        return scores   # shape (n,), higher = more anomalous

    pipeline = AblationPipeline(features_df, scorer=my_scorer)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr
from typing import Callable, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

ScoreFn = Callable[[np.ndarray], np.ndarray]


# ─────────────────────────────────────────────────────────────────────────────
# Scorer implementations
# ─────────────────────────────────────────────────────────────────────────────

def _knn_scorer(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Mean distance to k nearest neighbours (excluding self).
    Higher = more isolated = more anomalous.
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    return dists[:, 1:].mean(axis=1)


def _que_scorer(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Quantile-of-distance scorer (Que).
    Score = empirical quantile of each point's mean-kNN-distance within the
    global distance distribution.  Removes scale sensitivity: a point is
    anomalous if its neighbourhood is *relatively* sparse compared to all
    other points, not just absolutely far away.
    """
    raw = _knn_scorer(X, k=k)
    # Quantile rank within the observed distribution
    ranks = np.argsort(np.argsort(raw))          # double-argsort = rank
    return ranks / (len(raw) - 1)                # normalise to [0, 1]


def _naive_spectral_scorer(X: np.ndarray, k: int = 10, n_components: int = 5) -> np.ndarray:
    """
    Naive spectral anomaly score.

    1. Build a symmetric kNN affinity graph  W_ij = exp(-||xi - xj||² / σ²)
    2. Compute normalised graph Laplacian  L = I - D^{-1/2} W D^{-1/2}
    3. Take the bottom `n_components` eigenvectors (smoothest graph signals)
    4. Score each point by its squared residual when projected onto those
       eigenvectors.  Points in sparse / disconnected regions have high
       residual = high anomaly score.
    """
    n = X.shape[0]
    k_eff = min(k, n - 1)
    nc    = min(n_components, k_eff)

    # ── Affinity matrix ───────────────────────────────────────────────────────
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    dists, indices = nn.kneighbors(X)

    sigma = np.median(dists[:, 1:]) + 1e-10
    W = np.zeros((n, n))
    for i in range(n):
        for j_pos, j in enumerate(indices[i, 1:]):
            w = np.exp(-(dists[i, j_pos + 1] ** 2) / (sigma ** 2))
            W[i, j] = w
            W[j, i] = w          # symmetrise

    # ── Normalised Laplacian ──────────────────────────────────────────────────
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1).clip(1e-10)))
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    # ── Bottom eigenvectors (skip trivial constant eigenvector at index 0) ────
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    smooth = eigenvectors[:, 1 : nc + 1]         # shape (n, nc)

    # ── Reconstruction error ─────────────────────────────────────────────────
    X_proj      = smooth @ smooth.T @ X           # projection onto smooth subspace
    residuals   = np.linalg.norm(X - X_proj, axis=1)
    return residuals


# ─────────────────────────────────────────────────────────────────────────────
# Scorer registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScorerSpec:
    """Wraps a scorer function with its display name and default kwargs."""
    name:    str
    fn:      ScoreFn
    kwargs:  Dict = field(default_factory=dict)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.fn(X, **self.kwargs)


SCORER_REGISTRY: Dict[str, ScorerSpec] = {
    "KNN": ScorerSpec(
        name="KNN",
        fn=_knn_scorer,
        kwargs={"k": 10},
    ),
    "Que": ScorerSpec(
        name="Que",
        fn=_que_scorer,
        kwargs={"k": 10},
    ),
    "naive_spectral": ScorerSpec(
        name="naive_spectral",
        fn=_naive_spectral_scorer,
        kwargs={"k": 10, "n_components": 5},
    ),
}


def _resolve_scorer(scorer: Union[str, ScoreFn, None]) -> Dict[str, ScorerSpec]:
    """
    Returns an ordered dict of {name: ScorerSpec} to run.

    Accepts:
      "KNN" / "Que" / "naive_spectral"  → single scorer from registry
      "all"                              → all registered scorers
      callable                           → wrapped as a custom ScorerSpec
      None                               → defaults to KNN
    """
    if scorer is None or scorer == "KNN":
        return {"KNN": SCORER_REGISTRY["KNN"]}
    if scorer == "all":
        return dict(SCORER_REGISTRY)
    if isinstance(scorer, str):
        if scorer not in SCORER_REGISTRY:
            raise ValueError(
                f"Unknown scorer '{scorer}'. "
                f"Available: {list(SCORER_REGISTRY)} or 'all'."
            )
        return {scorer: SCORER_REGISTRY[scorer]}
    if callable(scorer):
        spec = ScorerSpec(name="custom", fn=scorer, kwargs={})
        return {"custom": spec}
    raise TypeError(f"scorer must be str, callable, or None; got {type(scorer)}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature group resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_groups(columns: List[str]) -> Dict[str, List[str]]:
    """
    Assign every feature column to exactly one group based on slim-pipeline
    naming conventions.  TX_COUNT is split out from AMOUNT intentionally.
    """
    def pick(patterns):
        return [c for c in columns
                if any(c.startswith(p) or p in c for p in patterns)]

    groups = {
        "AMOUNT":    pick(["amt_", "is_round_amount"]),
        "RECEIVER":  pick(["recv_"]),
        "PORTFOLIO": pick(["port_", "tx_share_of_daily_vol", "vol_7d_vs_30d"]),
        "TEMPORAL":  pick(["is_weekend", "is_off_hours", "seconds_since_last_tx",
                           "gap_zscore", "tx_count_24h", "velocity_7d_vs_30d"]),
        "CURRENCY":  pick(["ccy_", "currency_switched"]),
        "COMPOSITE": pick(["flag_"]),
    }

    tx_count_cols      = [c for c in groups["AMOUNT"] if c.endswith("_tx_count")]
    groups["AMOUNT"]   = [c for c in groups["AMOUNT"] if c not in tx_count_cols]
    groups["TX_COUNT"] = tx_count_cols

    seen = set()
    for name in list(groups):
        groups[name] = [c for c in groups[name] if c not in seen]
        seen.update(groups[name])
        if not groups[name]:
            del groups[name]

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Impute NaN with column median, then RobustScale."""
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    return RobustScaler().fit_transform(X)


def _recall_at_k(scores: np.ndarray, is_synthetic: np.ndarray, k: int) -> float:
    top_k   = set(np.argsort(scores)[-k:])
    synth   = set(np.where(is_synthetic)[0])
    return len(top_k & synth) / len(synth) if synth else 0.0


def _inject_synthetic_anomalies(
    df: pd.DataFrame,
    feature_cols: List[str],
    n: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Perturb n randomly chosen transactions to become synthetic anomalies.
    Each perturbation touches all feature groups so anomalies are detectable
    regardless of which group is ablated (marginal-contribution semantics).
    """
    base_idx = rng.choice(len(df), size=n, replace=False)
    synth    = df.iloc[base_idx][feature_cols].copy().reset_index(drop=True)

    for col in synth.columns:
        if "amt_" in col and "zscore"    in col: synth[col] = synth[col].fillna(0) + rng.uniform(8, 15, n)
        elif "amt_" in col and "vs_median" in col: synth[col] = rng.uniform(10, 20, n)
        elif "amt_" in col and "pct_rank"  in col: synth[col] = rng.uniform(0.97, 1.0, n)
        elif col == "is_round_amount":             synth[col] = 1
        elif "recv_" in col and "is_new"     in col: synth[col] = 1
        elif "recv_" in col and "tx_count"   in col: synth[col] = 0
        elif "recv_" in col and "amt_zscore" in col: synth[col] = synth[col].fillna(0) + rng.uniform(8, 15, n)
        elif "recv_" in col and "unique"     in col: synth[col] = synth[col].fillna(1) * rng.uniform(3, 6, n)
        elif "ccy_"  in col and "is_new"     in col: synth[col] = 1
        elif col == "currency_switched":             synth[col] = 1
        elif col in ("is_off_hours", "is_weekend"):  synth[col] = 1
        elif "vol_zscore" in col:                    synth[col] = synth[col].fillna(0) + rng.uniform(8, 15, n)
        elif "vol_share"  in col or "tx_share" in col: synth[col] = rng.uniform(0.85, 0.99, n)
        elif col.startswith("flag_"):               synth[col] = 1

    combined     = pd.concat([df[feature_cols].reset_index(drop=True), synth],
                              ignore_index=True)
    is_synth     = np.zeros(len(combined), dtype=bool)
    is_synth[len(df):] = True
    return combined, is_synth


# ─────────────────────────────────────────────────────────────────────────────
# Core ablation logic (scorer-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_scorer(
    df:           pd.DataFrame,
    feature_cols: List[str],
    groups:       Dict[str, List[str]],
    scorer:       ScorerSpec,
    top_n:        int,
    n_synthetic:  int,
    rng:          np.random.Generator,
) -> Dict:
    """Run all three ablation lenses for a single scorer. Returns a result dict."""

    X_full          = _prepare_matrix(df, feature_cols)
    baseline_scores = scorer(X_full)
    baseline_top    = set(np.argsort(baseline_scores)[-top_n:])

    # ── Lens 1: group leave-one-out ──────────────────────────────────────────
    grp_rows = []
    for gname, drop_cols in groups.items():
        kept = [c for c in feature_cols if c not in drop_cols]
        if not kept:
            continue
        scores   = scorer(_prepare_matrix(df, kept))
        sp_r, _  = spearmanr(baseline_scores, scores)
        abl_top  = set(np.argsort(scores)[-top_n:])
        overlap  = len(baseline_top & abl_top) / top_n
        grp_rows.append({
            "group":              gname,
            "n_dropped":          len(drop_cols),
            "spearman_r":         round(float(sp_r), 4),
            "top_n_overlap":      round(overlap, 4),
            "mean_score_delta":   round(float(np.mean(np.abs(scores - baseline_scores))), 6),
            "importance":         round((1 - float(sp_r)) + (1 - overlap), 4),
        })
    group_df = (pd.DataFrame(grp_rows)
                  .sort_values("importance", ascending=False)
                  .reset_index(drop=True))
    group_df["importance_rank"] = range(1, len(group_df) + 1)

    # ── Lens 2: window sensitivity ───────────────────────────────────────────
    windowed = {
        g: cols for g, cols in groups.items()
        if any(f"_{w}d_" in c or c.endswith(f"_{w}d")
               for c in cols for w in [7, 30, 90])
    }
    win_rows = []
    for gname, all_cols in windowed.items():
        for w in [7, 30, 90]:
            drop = [c for c in all_cols
                    if f"_{w}d_" in c or c.endswith(f"_{w}d")]
            if not drop:
                continue
            kept    = [c for c in feature_cols if c not in drop]
            scores  = scorer(_prepare_matrix(df, kept))
            sp_r, _ = spearmanr(baseline_scores, scores)
            abl_top = set(np.argsort(scores)[-top_n:])
            overlap = len(baseline_top & abl_top) / top_n
            win_rows.append({
                "group":       gname,
                "window_days": w,
                "n_dropped":   len(drop),
                "spearman_r":  round(float(sp_r), 4),
                "top_n_overlap": round(overlap, 4),
                "importance":  round((1 - float(sp_r)) + (1 - overlap), 4),
            })
    window_df = (pd.DataFrame(win_rows)
                   .sort_values(["group", "importance"], ascending=[True, False])
                   .reset_index(drop=True))

    # ── Lens 3: synthetic anomaly recall ────────────────────────────────────
    synth_rows = []
    combined_full, is_synth = _inject_synthetic_anomalies(
        df, feature_cols, n_synthetic, rng)
    base_recall = _recall_at_k(
        scorer(_prepare_matrix(combined_full, feature_cols)),
        is_synth, top_n)
    synth_rows.append({
        "group": "BASELINE",
        "recall_at_top_n": round(base_recall, 4),
        "recall_drop":     0.0,
        "recall_drop_pct": 0.0,
    })
    for gname, drop_cols in groups.items():
        kept = [c for c in feature_cols if c not in drop_cols]
        if not kept:
            continue
        combined_abl, is_synth_abl = _inject_synthetic_anomalies(
            df, kept, n_synthetic, rng)
        recall  = _recall_at_k(
            scorer(_prepare_matrix(combined_abl, kept)),
            is_synth_abl, top_n)
        drop     = base_recall - recall
        synth_rows.append({
            "group":           gname,
            "recall_at_top_n": round(recall, 4),
            "recall_drop":     round(drop, 4),
            "recall_drop_pct": round(drop / base_recall * 100 if base_recall else 0, 1),
        })
    synth_df = (pd.DataFrame(synth_rows)
                  .sort_values("recall_drop", ascending=False)
                  .reset_index(drop=True))

    return {
        "scorer":           scorer.name,
        "baseline_scores":  baseline_scores,
        "group_ablation":   group_df,
        "window_sensitivity": window_df,
        "synthetic":        synth_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AblationPipeline
# ─────────────────────────────────────────────────────────────────────────────

class AblationPipeline:
    """
    Parameters
    ----------
    features_df  : output of TransactionFeaturePipeline.fit_transform()
    scorer       : "KNN" | "Que" | "naive_spectral" | "all" | callable
                   Default = "KNN".
                   Pass "all" to run every registered scorer and get a
                   cross-scorer comparison lens.
                   Pass any (X: np.ndarray) -> np.ndarray callable for
                   a custom scorer.
    top_n        : top-N transactions to track for overlap and recall metrics
    n_synthetic  : number of synthetic anomalies injected for Lens 3
    seed         : RNG seed for reproducibility
    """

    RAW_COLS = {"transaction_id", "date", "timestamp",
                "debit_amount", "receiver_id", "currency"}

    def __init__(
        self,
        features_df:  pd.DataFrame,
        scorer:       Union[str, ScoreFn, None] = "KNN",
        top_n:        int = 20,
        n_synthetic:  int = 30,
        seed:         int = 42,
    ):
        self.df           = features_df
        self.scorers      = _resolve_scorer(scorer)
        self.top_n        = top_n
        self.n_synthetic  = n_synthetic
        self.rng          = np.random.default_rng(seed)
        self.feature_cols = [c for c in features_df.columns
                             if c not in self.RAW_COLS]
        self.groups       = _resolve_groups(self.feature_cols)

        ungrouped = [c for c in self.feature_cols
                     if not any(c in g for g in self.groups.values())]
        if ungrouped:
            print(f"[ablation] Warning: {len(ungrouped)} columns not in any group "
                  f"(always included): {ungrouped}")

    def run(self) -> Dict:
        """
        Run all lenses for every configured scorer.

        Returns
        -------
        dict with keys:
          "by_scorer"   : {scorer_name: {lens results}}
          "comparison"  : cross-scorer importance comparison DataFrame
                          (only present when >1 scorer is run)
          "groups"      : the feature group mapping used
        """
        print(f"[ablation] Groups:")
        for g, cols in self.groups.items():
            print(f"  {g:12s} ({len(cols):2d} cols)")
        print(f"[ablation] Scorers : {list(self.scorers)}")
        print(f"[ablation] top_n={self.top_n}, n_synthetic={self.n_synthetic}\n")

        by_scorer = {}
        for name, spec in self.scorers.items():
            print(f"[ablation] ── Running scorer: {name} ──")
            by_scorer[name] = _run_one_scorer(
                self.df, self.feature_cols, self.groups,
                spec, self.top_n, self.n_synthetic, self.rng,
            )

        result = {"by_scorer": by_scorer, "groups": self.groups}

        if len(by_scorer) > 1:
            result["comparison"] = self._cross_scorer_comparison(by_scorer)

        return result

    def _cross_scorer_comparison(
        self, by_scorer: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Build a group × scorer importance matrix plus agreement metrics.
        Agreement = std of importance ranks across scorers (low = high agreement).
        """
        frames = {}
        for sname, res in by_scorer.items():
            s = res["group_ablation"].set_index("group")["importance"]
            frames[sname] = s

        comp = pd.DataFrame(frames)
        comp.index.name = "group"

        # Rank within each scorer (1 = most important)
        rank_cols = []
        for col in comp.columns:
            rc = f"{col}_rank"
            comp[rc] = comp[col].rank(ascending=False).astype(int)
            rank_cols.append(rc)

        comp["rank_std"]      = comp[rank_cols].std(axis=1).round(2)
        comp["rank_mean"]     = comp[rank_cols].mean(axis=1).round(2)
        comp["agreement"]     = comp["rank_std"].apply(
            lambda s: "✓ agree" if s <= 1.0 else ("~ partial" if s <= 2.0 else "✗ disagree")
        )
        return comp.sort_values("rank_mean")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, report: Dict):
        sep = "─" * 72

        for sname, res in report["by_scorer"].items():
            print(f"\n{'':=<72}")
            print(f"  ABLATION REPORT — scorer: {sname}")
            print(f"{'':=<72}")

            print(f"\n📊 LENS 1 — Group leave-one-out")
            print(f"   spearman_r   : rank correlation vs baseline (1 = unchanged)")
            print(f"   top_n_overlap: fraction of top-{self.top_n} flagged txs surviving removal")
            print(f"   importance   : (1-spearman) + (1-overlap); higher = more important\n")
            print(res["group_ablation"].to_string(index=False))

            print(f"\n{sep}")
            print(f"\n🔍 LENS 2 — Window sensitivity\n")
            print(res["window_sensitivity"].to_string(index=False))

            print(f"\n{sep}")
            print(f"\n🎯 LENS 3 — Synthetic anomaly recall\n")
            print(res["synthetic"].to_string(index=False))

            print(f"\n{sep}")
            self._print_pruning_candidates(res)

        if "comparison" in report:
            print(f"\n{'':=<72}")
            print(f"  LENS 4 — Cross-scorer agreement")
            print(f"{'':=<72}")
            print("\n  rank_std ≤ 1 → scorers agree on this group's importance\n")
            print(report["comparison"].to_string())

    def _print_pruning_candidates(self, res: Dict):
        print(f"\n✅  PRUNING CANDIDATES (low importance + low recall_drop)")
        grp  = res["group_ablation"].set_index("group")
        syn  = res["synthetic"].set_index("group")
        found = False
        for g in grp.index:
            imp = grp.at[g, "importance"]
            rd  = syn.at[g, "recall_drop"] if g in syn.index else np.nan
            if imp < 0.15 and (np.isnan(rd) or rd < 0.05):
                print(f"  → {g:12s}  importance={imp:.3f}  recall_drop={rd:.3f}  "
                      f"cols={grp.at[g, 'n_dropped']}")
                found = True
        if not found:
            print("  (none below thresholds)")

    def plot_report(self, report: Dict, out_prefix: str = "/mnt/user-data/outputs/ablation"):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("[plot] matplotlib not available.")
            return

        n_scorers     = len(report["by_scorer"])
        has_comparison = "comparison" in report

        for sname, res in report["by_scorer"].items():
            n_rows = 3 if has_comparison and n_scorers > 1 else 2
            fig    = plt.figure(figsize=(14, 5 * n_rows))
            gs     = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)
            fig.suptitle(f"Ablation Report — {sname}", fontsize=13, fontweight="bold")

            grp  = res["group_ablation"]
            win  = res["window_sensitivity"]
            syn  = res["synthetic"]

            # Panel A — Spearman r
            ax = fig.add_subplot(gs[0, 0])
            colors = ["#d62728" if r < 0.95 else "#2ca02c" for r in grp["spearman_r"]]
            ax.barh(grp["group"], grp["spearman_r"], color=colors)
            ax.axvline(0.95, color="orange", linestyle=":", lw=1, label="0.95 threshold")
            ax.set_xlim(max(0, grp["spearman_r"].min() - 0.05), 1.02)
            ax.set_xlabel("Spearman r  (↑ = less important)")
            ax.set_title("A — Score-rank correlation after removal")
            ax.legend(fontsize=8)

            # Panel B — top-N overlap
            ax = fig.add_subplot(gs[0, 1])
            colors = ["#d62728" if o < 0.80 else "#2ca02c" for o in grp["top_n_overlap"]]
            ax.barh(grp["group"], grp["top_n_overlap"], color=colors)
            ax.axvline(0.80, color="orange", linestyle=":", lw=1, label="0.80 threshold")
            ax.set_xlim(0, 1.05)
            ax.set_xlabel(f"Overlap fraction (top-{self.top_n})")
            ax.set_title("B — Top-N flagged tx overlap after removal")
            ax.legend(fontsize=8)

            # Panel C — window heatmap
            ax = fig.add_subplot(gs[1, 0])
            if not win.empty:
                pivot = win.pivot(index="group", columns="window_days", values="importance")
                pivot = pivot.reindex(columns=sorted(pivot.columns))
                im    = ax.imshow(pivot.values, aspect="auto",
                                  cmap="RdYlGn_r", vmin=0, vmax=0.5)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([f"{c}d" for c in pivot.columns])
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                plt.colorbar(im, ax=ax, label="Importance")
                for r in range(len(pivot.index)):
                    for c in range(len(pivot.columns)):
                        v = pivot.values[r, c]
                        ax.text(c, r, f"{v:.2f}" if not np.isnan(v) else "–",
                                ha="center", va="center", fontsize=8,
                                color="white" if v > 0.3 else "black")
            ax.set_title("C — Window sensitivity heatmap")

            # Panel D — synthetic recall drop
            ax = fig.add_subplot(gs[1, 1])
            syn_plot = syn[syn["group"] != "BASELINE"].copy()
            colors   = ["#d62728" if d > 0.10 else "#2ca02c"
                        for d in syn_plot["recall_drop"]]
            ax.barh(syn_plot["group"], syn_plot["recall_drop_pct"], color=colors)
            ax.axvline(10, color="orange", linestyle=":", lw=1, label="10% threshold")
            ax.set_xlabel("Recall drop % when group removed")
            ax.set_title("D — Synthetic anomaly recall drop")
            ax.legend(fontsize=8)

            # Panel E — cross-scorer importance (only in multi-scorer run)
            if has_comparison and n_scorers > 1:
                ax  = fig.add_subplot(gs[2, :])
                comp = report["comparison"]
                imp_cols = [c for c in comp.columns
                            if c in report["by_scorer"]]
                x    = np.arange(len(comp))
                w    = 0.8 / len(imp_cols)
                cmap = plt.cm.get_cmap("tab10", len(imp_cols))
                for i, col in enumerate(imp_cols):
                    ax.bar(x + i * w, comp[col], width=w,
                           label=col, color=cmap(i), alpha=0.85)
                ax.set_xticks(x + w * (len(imp_cols) - 1) / 2)
                ax.set_xticklabels(comp.index, rotation=15, ha="right")
                ax.set_ylabel("Importance score")
                ax.set_title("E — Cross-scorer importance comparison")
                ax.legend(fontsize=9)
                for xi, (_, row) in zip(x, comp.iterrows()):
                    ax.text(xi + w * (len(imp_cols) - 1) / 2,
                            comp[imp_cols].loc[row.name].max() + 0.005,
                            row["agreement"], ha="center", fontsize=7, color="grey")

            path = f"{out_prefix}_{sname}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[plot] Saved → {path}")
            plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))
    from transaction_features import TransactionFeaturePipeline, _generate_sample_data

    print("Building features …")
    raw_df      = _generate_sample_data(n=200, seed=42)
    feat_pipe   = TransactionFeaturePipeline(lookback_windows=[7, 30, 90])
    features_df = feat_pipe.fit_transform(raw_df)

    # ── Single scorer ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Single scorer: KNN")
    pipe = AblationPipeline(features_df, scorer="KNN", top_n=20, n_synthetic=25)
    t0   = time.perf_counter()
    rep  = pipe.run()
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    pipe.print_report(rep)
    pipe.plot_report(rep)

    # ── All scorers + cross-comparison ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("All scorers (KNN + Que + naive_spectral)")
    pipe_all = AblationPipeline(features_df, scorer="all", top_n=20, n_synthetic=25)
    t0       = time.perf_counter()
    rep_all  = pipe_all.run()
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    pipe_all.print_report(rep_all)
    pipe_all.plot_report(rep_all)

    # Save comparison CSV
    if "comparison" in rep_all:
        rep_all["comparison"].to_csv(
            "/mnt/user-data/outputs/ablation_comparison.csv")
        print("Cross-scorer comparison saved.")

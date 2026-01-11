import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def ensure_derived_columns(df: pd.DataFrame, rollout_size: int) -> pd.DataFrame:
    df = df.copy()

    # learn_phase
    if "learn_phase" not in df.columns or df["learn_phase"].isna().all():
        df["learn_phase"] = (df["global_step"] // rollout_size).astype(int)

    # step_in_learn_phase (optional but handy)
    if "step_in_learn_phase" not in df.columns or df["step_in_learn_phase"].isna().all():
        df["step_in_learn_phase"] = (df["global_step"] % rollout_size).astype(int)

    # wealth norms
    if "wealth_t_norm" not in df.columns or df["wealth_t_norm"].isna().all():
        df["wealth_t_norm"] = df["wealth_t"] / df["initial_money"]
    if "wealth_next_norm" not in df.columns or df["wealth_next_norm"].isna().all():
        df["wealth_next_norm"] = df["wealth_next"] / df["initial_money"]

    # cash norms
    if "cash_t_norm" not in df.columns or df["cash_t_norm"].isna().all():
        df["cash_t_norm"] = df["cash_t"] / df["initial_money"]
    if "cash_next_norm" not in df.columns or df["cash_next_norm"].isna().all():
        df["cash_next_norm"] = df["cash_next"] / df["initial_money"]

    return df


###########################################################################
#import the data
###########################################################################
def import_files_raw():
        
    FILES = {
        "120k data": "model_value_lenght_120000_raw.xlsx",
        "60k data":  "model_value_lenght_60000_raw.xlsx",
        "40k data":  "model_value_lenght_40000_raw.xlsx",

        "higher learning rate":   "model_value_lenght_60000_learning_rate_0.0003_raw.xlsx",
        "lower entropy coefficient": "model_value_lenght_60000_ent_coef_0.005_raw.xlsx",
        "higher clip range":  "model_value_lenght_60000_clip_range_0.2_raw.xlsx",
    }


    BASE_DIR = Path(__file__).parent  # script directory

    DATASETS = {}
    print("Loading files...")
    for name, filename in FILES.items():
        path = (BASE_DIR / filename).resolve()
        print(f" - {name}: trying {path} | exists={path.exists()} | size={(path.stat().st_size if path.exists() else None)} bytes")
        df = pd.read_excel(path, sheet_name="transitions")
        df["model_id"] = name

        # run_id: one run per file
        if "run_id" not in df.columns:
            df["run_id"] = 0
        else:
            df["run_id"] = df["run_id"].fillna(0).astype(int)

        # env_id: ensure not NaN
        if "env_id" not in df.columns:
            df["env_id"] = 0
        else:
            df["env_id"] = df["env_id"].fillna(0).astype(int)
        DATASETS[name] = df
        ROLLOUT_SIZE = 2000 * 1
        DATASETS[name] = ensure_derived_columns(DATASETS[name], rollout_size=ROLLOUT_SIZE)

        print(
            f"   -> learn_phase all nan? {DATASETS[name]['learn_phase'].isna().all()} | "
            f"min={DATASETS[name]['learn_phase'].min()} max={DATASETS[name]['learn_phase'].max()}"
        )
        print(f"   -> loaded rows={len(DATASETS[name])}, cols={len(DATASETS[name].columns)}")

    print("files has been loaded")
    return DATASETS


###########################################################################
# Functions to help the results preparation
###########################################################################



def get_final_step(df):
    """
    Return rows corresponding to the final step of each episode.
    """
    return (
        df.sort_values("t_in_episode")
          .groupby(["run_id", "env_id", "episode_id"], as_index=False)
          .tail(1)
    )


def transform_wealth_cash(df, model_id):
    out = []

    group_keys = ["model_id", "learn_phase"]

    # ---------- TIME EVOLUTION ----------
    time_cols = ["wealth_t_norm", "cash_t_norm"]

    time_df = (
        df.groupby(group_keys + ["t_in_episode"], as_index=False)[time_cols]
          .mean()
    )

    for col in time_cols:
        tmp = time_df[["learn_phase", "t_in_episode", col]].copy()
        tmp["metric"] = col.replace("_t_norm", "").replace("_t", "")
        tmp["aggregation"] = "time"
        tmp["value"] = tmp[col]
        tmp["model_id"] = model_id
        out.append(tmp[["model_id","learn_phase","aggregation","metric","t_in_episode","value"]])

    # ---------- EPISODE MEAN ----------
    ep_mean = (
        df.groupby(group_keys + ["episode_id"], as_index=False)
          .agg({
              "wealth_t_norm": "mean",
              "cash_t_norm": "mean"
          })
          .groupby(group_keys, as_index=False)
          .mean()
    )

    for col in ["wealth_t_norm", "cash_t_norm"]:
        tmp = ep_mean[["learn_phase", col]].copy()
        tmp["metric"] = col.replace("_t_norm", "").replace("_t", "")
        tmp["aggregation"] = "episode_mean"
        tmp["value"] = tmp[col]
        tmp["t_in_episode"] = np.nan
        tmp["model_id"] = model_id
        out.append(tmp[["model_id","learn_phase","aggregation","metric","t_in_episode","value"]])

    # ---------- FINAL VALUE ----------
    final_df = get_final_step(df)

    final_agg = (
        final_df.groupby(group_keys, as_index=False)
                .agg({
                    "wealth_next_norm": "mean",
                    "cash_next_norm": "mean"
                })
    )

    for col in ["wealth_next_norm", "cash_next_norm"]:
        tmp = final_agg[["learn_phase", col]].copy()
        tmp["metric"] = col.replace("_next_norm", "").replace("_next", "")
        tmp["aggregation"] = "final"
        tmp["value"] = tmp[col]
        tmp["t_in_episode"] = np.nan
        tmp["model_id"] = model_id
        out.append(tmp[["model_id","learn_phase","aggregation","metric","t_in_episode","value"]])

    return pd.concat(out, ignore_index=True)


def transform_reward(df, model_id):
    out = []

    group_keys = ["model_id", "learn_phase"]

    # Episode mean
    ep_reward = (
        df.groupby(group_keys + ["episode_id"], as_index=False)["reward"]
          .mean()
          .groupby(group_keys, as_index=False)
          .mean()
    )

    ep_reward["metric"] = "reward"
    ep_reward["aggregation"] = "episode_mean"
    ep_reward["value"] = ep_reward["reward"]
    ep_reward["t_in_episode"] = np.nan
    ep_reward["model_id"] = model_id

    out.append(ep_reward[["model_id","learn_phase","aggregation","metric","t_in_episode","value"]])

    # Final reward
    final_df = get_final_step(df)

    final_reward = (
        final_df.groupby(group_keys, as_index=False)["reward"]
                .mean()
    )

    final_reward["metric"] = "reward"
    final_reward["aggregation"] = "final"
    final_reward["value"] = final_reward["reward"]
    final_reward["t_in_episode"] = np.nan
    final_reward["model_id"] = model_id

    out.append(final_reward[["model_id","learn_phase","aggregation","metric","t_in_episode","value"]])

    return pd.concat(out, ignore_index=True)


def transform_penalties(df, model_id):
    group_keys = ["model_id", "learn_phase"]

    pen = (
        df.groupby(group_keys + ["episode_id"], as_index=False)
          .agg({
              "budget_penalty": "mean",
              "resource_penalty": "mean"
          })
          .groupby(group_keys, as_index=False)
          .mean()
    )

    out = []
    for col in ["budget_penalty", "resource_penalty"]:
        tmp = pen[["learn_phase", col]].copy()
        tmp["metric"] = col
        tmp["aggregation"] = "episode_mean"
        tmp["value"] = tmp[col]
        tmp["t_in_episode"] = np.nan
        tmp["model_id"] = model_id
        out.append(tmp[["model_id","learn_phase","aggregation","metric","t_in_episode","value"]])

    return pd.concat(out, ignore_index=True)


def build_clean_dataset(df, model_id):
    """
    df      : raw dataset (Ds_A, Ds_B, ...)
    model_id: string identifier
    """
    parts = []

    parts.append(transform_wealth_cash(df, model_id))
    parts.append(transform_training_diagnostics(df, model_id))
    parts.append(transform_risk_quality(df, model_id))



    clean = pd.concat(parts, ignore_index=True)

    return clean.sort_values(
        ["model_id", "learn_phase", "aggregation", "metric", "t_in_episode"]
    ).reset_index(drop=True)


def transform_training_diagnostics(df, model_id):
    """
    Creates clean long-format rows for:
    - reward
    - budget_penalty
    - resource_penalty

    Aggregations:
    - time
    - episode_mean
    (no final)
    """
    out = []
    group_keys = ["run_id", "env_id", "learn_phase"]

    metrics = ["reward", "budget_penalty", "resource_penalty"]

    # ---------- TIME ----------
    time_df = (
        df.groupby(group_keys + ["t_in_episode"], as_index=False)[metrics]
          .mean()
    )

    for m in metrics:
        tmp = time_df[["learn_phase", "t_in_episode", m]].copy()
        tmp["model_id"] = model_id
        tmp["metric"] = m
        tmp["aggregation"] = "time"
        tmp["value"] = tmp[m]
        out.append(tmp[["model_id","learn_phase","aggregation","metric","t_in_episode","value"]])

    # ---------- EPISODE MEAN ----------
    # mean over t within each episode, then mean across episodes within learn_phase
    ep_df = (
        df.groupby(group_keys + ["episode_id"], as_index=False)[metrics]
          .mean()
          .groupby(group_keys, as_index=False)
          .mean()
    )

    for m in metrics:
        tmp = ep_df[["learn_phase", m]].copy()
        tmp["model_id"] = model_id
        tmp["metric"] = m
        tmp["aggregation"] = "episode_mean"
        tmp["value"] = tmp[m]
        tmp["t_in_episode"] = np.nan
        out.append(tmp[["model_id","learn_phase","aggregation","metric","t_in_episode","value"]])

    return pd.concat(out, ignore_index=True)


def transform_risk_quality(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """
    Risk / quality metrics (episode-mean only):
      - wealth_volatility
      - sharpe_ratio

    Definitions (simple returns):
      r_t = wealth_t_norm(t) / wealth_t_norm(t-1) - 1
      volatility_episode = std(r_t)
      sharpe_episode     = mean(r_t) / std(r_t)   (risk-free rate assumed 0)

    Aggregation:
      1) compute per (model_id, learn_phase, episode_id)
      2) average across episodes -> (model_id, learn_phase)
    """
    required = {"wealth_t_norm", "t_in_episode", "learn_phase", "episode_id"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"transform_risk_quality missing columns: {sorted(missing)}")

    d = df.sort_values(["model_id", "learn_phase", "episode_id", "t_in_episode"]).copy()

    grp_ep = ["model_id", "learn_phase", "episode_id"]
    # Simple returns within each episode
    d["ret_wealth"] = d.groupby(grp_ep)["wealth_t_norm"].pct_change()

    # Per-episode mean/std of returns
    ep = (
        d.groupby(grp_ep, as_index=False)
         .agg(ret_mean=("ret_wealth", "mean"), ret_std=("ret_wealth", "std"))
    )

    ep["wealth_volatility"] = ep["ret_std"]
    ep["sharpe_ratio"] = ep["ret_mean"] / ep["ret_std"]
    ep.loc[ep["ret_std"] == 0, "sharpe_ratio"] = np.nan

    # Average across episodes within learn_phase
    lp = (
        ep.groupby(["model_id", "learn_phase"], as_index=False)
          .agg(
              wealth_volatility=("wealth_volatility", "mean"),
              sharpe_ratio=("sharpe_ratio", "mean"),
          )
    )

    out = []
    for m in ["wealth_volatility", "sharpe_ratio"]:
        tmp = lp[["model_id", "learn_phase", m]].copy()
        tmp["metric"] = m
        tmp["aggregation"] = "episode_mean"
        tmp["value"] = tmp[m]
        tmp["t_in_episode"] = np.nan
        out.append(tmp[["model_id", "learn_phase", "aggregation", "metric", "t_in_episode", "value"]])

    return pd.concat(out, ignore_index=True)

###########################################################################
# PIPELINE: build clean datasets from raw ones
###########################################################################


def load_or_build_clean_all( clean_dir="ML", filename="CLEAN_ALL.csv"):
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    clean_path = clean_dir / filename

    # 1) Fast path: load cached CLEAN_ALL
    if clean_path.exists() and clean_path.stat().st_size > 0:
        print(f"[LOAD] Using cached CLEAN_ALL: {clean_path}")
        CLEAN_ALL = pd.read_csv(clean_path, low_memory=False)
        return CLEAN_ALL

    # 2) Slow path: build CLEAN_ALL from raw datasets
    print(f"[BUILD] CLEAN_ALL not found -> building and caching to: {clean_path}")

    DATASETS = import_files_raw()
    CLEAN_DATASETS = {}
    for model_id, df_raw in DATASETS.items():
        CLEAN_DATASETS[model_id] = build_clean_dataset(df_raw, model_id=model_id)

    CLEAN_ALL = pd.concat(CLEAN_DATASETS.values(), ignore_index=True)

    # cache (CSV)
    CLEAN_ALL.to_csv(clean_path, index=False)
    print(f"[SAVE] Cached CLEAN_ALL to: {clean_path} (rows={len(CLEAN_ALL):,})")

    return CLEAN_ALL

CLEAN_ALL = load_or_build_clean_all( clean_dir="ML", filename="CLEAN_ALL.csv")
###########################################################################
# functions to plots
###########################################################################
def last_learn_phase_per_model(
    cdf: pd.DataFrame,
    metric: str | None = None,
    aggregation: str | None = None,
    expected_points: int | None = None,
    min_coverage: float = 0.98,
) -> int | None:
    """
    "Safe last" learn_phase:
    returns the latest learn_phase with sufficient coverage for the given aggregation/metric.

    If metric/aggregation are None, it returns max learn_phase (fallback).
    Recommended usage: pass metric+aggregation where completeness matters (time/episode_mean).
    """
    # fallback: old behavior
    if metric is None or aggregation is None:
        lp = pd.to_numeric(cdf["learn_phase"], errors="coerce").dropna()
        return int(lp.max()) if len(lp) else None

    sub = cdf[
        (cdf["metric"] == metric) &
        (cdf["aggregation"] == aggregation)
    ].copy()

    if sub.empty:
        # fallback: max learn_phase
        lp = pd.to_numeric(cdf["learn_phase"], errors="coerce").dropna()
        return int(lp.max()) if len(lp) else None

    sub["learn_phase"] = pd.to_numeric(sub["learn_phase"], errors="coerce")
    sub = sub.dropna(subset=["learn_phase"])
    sub["learn_phase"] = sub["learn_phase"].astype(int)

    if aggregation == "time":
        sub = sub.dropna(subset=["t_in_episode"])
        counts = sub.groupby("learn_phase")["t_in_episode"].nunique().sort_index()
        if counts.empty:
            return int(sub["learn_phase"].max())

        if expected_points is None:
            expected_points = int(counts.max())

        needed = int(np.floor(min_coverage * expected_points))
        valid = counts[counts >= needed]
        return int(valid.index.max()) if not valid.empty else int(counts.idxmax())

    # For episode_mean/final: usually 1 value per phase is enough
    # But we can still do a coverage check (number of rows per phase)
    counts = sub.groupby("learn_phase").size().sort_index()
    if counts.empty:
        return int(sub["learn_phase"].max())

    # expected_points here means "typical rows per phase" (inferred if None)
    if expected_points is None:
        expected_points = int(counts.max())

    needed = int(np.floor(min_coverage * expected_points))
    valid = counts[counts >= needed]
    return int(valid.index.max()) if not valid.empty else int(counts.idxmax())

def save_fig(fig, filename: str):
    out = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

def plot_mature_final_bar(clean_all: pd.DataFrame, metric: str, filename: str):
    rows = []
    for model_id, cdf in clean_all.groupby("model_id"):
        lp_max = last_learn_phase_per_model(cdf, metric=metric, aggregation="final")
        sub = cdf[
            (cdf["metric"] == metric) &
            (cdf["aggregation"] == "final") &
            (cdf["learn_phase"] == lp_max)
        ]
        if sub.empty:
            continue
        # single value per model (already averaged across episodes in transform)
        val = float(sub["value"].iloc[0])
        rows.append((model_id, val))

    if not rows:
        print(f"[WARN] No data for mature-final bar plot metric={metric}")
        return

    # sort by value for readability
    rows = sorted(rows, key=lambda x: x[1])
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title(f"Mature (last learn_phase) — FINAL — {metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)

    save_fig(fig, filename)

def plot_learning_evolution_all_models(clean_all: pd.DataFrame, metric: str, filename: str):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    for model_id, cdf in clean_all.groupby("model_id"):
        sub = cdf[
            (cdf["metric"] == metric) &
            (cdf["aggregation"] == "episode_mean")
        ].sort_values("learn_phase")

        if sub.empty:
            continue

        ax.plot(sub["learn_phase"].values, sub["value"].values, label=model_id)

    ax.set_title(f"Learning evolution — EPISODE MEAN — {metric}")
    ax.set_xlabel("learn_phase")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)

    save_fig(fig, filename)

def plot_time_in_episode_per_model(cdf: pd.DataFrame, model_id: str, metric: str, filename: str):
    sub = cdf[
        (cdf["metric"] == metric) &
        (cdf["aggregation"] == "time")
    ].dropna(subset=["t_in_episode"]).copy()

    if sub.empty:
        print(f"[WARN] No time data for model={model_id} metric={metric}")
        return

    sub["learn_phase"] = pd.to_numeric(sub["learn_phase"], errors="coerce")
    sub = sub.dropna(subset=["learn_phase"])

    lp_min = int(sub["learn_phase"].min())
    lp_max = last_learn_phase_per_model(cdf, metric=metric, aggregation="time")

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # 1) all phases faint (behind)
    for lp, grp in sub.groupby("learn_phase"):
        lp_i = int(lp)
        if lp_i in (lp_min, lp_max):
            continue
        grp = grp.sort_values("t_in_episode")
        ax.plot(
            grp["t_in_episode"].values, grp["value"].values,
            linewidth=0.8, alpha=0.25, zorder=1
        )

    # 2) highlights on top with forced colors + outline
    def plot_phase(lp_i: int, label: str, color: str, lw: float = 4.0):
        grp = sub[sub["learn_phase"] == lp_i].sort_values("t_in_episode")
        if grp.empty:
            print(f"[WARN] highlight phase {lp_i} missing for {model_id}/{metric}")
            return
        (line,) = ax.plot(
            grp["t_in_episode"].values, grp["value"].values,
            linewidth=lw, alpha=1.0, color=color, label=label, zorder=100
        )
        # outline so it stays visible inside the spaghetti
        line.set_path_effects([pe.Stroke(linewidth=lw + 2.5, foreground="white"), pe.Normal()])

    plot_phase(lp_min, f"learn_phase {lp_min} (first)", color="tab:cyan")
    plot_phase(lp_max, f"learn_phase {lp_max} (last)",  color="tab:blue")

    ax.set_title(f"{model_id} — TIME in episode — {metric} (all learn_phases)")
    ax.set_xlabel("t_in_episode")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)

    save_fig(fig, filename)

def plot_diag_episode_mean_all_models(clean_all: pd.DataFrame, metric: str):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    for model_id, cdf in clean_all.groupby("model_id"):
        sub = cdf[
            (cdf["metric"] == metric) &
            (cdf["aggregation"] == "episode_mean")
        ].sort_values("learn_phase")

        if sub.empty:
            continue

        ax.plot(sub["learn_phase"].values, sub["value"].values, label=model_id)

    ax.set_title(f"Learning evolution — EPISODE MEAN — {metric}")
    ax.set_xlabel("learn_phase")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)

    save_fig(fig, f"diag_episode_mean_all_models__{metric}.png")

def plot_diag_time_in_episode_per_model(cdf, model_id: str, metric: str, highlight_phases=(0, 30)):
    sub = cdf[
        (cdf["metric"] == metric) &
        (cdf["aggregation"] == "time")
    ].dropna(subset=["t_in_episode"]).copy()

    if sub.empty:
        print(f"[WARN] No time data for model={model_id} metric={metric}")
        return

    hp0, hp1 = highlight_phases

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # Plot all learn_phases faintly first
    for lp, grp in sub.groupby("learn_phase"):
        grp = grp.sort_values("t_in_episode")
        ax.plot(
            grp["t_in_episode"].values,
            grp["value"].values,
            linewidth=0.8,
            alpha=0.25
        )

    # Highlight learn_phase 0 and 30 (if present)
    def highlight_phase(lp, label):
        grp = sub[sub["learn_phase"] == lp].sort_values("t_in_episode")
        if grp.empty:
            return
        ax.plot(
            grp["t_in_episode"].values,
            grp["value"].values,
            linewidth=2.8,
            label=label
        )

    highlight_phase(hp0, f"learn_phase {hp0}")
    highlight_phase(hp1, f"learn_phase {hp1}")

    ax.set_title(f"{model_id} — TIME in episode — {metric} (all learn_phases)")
    ax.set_xlabel("t_in_episode")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=9)

    save_fig(fig, f"diag_time_in_episode__{model_id}__{metric}.png")


##########################################################################
# PIPELINE: do the plots
##########################################################################


# 0 folder for plots
PLOTS_DIR = Path("ML/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("Generating plots...")
# 1 mature final bar plots wealth and cash
print("\n--- DIAG CLEAN_DATASETS ---")
for model_id, cdf in CLEAN_ALL.groupby("model_id"):
    print(
        model_id,
        "rows=", len(cdf),
        "learn_phase_all_nan=", cdf["learn_phase"].isna().all() if "learn_phase" in cdf.columns else "NO_COL",
        "learn_phase_max=", cdf["learn_phase"].max() if len(cdf) else "EMPTY_DF"
    )
plot_mature_final_bar(CLEAN_ALL, metric="wealth", filename="bar_mature_final__wealth.png")
plot_mature_final_bar(CLEAN_ALL, metric="cash", filename="bar_mature_final__cash.png")


# 2 learning evolution plots wealth and cash

plot_learning_evolution_all_models(CLEAN_ALL, metric="wealth", filename="learning_evolution_all_models__wealth.png")
plot_learning_evolution_all_models(CLEAN_ALL, metric="cash", filename="learning_evolution_all_models__cash.png")

# 3 time in episode plots per model for wealth and cash
for model_id, cdf in CLEAN_ALL.groupby("model_id"):
    plot_time_in_episode_per_model(
        cdf, model_id=model_id, metric="wealth",
        filename=f"time_in_episode__{model_id}__wealth.png"
    )
    plot_time_in_episode_per_model(
        cdf, model_id=model_id, metric="cash",
        filename=f"time_in_episode__{model_id}__cash.png"
    )

DIAG_METRICS = ["reward", "budget_penalty", "resource_penalty"]

# 4) Episode-mean evolution: all models together (one figure per metric)
for m in DIAG_METRICS:
    plot_diag_episode_mean_all_models(CLEAN_ALL, metric=m)

# 5) Time-in-episode: per model per metric
for model_id, cdf in CLEAN_ALL.groupby("model_id"):
    for m in DIAG_METRICS:
        plot_diag_time_in_episode_per_model(cdf, model_id=model_id, metric=m, highlight_phases=(0, 30))

RISK_METRICS = ["wealth_volatility", "sharpe_ratio"]

for m in RISK_METRICS:
    plot_diag_episode_mean_all_models(CLEAN_ALL, metric=m)

print(
    CLEAN_ALL.query("aggregation=='episode_mean' and metric=='sharpe_ratio'")
            .groupby("model_id")["learn_phase"]
            .agg(["min","max","nunique"])
)
print(f"Saved plots to: {PLOTS_DIR.resolve()}")
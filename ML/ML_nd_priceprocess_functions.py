import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import Union, Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch  
import time

#########################################################################################
# Global transition logger ("flight recorder")
#########################################################################################

# We store one row per env.step() as a plain dict (fast append). At the end of a run,
# call `log_to_dataframe()` to get a pandas DataFrame.
#
# IMPORTANT:
# - This is process-local. If you use SubprocVecEnv, each subprocess will have its own
#   copy of this global state (you would need a different logging strategy).

LOG_ROWS: list[dict] = []
LOG_META: dict = {}


def log_reset(
    *,
    run_id: str | None = None,
    n_steps: int | None = None,
    n_envs: int = 1,
    extra_meta: dict | None = None,
) -> None:
    """Reset the global logger."""
    global LOG_ROWS, LOG_META
    LOG_ROWS = []
    LOG_META = {
        "run_id": run_id,
        "n_steps": int(n_steps) if n_steps is not None else None,
        "n_envs": int(n_envs),
        "created_at_unix": time.time(),
    }
    if extra_meta:
        LOG_META.update(extra_meta)


def log_append(row: dict) -> None:
    """Append one transition row (dict) to the global logger."""
    # Shallow copy so later mutations don't affect stored history
    LOG_ROWS.append(dict(row))


def log_to_dataframe() -> pd.DataFrame:
    """Convert accumulated rows to a pandas DataFrame."""
    if not LOG_ROWS:
        return pd.DataFrame()
    return pd.DataFrame(LOG_ROWS)


def log_get_meta() -> dict:
    """Return a copy of logger metadata."""
    return dict(LOG_META)


#########################################################################################
# Helpers functions for matrix construction of price process parameters
#########################################################################################
def diag_from_dict(goods, values, default=0.0, name="diag"):
    """
    Create a diagonal DataFrame with labels.
    values can be dict {good: val} or scalar.
    """
    if np.isscalar(values):
        vals = {g: float(values) for g in goods}
    else:
        vals = {g: float(values.get(g, default)) for g in goods}

    df = pd.DataFrame(0.0, index=goods, columns=goods)
    for g, v in vals.items():
        df.loc[g, g] = v
    df.attrs["name"] = name
    return df


def matrix_from_pairs(goods, pairs, default=0.0, name="matrix", symmetric=False):
    """
    Build a full labeled matrix from a dict of pairwise values:
      pairs = {("food","metal"): 0.1, ("metal","food"): 0.1, ...}
    If symmetric=True, fills both (i,j) and (j,i) with the same value.
    """
    df = pd.DataFrame(default, index=goods, columns=goods, dtype=float)
    for (i, j), v in pairs.items():
        df.loc[i, j] = float(v)
        if symmetric:
            df.loc[j, i] = float(v)
    df.attrs["name"] = name
    return df


def correlation_from_pairs(goods, pairs, default_offdiag=0.0, name="Corr", fix_to_spd=True):
    """
    Create a correlation matrix with 1s on diagonal.
    Optionally "fix" it to be SPD for safe sampling of multivariate normals.
    """
    corr = pd.DataFrame(default_offdiag, index=goods, columns=goods, dtype=float)
    np.fill_diagonal(corr.values, 1.0)

    for (i, j), v in pairs.items():
        corr.loc[i, j] = float(v)
        corr.loc[j, i] = float(v)

    # Optional SPD fix (small eigenvalue floor)
    if fix_to_spd:
        A = corr.values
        # symmetrize
        A = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(A)
        w = np.clip(w, 1e-8, None)
        A_spd = (V * w) @ V.T
        # renormalize to correlation (diag=1)
        d = np.sqrt(np.diag(A_spd))
        A_spd = A_spd / np.outer(d, d)
        corr = pd.DataFrame(A_spd, index=goods, columns=goods)

    corr.attrs["name"] = name
    return corr


def covariance_from_corr(goods, corr, sigma, name="Sigma"):
    """
    Sigma = D * Corr * D, where D = diag(sigmas).
    sigma: dict {good: std_dev} or scalar
    """
    if np.isscalar(sigma):
        sig = pd.Series({g: float(sigma) for g in goods})
    else:
        sig = pd.Series({g: float(sigma[g]) for g in goods})

    D = np.diag(sig.values)
    Sigma = D @ corr.values @ D
    df = pd.DataFrame(Sigma, index=goods, columns=goods)
    df.attrs["name"] = name
    return df


def init_market_timeseries(goods, n_periods: int = 0, preallocate: bool = False):
    """
    Create separate time series containers for:
      - prices p_t (DataFrame)
      - transactions Q_t (DataFrame)
      - state s_t (DataFrame)
      - inventory inv_t (DataFrame)
      - cash cash_t (Series)

    Convention:
      - Q is defined for periods t = 0..n_periods-1  (flow during a period)
      - p, s, inv, cash are defined for t = 0..n_periods (state at boundaries, includes t=0 initial)

    If preallocate=False: empty but correctly typed containers.
    If preallocate=True: zero-filled containers with time index.
    """
    goods = list(goods)

    if not preallocate:
        p_ts   = pd.DataFrame({g: pd.Series(dtype="float64") for g in goods})
        Q_ts   = pd.DataFrame({g: pd.Series(dtype="float64") for g in goods})
        s_ts   = pd.DataFrame({g: pd.Series(dtype="float64") for g in goods})
        inv_ts = pd.DataFrame({g: pd.Series(dtype="float64") for g in goods})
        cash_ts = pd.Series(dtype="float64", name="cash")
        return p_ts, Q_ts, s_ts, inv_ts, cash_ts

    Q_index = range(n_periods)          # 0..n_periods-1
    state_index = range(n_periods + 1)  # 0..n_periods

    Q_ts    = pd.DataFrame(0.0, index=Q_index, columns=goods)
    p_ts    = pd.DataFrame(0.0, index=state_index, columns=goods)
    s_ts    = pd.DataFrame(0.0, index=state_index, columns=goods)
    inv_ts  = pd.DataFrame(0.0, index=state_index, columns=goods)
    cash_ts = pd.Series(0.0, index=state_index, name="cash")

    return p_ts, Q_ts, s_ts, inv_ts, cash_ts


###########################################################################################
# Initalization of some varaible and objects
###########################################################################################

#Initialize the timeseries dataframes
def init_market_timeseries(goods, n_periods: int = 0, preallocate: bool = False):
    """
    Create separate time series containers for:
      - prices p_t (DataFrame)
      - transactions Q_t (DataFrame)
      - state s_t (DataFrame)
      - inventory inv_t (DataFrame)
      - cash cash_t (Series)

    Convention:
      - Q is defined for periods t = 0..n_periods-1  (flow during a period)
      - p, s, inv, cash are defined for t = 0..n_periods (state at boundaries, includes t=0 initial)

    If preallocate=False: empty but correctly typed containers.
    If preallocate=True: zero-filled containers with time index.
    """
    goods = list(goods)

    if not preallocate:
        p_ts   = pd.DataFrame({g: pd.Series(dtype="float64") for g in goods})
        Q_ts   = pd.DataFrame({g: pd.Series(dtype="float64") for g in goods})
        s_ts   = pd.DataFrame({g: pd.Series(dtype="float64") for g in goods})
        inv_ts = pd.DataFrame({g: pd.Series(dtype="float64") for g in goods})
        cash_ts = pd.Series(dtype="float64", name="cash")
        return p_ts, Q_ts, s_ts, inv_ts, cash_ts

    Q_index = range(n_periods)          # 0..n_periods-1
    state_index = range(n_periods + 1)  # 0..n_periods

    Q_ts    = pd.DataFrame(0.0, index=Q_index, columns=goods)
    p_ts    = pd.DataFrame(0.0, index=state_index, columns=goods)
    s_ts    = pd.DataFrame(0.0, index=state_index, columns=goods)
    inv_ts  = pd.DataFrame(0.0, index=state_index, columns=goods)
    cash_ts = pd.Series(0.0, index=state_index, name="cash")

    return p_ts, Q_ts, s_ts, inv_ts, cash_ts

#define the initial money for the agent
def draw_initial_money(
    low: float = 50_000.0,
    high: float = 500_000.0,
    seed: int | None = None,
):
    """
    Draw initial money uniformly at random between low and high.
    """
    rng = np.random.default_rng(seed)
    return float(rng.uniform(low, high))


def init_market_timeseries_newround(
    p_ts: pd.DataFrame,
    Q_ts: pd.DataFrame,
    s_ts: pd.DataFrame,
    cash_ts: pd.Series | pd.DataFrame,
    params: dict,
    initial_cash: float,
):
    """
    Initialize time series at the beginning of a new round (t = 0).

    Rules:
    - p_ts[0] = exp(mu)
    - Q_ts[0] = 0
    - s_ts[0] = 0
    - cash_ts[0] = initial_cash
    - inv_ts is handled separately

    Returns updated (p_ts, Q_ts, s_ts, cash_ts)
    """

    goods = list(params["goods"])

    # --- Prices: p_0 = exp(mu) ---
    p0 = np.exp(params["mu"].reindex(goods).astype(float))
    p_ts.loc[0, goods] = p0.values

    # --- Trades: Q_0 = 0 ---
    Q_ts.loc[0, goods] = 0.0

    # --- Persistent impact: s_0 = 0 ---
    s_ts.loc[0, goods] = 0.0

    # --- Cash ---
    if isinstance(cash_ts, pd.DataFrame):
        cash_ts.loc[0, "cash"] = float(initial_cash)
    else:
        cash_ts.loc[0] = float(initial_cash)

    return p_ts, Q_ts, s_ts, cash_ts
###########################################################################################
# Market simulaiton of price. 
###########################################################################################

# Generate random Q time series for fine tuning
def fill_random_Q_ts(
    Q_ts: pd.DataFrame,
    goods,
    V,
    n_periods: int,
    max_frac: float = 0.10,
    intensity: float = 0.40,
    p_active: float = 0.60,
    seed: int | None = None,
):
    """
    Append n_periods of random Q[t, good] to an existing (possibly empty) Q_ts.
    """
    rng = np.random.default_rng(seed)
    goods = list(goods)

    # Determine time index
    start_t = 0 if Q_ts.empty else int(Q_ts.index.max()) + 1
    idx = range(start_t, start_t + n_periods)

    # Align V
    if isinstance(V, pd.Series):
        V_s = V.reindex(goods).astype(float)
    else:
        V_s = pd.Series(V, index=goods, dtype=float).reindex(goods)

    caps = max_frac * V_s.values

    active = rng.random((n_periods, len(goods))) < p_active

    a = 1.0
    b = 1.0 / max(1e-6, intensity)
    mag01 = rng.beta(a, b, size=(n_periods, len(goods)))

    sign = rng.choice([-1.0, 1.0], size=(n_periods, len(goods)))

    Q = active * sign * (mag01 * caps)

    chunk = pd.DataFrame(Q, index=idx, columns=goods)

    # Append
    Q_ts = pd.concat([Q_ts, chunk], axis=0)
    return Q_ts


def simulate_market(
    params: dict,
    n_periods: int,
    Q: pd.DataFrame | None = None,
    x0: pd.Series | None = None,
    s0: pd.Series | None = None,
    seed: int | None = None,
    plot: bool = True,
):
    """
    Simulate the market model:

    p_{t+1} = exp(x_{t+1})

    x_{t+1} = mu + R(x_t - mu) + A*u_t + B*s_t - C*s_t + eps_{t+1}
    u_t     = Q_t / V
    s_{t+1} = Lambda*s_t + u_t

    Notes:
    - We treat missing/NaN Q values as 0.
    - Uses correlated shocks eps ~ N(0, Sigma).
    - Returns x_df and p_df (both indexed by t=0..n_periods).
    """

    goods = list(params["goods"])
    N = len(goods)

    # --- Extract labeled objects ---
    mu = params["mu"].reindex(goods).astype(float)          # Series
    V = params["V"].reindex(goods).astype(float)            # Series

    R = params["R"].reindex(index=goods, columns=goods).astype(float).values
    A = params["A"].reindex(index=goods, columns=goods).astype(float).values
    B = params["B"].reindex(index=goods, columns=goods).astype(float).values
    Lam = params["Lambda"].reindex(index=goods, columns=goods).astype(float).values
    C = params["C"].reindex(index=goods, columns=goods).astype(float).values
    Sigma = params["Sigma"].reindex(index=goods, columns=goods).astype(float).values

    mu_np = mu.values
    V_np = V.values

    # --- Initial states ---
    if x0 is None:
        x = mu_np.copy()
    else:
        x = x0.reindex(goods).astype(float).values

    if s0 is None:
        s = np.zeros(N, dtype=float)
    else:
        s = s0.reindex(goods).astype(float).values

    # --- Build Q matrix (t=0..n_periods-1) ---
    # Q[t] is the trade executed during period t (affects x_{t+1})
    if Q is None:
        Q_df = pd.DataFrame(0.0, index=range(n_periods), columns=goods)
    else:
        # Reindex to required shape, missing -> NaN, then fill with 0
        Q_df = Q.copy()
        Q_df = Q_df.reindex(index=range(n_periods), columns=goods)
        Q_df = Q_df.fillna(0.0)

    Q_np = Q_df.values.astype(float)

    # --- RNG ---
    rng = np.random.default_rng(seed)

    # --- Storage for outputs (t=0..n_periods) ---
    x_path = np.zeros((n_periods + 1, N), dtype=float)
    s_path = np.zeros((n_periods + 1, N), dtype=float)

    x_path[0, :] = x
    s_path[0, :] = s

    # --- Simulation loop ---
    for t in range(n_periods):
        # order flow scaled by depth
        u = Q_np[t, :] / V_np

        # correlated shock in log space
        eps = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma)

        # update x_{t+1}
        # (using your sign convention: +A*u +B*s -C*s)
        x_next = (
            mu_np
            + R @ (x - mu_np)
            + A @ u
            + B @ s
            - C @ s
            + eps
        )

        # update s_{t+1} (persistence + current order flow)
        s_next = Lam @ s + u

        x, s = x_next, s_next
        x_path[t + 1, :] = x
        s_path[t + 1, :] = s

    # --- Convert to labeled DataFrames ---
    x_df = pd.DataFrame(x_path, index=range(n_periods + 1), columns=goods)
    p_df = np.exp(x_df)

    # --- Plot prices ---
    # --- Plot prices (ONE window, grid layout) ---
    if plot:
        n_goods = len(goods)
        n_cols = 2                     # change to 3 later if many goods
        n_rows = math.ceil(n_goods / n_cols)

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(12, 4 * n_rows),
            sharex=True
        )

        axes = np.array(axes).reshape(-1)

        for ax, g in zip(axes, goods):
            ax.plot(p_df.index, p_df[g])
            ax.set_title(f"Price — {g}")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for ax in axes[len(goods):]:
            ax.axis("off")

        axes[-1].set_xlabel("Period")
        plt.tight_layout()
        plt.show()

    return x_df, p_df

# this funciton add to the inventory at the last period 0.5% of V /!\ also used in the initialization process
def inv_random_add(
    inventory_ts: pd.DataFrame,
    market_size_matrix: np.ndarray,
    inventory_add_ratio: float = 0.005,
    prob_increase: float = 0.5,
    seed: int | None = None,
):
    """
    Randomly add inventory shocks at the LAST period (in-place).

    For each good i independently:
      - with prob prob_increase:  inv_i += frac * V_i
      - with prob 1-prob_increase: no change

    No new period is added.
    """
    if len(inventory_ts) == 0:
        raise ValueError("inventory_ts is empty (no period to update)")

    last_idx = inventory_ts.index[-1]
    n_goods = inventory_ts.shape[1]

    rng = np.random.default_rng(seed)

    # Bernoulli draws per good
    add_mask = rng.random(n_goods) < prob_increase

    # Increment vector
    delta = add_mask * (inventory_add_ratio * market_size_matrix)

    inventory_ts.loc[last_idx] += delta


def inv_random_add_vec(
    inv_vec,
    V_np,
    rng: np.random.Generator,
    p_add: float = 0.5,
    fraction: float = 0.005,
    integer_units: bool = True,
):
    """
    Vector version of inv_random_add.

    For each good independently:
      - with probability (1 - p_add): add nothing
      - with probability p_add: add (fraction * V_i) units to inventory

    Parameters
    ----------
    inv_vec : array-like, shape (n_goods,)
        Current inventory vector.
    V_np : array-like, shape (n_goods,)
        Market size / volume per good (used to scale random additions).
    rng : np.random.Generator
        RNG owned by the environment (self.rng).
    p_add : float
        Probability to add for each good (independent).
    fraction : float
        Fraction of V to add if adding occurs.
    integer_units : bool
        If True, additions are rounded to integers (recommended if goods must be integer).

    Returns
    -------
    inv_updated : np.ndarray, shape (n_goods,)
    """
    inv = np.asarray(inv_vec, dtype=np.float32).copy()
    V = np.asarray(V_np, dtype=np.float32)

    # independent Bernoulli per good
    add_mask = rng.random(size=inv.shape[0]) < p_add

    add_amount = fraction * V
    if integer_units:
        add_amount = np.round(add_amount)

    inv[add_mask] += add_amount[add_mask]
    return inv


def simulate_market_one_step(
    params: dict,
    p_ts: pd.DataFrame,
    Q_ts: pd.DataFrame,
    s_ts: pd.DataFrame,
    current_simulation_length: int, # who is = t
    rng: np.random.Generator,
):
    """
    One-step market simulation (t -> t+1) using the model:

      p_{t+1} = exp(x_{t+1})

      x_{t+1} = mu + R(x_t - mu) + A*u_t + B*s_t - C*s_t + eps_{t+1}
      u_t     = Q_t / V
      s_{t+1} = Lambda*s_t + u_t
      eps_{t+1} ~ N(0, Sigma)

    Assumptions (per your choice "option 2"):
      - s_ts is already initialized and s_ts.loc[current_simulation_length] exists
      - p_ts.loc[current_simulation_length] exists
      - Q_ts.loc[current_simulation_length] exists
      - params contains keys: "mu","R","A","B","C","Sigma","Lambda","V"
        where values are (preferably) pandas objects aligned on goods
        (Series/DataFrame) or numpy arrays of matching shape.

    Returns:
      p_next: pd.Series (indexed by goods) for time t+1
      s_ts:   the same DataFrame object, updated with row t+1
    """

    # --- Read parameters as numpy arrays ---
    mu = np.asarray(params["mu"].values if hasattr(params["mu"], "values") else params["mu"], dtype=float)
    R = np.asarray(params["R"].values if hasattr(params["R"], "values") else params["R"], dtype=float)
    A = np.asarray(params["A"].values if hasattr(params["A"], "values") else params["A"], dtype=float)
    B = np.asarray(params["B"].values if hasattr(params["B"], "values") else params["B"], dtype=float)
    C = np.asarray(params["C"].values if hasattr(params["C"], "values") else params["C"], dtype=float)
    Sigma = np.asarray(params["Sigma"].values if hasattr(params["Sigma"], "values") else params["Sigma"], dtype=float)
    Lam = np.asarray(params["Lambda"].values if hasattr(params["Lambda"], "values") else params["Lambda"], dtype=float)
    V = np.asarray(params["V"].values if hasattr(params["V"], "values") else params["V"], dtype=float)

    goods = list(p_ts.columns)
    N = len(goods)

    # --- Current state vectors ---
    p_t = p_ts.loc[current_simulation_length, goods].astype(float).values
    if np.any(p_t <= 0):
        raise ValueError(f"simulate_market_one_step: p_ts at t={current_simulation_length} has non-positive prices; cannot take log().")

    x_t = np.log(p_t)

    Q_t = Q_ts.loc[current_simulation_length, goods].astype(float).values
    u_t = Q_t / V  # order flow scaled by depth

    s_t = s_ts.loc[current_simulation_length, goods].astype(float).values

    # --- Shock ---
    eps = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma)

    # --- Update log-price ---
    x_next = (
        mu
        + R @ (x_t - mu)
        + A @ u_t
        + B @ s_t
        - C @ s_t
        + eps
    )

    # --- Update persistent impact ---
    s_next = Lam @ s_t + u_t

    # --- Convert back to price space ---
    p_next = np.exp(x_next)

    # --- Write next s into s_ts (option 2: s_ts must always be maintained) ---
    s_ts.loc[current_simulation_length + 1, goods] = s_next

    # Return next price as a labeled Series + updated s_ts
    return pd.Series(p_next, index=goods, name=current_simulation_length + 1), s_ts

############################################################################################
# ML functions
############################################################################################





#-----------------------------------------------------------------------------
# Window the data frame for the ML model
#-----------------------------------------------------------------------------


@dataclass
class WindowedMarketData:
    current_simulation_length: int
    window_size: int
    p_win: np.ndarray
    cash_win: np.ndarray
    inv_win: np.ndarray
    Q_win: np.ndarray


def _cash_to_series(cash_ts: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Accept Series or 1-col DataFrame; return Series."""
    if isinstance(cash_ts, pd.DataFrame):
        return cash_ts.iloc[:, 0]
    return cash_ts


def window_market_timeseries(
    p_ts: pd.DataFrame,
    Q_ts: pd.DataFrame,
    cash_ts: Union[pd.Series, pd.DataFrame],
    inv_ts: pd.DataFrame,
    current_simulation_length: int,
    window_size: int,
) -> WindowedMarketData:
    """
    Normal case (no padding): assumes enough history exists.

    At decision time t:
      - states (p, cash, inv) use [t-window_size+1 .. t]
      - actions (Q) use [t-window_size .. t-1]  (lagged; no Q_t)
    """
    cash_s = _cash_to_series(cash_ts)

    p_win = p_ts.loc[current_simulation_length - window_size + 1 : current_simulation_length]
    cash_win = cash_s.loc[current_simulation_length - window_size + 1 : current_simulation_length]
    inv_win = inv_ts.loc[current_simulation_length - window_size + 1 : current_simulation_length]
    Q_win = Q_ts.loc[current_simulation_length - window_size : current_simulation_length - 1]

    return WindowedMarketData(
    current_simulation_length=current_simulation_length,
    window_size=window_size,
    p_win=p_win,
    cash_win=cash_win,
    inv_win=inv_win,
    Q_win=Q_win,
)


def window_market_timeseries_padded(
    p_ts: pd.DataFrame,
    Q_ts: pd.DataFrame,
    cash_ts: Union[pd.Series, pd.DataFrame],
    inv_ts: pd.DataFrame,
    current_simulation_length: int,
    window_size: int,
    *,
    p_default: Union[pd.Series, dict, np.ndarray],
    initial_money: float,
) -> WindowedMarketData:
    """
    Early-simulation case: left-pad missing history with defaults, using ABSOLUTE time indices.

    Indices:
      - state window:  [current_simulation_length-window_size+1 .. current_simulation_length]
      - action window: [current_simulation_length-window_size .. current_simulation_length-1]

    Defaults used for missing times (<0):
      - p = p_default (e.g., exp(mu) per good)
      - cash = initial_money
      - inv = 0
      - Q = 0
    """
    cash_s = _cash_to_series(cash_ts)

    # Target indices (absolute timeline, may include negatives)
    state_idx = pd.Index(range(current_simulation_length - window_size + 1, current_simulation_length + 1), name="t_state")
    action_idx = pd.Index(range(current_simulation_length - window_size, current_simulation_length), name="t_action")  # ends at t-1

    # Build defaults aligned to columns
    p0 = pd.Series(p_default).reindex(p_ts.columns).astype(float)
    inv0 = pd.Series(0.0, index=inv_ts.columns, dtype=float)
    Q0 = pd.Series(0.0, index=Q_ts.columns, dtype=float)

    # Helper: build a full window by (1) creating a default frame (2) overwrite with real values where available
    def _build_df_window(ts: pd.DataFrame, idx: pd.Index, row_default: pd.Series) -> pd.DataFrame:
        win = pd.DataFrame(np.tile(row_default.values, (len(idx), 1)), index=idx, columns=row_default.index)
        # overwrite with available real rows (only for non-negative times)
        real = ts.loc[ts.index.intersection(pd.Index([i for i in idx if i >= 0]))]
        if len(real) > 0:
            win.loc[real.index, real.columns] = real
        return win

    def _build_series_window(ts: pd.Series, idx: pd.Index, default_value: float) -> pd.Series:
        win = pd.Series([default_value] * len(idx), index=idx, dtype=float)
        real_idx = [i for i in idx if i >= 0]
        if len(real_idx) > 0:
            real = ts.loc[ts.index.intersection(pd.Index(real_idx))]
            if len(real) > 0:
                win.loc[real.index] = real
        return win

    # Create padded windows
    p_win = _build_df_window(p_ts, state_idx, p0)
    inv_win = _build_df_window(inv_ts, state_idx, inv0)
    cash_win = _build_series_window(cash_s, state_idx, float(initial_money))
    Q_win = _build_df_window(Q_ts, action_idx, Q0)

    return WindowedMarketData(
    current_simulation_length=current_simulation_length,
    window_size=window_size,
    p_win=p_win,
    cash_win=cash_win,
    inv_win=inv_win,
    Q_win=Q_win,
)


def get_windowed_market_data(
    p_ts: pd.DataFrame,
    Q_ts: pd.DataFrame,
    cash_ts: Union[pd.Series, pd.DataFrame],
    inv_ts: pd.DataFrame,
    current_simulation_length: int,   # t
    window_size: int,                # W
    *,
    p_default: Optional[Union[pd.Series, dict, np.ndarray]] = None,
    initial_money: Optional[float] = None,
) -> WindowedMarketData:
    """
    Single entry point.

    If enough history exists (t >= window_size), use normal slicing.
    Otherwise, use padded windows (requires p_default and initial_money).
    """
    t = int(current_simulation_length)
    W = int(window_size)

    if t >= W:
        return window_market_timeseries(
            p_ts=p_ts,
            Q_ts=Q_ts,
            cash_ts=cash_ts,
            inv_ts=inv_ts,
            current_simulation_length=t,
            window_size=W,
        )

    if p_default is None or initial_money is None:
        raise ValueError("Early simulation (t < window_size) requires p_default and initial_money.")

    return window_market_timeseries_padded(
        p_ts=p_ts,
        Q_ts=Q_ts,
        cash_ts=cash_ts,
        inv_ts=inv_ts,
        current_simulation_length=t,
        window_size=W,
        p_default=p_default,
        initial_money=initial_money,
    )


def shape_windowed_data_in_vector_forML(win_datafor_ML, current_simulation_length=None):
    p = np.asarray(win_datafor_ML.p_win, dtype=np.float32).reshape(-1)
    Q = np.asarray(win_datafor_ML.Q_win, dtype=np.float32).reshape(-1)
    cash = np.asarray(win_datafor_ML.cash_win, dtype=np.float32).reshape(-1)
    inv = np.asarray(win_datafor_ML.inv_win, dtype=np.float32).reshape(-1)

    if current_simulation_length is None:
        t_val = float(win_datafor_ML.current_simulation_length)
    else:
        t_val = float(current_simulation_length)

    obs = np.concatenate([p, Q, cash, inv, np.array([t_val], dtype=np.float32)], axis=0)
    return obs


#-----------------------------------------------------------------------------
# Management of the Hyperparameters and the model architecture
#-----------------------------------------------------------------------------

def unwrap_base_env(env):
    """Return the underlying MarketEnv from VecEnv/Monitor wrappers."""
    if hasattr(env, "envs"):  # VecEnv (DummyVecEnv/SubprocVecEnv)
        env = env.envs[0]
    while hasattr(env, "env"):  # Monitor/TimeLimit/etc
        env = env.env
    return env

def push_hyperparams_to_env(env, ppo_kwargs: dict, total_training_steps: int | None = None):
    """
    Copy training hyperparameters into the base env so it can log timing
    without hardcoding values in the env.
    """
    base = unwrap_base_env(env)

    base.n_steps = int(ppo_kwargs.get("n_steps", 2048))
    base.n_epochs = int(ppo_kwargs.get("n_epochs", 0))
    base.batch_size = int(ppo_kwargs.get("batch_size", 0))
    base.learning_rate = float(ppo_kwargs.get("learning_rate", 0.0))
    base.gamma = float(ppo_kwargs.get("gamma", 0.0))

    if total_training_steps is not None:
        base.total_training_steps = int(total_training_steps)

    return base  # convenient if you want it

#sets of hyperparameters for the PPO model
#PPO_KWARGS = dict(learning_rate=1e-4,gamma=0.995,gae_lambda=0.95,clip_range=0.15,ent_coef=0.02,vf_coef=0.7,max_grad_norm=0.3,n_steps=2048,batch_size=256,n_epochs=10,policy_kwargs=dict(activation_fn=torch.nn.Tanh,net_arch=dict(pi=[256, 256], vf=[256, 256]),),)

def generate_model_architecture(dummy_env, ppo_kwargs):
  
    return PPO("MlpPolicy", dummy_env, **ppo_kwargs, verbose=0)


def make_vec_env(env_ctor, n_envs: int = 1):
    """
    Wraps your env constructor into a SB3 VecEnv (DummyVecEnv).
    env_ctor: callable with no args that returns a fresh env instance.
    """
    if n_envs != 1:
        # if later you want SubprocVecEnv etc, we can extend
        raise NotImplementedError("Only n_envs=1 supported in make_vec_env for now.")
    return DummyVecEnv([lambda: Monitor(env_ctor())])

def build_ppo_model(vec_env, ppo_kwargs: dict, seed: int = 0, verbose: int = 1):
    """
    PPO model factory. Hyperparameters come from Main_ML via ppo_kwargs.
    """
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=seed,
        verbose=verbose,
        **ppo_kwargs,
    )
    return model

#-----------------------------------------------------------------------------
# Gym Environment for the Market Simulation
#-----------------------------------------------------------------------------

class MarketEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        list_of_goods,
        params,                 # whatever you need for your price process
        window_size: int,
        max_simulation_length: int,
        initial_money: float,
        V_np: np.ndarray,       # INTERNAL only, model doesn't see it
        seed: int | None = None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.goods = list_of_goods
        self.n_goods = len(list_of_goods)
        self.V_np = np.asarray(V_np, dtype=np.float32) 

        self.params = params
        self.window_size = window_size
        self.max_simulation_length = max_simulation_length
        
        self.initial_money = float(initial_money)
        self.p_default = np.exp(np.asarray(self.params["mu"].values, dtype=np.float32))
        self.add_inventory_prob = 0.5        
        self.add_inv_fraction = 0.005

       
        

        # --- action space: unitless actions, one per good ---
        self.action_space = spaces.Box(
            low=-100.0, high=100.0, shape=(self.n_goods,), dtype=np.float32
        )

        # --- obs space: we will set obs_dim once we know it ---
        # We'll build a sample obs in reset() and then freeze the space.
        # --- obs space: fixed dimension (required by SB3 check_env) ---
        window_size = int(self.window_size)
        number_of_goods = int(self.n_goods)
        obs_dim = window_size * (3 * number_of_goods + 1) + 1  # p, Q, inv are (W*n), cash is (W), plus t

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # state holders
        self.current_simulation_length = 0
        self.global_step = 0
        self.episode_id = 0

        self.p_ts = None
        self.Q_ts = None
        self.cash_ts = None
        self.inv_ts = None

        # Initialize time measurement
        self.n_epochs = None
        self.batch_size = None
        self.n_steps = None
        self.total_training_steps = None
        self.wall_start_time = time.perf_counter()
        self.step_times = []
        self.learn_times = []

        self.last_step_time = None
        self.last_learn_time = None
        

    def _get_obs(self) -> np.ndarray:
        win = get_windowed_market_data(
            p_ts=self.p_ts,
            Q_ts=self.Q_ts,
            cash_ts=self.cash_ts,
            inv_ts=self.inv_ts,
            current_simulation_length=self.current_simulation_length,
            window_size=self.window_size,
            p_default=self.p_default,          # you already store exp(mu) in __init__ :contentReference[oaicite:1]{index=1}
            initial_money=self.initial_money,
        )
        
        
        obs = shape_windowed_data_in_vector_forML(win,current_simulation_length=win.current_simulation_length)
        obs = np.asarray(obs).reshape(-1)  # ensure 1D
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = obs.astype(np.float32, copy=False)
        assert obs.shape == self.observation_space.shape, (obs.shape, self.observation_space.shape)

        return obs.astype(np.float32, copy=False)


    def reset(self, seed=None, options=None):
        super().reset(seed=None)
        print("RESET CALLED", seed)
        self.current_simulation_length = 0
        self.episode_id += 1

        # Initialize your time series (prices, Q, cash, inv)
        # Use your init functions here:
        self.p_ts, self.Q_ts, self.s_ts, self.inv_ts, self.cash_ts = (
            init_market_timeseries(self.goods, preallocate=False)
        )

        # Set initial conditions for new round (you said you have/plan init_market_timeseries_newround)
        # Example logic (adapt to your function):
        self.p_ts.loc[self.current_simulation_length, :] = np.exp(self.params["mu"].values)  # if you store mu like this
        self.Q_ts.loc[self.current_simulation_length, :] = 0.0
        self.cash_ts.loc[self.current_simulation_length] = self.initial_money
        self.inv_ts.loc[self.current_simulation_length, :] = 0.0
        self.s_ts.loc[self.current_simulation_length, :] = 0.0

        obs = self._get_obs()

        # Freeze observation_space on first reset (needs fixed obs_dim)
        if self.observation_space is None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float32
            )

        return obs, {}


    def step(self, action):
        t0 = time.perf_counter()

        # --- 1) sanitize action ---
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -100.0, 100.0)

        # --- 2) map unitless action -> actual trade Q (internal cap uses V) ---
        q_max = self.V_np                                        # shape (n_goods,)
        Q_desired = a/100 * q_max                                # shape (n_goods,)

        # --- 3) apply feasibility / bounds (cash etc.) ---
        # You implement: adjust Q to avoid negative cash, apply trade caps, etc.
        cash_t = float(self.cash_ts.loc[self.current_simulation_length])
        p_t = self.p_ts.loc[self.current_simulation_length, :].values.astype(np.float32)
        inv_t = self.inv_ts.loc[self.current_simulation_length, :].values.astype(np.float32)

        Q_exec,budget_penalty,resource_penalty = self.apply_feasibility(Q_desired, p_t, inv_t, cash_t)
        # Save executed Q in timeseries at time t
        self.Q_ts.loc[self.current_simulation_length, :] = Q_exec
        # --- 4) transition dynamics: compute next prices, next inv, next cash ---
        # This is your market “reaction” function. Must output p_next.
        

        inv_next, cash_next, buy_cost, sell_proceeds = self.update_portfolio(inv_t, cash_t, p_t, Q_exec)

        p_next, _ = simulate_market_one_step(
            params=self.params,
            p_ts=self.p_ts,
            Q_ts=self.Q_ts,
            s_ts=self.s_ts,
            current_simulation_length=self.current_simulation_length,
            rng=self.rng
        )

        # store next state
        current_simulation_length_next = self.current_simulation_length + 1
        self.global_step += 1
        self.p_ts.loc[current_simulation_length_next, :] = p_next
        self.inv_ts.loc[current_simulation_length_next, :] = inv_next
        self.cash_ts.loc[current_simulation_length_next] = cash_next
        
        # --- 5) reward ---
        # Choose ONE (wealth-based is often better)
        wealth_t = cash_t + float(np.sum(inv_t * p_t))
        wealth_next = cash_next + float(np.sum(inv_next * p_next))
        reward = float(wealth_next - wealth_t - budget_penalty - resource_penalty)

        # --- 6) advance time & termination ---
        self.current_simulation_length = current_simulation_length_next
        terminated = (self.current_simulation_length >= self.max_simulation_length)      # horizon end
        truncated=False

        obs = self._get_obs()
        info = {
            "cash": cash_next,
            "wealth": wealth_next,
            "Q_exec": Q_exec,
        }

        #add info into the logging system
        row = {
                # Identifiers
                "run_id": getattr(self, "run_id", None),
                "env_id": getattr(self, "env_id", 0),
                "episode_id": self.episode_id,
                "t_in_episode": self.current_simulation_length,
                "global_step": self.global_step,
                "initial_money": self.initial_money,

                # BEFORE action (t)
                "cash_t": float(cash_t),
                "wealth_t": float(wealth_t),

                # AFTER transition (t+1)
                "cash_next": float(cash_next),
                "wealth_next": float(wealth_next),

                # Execution accounting
                "buy_cost": float(buy_cost),
                "sell_proceeds": float(sell_proceeds),
                "budget_penalty": float(budget_penalty),
                "resource_penalty": float(resource_penalty),

                # Learning signal
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
                # Per-good vectors (expanded to columns)
        for i, good in enumerate(self.goods):
            row[f"p_t_{good}"] = float(p_t[i])
            row[f"p_next_{good}"] = float(p_next.loc[good])

            row[f"inv_t_{good}"] = float(inv_t[i])
            row[f"inv_next_{good}"] = float(inv_next[i])

            row[f"s_t_{good}"] = float(self.s_ts.loc[self.current_simulation_length, good])
            

            row[f"a_t_{good}"] = float(action[i])
            row[f"Q_desired_{good}"] = float(Q_desired[i])
            row[f"Q_exec_{good}"] = float(Q_exec[i])

        log_append(row)

        if terminated:
            self._print_episode_timing()
        t1 = time.perf_counter()
        self.step_times.append(t1 - t0)
        
        return obs, reward, terminated,truncated, info


    def apply_feasibility(self,Q_desired, p_t, inv_t, cash_t):
        """
        Enforce:
        1) Budget feasibility on buys (Q>0): scale down buys proportionally if cost > cash
        2) Resource feasibility on sells (Q<0): cap sells so you don't sell more than inventory

        Conventions:
        - Q[i] > 0 : buy quantity of good i
        - Q[i] < 0 : sell quantity of good i
        - inv_t[i] >= 0 assumed (no short inventory)
        - p_t[i] > 0

        Returns:
        Q_exec: feasible trade vector (float)
        budget_penalty: cost_raw - cost_adjusted  (>= 0)
        resource_penalty: total_sell_raw - total_sell_adjusted (>= 0), measured in units sold
        """
        Q = np.asarray(Q_desired, dtype=np.float32).copy()
        p = np.asarray(p_t, dtype=np.float32)
        inv = np.asarray(inv_t, dtype=np.float32)
        cash = float(cash_t)

        # ----------------------------
        # 1) Budget feasibility (buys)
        # ----------------------------
        buy = np.maximum(Q, 0.0)                   # keep only buys
        cost_raw = float(np.sum(buy * p))

        if cost_raw > cash and cost_raw > 0.0:
            scale = cash / cost_raw               # same % scaling for all positive Q
            buy_adj = buy * scale
            Q[Q > 0.0] = buy_adj[Q > 0.0]
            cost_adj = float(np.sum(buy_adj * p))
        else:
            cost_adj = cost_raw

        budget_penalty = max(0.0, cost_raw - cost_adj)

        # --------------------------------
        # 2) Resource feasibility (sells)
        # --------------------------------
        # Desired sells in *positive units*
        sell_raw_units = np.maximum(-Q, 0.0)      # if Q=-5 => sell_raw_units=5

        # Cap sells by available inventory
        sell_adj_units = np.minimum(sell_raw_units, inv)

        # Put back into Q (negative for sells)
        Q[Q < 0.0] = -sell_adj_units[Q < 0.0]

        resource_penalty = float(np.sum(sell_raw_units) - np.sum(sell_adj_units))
        if resource_penalty < 0.0:  # numerical safety
            resource_penalty = 0.0

        return Q, budget_penalty, resource_penalty


    def update_portfolio(self, inv_t, cash_t, p_t, Q_exec):
        rng = self.rng
        V_np = self.V_np
        """
        Update cash and inventory based on executed trades at time t, using current prices p_t.
        This is called BEFORE simulating p_{t+1}, consistent with your assumption that trades
        impact next-period prices (not current-period execution price).
        
        Conventions:
        - Q_exec[i] > 0 : buy quantity of good i
        - Q_exec[i] < 0 : sell quantity of good i
        - Trades clear at price p_t (current period)

        Steps:
        1) cash_next = cash_t - sum(buys * p_t) + sum(sells * p_t)
        2) inv_next  = inv_t + Q_exec
        3) inv_next  = inv_random_add_vec(inv_next, V_np, rng, ...)

        Returns
        -------
        inv_next : np.ndarray, shape (n_goods,)
        cash_next : float
        buy_cost : float
        sell_proceeds : float
        """
        inv = np.asarray(inv_t, dtype=np.float32)
        Q = np.asarray(Q_exec, dtype=np.float32)
        p = np.asarray(p_t, dtype=np.float32)

        buy_qty = np.maximum(Q, 0.0)
        sell_qty = np.maximum(-Q, 0.0)

        buy_cost = float(np.sum(buy_qty * p))
        sell_proceeds = float(np.sum(sell_qty * p))

        cash_next = float(cash_t) - buy_cost + sell_proceeds

        inv_next = inv + Q
        # If you disallow negative inventory, feasibility should have prevented it,
        # but we can keep a hard safety floor at 0 if desired:
        # inv_next = np.maximum(inv_next, 0.0)

        inv_next = inv_random_add_vec(
            inv_vec=inv_next,
            V_np=self.V_np,
            rng=self.rng,
            p_add=self.add_inventory_prob,
            fraction=self.add_inv_fraction,
        )
        inv_next = np.round(inv_next)#be sure that we don't have fractional inventory

        return inv_next, cash_next, buy_cost, sell_proceeds


    def _print_episode_timing(self):
    
        elapsed = time.perf_counter() - self.wall_start_time

        avg_step = np.mean(self.step_times[-self.max_simulation_length:])
        total_steps_done = len(self.step_times)

        # PPO rollout logic
        n_steps = self.n_steps  # pass this into env from main!
        total_training_steps = self.total_training_steps  # same
        remaining_steps = max(total_training_steps - total_steps_done, 0)

        est_rollout_time = remaining_steps * avg_step

        avg_learn = np.mean(self.learn_times) if self.learn_times else 0.0
        remaining_updates = remaining_steps // n_steps
        est_learn_time = remaining_updates * avg_learn

        print(
            "\n[EPISODE END]"
            f"\n  Avg step time        : {avg_step:.4f}s"
            f"\n  Total elapsed time   : {elapsed/60:.2f} min"
            f"\n  Est rollout remaining: {est_rollout_time/60:.2f} min"
            f"\n  Est learn remaining  : {est_learn_time/60:.2f} min"
            f"\n  Est total remaining  : {(est_rollout_time + est_learn_time)/60:.2f} min\n"
        )

def smoke_test_env(env, n_episodes: int = 3, seed: int = 0):
    # Gymnasium reset signature: obs, info
    obs, info = env.reset(seed=seed)

    for ep in range(n_episodes):
        terminated = False
        truncated = False
        ep_reward = 0.0
        step_count = 0

        last_info = {}
        while not (terminated or truncated):
            action = env.action_space.sample()  # random policy
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            step_count += 1
            last_info = info

            # Optional: quick NaN check
            if not np.isfinite(obs).all():
                raise ValueError(f"Non-finite obs at step {step_count} (episode {ep})")
            if not np.isfinite(reward):
                raise ValueError(f"Non-finite reward at step {step_count} (episode {ep})")

        print(
            f"[Episode {ep}] steps={step_count} "
            f"sum_reward={ep_reward:,.2f} "
            f"cash={last_info.get('cash', None)} "
            f"wealth={last_info.get('wealth', None)}"
        )

        obs, info = env.reset(seed=seed + ep + 1)




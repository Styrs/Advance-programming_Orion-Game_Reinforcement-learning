import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from copy import deepcopy

import ML_nd_priceprocess_functions as fn
from ML_nd_priceprocess_functions import log_reset
import torch
import time
import os
import torch
import prices_process 


# -------------------------------------------------
# 1) CONFIG BUILDER FUNCTION
# -------------------------------------------------
#There you can change the hyperparameters of the PPO and the ML process
def make_config(
    *,
    # Exogenous env settings
    seed: int = 0,
    list_of_goods=None,
    window_size: int = 20,
    episode_len_t: int = 200,
    train_episodes_chunks: int = 300,

    # PPO hyperparameters
    learning_rate: float = 1e-4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.15,
    ent_coef: float = 0.02,
    vf_coef: float = 0.7,
    max_grad_norm: float = 0.3,
    n_steps: int = 2000,
    batch_size: int = 256,
    n_epochs: int = 10,

    # Policy architecture
    activation_fn=torch.nn.Tanh,
    pi_arch=(256, 256),
    vf_arch=(256, 256),
):
    if list_of_goods is None:
        list_of_goods = ["food", "metal", "industrial", "consumption"]

    ppo_kwargs = dict(
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        policy_kwargs=dict(
            activation_fn=activation_fn,
            net_arch=dict(pi=list(pi_arch), vf=list(vf_arch)),
        ),
    )

    return dict(
        seed=seed,
        list_of_goods=list_of_goods,
        window_size=window_size,
        episode_len_t=episode_len_t,
        train_episodes_chunks=train_episodes_chunks,
        ppo_kwargs=ppo_kwargs,
    )

# -------------------------------------------------
# 2) Define RUN ID and OUTPUT PATH builder
# -------------------------------------------------
#define the name of the output files and folders
def build_run_id_and_out_path(cfg: dict, base_cfg: dict):
    total_len = cfg["episode_len_t"] * cfg["train_episodes_chunks"]

    # Only suffix the parameters you said you vary
    suffixes = []
    for key in ["learning_rate", "ent_coef", "clip_range"]:
        if cfg["ppo_kwargs"][key] != base_cfg["ppo_kwargs"][key]:
            suffixes.append(f"{key}_{cfg['ppo_kwargs'][key]}")

    suffix = ("_" + "_".join(suffixes)) if suffixes else ""

    run_id = f"len_{total_len}{suffix}"
    out_path = f"ML/model_value_lenght_{total_len}{suffix}_raw.xlsx"
    return run_id, out_path

# -------------------------------------------------
# 3) Single EXPERIMENT RUNNER (one config)
# -------------------------------------------------
#function that runs one experiment given a config and a base config
def run_one_experiment(cfg: dict, base_cfg: dict):
    run_id, out_path = build_run_id_and_out_path(cfg, base_cfg)

    print(f"\n=== RUN {run_id} ===")

    # Make sure output folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Reset logger for this run
    fn.log_reset(
        run_id=run_id,
        n_steps=cfg["ppo_kwargs"]["n_steps"],
        n_envs=1,
        extra_meta={
            "run_id": run_id,
            "seed": cfg["seed"],
            "list_of_goods": deepcopy(cfg["list_of_goods"]),
            "window_size": cfg["window_size"],
            "episode_len_t": cfg["episode_len_t"],
            "train_episodes_chunks": cfg["train_episodes_chunks"],
            **deepcopy(cfg["ppo_kwargs"]),
        },
    )

    # -------------------------
    # Build market params (YOUR function signature)
    # -------------------------
    params = prices_process.build_market_matrices(cfg["list_of_goods"])
    params["goods"] = cfg["list_of_goods"]

    # V_np is used internally by the env (not observed by model)
    V_np = np.asarray(params["V"].values, dtype=np.float32)

    # initial_money as you currently do it
    initial_money = fn.draw_initial_money(seed=cfg["seed"])

 
    # -------------------------
    # Build env (YOUR constructor signature)
    # -------------------------
    def env_ctor():
        return fn.MarketEnv(
            list_of_goods=cfg["list_of_goods"],
            params=params,
            window_size=cfg["window_size"],
            max_simulation_length=cfg["episode_len_t"],
            initial_money=initial_money,
            V_np=V_np,
            seed=cfg["seed"],
        )

    vec_env = fn.make_vec_env(env_ctor, n_envs=1)
    
    total_timesteps = cfg["episode_len_t"] * cfg["train_episodes_chunks"]

    #give the env the training horizon + PPO rollout params
    fn.push_hyperparams_to_env(
        vec_env,
        cfg["ppo_kwargs"],
        total_training_steps=total_timesteps
    )
    # Build PPO model
    model = fn.build_ppo_model(
        vec_env=vec_env,
        ppo_kwargs=cfg["ppo_kwargs"],
        seed=cfg["seed"],
        verbose=1,
    )

    # Total timesteps
    total_timesteps = cfg["episode_len_t"] * cfg["train_episodes_chunks"]

    # Train
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    print("Total learn() wall time:", time.perf_counter() - t0)

    # Export transition log
    df = fn.log_to_dataframe()

    # --- Normalized columns (needed for downstream analysis) ---
    # (we keep the checks to avoid crashing if logging schema changes)
    if "initial_money" in df.columns:
        if "wealth_t" in df.columns:
            df["wealth_t_norm"] = df["wealth_t"] / df["initial_money"]
        if "wealth_next" in df.columns:
            df["wealth_next_norm"] = df["wealth_next"] / df["initial_money"]
        if "cash_t" in df.columns:
            df["cash_t_norm"] = df["cash_t"] / df["initial_money"]
        if "cash_next" in df.columns:
            df["cash_next_norm"] = df["cash_next"] / df["initial_money"]

    # Learning period index (same idea as before)
    if "global_step" in df.columns:
        df["learn_phase"] = (df["global_step"] // cfg["episode_len_t"]).astype(int)
    else:
        # fallback if your logger uses a different column
        df["learn_phase"] = 0

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="transitions", index=False)
        pd.DataFrame([{
            "run_id": run_id,
            "out_path": out_path,
            "total_timesteps": total_timesteps,
            **cfg,
        }]).to_excel(writer, sheet_name="meta", index=False)

    print(f"Saved: {out_path} (rows={len(df):,})")


# -------------------------------------------------
# 4) Run the models with different configs
# -------------------------------------------------

if __name__ == "__main__":

    base_cfg = make_config()

    runs = [
        # 3 horizon variants
        make_config(train_episodes_chunks=200),
        make_config(train_episodes_chunks=300),
        make_config(train_episodes_chunks=600),

        # 3 hyperparam variants
        make_config(learning_rate=3e-4),
        make_config(ent_coef=0.005),
        make_config(clip_range=0.2),
    ]

    for cfg in runs:
        run_one_experiment(cfg, base_cfg)

"""
# -------------------------
# CENTRALIZED CONFIG (Main_ML)
# -------------------------
SEED = 0
LIST_OF_GOODS = ["food", "metal", "industrial", "consumption"]

WINDOW_SIZE = 20
EPISODE_LEN_T = 200


TRAIN_EPISODES_CHUNKS = 200 

PPO_KWARGS = dict(
    learning_rate=1e-4,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.02,
    vf_coef=0.7,
    max_grad_norm=0.3,
    n_steps=2000,
    batch_size=256,
    n_epochs=10,
    policy_kwargs=dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
)


# -------------------------
# Build market params
# -------------------------
params = prices_process.build_market_matrices(LIST_OF_GOODS)
params["goods"] = LIST_OF_GOODS
V_np = np.asarray(params["V"].values, dtype=np.float32)

initial_money = fn.draw_initial_money(seed=SEED)
#contruction of the logger
log_reset(
    run_id="experiment_001",
    n_steps=PPO_KWARGS["n_steps"],
    n_envs=1,
    extra_meta={"algo": "PPO","policy": "MlpPolicy",},)
# -------------------------
# Env constructor (no-arg)
# -------------------------
def env_ctor():
    return fn.MarketEnv(
        list_of_goods=LIST_OF_GOODS,
        params=params,
        window_size=WINDOW_SIZE,
        max_simulation_length=EPISODE_LEN_T,
        initial_money=initial_money,
        V_np=V_np,
        seed=SEED,
    )


# -------------------------
# Quick sanity check
# -------------------------
env0 = env_ctor()
check_env(env0, warn=True)


# -------------------------
# Vectorize + build PPO via ML_nd_...
# -------------------------
vec_env = fn.make_vec_env(env_ctor, n_envs=1)
model = fn.build_ppo_model(vec_env, PPO_KWARGS, seed=SEED, verbose=1)




TOTAL_TIMESTEPS = TRAIN_EPISODES_CHUNKS * EPISODE_LEN_T  # or your chunk size
fn.push_hyperparams_to_env(vec_env, PPO_KWARGS, total_training_steps=TOTAL_TIMESTEPS)
t0 = time.perf_counter()
model.learn(total_timesteps=TOTAL_TIMESTEPS)
learn_total = time.perf_counter() - t0

print(f"Total learn() wall time: {learn_total:.2f} seconds")


model.save("ppo_market_env")


# -------------------------
# Save the results to Excel
# -------------------------
df = fn.log_to_dataframe()
meta = fn.log_get_meta()

if df.empty:
    print("Logger is empty: no rows were recorded.")
else:
    n_steps = int(meta.get("n_steps") or 0)
    n_envs = int(meta.get("n_envs") or 1)

    if n_steps <= 0:
        raise ValueError(
            "LOG_META['n_steps'] is missing/invalid. "
            "Make sure you called log_reset(..., n_steps=PPO_KWARGS['n_steps'], n_envs=...) before training."
        )

    rollout_size = n_steps * n_envs

    # PPO "learning phase" index (policy update count)
    df["learn_phase"] = (df["global_step"] // rollout_size).astype(int)
    df["step_in_learn_phase"] = (df["global_step"] % rollout_size).astype(int)

    
    if "wealth_t" in df.columns and "initial_money" in df.columns:
        df["wealth_t_norm"] = df["wealth_t"] / df["initial_money"]
    if "wealth_next" in df.columns and "initial_money" in df.columns:
        df["wealth_next_norm"] = df["wealth_next"] / df["initial_money"]
    if "cash_t" in df.columns and "initial_money" in df.columns:
        df["cash_t_norm"] = df["cash_t"] / df["initial_money"]
    if "cash_next" in df.columns and "initial_money" in df.columns:
        df["cash_next_norm"] = df["cash_next"] / df["initial_money"]

    out_path = f"ML/model_value_lenght_{EPISODE_LEN_T*TRAIN_EPISODES_CHUNKS}_raw.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="transitions", index=False)
        pd.DataFrame([meta]).to_excel(writer, sheet_name="meta", index=False)

    print(f"Saved transition log to: {out_path}  (rows={len(df):,})")

    """
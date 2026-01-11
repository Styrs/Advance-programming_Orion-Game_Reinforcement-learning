import numpy as np
import pandas as pd
import ML_nd_priceprocess_functions
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv



#base model:
# Pt+1 = e^xt+1
# xt+1 = mu +R(xt - mu) +A*ut + B*st - C*st + random_shockt+1
# ut = Qt/V ; Qt net buy/selling amount of good ; V market volume ; instant impact of selling/buying
# st = lambda*st-1 + ut-1 ; lambda persistence of impact ; st persistent impact of selling/buying
# C impact of past sell/buy of the other goods (no instant impact but delayed impact)
# random_shockt+1 ~ N(0,sigma); sigma matrix of variancce and covariance of the good's shocks


list_of_goods = ["food","metal","industrial","consumption"]
number_of_goods = len(list_of_goods)

#########################################################################################
# Parameters definition and building
#########################################################################################
def build_market_matrices(goods):
    """
    Returns a dict of labeled DataFrames + useful vectors.
    Everything is indexed by good names so you can tune easily.
    """
    # --- Core diagonals (start with reasonable defaults; tune later) ---
    # Mean reversion: R (0..1). 0.98 means slow return to mean.
    R = ML_nd_priceprocess_functions.diag_from_dict(goods, {g: 0.98 for g in goods}, name="R")

    # Instant impact: A (>=0). Larger -> stronger immediate effect of u on x.
    A = ML_nd_priceprocess_functions.diag_from_dict(goods, {g: 0.05 for g in goods}, name="A")

    # Persistent impact scale: B (>=0). Larger -> stronger effect from s on x.
    B = ML_nd_priceprocess_functions.diag_from_dict(goods, {g: 0.03 for g in goods}, name="B")

    # Persistence/decay of s: Lambda (0..1). 0.9 decays slowly.
    Lambda = ML_nd_priceprocess_functions.diag_from_dict(goods, {g: 0.90 for g in goods}, name="Lambda")

    # --- Cross-good delayed impact: C (sparse, readable) ---
    # In *your* equation: x += ... - C * s_t
    # So if C[i,j] > 0, then positive s_j pushes x_i DOWN (because subtract).
    # Choose sign based on your economic meaning.
    C_pairs = {
        # examples (tune/remove):
        ("food", "consumption"): -0.02,
        ("consumption", "food"): 0.01,
        ("industrial", "metal"): 0.03,
        ("metal", "industrial"): 0.02,
        ("industrial", "consumption"): 0.01,
        ("consumption", "industrial"): 0.01,
    }
    C = ML_nd_priceprocess_functions.matrix_from_pairs(goods, C_pairs, default=0.0, name="C", symmetric=False)
    np.fill_diagonal(C.values, 0.0)

    # --- Correlated shocks: Corr and Sigma ---
    # Corr controls co-movement of exogenous shocks.
    Corr_pairs = {
        ("metal", "industrial"): 0.50,
        ("food", "consumption"): 0.30,
        ("industrial", "consumption"): 0.10,
    }
    Corr = ML_nd_priceprocess_functions.correlation_from_pairs(goods, Corr_pairs, default_offdiag=0.0, name="Corr", fix_to_spd=True)

    # Standard deviations of shocks per tick (log space)
    # e.g. 0.02 ~ ~2% typical one-sigma move (roughly)
    sigma = {
        "food": 0.015,
        "metal": 0.025,
        "industrial": 0.020,
        "consumption": 0.018,
    }
    Sigma = ML_nd_priceprocess_functions.covariance_from_corr(goods, Corr, sigma, name="Sigma")

    # --- Market depth / volume vector V (used in u = Q/V) ---
    V = pd.Series(
        {"food": 10000.0, "metal": 6000.0, "industrial": 8000.0, "consumption": 9000.0},
        name="V"
    )

    # --- Mean log price mu (log of baseline prices) ---
    # Set baseline prices in normal price units then log them.
    baseline_prices = pd.Series({"food": 100.0, "metal": 300.0, "industrial": 200.0, "consumption": 150.0}, name="p0")
    mu = np.log(baseline_prices).rename("mu")

    return {
        "goods": goods,
        "mu": mu,                 # Series indexed by goods
        "V": V,                   # Series indexed by goods
        "R": R,                   # DataFrame NxN
        "A": A,                   # DataFrame NxN
        "B": B,                   # DataFrame NxN
        "Lambda": Lambda,         # DataFrame NxN
        "C": C,                   # DataFrame NxN
        "Corr": Corr,             # DataFrame NxN
        "Sigma": Sigma,           # DataFrame NxN
    }

params = build_market_matrices(list_of_goods)

def print_param(name, obj):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'-'*60}")
    print(obj)

# Look at matrices in a readable way:
print_param("R — Mean Reversion Matrix", params["R"])
print_param("A — Instant Impact Matrix", params["A"])
print_param("B — Persistent Impact Scale Matrix", params["B"])
print_param("Lambda — Impact Persistence Matrix", params["Lambda"])
print_param("C — Cross-Good Delayed Impact Matrix", params["C"])
print_param("Corr — Shock Correlation Matrix", params["Corr"])
print_param("Sigma — Shock Covariance Matrix", params["Sigma"])
print_param("V — Market Depth Vector", params["V"])
print_param("mu — Long-Run Mean Log-Prices", params["mu"])

# Get NumPy arrays when you simulate:
R_np = params["R"].values
A_np = params["A"].values
B_np = params["B"].values
Lam_np = params["Lambda"].values
C_np = params["C"].values
Sigma_np = params["Sigma"].values
mu_np = params["mu"].values
V_np = params["V"].values



############## Initialization of the timeseries dataframes and important variable ##########################

p_ts, Q_ts, s_ts, inv_ts, cash_ts = ML_nd_priceprocess_functions.init_market_timeseries(list_of_goods, preallocate=False)

normal_market_size = np.sum(V_np * np.exp(mu_np + 0.5 * Sigma_np**2))

print(f"\nNormal market size (sum of V * baseline prices): {normal_market_size:.2f}")



##############################################################################
#exploratory tests of the simulate the market
##############################################################################
#do you test the simulation of the market process alone
test = False 
if test == True:
    ############### Simulation call with only chocks #############################
    n_periods = 200

    x_df, p_df = ML_nd_priceprocess_functions.simulate_market(
        params=params,
        n_periods=n_periods,
        Q=None,        # or your DataFrame of net trades
        seed=42,
        plot=True
    )

    ############### Simulation call random sell/buy ##############################



    n_periods = 200
    Q_ts_test = ML_nd_priceprocess_functions.fill_random_Q_ts(
        Q_ts,
        list_of_goods,
        V_np,
        n_periods=200,
        seed=123
    )


    x_df, p_df = ML_nd_priceprocess_functions.simulate_market(
        params=params,
        n_periods=n_periods,
        Q=Q_ts_test,
        seed=42,
        plot=True
    )

##############################################################################
# ML process
##############################################################################
trash = True
if trash == False:
    #-----------------------------------------------------------------------------
    # def of general varaible and start training 
    #-----------------------------------------------------------------------------
    number_of_simulation_rounds_max = 1000
    current_simulation_round = 0

    max_simulation_length = 200
    windows_size = 30

    p_default = np.exp(mu_np)  # default price when padding the window

    model_policy = ML_nd_priceprocess_functions.generate_model_architecture(dummy_env)

    #-----------------------------------------------------------------------------
    # Sart of the loop, definition of the initial varaible
    #-----------------------------------------------------------------------------
    initial_money = ML_nd_priceprocess_functions.draw_initial_money(low=100_000, high=300_000, seed=42)


    p_ts, Q_ts, s_ts, cash_ts = ML_nd_priceprocess_functions.init_market_timeseries_newround(
        p_ts=p_ts,
        Q_ts=Q_ts,
        s_ts=s_ts,
        cash_ts=cash_ts,
        params=params,
        initial_cash=initial_money
    )

    ML_nd_priceprocess_functions.inv_random_add(inventory_ts = inv_ts, market_size_matrix=V_np,inventory_add_ratio=0.005,prob_increase=0.5, seed=42)

    current_simulation_length = 1
    #-----------------------------------------------------------------------------
    # Window the data frame for the ML model
    #-----------------------------------------------------------------------------

    win_datafor_ML = ML_nd_priceprocess_functions.get_windowed_market_data(
        p_ts=p_ts,
        Q_ts=Q_ts,
        cash_ts=cash_ts,
        inv_ts=inv_ts,
        t=current_simulation_length,   # t == current_simulation_length
        W=windows_size,                # W == windows_size
        p_default=p_default,           # required when t < W (padding)
        initial_money=initial_money,   # required when t < W (padding)
    )

    data_flatten = ML_nd_priceprocess_functions.shape_windowed_data_in_vector_forML(win_datafor_ML,t=current_simulation_length)

    #-----------------------------------------------------------------------------
    # Put in the model the data and get the actions
    #-----------------------------------------------------------------------------

    action_unit, _ = model_policy.predict(data_flatten, deterministic=False)

    #-----------------------------------------------------------------------------
    # Feasability check of the actions and enironment update
    #-----------------------------------------------------------------------------

    #the inventory update after the actions
    ML_nd_priceprocess_functions.inv_random_add(inventory_ts = inv_ts, market_size_matrix=V_np,inventory_add_ratio=0.005,prob_increase=0.5, seed=42)
    #-----------------------------------------------------------------------------
    #reward calcultion
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # End of the loop, reinitialization of some varaible 
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # Prepare the data for the ML learning
    #-----------------------------------------------------------------------------




    #-----------------------------------------------------------------------------
    #Update the model
    #-----------------------------------------------------------------------------





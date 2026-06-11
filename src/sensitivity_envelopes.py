"""
Sensitivity-envelope module for the 6-DOF system  S1 -- S3 -- S2.

Graph:  V1(S1) --- e13 --- V3(S3) --- e32 --- V2(S2)

Two removal scenarios:
  A) remove k*  (k34_est -> 0):  source at S3
  B) remove m*  (m*_est  -> 0):  source at S3

Two propagation methods:
  1) 1-hop weighted average
  2) Heat-kernel diffusion  H = exp(-beta L)

Includes:
  - Full monolithic re-solve for ground-truth validation
  - Wall-clock timing comparison (re-solve vs diffusion)
  - All 5 reviewer corrections [C1]-[C5]

"""

import time
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#  Colour-blind safe palette (Okabe-Ito)                              #
# ------------------------------------------------------------------ #
OI = {
    "black":  "#000000",
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
}

# ================================================================== #
#                       HELPERS                                       #
# ================================================================== #

def _rms(x):
    return np.sqrt(np.mean(np.asarray(x) ** 2))


def _normalise_weights(w13_raw, w32_raw, mode="sum"):
    """[C1] Normalise raw edge weights to dimensionless affinities."""
    if mode == "sum":
        denom = w13_raw + w32_raw
    elif mode == "max":
        denom = max(w13_raw, w32_raw)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    denom = max(denom, 1e-30)
    return w13_raw / denom, w32_raw / denom


def _build_graph_laplacian(w13, w32, laplacian_type="combinatorial"):
    """
    [C2] Weighted Laplacian for the 3-node chain  S1--S3--S2.
    Node ordering: [S1, S3, S2] -> indices 0, 1, 2.
    """
    W = np.array([
        [0.0,  w13, 0.0],
        [w13,  0.0, w32],
        [0.0,  w32, 0.0],
    ])
    D = np.diag(W.sum(axis=1))
    if laplacian_type == "combinatorial":
        L = D - W
    elif laplacian_type == "random_walk":
        D_inv = np.diag([1.0/d if d > 0 else 0.0 for d in np.diag(D)])
        L = np.eye(3) - D_inv @ W
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")
    return L, W


def _heat_kernel(L, beta):
    return scipy.linalg.expm(-beta * L)


def _match_alpha_to_beta(L, beta, source_idx=1):
    """[C5] Auto-match alpha so total 1-hop leak = heat-kernel off-diagonal mass."""
    H = _heat_kernel(L, beta)
    off = 1.0 - H[source_idx, source_idx]
    if off >= 1.0:
        return 0.99
    if off <= 0.0:
        return 0.0
    return min(off / (1.0 - off), 0.99)


# ================================================================== #
#   METHOD 1 -- 1-HOP                                                #
# ================================================================== #

def _diffuse_1hop(delta_f, alpha, eta13, eta32):
    """Source at S3 (node 1).  Neighbours: S1 (eta13), S2 (eta32)."""
    q  = np.abs(delta_f)
    s3 = q / (1.0 + alpha)
    nb = alpha * q / (1.0 + alpha)
    s1 = eta13 * nb
    s2 = eta32 * nb
    return s1, s3, s2


# ================================================================== #
#   METHOD 2 -- HEAT-KERNEL DIFFUSION                                #
# ================================================================== #

def _diffuse_laplacian(delta_f, L, beta, source_idx=1):
    """Source at node source_idx (default S3 = node 1)."""
    H = _heat_kernel(L, beta)
    q = np.abs(delta_f)
    s0 = H[0, source_idx] * q
    s1 = H[1, source_idx] * q
    s2 = H[2, source_idx] * q
    return s0, s1, s2


# ================================================================== #
#   FULL MONOLITHIC RE-SOLVE (ground truth for validation)           #
# ================================================================== #

def _build_MKC_modified(
    m1, m2, m3, m4, m5, m6,
    k1, k2, k3, k4, k5, k34, k6, k7, k8, k9,
    c1, c2, c3, c4, c5, c6, c7, c8, c9,
    m_star=0.0,
):
    """
    Build monolithic M, K, C for the 6-DOF system.
    m_star is added to m3.  k34 is the parallel spring between DOF3-4.
    """
    M = np.diag([m1, m2, m3 + m_star, m4, m5, m6])
    K = np.zeros((6, 6))
    C = np.zeros((6, 6))

    # S1 internal
    K[0,0] += k1 + k3;  K[0,1] += -k3;  K[1,0] += -k3;  K[1,1] += k2 + k3
    C[0,0] += c1 + c3;  C[0,1] += -c3;  C[1,0] += -c3;  C[1,1] += c2 + c3

    # S3 internal
    k_34tot = k5 + k34
    K[2,2] += k_34tot;  K[2,3] += -k_34tot;  K[3,2] += -k_34tot
    K[3,3] += k_34tot + k6
    C[2,2] += c5;  C[2,3] += -c5;  C[3,2] += -c5;  C[3,3] += c5 + c6

    # S2 internal
    K[4,4] += k8 + k9;  K[4,5] += -k9;  K[5,4] += -k9;  K[5,5] += k9
    C[4,4] += c8 + c9;  C[4,5] += -c9;  C[5,4] += -c9;  C[5,5] += c9

    # coupling 2-3
    K[1,1] += k4;  K[1,2] += -k4;  K[2,1] += -k4;  K[2,2] += k4
    C[1,1] += c4;  C[1,2] += -c4;  C[2,1] += -c4;  C[2,2] += c4

    # coupling 4-5
    K[3,3] += k7;  K[3,4] += -k7;  K[4,3] += -k7;  K[4,4] += k7
    C[3,3] += c7;  C[3,4] += -c7;  C[4,3] += -c7;  C[4,4] += c7

    return M, K, C


def resolve_monolithic(
    forces, dt, N,
    phys,
    remove_kstar=False,
    remove_mstar=False,
):
    """
    Full forward simulation of the 6-DOF system.

    Parameters
    ----------
    forces : (N,) or (N+1,) external force on DOF4
    dt     : time step
    N      : number of steps
    phys   : dict of ALL physical parameters (see below)
    remove_kstar : if True, set k34 = 0
    remove_mstar : if True, set m_star = 0

    Returns
    -------
    Q, V : (N+1, 6) displacement and velocity histories
    elapsed : wall-clock time [s]
    """
    p = phys.copy()
    if remove_kstar:
        p["k34"] = 0.0
    if remove_mstar:
        p["m_star"] = 0.0

    M, K, C = _build_MKC_modified(
        p["m1"], p["m2"], p["m3"], p["m4"], p["m5"], p["m6"],
        p["k1"], p["k2"], p["k3"], p["k4"], p["k5"], p["k34"],
        p["k6"], p["k7"], p["k8"], p["k9"],
        p["c1"], p["c2"], p["c3"], p["c4"], p["c5"],
        p["c6"], p["c7"], p["c8"], p["c9"],
        m_star=p["m_star"],
    )
    Minv = np.diag(1.0 / np.diag(M))

    Q = np.zeros((N+1, 6))
    V = np.zeros((N+1, 6))
    q = np.zeros(6)
    v = np.zeros(6)

    tic = time.perf_counter()
    for k in range(N):
        F = np.zeros(6)
        F[3] = forces[k] if k < len(forces) else 0.0
        a = Minv @ (F - C @ v - K @ q)
        v = v + dt * a
        q = q + dt * v
        Q[k+1] = q
        V[k+1] = v
    elapsed = time.perf_counter() - tic

    return Q, V, elapsed


# ================================================================== #
#                 COMPUTATION                                         #
# ================================================================== #

def compute_sensitivity_envelopes(
    t, Q_est, V_est, params_hist, F23_hist, F45_hist,
    P1, P2, P3,
    *,
    alpha=None,          # None = auto-match to beta [C5]
    beta=0.5,
    laplacian_type="combinatorial",  # [C2]
    weight_norm="sum",               # [C1]
    use_time_varying_params=False,
    t_max_for_weights=None,
    k5=25000.0, c5=480.0, m3=500.0,
    # [C3] optional time-varying sigma
    sig_x1_trace=None,
    sig_x6_trace=None,
):
    """
    Compute sensitivity envelopes for x1 and x6 under removal of
    k* (k34) and m*, using 1-hop and heat-kernel methods.

    Timed internally for benchmarking.
    """
    tic = time.perf_counter()

    t = np.asarray(t)
    Q_est = np.asarray(Q_est)
    V_est = np.asarray(V_est)
    params_hist = np.asarray(params_hist)
    F23_hist = np.asarray(F23_hist)
    F45_hist = np.asarray(F45_hist)
    Np1 = len(t)

    # pad forces
    def _pad(arr):
        out = np.zeros(Np1)
        n = min(len(arr), Np1)
        out[:n] = arr[:n]
        if n < Np1:
            out[n:] = arr[-1]
        return out

    F23 = _pad(F23_hist)
    F45 = _pad(F45_hist)

    # Interface force levels define the graph weights between subsystems.
    # ------ [C1] Edge weights ------
    if t_max_for_weights is not None:
        idx_w = t <= t_max_for_weights
    else:
        idx_w = np.ones(Np1, dtype=bool)

    w13_raw = _rms(F23[idx_w])
    w32_raw = _rms(F45[idx_w])
    w13, w32 = _normalise_weights(w13_raw, w32_raw, mode=weight_norm)
    denom = w13 + w32 if (w13 + w32) > 0 else 1.0
    eta13 = w13 / denom
    eta32 = w32 / denom

    # ------ [C2] Laplacian ------
    L, W_adj = _build_graph_laplacian(w13, w32, laplacian_type)

    # ------ [C5] Match alpha ------
    if alpha is None:
        alpha = _match_alpha_to_beta(L, beta, source_idx=1)

    # ------ Parameter trajectories ------
    if use_time_varying_params:
        mstar = params_hist[:, 2]
        kstar = params_hist[:, 5]
    else:
        mstar = np.full(Np1, params_hist[-1, 2])
        kstar = np.full(Np1, params_hist[-1, 5])

    # ------ Baseline S3 states ------
    x3, x4 = Q_est[:, 2], Q_est[:, 3]
    v3, v4 = V_est[:, 2], V_est[:, 3]

    # Defect forces approximate the effect of removing each inferred property.
    # ------ [C4] Defect forces (no artificial clipping) ------
    # Scenario A: remove k*
    delta_f_k = -kstar * (x3 - x4)

    # Scenario B: remove m*
    M3_eff = np.maximum(m3 + mstar, 1e-6)   # [C4] only guard denominator
    k_34tot = k5 + kstar                      # [C4] no max on kstar
    a3_base = (F23 - k_34tot * (x3 - x4) - c5 * (v3 - v4)) / M3_eff
    delta_f_m = -mstar * a3_base

    # ------ [C3] Covariance scaling (shared by both methods) ------
    using_terminal_cov = (sig_x1_trace is None) or (sig_x6_trace is None)

    if sig_x1_trace is not None:
        sig_x1 = np.asarray(sig_x1_trace)
    else:
        sig_x1 = np.full(Np1, np.sqrt(max(P1[0, 0], 0.0)))

    if sig_x6_trace is not None:
        sig_x6 = np.asarray(sig_x6_trace)
    else:
        sig_x6 = np.full(Np1, np.sqrt(max(P2[1, 1], 0.0)))

    x1_base = Q_est[:, 0]
    x6_base = Q_est[:, 5]

    # Build two envelope families from the same defect forces.
    # Time shared by both (graph construction + defect forces)
    elapsed_shared = time.perf_counter() - tic

    # ------ METHOD A: 1-hop (timed separately) ------
    tic_A = time.perf_counter()
    s1_k_A, s3_k_A, s2_k_A = _diffuse_1hop(delta_f_k, alpha, eta13, eta32)
    s1_m_A, s3_m_A, s2_m_A = _diffuse_1hop(delta_f_m, alpha, eta13, eta32)
    dx1_k_A = s1_k_A * sig_x1;  dx6_k_A = s2_k_A * sig_x6
    dx1_m_A = s1_m_A * sig_x1;  dx6_m_A = s2_m_A * sig_x6
    elapsed_1hop = time.perf_counter() - tic_A

    # ------ METHOD B: Heat-kernel (timed separately) ------
    tic_B = time.perf_counter()
    s1_k_B, s3_k_B, s2_k_B = _diffuse_laplacian(delta_f_k, L, beta)
    s1_m_B, s3_m_B, s2_m_B = _diffuse_laplacian(delta_f_m, L, beta)
    dx1_k_B = s1_k_B * sig_x1;  dx6_k_B = s2_k_B * sig_x6
    dx1_m_B = s1_m_B * sig_x1;  dx6_m_B = s2_m_B * sig_x6
    elapsed_heat_kernel = time.perf_counter() - tic_B

    # Total for each method = shared prep + propagation + envelope
    elapsed_1hop_total = elapsed_shared + elapsed_1hop
    elapsed_hk_total   = elapsed_shared + elapsed_heat_kernel

    env = {
        "t": t,
        "x1_base": x1_base,
        "x6_base": x6_base,

        # Method A: 1-hop
        "x1_band_k_A": (x1_base - dx1_k_A, x1_base + dx1_k_A),
        "x1_band_m_A": (x1_base - dx1_m_A, x1_base + dx1_m_A),
        "x6_band_k_A": (x6_base - dx6_k_A, x6_base + dx6_k_A),
        "x6_band_m_A": (x6_base - dx6_m_A, x6_base + dx6_m_A),

        # Method B: Laplacian
        "x1_band_k_B": (x1_base - dx1_k_B, x1_base + dx1_k_B),
        "x1_band_m_B": (x1_base - dx1_m_B, x1_base + dx1_m_B),
        "x6_band_k_B": (x6_base - dx6_k_B, x6_base + dx6_k_B),
        "x6_band_m_B": (x6_base - dx6_m_B, x6_base + dx6_m_B),

        # Raw scores
        "scores_A": {
            "k": {"s1": s1_k_A, "s3": s3_k_A, "s2": s2_k_A},
            "m": {"s1": s1_m_A, "s3": s3_m_A, "s2": s2_m_A},
        },
        "scores_B": {
            "k": {"s1": s1_k_B, "s3": s3_k_B, "s2": s2_k_B},
            "m": {"s1": s1_m_B, "s3": s3_m_B, "s2": s2_m_B},
        },

        # Defect forces
        "delta_f_k": delta_f_k,
        "delta_f_m": delta_f_m,

        # Graph
        "weights_raw": {"w13": w13_raw, "w32": w32_raw},
        "weights_norm": {"w13": w13, "w32": w32, "eta13": eta13, "eta32": eta32},
        "laplacian": L,
        "laplacian_type": laplacian_type,
        "adjacency": W_adj,
        "heat_kernel": _heat_kernel(L, beta),

        # Parameters
        "alpha": alpha,
        "beta": beta,
        "using_terminal_cov": using_terminal_cov,
        "sig_x1": sig_x1,
        "sig_x6": sig_x6,

        # Timing (separate for each method)
        "time_shared_s":       elapsed_shared,
        "time_1hop_s":         elapsed_1hop_total,
        "time_heat_kernel_s":  elapsed_hk_total,
    }
    return env


# ================================================================== #
#   TIMING BENCHMARK                                                  #
# ================================================================== #

def run_timing_benchmark(
    forces, dt, N, phys, env,
    *,
    time_ukf=None,
    n_repeats=5,
):
    """
    Compare wall-clock times:
      1) Distributed UKF loop (measured externally, passed as time_ukf)
      2) Full monolithic re-solve (2 scenarios: remove k*, remove m*)
      3) 1-hop diffusion (timed inside compute_sensitivity_envelopes)
      4) Heat-kernel diffusion (timed inside compute_sensitivity_envelopes)

    Parameters
    ----------
    forces    : (N,) excitation force on DOF4
    dt, N     : time step, number of steps
    phys      : dict of physical parameters (true values)
    env       : output of compute_sensitivity_envelopes
    time_ukf  : wall-clock time [s] of the distributed UKF loop
                (measure with time.perf_counter)
    n_repeats : number of timing repeats for stable measurement

    Returns
    -------
    timing : dict with all timing results
    Q_no_kstar, V_no_kstar : re-solved states with k*=0
    Q_no_mstar, V_no_mstar : re-solved states with m*=0
    """
    # Re-solve the perturbed monolithic system for comparison.
    # ---- Full re-solve: remove k* ----
    times_kstar = []
    for _ in range(n_repeats):
        Q_nk, V_nk, el = resolve_monolithic(
            forces, dt, N, phys, remove_kstar=True)
        times_kstar.append(el)

    # ---- Full re-solve: remove m* ----
    times_mstar = []
    for _ in range(n_repeats):
        Q_nm, V_nm, el = resolve_monolithic(
            forces, dt, N, phys, remove_mstar=True)
        times_mstar.append(el)

    # ---- Diffusion times (from env, separately timed) ----
    t_1hop = env["time_1hop_s"]
    t_hk   = env["time_heat_kernel_s"]

    # ---- Re-solve of BOTH scenarios ----
    t_resolve_both = np.mean(times_kstar) + np.mean(times_mstar)

    timing = {
        # UKF loop (externally measured)
        "ukf_loop_s":                   time_ukf,

        # Monolithic re-solve
        "resolve_remove_kstar_mean_s":  np.mean(times_kstar),
        "resolve_remove_kstar_std_s":   np.std(times_kstar),
        "resolve_remove_mstar_mean_s":  np.mean(times_mstar),
        "resolve_remove_mstar_std_s":   np.std(times_mstar),
        "resolve_both_mean_s":          t_resolve_both,

        # Diffusion (each method separately)
        "diffusion_1hop_s":             t_1hop,
        "diffusion_heat_kernel_s":      t_hk,

        # Speedups
        "speedup_1hop_vs_ukf":         (time_ukf / max(t_1hop, 1e-12)
                                         if time_ukf is not None else None),
        "speedup_hk_vs_ukf":           (time_ukf / max(t_hk, 1e-12)
                                         if time_ukf is not None else None),
        "speedup_1hop_vs_resolve":      t_resolve_both / max(t_1hop, 1e-12),
        "speedup_hk_vs_resolve":        t_resolve_both / max(t_hk, 1e-12),

        "n_repeats":                    n_repeats,
        "N_steps":                      N,
        "dt":                           dt,
    }

    return timing, Q_nk, V_nk, Q_nm, V_nm


# ================================================================== #
#                         PLOTS                                       #
# ================================================================== #

def _apply_style():
    plt.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["Times New Roman", "DejaVu Serif"],
        "font.size":       10,
        "axes.labelsize":  11,
        "legend.fontsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth":  0.6,
    })


def plot_comparison(env, *, t_max=20.0, dpi=600):
    """4-panel: x1 and x6 envelopes for both scenarios, both methods."""
    _apply_style()
    t = env["t"]
    idx = t <= t_max

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)

    panels = [
        (0, 0, "x1", "k", r"$x_1$ [m]"),
        (0, 1, "x1", "m", r"$x_1$ [m]"),
        (1, 0, "x6", "k", r"$x_6$ [m]"),
        (1, 1, "x6", "m", r"$x_6$ [m]"),
    ]
    titles = {"k": r"remove $k^{*}$", "m": r"remove $m^{*}$"}

    for row, col, state, scen, ylabel in panels:
        ax = axes[row, col]
        base = env[f"{state}_base"]
        lo_A, hi_A = env[f"{state}_band_{scen}_A"]
        lo_B, hi_B = env[f"{state}_band_{scen}_B"]

        ax.plot(t[idx], base[idx], color=OI["black"], lw=1.0,
                label="baseline", zorder=3)
        ax.fill_between(t[idx], lo_A[idx], hi_A[idx],
                        color=OI["blue"], alpha=0.22, label="1-hop")
        ax.fill_between(t[idx], lo_B[idx], hi_B[idx],
                        color=OI["orange"], alpha=0.22, label="heat-kernel")

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.4)
        if row == 0: ax.set_title(titles[scen], fontsize=11)
        if row == 1: ax.set_xlabel("time [s]")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_comparison_with_resolve(
    env, Q_no_kstar, Q_no_mstar, Q_baseline_true,
    *, t_max=20.0, dpi=600,
):
    """
    KEY FIGURE: overlay diffusion envelopes AND re-solved ground truth.
    This validates whether the envelopes actually capture the true
    perturbed response.

    4 panels: x1 & x6 for each removal scenario.
    """
    _apply_style()
    t = env["t"]
    idx = t <= t_max
    Np1 = len(t)

    # Differences from baseline
    dx1_true_k = Q_no_kstar[:Np1, 0] - Q_baseline_true[:Np1, 0]
    dx6_true_k = Q_no_kstar[:Np1, 5] - Q_baseline_true[:Np1, 5]
    dx1_true_m = Q_no_mstar[:Np1, 0] - Q_baseline_true[:Np1, 0]
    dx6_true_m = Q_no_mstar[:Np1, 5] - Q_baseline_true[:Np1, 5]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)

    panels = [
        (0, 0, "x1", "k", dx1_true_k, r"$x_1$ [m]"),
        (0, 1, "x1", "m", dx1_true_m, r"$x_1$ [m]"),
        (1, 0, "x6", "k", dx6_true_k, r"$x_6$ [m]"),
        (1, 1, "x6", "m", dx6_true_m, r"$x_6$ [m]"),
    ]
    titles = {"k": r"remove $k^{*}$", "m": r"remove $m^{*}$"}

    for row, col, state, scen, dx_true, ylabel in panels:
        ax = axes[row, col]
        base = env[f"{state}_base"]

        lo_A, hi_A = env[f"{state}_band_{scen}_A"]
        lo_B, hi_B = env[f"{state}_band_{scen}_B"]

        # Baseline
        ax.plot(t[idx], base[idx], color=OI["black"], lw=0.8,
                label="baseline", zorder=2)

        # 1-hop band
        ax.fill_between(t[idx], lo_A[idx], hi_A[idx],
                        color=OI["blue"], alpha=0.18, label="1-hop envelope")

        # Heat-kernel band
        ax.fill_between(t[idx], lo_B[idx], hi_B[idx],
                        color=OI["orange"], alpha=0.18, label="heat-kernel envelope")

        # True perturbed response
        ax.plot(t[idx], (base + dx_true)[idx],
                color=OI["red"], lw=0.9, ls="--",
                label="re-solved (truth)", zorder=3)

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.4)
        if row == 0: ax.set_title(titles[scen], fontsize=11)
        if row == 1: ax.set_xlabel("time [s]")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_timing_bar(timing, *, dpi=600):
    """
    Bar chart comparing wall-clock times:
      - Distributed UKF loop, when available
      - 1-hop diffusion
      - Heat-kernel diffusion
    """
    _apply_style()

    labels = []
    values = []
    colours = []

    # Bar 1: Distributed UKF loop
    if timing["ukf_loop_s"] is not None:
        labels.append("Distributed\nUKF loop")
        values.append(timing["ukf_loop_s"])
        colours.append(OI["purple"])

    # Bar 2: 1-hop diffusion
    labels.append("1-hop\ndiffusion")
    values.append(timing["diffusion_1hop_s"])
    colours.append(OI["blue"])

    # Bar 3: Heat-kernel diffusion
    labels.append("Heat-kernel\ndiffusion")
    values.append(timing["diffusion_heat_kernel_s"])
    colours.append(OI["orange"])

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values,
                  color=colours, alpha=0.75, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)

    # Add time labels on top of each bar
    for i, (v, bar) in enumerate(zip(values, bars)):
        if v >= 1.0:
            txt = f"{v:.2f} s"
        elif v >= 1e-3:
            txt = f"{v*1e3:.2f} ms"
        else:
            txt = f"{v*1e6:.0f} µs"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                txt, ha='center', va='bottom', fontsize=9)

    # Annotate speedups
    if timing["speedup_1hop_vs_ukf"] is not None:
        su_1hop = timing["speedup_1hop_vs_ukf"]
        su_hk   = timing["speedup_hk_vs_ukf"]
        # Put speedup text between UKF bar and diffusion bars
        y_mid = max(values) * 0.45
        ax.text(1, y_mid,
                f"${su_1hop:.0f}\\times$",
                ha='center', fontsize=11, fontweight='bold', color=OI["blue"])
        ax.text(2, y_mid,
                f"${su_hk:.0f}\\times$",
                ha='center', fontsize=11, fontweight='bold', color=OI["orange"])
        ax.text(1.5, y_mid * 0.65,
                "faster", ha='center', fontsize=9, color=OI["black"])

    ax.set_ylabel("wall-clock time [s]")
    ax.set_title(f"Computational cost comparison  "
                 f"(N = {timing['N_steps']:,} steps, "
                 f"dt = {timing['dt']*1e3:.1f} ms)", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)

    plt.tight_layout()
    plt.show()


def plot_score_profiles(env, *, t_max=20.0, dpi=600):
    """2x3 panel: rows = scenario, cols = S1/S3/S2."""
    _apply_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8})
    t = env["t"]
    idx = t <= t_max

    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.4), sharex=True)
    node_labels = ["S1", "S3", "S2"]
    node_keys = ["s1", "s3", "s2"]
    scen_labels = {"k": r"remove $k^{*}$", "m": r"remove $m^{*}$"}

    for row, scen in enumerate(["k", "m"]):
        for col, (nl, nk) in enumerate(zip(node_labels, node_keys)):
            ax = axes[row, col]
            ax.plot(t[idx], env["scores_A"][scen][nk][idx],
                    color=OI["blue"], lw=0.9, label="1-hop")
            ax.plot(t[idx], env["scores_B"][scen][nk][idx],
                    color=OI["orange"], lw=0.9, ls="--", label="heat-kernel")
            ax.set_ylabel(f"$s_{{\\mathrm{{{nl}}}}}$")
            ax.grid(alpha=0.25, linewidth=0.4)
            if row == 0: ax.set_title(nl, fontsize=10)
            if row == 1: ax.set_xlabel("time [s]")
            if row == 0 and col == 0: ax.legend(frameon=False)
        axes[row, -1].annotate(scen_labels[scen], xy=(1.08, 0.5),
            xycoords="axes fraction", fontsize=10, ha="left", va="center", rotation=-90)

    plt.tight_layout()
    plt.show()


def plot_defect_forces(env, *, t_max=20.0, dpi=600):
    _apply_style()
    t = env["t"]
    idx = t <= t_max

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)

    ax1.plot(t[idx], env["delta_f_k"][idx], color=OI["blue"], lw=0.8)
    ax1.set_ylabel(r"$\Delta f_{k^{*}}$ [N]"); ax1.set_xlabel("time [s]")
    ax1.set_title(r"Defect force -- remove $k^{*}$", fontsize=10)
    ax1.grid(alpha=0.25, linewidth=0.4)

    ax2.plot(t[idx], env["delta_f_m"][idx], color=OI["orange"], lw=0.8)
    ax2.set_ylabel(r"$\Delta f_{m^{*}}$ [N]"); ax2.set_xlabel("time [s]")
    ax2.set_title(r"Defect force -- remove $m^{*}$", fontsize=10)
    ax2.grid(alpha=0.25, linewidth=0.4)

    plt.tight_layout()
    plt.show()


def plot_heat_kernel_decay(env, *, beta_range=None, dpi=600):
    _apply_style()
    w13 = env["weights_norm"]["w13"]
    w32 = env["weights_norm"]["w32"]
    L, _ = _build_graph_laplacian(w13, w32, env["laplacian_type"])

    if beta_range is None:
        beta_range = np.linspace(0.001, 3.0, 400)

    H_trace = np.zeros((3, len(beta_range)))
    for i, b in enumerate(beta_range):
        H = _heat_kernel(L, b)
        H_trace[:, i] = H[:, 1]   # source at S3 (node 1)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    names = ["S1", "S3", "S2"]
    cols = [OI["blue"], OI["red"], OI["orange"]]
    for j in range(3):
        ax.plot(beta_range, H_trace[j], color=cols[j], lw=1.2,
                label=rf"$H_{{S3 \to {names[j]}}}$")
    ax.axvline(env["beta"], color=OI["black"], ls=":", lw=0.8,
               label=rf"$\beta = {env['beta']:.2f}$")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$H_{ij}(\beta)$")
    ax.set_title(f"Heat-kernel entries ({env['laplacian_type']} Laplacian)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tight_layout()
    plt.show()


# ================================================================== #
#   SUMMARY PRINTOUT                                                  #
# ================================================================== #

def print_summary(env, timing=None):
    """Print graph quantities, scaling, and optional timing."""
    wr = env["weights_raw"]
    wn = env["weights_norm"]
    H  = env["heat_kernel"]
    L  = env["laplacian"]
    alpha = env["alpha"]

    print("=" * 64)
    print("  SENSITIVITY ENVELOPE -- SUMMARY")
    print("=" * 64)

    print(f"\n  Raw edge weights (RMS interface forces):")
    print(f"    w13_raw = {wr['w13']:.2f} N   (S1 <-> S3)")
    print(f"    w32_raw = {wr['w32']:.2f} N   (S3 <-> S2)")

    print(f"\n  [C1] Normalised weights (mode = {env.get('weight_norm','sum')}):")
    print(f"    w13 = {wn['w13']:.6f}")
    print(f"    w32 = {wn['w32']:.6f}")
    print(f"    eta13 = {wn['eta13']:.4f}")
    print(f"    eta32 = {wn['eta32']:.4f}")

    print(f"\n  [C2] Laplacian type: {env['laplacian_type']}")
    print(f"  L (node order: S1, S3, S2):")
    for i, nm in enumerate(["S1", "S3", "S2"]):
        row_str = "  ".join(f"{L[i,j]:+10.6f}" for j in range(3))
        print(f"    {nm}: [ {row_str} ]")

    print(f"\n  Heat kernel H (beta = {env['beta']:.4f}):")
    for i, nm in enumerate(["S1", "S3", "S2"]):
        row_str = "  ".join(f"{H[i,j]:+.6f}" for j in range(3))
        print(f"    {nm}: [ {row_str} ]")

    h_off = 1.0 - H[1, 1]
    a_leak = alpha / (1.0 + alpha)
    print(f"\n  [C5] alpha = {alpha:.4f}")
    print(f"    1-hop total leak:   alpha/(1+alpha) = {a_leak:.4f}")
    print(f"    Laplacian total leak: 1-H[S3,S3]   = {h_off:.4f}")
    if (H[0,1] + H[2,1]) > 0:
        print(f"    1-hop split (S1):   {wn['eta13']:.4f}"
              f"   Laplacian: {H[0,1]/(H[0,1]+H[2,1]):.4f}")
        print(f"    1-hop split (S2):   {wn['eta32']:.4f}"
              f"   Laplacian: {H[2,1]/(H[0,1]+H[2,1]):.4f}")

    print(f"\n  [C3] Covariance: "
          f"{'terminal (constant)' if env['using_terminal_cov'] else 'time-varying'}")
    print(f"    sigma(x1) = {env['sig_x1'][-1]:.6e}")
    print(f"    sigma(x6) = {env['sig_x6'][-1]:.6e}")

    if timing is not None:
        print(f"\n  TIMING ({timing['N_steps']:,} steps, "
              f"dt = {timing['dt']*1e3:.1f} ms):")
        if timing["ukf_loop_s"] is not None:
            print(f"    Distributed UKF loop:  "
                  f"{timing['ukf_loop_s']:.4f} s")
        print(f"    1-hop diffusion:       "
              f"{timing['diffusion_1hop_s']*1e3:.4f} ms")
        print(f"    Heat-kernel diffusion:  "
              f"{timing['diffusion_heat_kernel_s']*1e3:.4f} ms")
        if timing["speedup_1hop_vs_ukf"] is not None:
            print(f"    Speedup (1-hop vs UKF):       "
                  f"{timing['speedup_1hop_vs_ukf']:.0f}x")
            print(f"    Speedup (heat-kernel vs UKF): "
                  f"{timing['speedup_hk_vs_ukf']:.0f}x")

    print("=" * 64)

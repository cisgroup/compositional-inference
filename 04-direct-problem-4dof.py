
"""
Forward 4-DOF coupling-scheme comparison.

The script solves the same 4-DOF system with monolithic, Jacobi,
Gauss-Seidel, and AB2 coupling updates, then compares the resulting
state trajectories and errors.
"""

import numpy as np
import pandas as pd

# ================== Parameters ==================
m1 = m2 = m3 = m4 = 500.0
k1 = k2 = k3 = k4 = 50_000.0
c1 = c2 = c3 = c4 = 300.0

dt = 1e-3
T = 5.0
N = int(T / dt)
sigmaF = 50.0
SEED = 123

# ================== Helpers ==================
def M_inv_2dof(m_a, m_b):
    return np.diag([1.0 / m_a, 1.0 / m_b])

def self_KC_S1(k1, k2, c1, c2):
    K = np.array([[k1 + k2, -k2],
                  [-k2,      k2]], dtype=float)
    C = np.array([[c1 + c2, -c2],
                  [-c2,      c2]], dtype=float)
    return K, C

def self_KC_S2(k3, k4, c3, c4):
    K = np.array([[k4,  -k4],
                  [-k4,  k4]], dtype=float)
    C = np.array([[c4,  -c4],
                  [-c4,  c4]], dtype=float)
    return K, C

def edge_force(x2, v2, x3, v3, k3, c3):
    return k3 * (x2 - x3) + c3 * (v2 - v3)

def step_2dof_heun(q, v, M_inv, K, C, load, dt):
    def accel(q_, v_):
        L = load(q_, v_) if callable(load) else load
        return M_inv @ (L - C @ v_ - K @ q_)

    a_n = accel(q, v)
    q_pred = q + dt * v
    v_pred = v + dt * a_n

    a_pred = accel(q_pred, v_pred)
    q_new = q + 0.5 * dt * (v + v_pred)
    v_new = v + 0.5 * dt * (a_n + a_pred)
    return q_new, v_new

def s1_step(state1, x3, v3, u1, dt):
    x1, x2, v1, v2 = state1
    q = np.array([x1, x2], dtype=float)
    v = np.array([v1, v2], dtype=float)
    K, C = self_KC_S1(k1, k2, c1, c2)
    Minv = M_inv_2dof(m1, m2)
    Fb = edge_force(x2, v2, x3, v3, k3, c3)
    load = u1 + np.array([0.0, -Fb])
    q_new, v_new = step_2dof_heun(q, v, Minv, K, C, load, dt)
    return np.array([q_new[0], q_new[1], v_new[0], v_new[1]], dtype=float)

def s2_step(state2, x2, v2, u2, dt):
    x3, x4, v3, v4 = state2
    q = np.array([x3, x4], dtype=float)
    v = np.array([v3, v4], dtype=float)
    K, C = self_KC_S2(k3, k4, c3, c4)
    Minv = M_inv_2dof(m3, m4)
    Fb = edge_force(x2, v2, x3, v3, k3, c3)
    load = u2 + np.array([Fb, 0.0])
    q_new, v_new = step_2dof_heun(q, v, Minv, K, C, load, dt)
    return np.array([q_new[0], q_new[1], v_new[0], v_new[1]], dtype=float)

# ================== Coupling schemes ==================
def jacobi_step(state1, state2, u_all, dt):
    # True Jacobi: both subsystems use interface data from time level n
    x2, v2 = state1[1], state1[3]
    x3, v3 = state2[0], state2[2]
    u1 = u_all[:2].copy()
    u2 = u_all[2:].copy()

    state1_next = s1_step(state1, x3, v3, u1, dt)
    state2_next = s2_step(state2, x2, v2, u2, dt)
    return state1_next, state2_next

def gauss_seidel_step(state1, state2, u_all, dt):
    # Gauss-Seidel: S2 uses updated interface values from S1
    x3, v3 = state2[0], state2[2]
    u1 = u_all[:2].copy()
    u2 = u_all[2:].copy()

    state1_next = s1_step(state1, x3, v3, u1, dt)
    x2_new, v2_new = state1_next[1], state1_next[3]
    state2_next = s2_step(state2, x2_new, v2_new, u2, dt)
    return state1_next, state2_next

def jacobi_step_ab2(state1, state2, prev_state1, prev_state2, u_all, dt):
    x2, v2 = state1[1], state1[3]
    x3, v3 = state2[0], state2[2]
    x2p, v2p = prev_state1[1], prev_state1[3]
    x3p, v3p = prev_state2[0], prev_state2[2]

    # AB2 / linear extrapolation of the interface state
    x2_hat = 1.5 * x2 - 0.5 * x2p
    v2_hat = 1.5 * v2 - 0.5 * v2p
    x3_hat = 1.5 * x3 - 0.5 * x3p
    v3_hat = 1.5 * v3 - 0.5 * v3p

    u1 = u_all[:2].copy()
    u2 = u_all[2:].copy()

    state1_next = s1_step(state1, x3_hat, v3_hat, u1, dt)
    state2_next = s2_step(state2, x2_hat, v2_hat, u2, dt)
    return state1_next, state2_next

# ================== Monolithic model ==================
def mono_step(state, u_all, dt):
    x1, x2, x3, x4, v1, v2, v3, v4 = state
    q = np.array([x1, x2, x3, x4], dtype=float)
    v = np.array([v1, v2, v3, v4], dtype=float)
    Minv = np.diag([1/m1, 1/m2, 1/m3, 1/m4])

    K = np.array([[k1+k2,   -k2,      0,      0],
                  [  -k2, k2+k3,    -k3,      0],
                  [    0,   -k3,  k3+k4,   -k4],
                  [    0,     0,    -k4,    k4]], dtype=float)

    C = np.array([[c1+c2,   -c2,      0,      0],
                  [  -c2, c2+c3,    -c3,      0],
                  [    0,   -c3,  c3+c4,   -c4],
                  [    0,     0,    -c4,    c4]], dtype=float)

    a = Minv @ (u_all - C @ v - K @ q)
    v_new = v + dt * a
    q_new = q + dt * v_new
    return np.concatenate([q_new, v_new])

# ================== Simulation ==================
def simulate_method(method_name, u_hist):
    state1 = np.array([0.01, 0.0, 0.01, 0.0], dtype=float)
    state2 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    t = np.linspace(0.0, T, N + 1)
    Q = np.zeros((N + 1, 4), dtype=float)
    V = np.zeros((N + 1, 4), dtype=float)

    Q[0] = [state1[0], state1[1], state2[0], state2[1]]
    V[0] = [state1[2], state1[3], state2[2], state2[3]]

    prev_state1 = state1.copy()
    prev_state2 = state2.copy()

    for k in range(N):
        u = u_hist[k]

        if method_name == "jacobi":
            next_state1, next_state2 = jacobi_step(state1, state2, u, dt)
        elif method_name == "gauss_seidel":
            next_state1, next_state2 = gauss_seidel_step(state1, state2, u, dt)
        elif method_name == "ab2":
            if k == 0:
                next_state1, next_state2 = jacobi_step(state1, state2, u, dt)
            else:
                next_state1, next_state2 = jacobi_step_ab2(
                    state1, state2, prev_state1, prev_state2, u, dt
                )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        prev_state1 = state1.copy()
        prev_state2 = state2.copy()
        state1, state2 = next_state1, next_state2

        Q[k + 1] = [state1[0], state1[1], state2[0], state2[1]]
        V[k + 1] = [state1[2], state1[3], state2[2], state2[3]]

    return t, Q, V

def simulate_monolithic(u_hist):
    state_m = np.array([0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0], dtype=float)
    t = np.linspace(0.0, T, N + 1)
    Q = np.zeros((N + 1, 4), dtype=float)
    V = np.zeros((N + 1, 4), dtype=float)

    Q[0] = state_m[:4]
    V[0] = state_m[4:]

    for k in range(N):
        state_m = mono_step(state_m, u_hist[k], dt)
        Q[k + 1] = state_m[:4]
        V[k + 1] = state_m[4:]

    return t, Q, V

def make_state_df(t, Q, V):
    df = pd.DataFrame({
        "time": t,
        "x1": Q[:, 0],
        "x2": Q[:, 1],
        "x3": Q[:, 2],
        "x4": Q[:, 3],
        "v1": V[:, 0],
        "v2": V[:, 1],
        "v3": V[:, 2],
        "v4": V[:, 3],
    })
    return df

def tukey_mask(x, k=1.5):
    """
    Return boolean mask for values kept by Tukey's rule.
    Keeps points inside [Q1 - k*IQR, Q3 + k*IQR].
    """
    x = np.asarray(x, dtype=float)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (x >= lower) & (x <= upper)


def make_error_dfs(t, Q_method, V_method, Q_mono, V_mono):
    err_q = Q_method - Q_mono
    err_v = V_method - V_mono

    err_df = pd.DataFrame({
        "time": t,
        "err_x1": err_q[:, 0],
        "err_x2": err_q[:, 1],
        "err_x3": err_q[:, 2],
        "err_x4": err_q[:, 3],
        "err_v1": err_v[:, 0],
        "err_v2": err_v[:, 1],
        "err_v3": err_v[:, 2],
        "err_v4": err_v[:, 3],
        "err_q_l2": np.linalg.norm(err_q, axis=1),
        "err_v_l2": np.linalg.norm(err_v, axis=1),
    })

    trimmed_df = err_df.copy()

    error_cols = [
        "err_x1", "err_x2", "err_x3", "err_x4",
        "err_v1", "err_v2", "err_v3", "err_v4",
        "err_q_l2", "err_v_l2",
    ]

    # remove outliers column-by-column by setting them to NaN
    # this preserves time alignment
    for col in error_cols:
        mask = tukey_mask(trimmed_df[col].values, k=1.5)
        trimmed_df.loc[~mask, col] = np.nan

    return err_df, trimmed_df

def summarize_error(method_name, err_df):
    return {
        "method": method_name,
        "max_err_q_l2": float(err_df["err_q_l2"].max()),
        "max_err_v_l2": float(err_df["err_v_l2"].max()),
        "rmse_q_all_dofs": float(np.sqrt(np.mean(err_df[["err_x1","err_x2","err_x3","err_x4"]].values**2))),
        "rmse_v_all_dofs": float(np.sqrt(np.mean(err_df[["err_v1","err_v2","err_v3","err_v4"]].values**2))),
        "final_err_q_l2": float(err_df["err_q_l2"].iloc[-1]),
        "final_err_v_l2": float(err_df["err_v_l2"].iloc[-1]),
    }

def main():
    rng = np.random.default_rng(SEED)
    u_hist = rng.normal(0.0, sigmaF, size=(N, 4))

    t_mono, Q_mono, V_mono = simulate_monolithic(u_hist)
    state_dfs = {"monolithic": make_state_df(t_mono, Q_mono, V_mono)}
    error_dfs = {}
    error_dfs_no_outliers = {}

    summary_rows = []
    for method_name in ["jacobi", "gauss_seidel", "ab2"]:
        t, Q, V = simulate_method(method_name, u_hist)
        state_dfs[method_name] = make_state_df(t, Q, V)

        err_df, err_df_no_outliers = make_error_dfs(t, Q, V, Q_mono, V_mono)
        error_dfs[method_name] = err_df
        error_dfs_no_outliers[method_name] = err_df_no_outliers

        summary_rows.append(summarize_error(method_name, err_df))

    summary_df = pd.DataFrame(summary_rows)
    return state_dfs, error_dfs, error_dfs_no_outliers, summary_df

import matplotlib.pyplot as plt

def plot_comparison(mono_df, jacobi_df, gs_df, ab2_df,
                    err_jacobi_df=None, err_gs_df=None, err_ab2_df=None):
    """
    Expected columns in result dfs:
        time, x1, x2, x3, x4, v1, v2, v3, v4

    Expected columns in error dfs:
        time, err_q_l2, err_v_l2
    """

    t = mono_df["time"].values
    dofs = [1, 2, 3, 4]

    # -------------------------
    # 1) Displacement comparison
    # -------------------------
    fig1, axes1 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for i, dof in enumerate(dofs):
        ax = axes1[i]
        ax.plot(t, mono_df[f"x{dof}"], label="Monolithic", linewidth=2)
        ax.plot(t, jacobi_df[f"x{dof}"], "--", label="Jacobi")
        ax.plot(t, gs_df[f"x{dof}"], "-.", label="Gauss-Seidel")
        ax.plot(t, ab2_df[f"x{dof}"], ":", label="AB2")
        ax.set_ylabel(f"x{dof} [m]")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(ncol=4, frameon=False)

    axes1[-1].set_xlabel("Time [s]")
    fig1.suptitle("Displacement Comparison: Monolithic vs Jacobi vs Gauss-Seidel vs AB2")
    fig1.tight_layout()
    plt.show()

    # -------------------------
    # 2) Velocity comparison
    # -------------------------
    fig2, axes2 = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for i, dof in enumerate(dofs):
        ax = axes2[i]
        ax.plot(t, mono_df[f"v{dof}"], label="Monolithic", linewidth=2)
        ax.plot(t, jacobi_df[f"v{dof}"], "--", label="Jacobi")
        ax.plot(t, gs_df[f"v{dof}"], "-.", label="Gauss-Seidel")
        ax.plot(t, ab2_df[f"v{dof}"], ":", label="AB2")
        ax.set_ylabel(f"v{dof} [m/s]")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(ncol=4, frameon=False)

    axes2[-1].set_xlabel("Time [s]")
    fig2.suptitle("Velocity Comparison: Monolithic vs Jacobi vs Gauss-Seidel vs AB2")
    fig2.tight_layout()
    plt.show()

    # -------------------------
    # 3) Error norms
    # -------------------------
    if err_jacobi_df is not None and err_gs_df is not None and err_ab2_df is not None:
        fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # displacement norm error
        axes3[0].plot(err_jacobi_df["time"], err_jacobi_df["err_q_l2"], label="Jacobi")
        axes3[0].plot(err_gs_df["time"], err_gs_df["err_q_l2"], label="Gauss-Seidel")
        axes3[0].plot(err_ab2_df["time"], err_ab2_df["err_q_l2"], label="AB2")
        axes3[0].set_ylabel(r"$||\Delta x||_2$")
        axes3[0].set_title("Displacement Error Norm vs Monolithic")
        axes3[0].grid(True, alpha=0.3)
        axes3[0].legend(frameon=False)

        # velocity norm error
        axes3[1].plot(err_jacobi_df["time"], err_jacobi_df["err_v_l2"], label="Jacobi")
        axes3[1].plot(err_gs_df["time"], err_gs_df["err_v_l2"], label="Gauss-Seidel")
        axes3[1].plot(err_ab2_df["time"], err_ab2_df["err_v_l2"], label="AB2")
        axes3[1].set_ylabel(r"$||\Delta v||_2$")
        axes3[1].set_xlabel("Time [s]")
        axes3[1].set_title("Velocity Error Norm vs Monolithic")
        axes3[1].grid(True, alpha=0.3)
        axes3[1].legend(frameon=False)

        fig3.tight_layout()
        plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_error_boxplots(err_jacobi_df, err_gs_df, err_ab2_df):
    """
    Expected columns:
      time,
      err_x1 ... err_x4,
      err_v1 ... err_v4

    This plots absolute errors as boxplots:
      left  = displacement errors
      right = velocity errors
    """

    methods = {
        "Jacobi": err_jacobi_df,
        "GS": err_gs_df,
        "AB2": err_ab2_df,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # -------------------------
    # Left: displacement errors
    # -------------------------
    ax = axes[0]

    positions = []
    box_data = []
    labels = []

    # grouped positions for each DOF
    # DOF1: 1,2,3   DOF2: 5,6,7   DOF3: 9,10,11   DOF4: 13,14,15
    base_positions = [1, 5, 9, 13]
    offsets = [0, 1, 2]

    method_names = list(methods.keys())

    for dof_idx, base in enumerate(base_positions, start=1):
        for j, method in enumerate(method_names):
            df = methods[method]
            data = np.abs(df[f"err_x{dof_idx}"].values)
            box_data.append(data)
            positions.append(base + offsets[j])
            labels.append(method)

    bp1 = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False
    )

    colors = ["lightblue", "lightgreen", "salmon"] * 4
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticks([2, 6, 10, 14])
    ax.set_xticklabels(["DOF 1", "DOF 2", "DOF 3", "DOF 4"])
    ax.set_ylabel("Absolute displacement error [m]")
    ax.set_title("Displacement Error Distribution")
    ax.grid(True, alpha=0.3)

    # fake legend
    for color, name in zip(["lightblue", "lightgreen", "salmon"], method_names):
        ax.plot([], [], color=color, linewidth=10, label=name)
    ax.legend(frameon=False)

    # -------------------------
    # Right: velocity errors
    # -------------------------
    ax = axes[1]

    positions = []
    box_data = []

    for dof_idx, base in enumerate(base_positions, start=1):
        for j, method in enumerate(method_names):
            df = methods[method]
            data = np.abs(df[f"err_v{dof_idx}"].values)
            box_data.append(data)
            positions.append(base + offsets[j])

    bp2 = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False
    )

    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticks([2, 6, 10, 14])
    ax.set_xticklabels(["DOF 1", "DOF 2", "DOF 3", "DOF 4"])
    ax.set_ylabel("Absolute velocity error [m/s]")
    ax.set_title("Velocity Error Distribution")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Error Comparison by DOF and Method")
    fig.tight_layout()
    plt.show()
#%%
if __name__ == "__main__":
    state_dfs, error_dfs, error_dfs_no_outliers, summary_df = main()

    plot_comparison(
        state_dfs["monolithic"],
        state_dfs["jacobi"],
        state_dfs["gauss_seidel"],
        state_dfs["ab2"],
        error_dfs["jacobi"],
        error_dfs["gauss_seidel"],
        error_dfs["ab2"]
    )

    plot_error_boxplots(
        error_dfs_no_outliers["jacobi"],
        error_dfs_no_outliers["gauss_seidel"],
        error_dfs_no_outliers["ab2"]
    )
#%%

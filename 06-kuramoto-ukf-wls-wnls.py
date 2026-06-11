
"""
Kuramoto/PYPOWER estimator benchmark.

The script builds network coupling matrices from PYPOWER cases and compares
centralized and distributed UKF, WLS, and WNLS estimators across several
test networks.
"""

# %%# Imports
import time
from typing import Callable, List, Tuple, Dict, Any

import numpy as np
import numpy.linalg as npl
import scipy.linalg
import matplotlib.pyplot as plt

import pypower.api as ppa
from pypower.makeYbus import makeYbus
from pypower.idx_bus import BUS_I
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_gen import GEN_BUS


# %%# UKF / WLS / WNLS core
from src.filters import dynamic_wls_step, dynamic_wnls_step, robust_inverse, ukf_step


# %%# PYPOWER helpers (general for any case)
def get_generator_buses_case(ppc: Dict[str, Any]) -> List[int]:
    """Return generator bus indices in 0..nb-1 (0-based indices in bus array order)."""
    bus = ppc["bus"]
    gen = ppc["gen"]

    bus_numbers = bus[:, BUS_I].astype(int)
    bus_map = {bus_no: idx for idx, bus_no in enumerate(bus_numbers)}

    gen_bus_numbers = gen[:, GEN_BUS].astype(int)
    gen_idx = [bus_map[b] for b in gen_bus_numbers if b in bus_map]

    seen = set()
    out = []
    for g in gen_idx:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def build_coupling_from_pypower_case(
    case_fn: Callable[[], Dict[str, Any]],
    *,
    coupling: str = "susceptance",
    thresh: float = 1e-12,
) -> Tuple[np.ndarray, List[List[int]], Dict[str, Any]]:
    """Build coupling matrix K + neighbor list from any PYPOWER case."""
    ppc = case_fn()
    baseMVA = ppc["baseMVA"]
    bus = ppc["bus"].copy()
    branch = ppc["branch"].copy()

    bus_numbers = bus[:, BUS_I].astype(int)
    nb = bus.shape[0]
    bus_map = {bus_no: idx for idx, bus_no in enumerate(bus_numbers)}

    f = branch[:, F_BUS].astype(int)
    t = branch[:, T_BUS].astype(int)
    try:
        branch[:, F_BUS] = np.array([bus_map[x] for x in f], dtype=float)
        branch[:, T_BUS] = np.array([bus_map[x] for x in t], dtype=float)
    except KeyError as e:
        raise ValueError(f"Branch references unknown bus number: {e}")

    Ybus, _, _ = makeYbus(baseMVA, bus, branch)
    Ybus = Ybus.toarray()

    if coupling == "susceptance":
        K = -np.imag(Ybus)
    elif coupling == "conductance":
        K = -np.real(Ybus)
    elif coupling == "magnitude":
        K = np.abs(Ybus)
    else:
        raise ValueError("coupling must be one of {'susceptance','conductance','magnitude'}")

    np.fill_diagonal(K, 0.0)
    neighbors = [list(np.where(np.abs(K[i]) > thresh)[0]) for i in range(nb)]
    return K, neighbors, ppc


# %%# Partitioning (generator-seeded, max size; internal vs cut ratio as secondary objective)
def generator_centered_partition_maxsize(
    K,
    gen_buses,
    max_size=5,
    use_abs=True,
    eps=1e-12,
):
    K = np.asarray(K)
    N = K.shape[0]
    W = np.abs(K) if use_abs else K

    clusters = [[g] for g in gen_buses]
    assigned = np.full(N, -1, dtype=int)
    for ci, g in enumerate(gen_buses):
        assigned[g] = ci

    all_idx = np.arange(N)

    def internal(ci, v):
        return float(W[v, clusters[ci]].sum())

    def cut(ci, v):
        mask = np.ones(N, dtype=bool)
        mask[clusters[ci]] = False
        return float(W[v, all_idx[mask]].sum())

    def ratio(ci, v):
        return internal(ci, v) / (cut(ci, v) + eps)

    def pick_best(ci, candidates):
        return max(candidates, key=lambda v: (internal(ci, v), ratio(ci, v)))

    changed = True
    while changed:
        changed = False
        for ci in range(len(clusters)):
            if len(clusters[ci]) >= max_size:
                continue
            cand = [v for v in range(N) if assigned[v] == -1 and internal(ci, v) > 0]
            if not cand:
                continue
            v_best = pick_best(ci, cand)
            clusters[ci].append(v_best)
            assigned[v_best] = ci
            changed = True

    for v in np.where(assigned == -1)[0]:
        feasible = [ci for ci in range(len(clusters)) if len(clusters[ci]) < max_size]
        if feasible:
            ci_best = max(feasible, key=lambda ci: (internal(ci, v), ratio(ci, v)))
            clusters[ci_best].append(v)
            assigned[v] = ci_best
        else:
            clusters.append([v])
            assigned[v] = len(clusters) - 1

    return [sorted(c) for c in clusters]


# %%# Kuramoto / swing model simulation helpers
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def heun_step(theta, omega, d, Omega, K, dt):
    N = len(theta)

    def deriv(th, w):
        coup = np.zeros(N)
        for i in range(N):
            coup[i] = np.sum(K[i, :] * np.sin(th - th[i]))
        th_dot = w.copy()
        w_dot = -d * w + Omega + coup
        return th_dot, w_dot

    k1t, k1w = deriv(theta, omega)
    th2 = wrap_angle(theta + dt * k1t)
    w2 = omega + dt * k1w
    k2t, k2w = deriv(th2, w2)

    thn = wrap_angle(theta + 0.5 * dt * (k1t + k2t))
    wn = omega + 0.5 * dt * (k1w + k2w)
    return thn, wn


def heun_step_sub(theta_local, omega_local, d_local, Omega_local, K, local_ids, ext_theta, dt):
    m = len(local_ids)
    local_set = set(local_ids)
    local_pos = {nid: a for a, nid in enumerate(local_ids)}

    def deriv(th_l, w_l):
        th_dot = w_l.copy()
        w_dot = np.zeros(m)
        for a, i in enumerate(local_ids):
            kij_row = K[i]
            sum_c = 0.0
            for j in range(K.shape[0]):
                kij = kij_row[j]
                if kij == 0.0 or j == i:
                    continue
                thj = th_l[local_pos[j]] if j in local_set else ext_theta[j]
                sum_c += kij * np.sin(thj - th_l[a])
            w_dot[a] = -d_local[a] * w_l[a] + Omega_local[a] + sum_c
        return th_dot, w_dot

    k1t, k1w = deriv(theta_local, omega_local)
    th2 = np.array([wrap_angle(theta_local[a] + dt * k1t[a]) for a in range(m)])
    w2 = omega_local + dt * k1w
    k2t, k2w = deriv(th2, w2)

    thn = np.array([wrap_angle(theta_local[a] + 0.5 * dt * (k1t[a] + k2t[a])) for a in range(m)])
    wn = omega_local + 0.5 * dt * (k1w + k2w)
    return thn, wn


# %%# Random truth parameter generators
def make_d_true(num_nodes, low=0.1, high=0.30, seed=None):
    rng = np.random.default_rng(seed)
    d = rng.uniform(low, high, size=num_nodes)
    return np.round(d, 2)


def make_Omega_true(num_nodes, seed=None):
    rng = np.random.default_rng(seed)
    choices = np.round(np.linspace(-1.0, 1.0, 21), 1)
    return rng.choice(choices, size=num_nodes, replace=True)


# %%# Shared model builders
def make_central_models(K, d_true, Nbus):
    def f_central(x, u, dt_):
        th = x[0:Nbus]
        w = x[Nbus:2 * Nbus]
        Om = x[2 * Nbus:3 * Nbus]
        thn, wn = heun_step(th, w, d_true, Om, K, dt_)
        return np.concatenate([thn, wn, Om])

    def h_central(x, u, dt_):
        th = x[0:Nbus]
        w = x[Nbus:2 * Nbus]
        return np.concatenate([wrap_angle(th), w])

    return f_central, h_central


def make_subsystem_models(K, d_true, ids):
    ids = list(ids)
    m = len(ids)

    def f_sub(x, u, dt_):
        th_l = x[0:m]
        w_l = x[m:2 * m]
        Om_l = x[2 * m:3 * m]
        d_l = d_true[ids]
        thn, wn = heun_step_sub(th_l, w_l, d_l, Om_l, K, ids, u, dt_)
        return np.concatenate([thn, wn, Om_l])

    def h_sub(x, u, dt_):
        th_l = x[0:m]
        w_l = x[m:2 * m]
        return np.concatenate([wrap_angle(th_l), w_l])

    return f_sub, h_sub


# %%# Single-case benchmark (returns times for all methods)
def run_case_benchmark(
    case_fn: Callable[[], Dict[str, Any]],
    *,
    coupling: str = "susceptance",
    max_cluster_size: int = 5,
    jacobi_iters: int = 1,
    dt: float = 0.01,
    T: float = 3.0,
    seed: int = 42,
    wnls_inner_iters: int = 5,
) -> Dict[str, Any]:
    """
    Run central and distributed UKF / WLS / WNLS for a single PYPOWER case.
    """
    K, _neighbors, ppc = build_coupling_from_pypower_case(case_fn, coupling=coupling)
    Nbus = K.shape[0]
    rng = np.random.default_rng(seed)

    d_true = make_d_true(Nbus, seed=seed)
    Omega_true = make_Omega_true(Nbus, seed=seed)
    steps = int(T / dt)

    theta0 = rng.uniform(-0.5, 0.5, Nbus)
    omega0 = rng.uniform(-0.2, 0.2, Nbus)

    Theta_true = np.zeros((steps + 1, Nbus))
    W_true = np.zeros((steps + 1, Nbus))
    Theta_true[0] = theta0
    W_true[0] = omega0
    for k in range(steps):
        Theta_true[k + 1], W_true[k + 1] = heun_step(Theta_true[k], W_true[k], d_true, Omega_true, K, dt)

    meas_std_th = 0.02
    meas_std_w = 0.02
    Y_th = wrap_angle(Theta_true + rng.normal(0, meas_std_th, Theta_true.shape))
    Y_w = W_true + rng.normal(0, meas_std_w, W_true.shape)

    x0t = wrap_angle(theta0 + rng.normal(0, 0.2, Nbus))
    x0w = omega0 + rng.normal(0, 0.2, Nbus)
    x0O = np.zeros(Nbus)

    Q_c = np.diag(np.concatenate([1e-4 * np.ones(Nbus), 1e-4 * np.ones(Nbus), 1e-4 * np.ones(Nbus)]))
    R_c = np.diag(np.concatenate([(meas_std_th ** 2) * np.ones(Nbus), (meas_std_w ** 2) * np.ones(Nbus)]))
    P0_c = np.diag(np.concatenate([(0.5 ** 2) * np.ones(Nbus), (0.5 ** 2) * np.ones(Nbus), (1.0 ** 2) * np.ones(Nbus)]))

    f_central, h_central = make_central_models(K, d_true, Nbus)

    # -------- CENTRAL UKF
    gamma = 1.0
    xc_ukf = np.concatenate([x0t, x0w, x0O])
    Pc_ukf = P0_c.copy()

    t0 = time.perf_counter()
    for k in range(steps):
        z = np.concatenate([Y_th[k + 1], Y_w[k + 1]])
        xc_ukf, Pc_ukf = ukf_step(xc_ukf, Pc_ukf, R_c, Q_c, gamma, z, dt, f_central, h_central, None)
        xc_ukf[0:Nbus] = wrap_angle(xc_ukf[0:Nbus])
    t_central_ukf = time.perf_counter() - t0

    # -------- CENTRAL WLS
    xc_wls = np.concatenate([x0t, x0w, x0O])
    Pc_wls = P0_c.copy()

    t0 = time.perf_counter()
    for k in range(steps):
        z = np.concatenate([Y_th[k + 1], Y_w[k + 1]])
        xc_wls, Pc_wls = dynamic_wls_step(xc_wls, Q_c, R_c, z, dt, f_central, h_central, None)
        xc_wls[0:Nbus] = wrap_angle(xc_wls[0:Nbus])
    t_central_wls = time.perf_counter() - t0

    # -------- CENTRAL WNLS
    xc_wnls = np.concatenate([x0t, x0w, x0O])
    Pc_wnls = P0_c.copy()

    t0 = time.perf_counter()
    for k in range(steps):
        z = np.concatenate([Y_th[k + 1], Y_w[k + 1]])
        xc_wnls, Pc_wnls = dynamic_wnls_step(
            xc_wnls, Q_c, R_c, z, dt, f_central, h_central, None, max_iter=wnls_inner_iters
        )
        xc_wnls[0:Nbus] = wrap_angle(xc_wnls[0:Nbus])
    t_central_wnls = time.perf_counter() - t0

    # -------- DISTRIBUTED PREP
    gen_buses = get_generator_buses_case(ppc)
    if len(gen_buses) == 0:
        gen_buses = [0]
    subsystems = generator_centered_partition_maxsize(K, gen_buses, max_size=max_cluster_size)
    n_clusters = len(subsystems)

    def init_dist_states():
        xS = []
        PS = []
        for ids in subsystems:
            m = len(ids)
            x_init = np.concatenate([x0t[ids], x0w[ids], np.zeros(m)])
            P_init = np.diag(np.concatenate([(0.5 ** 2) * np.ones(m), (0.5 ** 2) * np.ones(m), (1.0 ** 2) * np.ones(m)]))
            xS.append(x_init)
            PS.append(P_init)
        return xS, PS

    Q_base_th, Q_base_w, Q_base_O = 1e-4, 1e-4, 1e-6

    # -------- DISTRIBUTED UKF
    xS_ukf, PS_ukf = init_dist_states()
    theta_global_est = x0t.copy()

    t1 = time.perf_counter()
    for k in range(steps):
        theta_jacobi = theta_global_est.copy()
        for _ in range(jacobi_iters):
            theta_updates = theta_jacobi.copy()
            for s, ids in enumerate(subsystems):
                m = len(ids)
                f_sub, h_sub = make_subsystem_models(K, d_true, ids)

                Q_s = np.diag(np.concatenate([Q_base_th * np.ones(m), Q_base_w * np.ones(m), Q_base_O * np.ones(m)]))
                R_s = np.diag(np.concatenate([(meas_std_th ** 2) * np.ones(m), (meas_std_w ** 2) * np.ones(m)]))
                z_s = np.concatenate([Y_th[k + 1, ids], Y_w[k + 1, ids]])

                x_new, P_new = ukf_step(xS_ukf[s], PS_ukf[s], R_s, Q_s, gamma, z_s, dt, f_sub, h_sub, theta_jacobi)
                x_new[0:m] = wrap_angle(x_new[0:m])

                xS_ukf[s] = x_new
                PS_ukf[s] = P_new
                theta_updates[ids] = x_new[0:m]
            theta_jacobi = theta_updates
        theta_global_est = theta_jacobi.copy()
    t_dist_ukf = time.perf_counter() - t1

    # -------- DISTRIBUTED WLS
    xS_wls, PS_wls = init_dist_states()
    theta_global_est = x0t.copy()

    t1 = time.perf_counter()
    for k in range(steps):
        theta_jacobi = theta_global_est.copy()
        for _ in range(jacobi_iters):
            theta_updates = theta_jacobi.copy()
            for s, ids in enumerate(subsystems):
                m = len(ids)
                f_sub, h_sub = make_subsystem_models(K, d_true, ids)

                Q_s = np.diag(np.concatenate([Q_base_th * np.ones(m), Q_base_w * np.ones(m), Q_base_O * np.ones(m)]))
                R_s = np.diag(np.concatenate([(meas_std_th ** 2) * np.ones(m), (meas_std_w ** 2) * np.ones(m)]))
                z_s = np.concatenate([Y_th[k + 1, ids], Y_w[k + 1, ids]])

                x_new, P_new = dynamic_wls_step(xS_wls[s], Q_s, R_s, z_s, dt, f_sub, h_sub, theta_jacobi)
                x_new[0:m] = wrap_angle(x_new[0:m])

                xS_wls[s] = x_new
                PS_wls[s] = P_new
                theta_updates[ids] = x_new[0:m]
            theta_jacobi = theta_updates
        theta_global_est = theta_jacobi.copy()
    t_dist_wls = time.perf_counter() - t1

    # -------- DISTRIBUTED WNLS
    xS_wnls, PS_wnls = init_dist_states()
    theta_global_est = x0t.copy()

    t1 = time.perf_counter()
    for k in range(steps):
        theta_jacobi = theta_global_est.copy()
        for _ in range(jacobi_iters):
            theta_updates = theta_jacobi.copy()
            for s, ids in enumerate(subsystems):
                m = len(ids)
                f_sub, h_sub = make_subsystem_models(K, d_true, ids)

                Q_s = np.diag(np.concatenate([Q_base_th * np.ones(m), Q_base_w * np.ones(m), Q_base_O * np.ones(m)]))
                R_s = np.diag(np.concatenate([(meas_std_th ** 2) * np.ones(m), (meas_std_w ** 2) * np.ones(m)]))
                z_s = np.concatenate([Y_th[k + 1, ids], Y_w[k + 1, ids]])

                x_new, P_new = dynamic_wnls_step(
                    xS_wnls[s], Q_s, R_s, z_s, dt, f_sub, h_sub, theta_jacobi, max_iter=wnls_inner_iters
                )
                x_new[0:m] = wrap_angle(x_new[0:m])

                xS_wnls[s] = x_new
                PS_wnls[s] = P_new
                theta_updates[ids] = x_new[0:m]
            theta_jacobi = theta_updates
        theta_global_est = theta_jacobi.copy()
    t_dist_wnls = time.perf_counter() - t1

    return {
        "Nbus": Nbus,
        "n_clusters": n_clusters,
        "central_ukf": t_central_ukf,
        "distributed_ukf": t_dist_ukf,
        "central_wls": t_central_wls,
        "distributed_wls": t_dist_wls,
        "central_wnls": t_central_wnls,
        "distributed_wnls": t_dist_wnls,
    }


# %%# Multi-case runner — returns results only, no plotting
def benchmark_cases(
    case_names: List[str],
    *,
    coupling: str = "susceptance",
    max_cluster_size: int = 5,
    jacobi_iters: int = 1,
    dt: float = 0.01,
    T: float = 3.0,
    seed: int = 42,
    wnls_inner_iters: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run central and distributed UKF / WLS / WNLS for each named PYPOWER case.
    """
    results = []
    for name in case_names:
        case_fn = getattr(ppa, name, None)
        if case_fn is None or not callable(case_fn):
            print(f"Skipping unknown case: {name}")
            continue

        print(f"Running {name} ...")
        out = run_case_benchmark(
            case_fn,
            coupling=coupling,
            max_cluster_size=max_cluster_size,
            jacobi_iters=jacobi_iters,
            dt=dt,
            T=T,
            seed=seed,
            wnls_inner_iters=wnls_inner_iters,
        )
        row = {"case_name": name, **out}
        results.append(row)

        print(
            f"  buses={out['Nbus']:4d} | clusters={out['n_clusters']:2d} | "
            f"UKF(C,D)=({out['central_ukf']:.3f}s, {out['distributed_ukf']:.3f}s) | "
            f"WLS(C,D)=({out['central_wls']:.3f}s, {out['distributed_wls']:.3f}s) | "
            f"WNLS(C,D)=({out['central_wnls']:.3f}s, {out['distributed_wnls']:.3f}s)"
        )

    if not results:
        print("No cases ran.")

    return results


# %%# Standalone plot function — call any time you have results
def plot_benchmark(
    results: List[Dict[str, Any]],
    *,
    save_plot_path: str = "benchmark_times_all_methods.png",
) -> None:
    """
    Plot central vs distributed computation times for UKF / WLS / WNLS.
    """
    if not results:
        print("No results to plot.")
        return

    case_names = [r["case_name"] for r in results]
    central_ukf = np.array([r["central_ukf"] for r in results], dtype=float)
    dist_ukf = np.array([r["distributed_ukf"] for r in results], dtype=float)
    central_wls = np.array([r["central_wls"] for r in results], dtype=float)
    dist_wls = np.array([r["distributed_wls"] for r in results], dtype=float)
    central_wnls = np.array([r["central_wnls"] for r in results], dtype=float)
    dist_wnls = np.array([r["distributed_wnls"] for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(max(12, 1.5 * len(results)), 6))
    ax.plot(case_names, central_ukf, marker="o", linewidth=2, label="Central UKF")
    ax.plot(case_names, dist_ukf, marker="o", linestyle="--", linewidth=2, label="Distributed UKF")
    ax.plot(case_names, central_wls, marker="s", linewidth=2, label="Central WLS")
    ax.plot(case_names, dist_wls, marker="s", linestyle="--", linewidth=2, label="Distributed WLS")
    ax.plot(case_names, central_wnls, marker="^", linewidth=2, label="Central WNLS")
    ax.plot(case_names, dist_wnls, marker="^", linestyle="--", linewidth=2, label="Distributed WNLS")

    ax.set_yscale("log")
    ax.set_xlabel("Case")
    ax.set_ylabel("Computation Time [s] (log scale)")
    ax.legend(ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    plt.show()


# %%# CLI entrypoint
def main():
    case_names = [
        "case9",
        "case14",
        "case30",
        "case39",
        "case57",
        "case118",
        "case300",
    ]

    results = benchmark_cases(
        case_names,
        coupling="magnitude",
        max_cluster_size=5,
        jacobi_iters=1,
        dt=0.01,
        T=3.0,
        seed=42,
        wnls_inner_iters=5,
    )

    plot_benchmark(results)
    return results


# %%
if __name__ == "__main__":
    results = main()
#%%

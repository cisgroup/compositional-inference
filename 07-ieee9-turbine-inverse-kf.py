"""
IEEE-9 + Pelton Turbine Hybrid Simulator — inverse variant
==========================================================

This script couples an IEEE-9 network model with embedded Pelton turbine
subsystems. It generates synthetic turbine-network measurements and recovers
hidden generator and runner states with local Kalman-filter updates.

Measurements assumed available per generator:
    - xG, yG : generator lateral displacement
    - xR, yR : runner lateral displacement
    - q      : turbine flow rate
    - omega  : turbine rotational speed

The non-generator network buses, governor, hydraulics, and rotational
subsystems remain deterministic and use the same equations as the
forward model. The generator and runner states are updated through a
linearized Kalman filter with Heun mean propagation and local
message passing.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time



# ---------------------------------------------------------------------------
# Progress / timing helpers (so long runs show ETA in the terminal)
# ---------------------------------------------------------------------------

def _format_hms(seconds: float) -> str:
    seconds = float(max(seconds, 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - 60 * m - 3600 * h
    if h > 0:
        return f"{h:d}h{m:02d}m{s:04.1f}s"
    if m > 0:
        return f"{m:d}m{s:04.1f}s"
    return f"{s:.1f}s"


def _progress_report(k: int, steps: int, t0: float, label: str, every: int) -> None:
    if every <= 0:
        return
    if (k + 1) % every != 0 and (k + 1) != steps:
        return
    elapsed = time.perf_counter() - t0
    done = k + 1
    rate = done / max(elapsed, 1e-12)
    eta = (steps - done) / max(rate, 1e-12)
    pct = 100.0 * done / max(steps, 1)
    print(f"[{label}] {done}/{steps} ({pct:5.1f}%)  elapsed={_format_hms(elapsed)}  eta={_format_hms(eta)}  rate={rate:6.1f} it/s")

# ============================================================================
# 1. IEEE-9 network
# ============================================================================

def build_coupling_from_case9():
    from pypower.case9 import case9
    from pypower.idx_brch import F_BUS, T_BUS
    from pypower.idx_bus import BUS_I
    from pypower.makeYbus import makeYbus

    ppc = case9()
    baseMVA = ppc["baseMVA"]
    bus = ppc["bus"].copy()
    branch = ppc["branch"].copy()

    bus_numbers = bus[:, BUS_I].astype(int)
    nb = bus.shape[0]
    bus_map = {bus_no: idx for idx, bus_no in enumerate(bus_numbers)}

    f = branch[:, F_BUS].astype(int)
    t = branch[:, T_BUS].astype(int)
    branch[:, F_BUS] = np.array([bus_map[x] for x in f], dtype=float)
    branch[:, T_BUS] = np.array([bus_map[x] for x in t], dtype=float)

    Ybus, _, _ = makeYbus(baseMVA, bus, branch)
    Ybus = Ybus.toarray()

    B = np.imag(Ybus)
    K = np.zeros_like(B)
    for i in range(nb):
        for j in range(nb):
            if i != j:
                K[i, j] = B[i, j]
    np.fill_diagonal(K, 0.0)

    neighbors = [list(np.where(np.abs(K[i]) > 1e-12)[0]) for i in range(nb)]
    return K, neighbors


def compute_Pe_all(delta, K):
    Pe = np.empty(len(delta))
    for i in range(len(delta)):
        Pe[i] = np.dot(K[i], np.sin(delta - delta[i]))
    return Pe


def _dc_warm_start(K, P_inj, slack=0):
    nb = K.shape[0]
    B = np.diag(K.sum(1)) - K
    idx = [i for i in range(nb) if i != slack]
    d = np.zeros(nb)
    d[idx] = np.linalg.solve(B[np.ix_(idx, idx)], P_inj[idx])
    return d


def _pf_jacobian(K, delta):
    nb = K.shape[0]
    J = np.zeros((nb, nb))
    for i in range(nb):
        J[i, i] = -np.dot(K[i], np.cos(delta - delta[i]))
        for j in range(nb):
            if i != j:
                J[i, j] = K[i, j] * np.cos(delta[j] - delta[i])
    return J


def _nr_power_flow(K, P_inj, slack_bus=0, tol=1e-12, max_iter=50):
    nb = K.shape[0]
    delta = _dc_warm_start(K, P_inj, slack_bus)
    idx = [i for i in range(nb) if i != slack_bus]
    for _ in range(max_iter):
        Pe = compute_Pe_all(delta, K)
        F = P_inj + Pe
        fr = F[idx]
        if np.max(np.abs(fr)) < tol:
            break
        J = _pf_jacobian(K, delta)
        delta[idx] += np.linalg.solve(J[np.ix_(idx, idx)], -fr)
    return delta


def build_ieee9_network():
    from pypower.case9 import case9
    from pypower.idx_bus import BUS_I, PD
    from pypower.idx_gen import GEN_BUS, PG

    K, neighbors = build_coupling_from_case9()
    nb = K.shape[0]

    ppc = case9()
    baseMVA = float(ppc["baseMVA"])
    bus = ppc["bus"]
    gen = ppc["gen"]

    bus_numbers = bus[:, BUS_I].astype(int)
    bus_map = {bn: idx for idx, bn in enumerate(bus_numbers)}

    gen_buses_1based = gen[:, GEN_BUS].astype(int)
    gen_buses = [bus_map[b] for b in gen_buses_1based]
    P_gen_ppc = gen[:, PG].copy().astype(float)

    P_load_total = bus[:, PD].sum()
    slack_i = next(i for i, pg in enumerate(P_gen_ppc) if pg == 0.0)
    P_gen_ppc[slack_i] = P_load_total - P_gen_ppc[P_gen_ppc > 0].sum()
    P_gen_MW = P_gen_ppc * P_load_total / P_gen_ppc.sum()

    load_mask = bus[:, PD] > 0
    load_buses = [bus_map[int(bus[i, BUS_I])] for i in range(nb) if bus[i, PD] > 0]
    P_load_raw = bus[load_mask, PD]

    P_inj_pu = np.zeros(nb)
    for bus_idx, pmw in zip(gen_buses, P_gen_MW):
        P_inj_pu[bus_idx] += pmw / baseMVA
    for bus_idx, pmw in zip(load_buses, P_load_raw):
        P_inj_pu[bus_idx] -= pmw / baseMVA

    slack_bus = gen_buses[slack_i]
    delta_ss = _nr_power_flow(K, P_inj_pu, slack_bus=slack_bus)

    return K, neighbors, baseMVA, P_gen_MW, gen_buses, load_buses, P_inj_pu, delta_ss


# ============================================================================
# 2. Parameters
# ============================================================================

@dataclass
class Parameters:
    Q_rated: float = 27.0
    H_rated: float = 595.0
    eta: float = 0.92
    n_rated: float = 375.0
    Z_n: int = 6
    rho: float = 1000.0
    g: float = 9.81
    T_e: float = 0.5155
    phi_j: float = 0.985
    s_e: float = 0.102
    h0: float = 1.0
    y_r: float = 0.9
    T_q: float = 0.05
    k_p: float = 5.0
    k_i: float = 2.12
    k_d: float = 0.5
    T_y: float = 0.01
    m1: float = 6e5
    m2: float = 3e5
    J1: float = 6.8e6
    J2: float = 3.4e6
    e1: float = 0.5e-3
    e2: float = 0.5e-3
    E: float = 200e9
    G: float = 80e9
    l_shaft: float = 10.3
    d_H: float = 1.15
    d_B: float = 0.3
    h_shaft: float = 11.995
    phi_m: float = 0.01
    a: float = 1.5
    b: float = 7.3
    c: float = 1.5
    d: float = 1.0
    k_gen: float = 2e9
    k_run: float = 8e9
    k_cross: float = 2e7
    c_gen: float = 1e6
    c_run: float = 1e6
    c_cross: float = 2e4
    c_t: float = 1e5
    D_omega: float = 1e5
    T_ab: float = 10.0
    delta_0: float = 4e-3
    delta_2: float = 3.5e-3
    k_r: float = 6e9
    f_fric: float = 0.02
    K_seal: float = 3e7
    D_seal: float = 1e5
    tau_seal: float = 0.3
    e_x: float = 1.0
    e_y: float = 0.5
    e_h: float = 1.5
    A_vortex: float = 2e4
    f_vortex_ratio: float = 0.25
    k_bld_load: float = 2.5e4
    k_hyd_load: float = 2.0e4
    k_torque_radial: float = 0.12
    k_radial_dc: float = 1.0
    r_runner_force: float = 0.55
    omega_rated: float = field(init=False)
    P_rated: float = field(init=False)
    M_gB: float = field(init=False)
    k_y: float = field(init=False)
    J1_eff: float = field(init=False)
    J2_eff: float = field(init=False)
    J_tot: float = field(init=False)
    J_eq: float = field(init=False)
    m_01: float = field(init=False)
    m_02: float = field(init=False)
    C_q: float = field(init=False)
    K11: float = field(init=False)
    K22: float = field(init=False)
    K12: float = field(init=False)

    def __post_init__(self):
        self.omega_rated = 2.0 * np.pi * self.n_rated / 60.0
        self.P_rated = self.rho * self.g * self.Q_rated * self.H_rated * self.eta
        self.M_gB = self.P_rated / self.omega_rated
        J_p = np.pi / 32.0 * (self.d_H**4 - self.d_B**4)
        self.k_y = self.G * J_p / self.l_shaft
        self.J1_eff = self.J1 + 2.0 * self.m1 * self.e1**2
        self.J2_eff = self.J2 + 2.0 * self.m2 * (
            self.e2 * np.cos(self.phi_m) + self.h_shaft * np.sin(self.phi_m)
        ) ** 2
        self.J_tot = self.J1_eff + self.J2_eff
        self.J_eq = (self.J1_eff * self.J2_eff) / (self.J1_eff + self.J2_eff)
        alpha_n = np.radians(45.0)
        beta_n = np.radians(62.0)
        self.m_01 = np.pi * np.tan(alpha_n / 2) * 0.218 * np.cos((alpha_n + beta_n) / 4)
        self.m_02 = np.pi * np.tan(alpha_n / 2) ** 2 * np.cos((alpha_n + beta_n) / 4)
        x_eq = 0.8
        A_eq = (
            self.m_01 * self.s_e * (x_eq + 1.0)
            - self.m_02 * self.s_e**2 * (x_eq + 1.0) ** 2
        )
        self.C_q = 1.0 / (A_eq * np.sqrt(2.0 * self.g * self.H_rated))
        I = np.pi / 64.0 * (self.d_H**4 - self.d_B**4)
        k_sB = 3.0 * self.E * I / self.b**3
        k_G = k_sB * (self.a**2 + self.b**2 + 3.0 * self.a * self.b) / (3.0 * self.a * self.b)
        k_R = k_sB * (self.c**2 + self.b**2 + 3.0 * self.c * self.b) / (3.0 * self.c * self.b)
        k_cpl = -k_sB * (self.a + self.b) * (self.b + self.c) / (3.0 * self.a * self.b)
        self.K11 = 4.0 * k_G
        self.K22 = 4.0 * k_R
        self.K12 = 4.0 * k_cpl


# ============================================================================
# 3. Generic integration and KF helpers
# ============================================================================

def heun_step_cont(f_cont, x, u, t, dt):
    k1 = f_cont(x, u, t)
    x_pred = x + dt * k1
    k2 = f_cont(x_pred, u, t + dt)
    return x + 0.5 * dt * (k1 + k2)


from src.filters import linear_kf_predict_heun, linear_kf_update, numerical_jacobian


def moving_average(signal, window):
    window = max(1, int(window))
    if window == 1:
        return np.asarray(signal, dtype=float).copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(np.asarray(signal, dtype=float), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ============================================================================
# 4. Deterministic subsystems
# ============================================================================

def governor_rhs_forward(x_gov_vec, t, p: Parameters, n_state: float, n_dot: float, n_ref: float):
    x_n, x_gov = x_gov_vec
    error = n_ref - n_state
    u_pid = p.k_p * error + p.k_i * x_gov - p.k_d * n_dot
    u_des = np.clip(u_pid, 0.0, 1.0)
    # Conditional-integration anti-windup: freeze integrator only when
    # the output is saturated AND the error would push it deeper into saturation.
    saturated_high = (u_pid >= 1.0) and (error > 0.0)
    saturated_low = (u_pid <= 0.0) and (error < 0.0)
    dx_gov = 0.0 if (saturated_high or saturated_low) else error
    dx_n = (u_des - x_n) / p.T_y
    return np.array([dx_n, dx_gov])


def governor_heun_step_forward(x_n, x_gov, n_state, n_dot, p: Parameters, dt: float, t: float, n_ref: float):
    x_vec = np.array([x_n, x_gov], dtype=float)
    k1 = governor_rhs_forward(x_vec, t, p, n_state, n_dot, n_ref)
    x_pred = x_vec + dt * k1
    k2 = governor_rhs_forward(x_pred, t + dt, p, n_state, n_dot, n_ref)
    x_new = x_vec + 0.5 * dt * (k1 + k2)
    return float(x_new[0]), float(x_new[1])


def f_hydraulics_cont_forward(x, u, t):
    p = u["p"]
    msg = u["msg"]
    x1, x2, x3, q, Cq_s = x
    C_q = Cq_s * p.C_q

    x_n = msg["x_n"]
    x_n_eff = np.clip(x_n, 0.0, 1.0)
    h = x3
    A_n = max(
        p.m_01 * p.s_e * (x_n_eff + 1.0) - p.m_02 * p.s_e**2 * (x_n_eff + 1.0) ** 2,
        1e-8,
    )
    q_target = C_q * A_n * np.sqrt(2 * p.g * p.H_rated * max(h + 1.0, 0.1))

    dx1 = x2
    dx2 = x3
    dx3 = -(np.pi**2 / p.T_e**2) * x2 + (1.0 / (p.Z_n * p.T_e**3)) * (p.h0 - x3 / p.y_r**2 - q**2)
    dq = (q_target - q) / p.T_q
    return np.array([dx1, dx2, dx3, dq, 0.0])


def f_rotational_cont_forward(x, u, t):
    p = u["p"]
    msg = u["msg"]
    n_state, omega, delta_state, e_x_s, e_y_s, e_h_s = x
    q = msg["q"]
    h = msg["h"]
    Te_grid = msg["Te_grid"]

    e_x = e_x_s * p.e_x
    e_y = e_y_s * p.e_y
    e_h = e_h_s * p.e_h
    m_t = e_x * n_state + e_y * (q - 1.0) + e_h * h
    M_t = p.M_gB * (1.0 + m_t)
    omega_err = omega - p.omega_rated
    M_damp = p.D_omega * omega_err
    omega_dot = (M_t - Te_grid - M_damp) / p.J_tot
    n_dot = omega_dot / p.omega_rated
    delta_dot = omega - p.omega_rated
    return np.array([n_dot, omega_dot, delta_dot, 0.0, 0.0, 0.0])


def torque_from_rot_state(x_rot, p: Parameters, q: float, h: float):
    n_state, omega, _, e_x_s, e_y_s, e_h_s = x_rot
    e_x = e_x_s * p.e_x
    e_y = e_y_s * p.e_y
    e_h = e_h_s * p.e_h
    m_t = e_x * n_state + e_y * (q - 1.0) + e_h * h
    return p.M_gB * (1.0 + m_t)


# ============================================================================
# 5. Estimation subsystems
# ============================================================================

def f_generator_lv_cont_forward(x, u, t):
    p = u["p"]
    msg = u["msg"]
    xG, yG, vxG, vyG = x
    omega = msg["omega"]
    phi1 = msg["phi1"]
    xR = msg["xR"]
    yR = msg["yR"]

    rG = np.sqrt(xG**2 + yG**2) + 1e-12
    Fx_kG = -p.k_gen * xG - p.K11 * xG - p.K12 * xR
    Fy_kG = -p.k_gen * yG - p.K11 * yG - p.K12 * yR
    Fx_kcG = -p.k_cross * yG
    Fy_kcG = p.k_cross * xG
    Fx_cG = -p.c_gen * vxG
    Fy_cG = -p.c_gen * vyG
    Fx_ccG = -p.c_cross * vyG
    Fy_ccG = p.c_cross * vxG
    Fx_unbG = p.m1 * p.e1 * omega**2 * np.cos(phi1)
    Fy_unbG = p.m1 * p.e1 * omega**2 * np.sin(phi1)

    Fx_rubG = 0.0
    Fy_rubG = 0.0
    if rG > p.delta_0:
        pen = (rG - p.delta_0) * p.k_r / rG
        Fx_rubG = -pen * (xG - p.f_fric * yG)
        Fy_rubG = -pen * (p.f_fric * xG + yG)

    Fx_G = Fx_kG + Fx_kcG + Fx_cG + Fx_ccG + Fx_unbG + Fx_rubG
    Fy_G = Fy_kG + Fy_kcG + Fy_cG + Fy_ccG + Fy_unbG + Fy_rubG

    return np.array([vxG, vyG, Fx_G / p.m1, Fy_G / p.m1])


def generator_force_components(x_gen, p, omega, phi1, xR, yR):
    xG, yG, vxG, vyG = x_gen
    rG = np.sqrt(xG**2 + yG**2) + 1e-12

    Fx_kG = -p.k_gen * xG - p.K11 * xG - p.K12 * xR
    Fy_kG = -p.k_gen * yG - p.K11 * yG - p.K12 * yR
    Fx_kcG = -p.k_cross * yG
    Fy_kcG = p.k_cross * xG
    Fx_cG = -p.c_gen * vxG
    Fy_cG = -p.c_gen * vyG
    Fx_ccG = -p.c_cross * vyG
    Fy_ccG = p.c_cross * vxG
    Fx_unbG = p.m1 * p.e1 * omega**2 * np.cos(phi1)
    Fy_unbG = p.m1 * p.e1 * omega**2 * np.sin(phi1)

    Fx_rubG = 0.0
    Fy_rubG = 0.0
    if rG > p.delta_0:
        pen = (rG - p.delta_0) * p.k_r / rG
        Fx_rubG = -pen * (xG - p.f_fric * yG)
        Fy_rubG = -pen * (p.f_fric * xG + yG)

    Fx_tot = Fx_kG + Fx_kcG + Fx_cG + Fx_ccG + Fx_unbG + Fx_rubG
    Fy_tot = Fy_kG + Fy_kcG + Fy_cG + Fy_ccG + Fy_unbG + Fy_rubG
    return {
        "unb_mag": np.hypot(Fx_unbG, Fy_unbG),
        "total_mag": np.hypot(Fx_tot, Fy_tot),
    }


def gen_accel_from_state_forward(x_gen, p, omega, phi1, xR, yR):
    return f_generator_lv_cont_forward(
        x_gen,
        {"p": p, "msg": {"omega": omega, "phi1": phi1, "xR": xR, "yR": yR}},
        0.0,
    )[2:]


def f_runner_torsion_stateonly_cont(x, u, t):
    p = u["p"]
    msg = u["msg"]
    xR, yR, vxR, vyR, alpha_t, beta_t = x

    omega = msg["omega"]
    phi1 = msg["phi1"]
    M_t = msg["M_t"]
    Te_grid = msg["Te_grid"]
    q = msg["q"]
    h = msg.get("h", 0.0)
    xG = msg["xG"]
    yG = msg["yG"]
    dvxG = msg["dvxG"]
    dvyG = msg["dvyG"]

    Fx_kR = -p.k_run * xR - p.K22 * xR - p.K12 * xG
    Fy_kR = -p.k_run * yR - p.K22 * yR - p.K12 * yG
    Fx_kcR = -0.8 * p.k_cross * yR
    Fy_kcR = 0.8 * p.k_cross * xR
    Fx_seal = -p.K_seal * xR - p.tau_seal * omega * p.D_seal * yR
    Fy_seal = p.tau_seal * omega * p.D_seal * xR - p.K_seal * yR
    Fx_cR = -p.c_run * vxR
    Fy_cR = -p.c_run * vyR

    phi2 = phi1 - alpha_t
    Fx_unbR = p.m2 * p.e2 * omega**2 * np.cos(phi2)
    Fy_unbR = p.m2 * p.e2 * omega**2 * np.sin(phi2)
    fv = p.f_vortex_ratio * omega
    q_head_scale = q * np.sqrt(max(h + 1.0, 0.1))
    F_hyd_amp = p.A_vortex * q + p.k_hyd_load * q_head_scale**2
    Fx_hyd = F_hyd_amp * np.sin(fv * t)
    Fy_hyd = F_hyd_amp * np.cos(fv * t)
    fb = p.Z_n * omega
    F_bld = 0.1 * p.A_vortex + p.k_bld_load * q_head_scale**2
    Fx_bld = F_bld * np.sin(fb * t)
    Fy_bld = F_bld * np.cos(fb * t)
    F_torque = p.k_torque_radial * abs(M_t) / max(p.r_runner_force, 1e-3)
    Fx_torque = F_torque * np.cos(phi2)
    Fy_torque = F_torque * np.sin(phi2)
    F_radial_dc = p.k_radial_dc * (q_head_scale - 1.0) * p.M_gB / max(p.r_runner_force, 1e-3)
    # Fixed jet direction in the stationary frame: this creates an orbit-center shift.
    Fx_dc = F_radial_dc
    Fy_dc = 0.0

    Fx_R = Fx_kR + Fx_kcR + Fx_seal + Fx_cR + Fx_unbR + Fx_hyd + Fx_bld + Fx_torque + Fx_dc
    Fy_R = Fy_kR + Fy_kcR + Fy_seal + Fy_cR + Fy_unbR + Fy_hyd + Fy_bld + Fy_torque + Fy_dc

    dvxR = Fx_R / p.m2
    dvyR = Fy_R / p.m2
    c_gen = (-p.m1 * p.e1 * (dvyG * np.cos(phi1) - dvxG * np.sin(phi1))) / p.J1_eff
    c_run = (p.m2 * p.e2 * (dvyR * np.cos(phi2) - dvxR * np.sin(phi2))) / p.J2_eff
    dalpha = beta_t
    dbeta = (
        M_t / p.J1_eff
        + Te_grid / p.J2_eff
        - p.k_y * alpha_t / p.J_eq
        - p.c_t * beta_t / p.J_eq
        + c_gen
        - c_run
    )
    return np.array([vxR, vyR, dvxR, dvyR, dalpha, dbeta])


def runner_force_components(x_rt, p, omega, phi1, q, h, M_t, xG, yG, t):
    xR, yR, vxR, vyR, alpha_t, _ = x_rt

    Fx_kR = -p.k_run * xR - p.K22 * xR - p.K12 * xG
    Fy_kR = -p.k_run * yR - p.K22 * yR - p.K12 * yG
    Fx_kcR = -0.8 * p.k_cross * yR
    Fy_kcR = 0.8 * p.k_cross * xR
    Fx_seal = -p.K_seal * xR - p.tau_seal * omega * p.D_seal * yR
    Fy_seal = p.tau_seal * omega * p.D_seal * xR - p.K_seal * yR
    Fx_cR = -p.c_run * vxR
    Fy_cR = -p.c_run * vyR

    phi2 = phi1 - alpha_t
    Fx_unbR = p.m2 * p.e2 * omega**2 * np.cos(phi2)
    Fy_unbR = p.m2 * p.e2 * omega**2 * np.sin(phi2)
    fv = p.f_vortex_ratio * omega
    q_head_scale = q * np.sqrt(max(h + 1.0, 0.1))
    F_hyd_amp = p.A_vortex * q + p.k_hyd_load * q_head_scale**2
    Fx_hyd = F_hyd_amp * np.sin(fv * t)
    Fy_hyd = F_hyd_amp * np.cos(fv * t)
    fb = p.Z_n * omega
    F_bld = 0.1 * p.A_vortex + p.k_bld_load * q_head_scale**2
    Fx_bld = F_bld * np.sin(fb * t)
    Fy_bld = F_bld * np.cos(fb * t)
    F_torque = p.k_torque_radial * abs(M_t) / max(p.r_runner_force, 1e-3)
    Fx_torque = F_torque * np.cos(phi2)
    Fy_torque = F_torque * np.sin(phi2)
    F_radial_dc = p.k_radial_dc * (q_head_scale - 1.0) * p.M_gB / max(p.r_runner_force, 1e-3)
    Fx_dc = F_radial_dc
    Fy_dc = 0.0

    Fx_tot = Fx_kR + Fx_kcR + Fx_seal + Fx_cR + Fx_unbR + Fx_hyd + Fx_bld + Fx_torque + Fx_dc
    Fy_tot = Fy_kR + Fy_kcR + Fy_seal + Fy_cR + Fy_unbR + Fy_hyd + Fy_bld + Fy_torque + Fy_dc
    return {
        "hyd_mag": np.hypot(Fx_hyd, Fy_hyd),
        "blade_mag": np.hypot(Fx_bld, Fy_bld),
        "torque_mag": np.hypot(Fx_torque, Fy_torque),
        "dc_mag": np.hypot(Fx_dc, Fy_dc),
        "total_mag": np.hypot(Fx_tot, Fy_tot),
    }


GEN_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
RUN_H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])


@dataclass
class KFConfig:
    q_gen: float = 1e-12
    q_run: float = 1e-12
    q_torsion: float = 1e-9
    r_xy_gen: float = 1e-10
    r_xy_run: float = 1e-10
    p0_gen: float = 1e-8
    p0_run: float = 1e-8
    p0_torsion: float = 1e-6


@dataclass
class SensorSample:
    xG: float
    yG: float
    xR: float
    yR: float
    q: float
    omega: float


@dataclass
class TurbineModulesForward:
    x_h: np.ndarray
    x_rot: np.ndarray
    x_gen: np.ndarray
    x_rt: np.ndarray
    x_gov: np.ndarray


@dataclass
class TurbineModulesInverse:
    x_h: np.ndarray
    x_rot: np.ndarray
    x_gen: np.ndarray
    P_gen: np.ndarray
    x_rt: np.ndarray
    P_rt: np.ndarray
    x_gov: np.ndarray


def unpack_19_to_modules_forward(x19):
    x_h = np.array([x19[0], x19[1], x19[2], x19[3], 1.0], dtype=float)
    x_gov = np.array([x19[4], x19[5]], dtype=float)
    x_rot = np.array([x19[6], x19[7], x19[18], 1.0, 1.0, 1.0], dtype=float)
    x_gen = np.array([x19[8], x19[9], x19[10], x19[11]], dtype=float)
    x_rt = np.array([x19[12], x19[13], x19[14], x19[15], x19[16], x19[17], 1.0, 1.0, 1.0], dtype=float)
    return TurbineModulesForward(x_h=x_h, x_rot=x_rot, x_gen=x_gen, x_rt=x_rt, x_gov=x_gov)


def pack_modules_forward_to_19(mod: TurbineModulesForward, x19_template):
    x19 = x19_template.copy()
    x19[0:4] = mod.x_h[:4]
    x19[4:6] = mod.x_gov
    x19[6] = mod.x_rot[0]
    x19[7] = mod.x_rot[1]
    x19[18] = mod.x_rot[2]
    x19[8:12] = mod.x_gen
    x19[12] = mod.x_rt[0]
    x19[13] = mod.x_rt[1]
    x19[14] = mod.x_rt[2]
    x19[15] = mod.x_rt[3]
    x19[16] = mod.x_rt[4]
    x19[17] = mod.x_rt[5]
    return x19


def unpack_19_to_modules_inverse(x19, kf_cfg: KFConfig):
    x_h = np.array([x19[0], x19[1], x19[2], x19[3], 1.0], dtype=float)
    x_gov = np.array([x19[4], x19[5]], dtype=float)
    x_rot = np.array([x19[6], x19[7], x19[18], 1.0, 1.0, 1.0], dtype=float)
    x_gen = np.array([x19[8], x19[9], x19[10], x19[11]], dtype=float)
    x_rt = np.array([x19[12], x19[13], x19[14], x19[15], x19[16], x19[17]], dtype=float)
    P_gen = np.diag([kf_cfg.p0_gen, kf_cfg.p0_gen, kf_cfg.p0_gen, kf_cfg.p0_gen])
    P_rt = np.diag([
        kf_cfg.p0_run,
        kf_cfg.p0_run,
        kf_cfg.p0_run,
        kf_cfg.p0_run,
        kf_cfg.p0_torsion,
        kf_cfg.p0_torsion,
    ])
    return TurbineModulesInverse(
        x_h=x_h,
        x_rot=x_rot,
        x_gen=x_gen,
        P_gen=P_gen,
        x_rt=x_rt,
        P_rt=P_rt,
        x_gov=x_gov,
    )


def pack_modules_inverse_to_19(mod: TurbineModulesInverse, x19_template):
    x19 = x19_template.copy()
    x19[0:4] = mod.x_h[:4]
    x19[4:6] = mod.x_gov
    x19[6] = mod.x_rot[0]
    x19[7] = mod.x_rot[1]
    x19[18] = mod.x_rot[2]
    x19[8:12] = mod.x_gen
    x19[12] = mod.x_rt[0]
    x19[13] = mod.x_rt[1]
    x19[14] = mod.x_rt[2]
    x19[15] = mod.x_rt[3]
    x19[16] = mod.x_rt[4]
    x19[17] = mod.x_rt[5]
    return x19


def make_process_covariances(kf_cfg: KFConfig, dt: float):
    dt_eff = max(float(dt), 1e-9)
    Q_gen = dt_eff * np.diag([kf_cfg.q_gen, kf_cfg.q_gen, 100.0 * kf_cfg.q_gen, 100.0 * kf_cfg.q_gen])
    Q_rt = dt_eff * np.diag([
        kf_cfg.q_run,
        kf_cfg.q_run,
        100.0 * kf_cfg.q_run,
        100.0 * kf_cfg.q_run,
        kf_cfg.q_torsion,
        10.0 * kf_cfg.q_torsion,
    ])
    R_gen = np.diag([kf_cfg.r_xy_gen, kf_cfg.r_xy_gen])
    R_rt = np.diag([kf_cfg.r_xy_run, kf_cfg.r_xy_run])
    return Q_gen, Q_rt, R_gen, R_rt


# ============================================================================
# 6. Forward turbine step for synthetic truth generation
# ============================================================================

def jacobi_forward_step_modules(mod: TurbineModulesForward, p: Parameters, Te_grid: float, t: float, dt: float, n_ref: float, jacobi_iters: int = 3):
    mod_it = TurbineModulesForward(
        x_h=mod.x_h.copy(),
        x_rot=mod.x_rot.copy(),
        x_gen=mod.x_gen.copy(),
        x_rt=mod.x_rt.copy(),
        x_gov=mod.x_gov.copy(),
    )

    for _ in range(jacobi_iters):
        x_h = mod_it.x_h
        x_rot = mod_it.x_rot
        x_gen = mod_it.x_gen
        x_rt = mod_it.x_rt
        x_gov = mod_it.x_gov

        q_msg = float(x_h[3])
        h_msg = float(x_h[2])
        n_state = float(x_rot[0])
        omega = float(x_rot[1])
        delta_state = float(x_rot[2])
        phi1 = delta_state + p.omega_rated * t

        M_t_msg = torque_from_rot_state(x_rot, p, q_msg, h_msg)
        omega_err = omega - p.omega_rated
        M_damp = p.D_omega * omega_err
        omega_dot = (M_t_msg - Te_grid - M_damp) / p.J_tot
        n_dot_msg = omega_dot / p.omega_rated
        dvxG_msg, dvyG_msg = gen_accel_from_state_forward(x_gen, p, omega, phi1, float(x_rt[0]), float(x_rt[1]))

        x_n_new, x_gov_new = governor_heun_step_forward(
            x_n=float(x_gov[0]),
            x_gov=float(x_gov[1]),
            n_state=n_state,
            n_dot=n_dot_msg,
            p=p,
            dt=dt,
            t=t,
            n_ref=n_ref,
        )

        x_h_new = heun_step_cont(f_hydraulics_cont_forward, x_h, {"p": p, "msg": {"x_n": x_n_new}}, t, dt)
        q_msg_updated = float(x_h_new[3])
        h_msg_updated = float(x_h_new[2])
        x_rot_new = heun_step_cont(
            f_rotational_cont_forward,
            x_rot,
            {"p": p, "msg": {"q": q_msg_updated, "h": h_msg_updated, "Te_grid": float(Te_grid)}},
            t,
            dt,
        )
        x_gen_new = heun_step_cont(
            f_generator_lv_cont_forward,
            x_gen,
            {"p": p, "msg": {"omega": omega, "phi1": phi1, "xR": float(x_rt[0]), "yR": float(x_rt[1])}},
            t,
            dt,
        )
        x_rt_new_6 = heun_step_cont(
            f_runner_torsion_stateonly_cont,
            x_rt[:6],
            {
                "p": p,
                "msg": {
                    "omega": omega,
                    "phi1": phi1,
                    "M_t": float(M_t_msg),
                    "Te_grid": float(Te_grid),
                    "q": q_msg,
                    "h": h_msg,
                    "xG": float(x_gen[0]),
                    "yG": float(x_gen[1]),
                    "dvxG": float(dvxG_msg),
                    "dvyG": float(dvyG_msg),
                },
            },
            t,
            dt,
        )
        x_rt_new = np.array([x_rt_new_6[0], x_rt_new_6[1], x_rt_new_6[2], x_rt_new_6[3], x_rt_new_6[4], x_rt_new_6[5], 1.0, 1.0, 1.0], dtype=float)

        mod_it = TurbineModulesForward(
            x_h=x_h_new,
            x_rot=x_rot_new,
            x_gen=x_gen_new,
            x_rt=x_rt_new,
            x_gov=np.array([x_n_new, x_gov_new], dtype=float),
        )

    return mod_it


def decomposed_turbine_step_19(x19, t, dt, p: Parameters, Te_grid: float, n_ref: float = 0.0, n_substeps: int = 10, jacobi_iters: int = 3):
    dt_sub = dt / n_substeps
    x = x19.copy()
    for s in range(n_substeps):
        ts = t + s * dt_sub
        mod0 = unpack_19_to_modules_forward(x)
        mod1 = jacobi_forward_step_modules(mod0, p, Te_grid, ts, dt_sub, n_ref, jacobi_iters=jacobi_iters)
        x = pack_modules_forward_to_19(mod1, x)
    return x


# ============================================================================
# 7. Inverse turbine step
# ============================================================================

def jacobi_inverse_step_modules(mod: TurbineModulesInverse, p: Parameters, Te_grid: float, sensor: SensorSample, kf_cfg: KFConfig, t: float, dt: float, n_ref: float, jacobi_iters: int = 3, do_update: bool = True):
    Q_gen, Q_rt, R_gen, R_rt = make_process_covariances(kf_cfg, dt)
    mod_it = TurbineModulesInverse(
        x_h=mod.x_h.copy(),
        x_rot=mod.x_rot.copy(),
        x_gen=mod.x_gen.copy(),
        P_gen=mod.P_gen.copy(),
        x_rt=mod.x_rt.copy(),
        P_rt=mod.P_rt.copy(),
        x_gov=mod.x_gov.copy(),
    )

    for _ in range(jacobi_iters):
        x_h = mod_it.x_h.copy()
        x_rot = mod_it.x_rot.copy()
        x_gen = mod_it.x_gen
        P_gen = mod_it.P_gen
        x_rt = mod_it.x_rt
        P_rt = mod_it.P_rt
        x_gov = mod_it.x_gov

        omega_meas = float(sensor.omega)
        q_meas = float(sensor.q)

        q_msg = float(x_h[3])
        h_msg = float(x_h[2])
        n_state = float(x_rot[0])
        delta_state = float(x_rot[2])
        phi1 = delta_state + p.omega_rated * t

        M_t_msg = torque_from_rot_state(x_rot, p, q_msg, h_msg)
        omega_det = float(x_rot[1])
        omega_err = omega_det - p.omega_rated
        M_damp = p.D_omega * omega_err
        omega_dot = (M_t_msg - Te_grid - M_damp) / p.J_tot
        n_dot_msg = omega_dot / p.omega_rated

        x_n_new, x_gov_new = governor_heun_step_forward(
            x_n=float(x_gov[0]),
            x_gov=float(x_gov[1]),
            n_state=n_state,
            n_dot=n_dot_msg,
            p=p,
            dt=dt,
            t=t,
            n_ref=n_ref,
        )
        x_h_new = heun_step_cont(f_hydraulics_cont_forward, x_h, {"p": p, "msg": {"x_n": x_n_new}}, t, dt)
        q_msg_updated = float(x_h_new[3])
        h_msg_updated = float(x_h_new[2])
        x_rot_new = heun_step_cont(
            f_rotational_cont_forward,
            x_rot,
            {"p": p, "msg": {"q": q_msg_updated, "h": h_msg_updated, "Te_grid": float(Te_grid)}},
            t,
            dt,
        )

        gen_u = {"p": p, "msg": {"omega": omega_meas, "phi1": phi1, "xR": float(x_rt[0]), "yR": float(x_rt[1])}}
        x_gen_pred, P_gen_pred = linear_kf_predict_heun(f_generator_lv_cont_forward, x_gen, P_gen, gen_u, t, dt, Q_gen)
        if do_update:
            z_gen = np.array([sensor.xG, sensor.yG], dtype=float)
            x_gen_new, P_gen_new = linear_kf_update(x_gen_pred, P_gen_pred, z_gen, GEN_H, R_gen)
        else:
            x_gen_new, P_gen_new = x_gen_pred, P_gen_pred

        dvxG_msg, dvyG_msg = gen_accel_from_state_forward(x_gen_new, p, omega_meas, phi1, float(x_rt[0]), float(x_rt[1]))
        rt_u = {
            "p": p,
            "msg": {
                "omega": omega_meas,
                "phi1": phi1,
                "M_t": float(M_t_msg),
                "Te_grid": float(Te_grid),
                "q": q_meas,
                "h": h_msg_updated,
                "xG": float(x_gen_new[0]),
                "yG": float(x_gen_new[1]),
                "dvxG": float(dvxG_msg),
                "dvyG": float(dvyG_msg),
            },
        }
        x_rt_pred, P_rt_pred = linear_kf_predict_heun(f_runner_torsion_stateonly_cont, x_rt, P_rt, rt_u, t, dt, Q_rt)
        if do_update:
            z_rt = np.array([sensor.xR, sensor.yR], dtype=float)
            x_rt_new, P_rt_new = linear_kf_update(x_rt_pred, P_rt_pred, z_rt, RUN_H, R_rt)
        else:
            x_rt_new, P_rt_new = x_rt_pred, P_rt_pred

        mod_it = TurbineModulesInverse(
            x_h=x_h_new,
            x_rot=x_rot_new,
            x_gen=x_gen_new,
            P_gen=P_gen_new,
            x_rt=x_rt_new,
            P_rt=P_rt_new,
            x_gov=np.array([x_n_new, x_gov_new], dtype=float),
        )

    return mod_it


def inverse_turbine_step_19(x19, P_gen, P_rt, t, dt, p: Parameters, Te_grid: float, sensor: SensorSample, kf_cfg: KFConfig, n_ref: float = 0.0, n_substeps: int = 1, jacobi_iters: int = 3, do_update: bool = True):
    dt_sub = dt / n_substeps
    x = x19.copy()
    Pg = P_gen.copy()
    Pr = P_rt.copy()
    for s in range(n_substeps):
        ts = t + s * dt_sub
        mod = unpack_19_to_modules_inverse(x, kf_cfg)
        mod.P_gen = Pg
        mod.P_rt = Pr
        mod_next = jacobi_inverse_step_modules(
            mod,
            p,
            Te_grid,
            sensor,
            kf_cfg,
            ts,
            dt_sub,
            n_ref,
            jacobi_iters=jacobi_iters,
            do_update=do_update,
        )
        x = pack_modules_inverse_to_19(mod_next, x)
        Pg = mod_next.P_gen
        Pr = mod_next.P_rt
    return x, Pg, Pr


# ============================================================================
# 8. Network-level forward and inverse simulations
# ============================================================================

def te_from_pe(baseMVA, p_obj, pe_code, torque_scale=1.0):
    return -torque_scale * (baseMVA * 1e6 / p_obj.omega_rated) * pe_code


def initialize_turbine_state(delta_bus, p: Parameters):
    x_n0 = 0.8
    x0 = np.zeros(19)
    x0[3] = 1.0
    x0[4] = x_n0
    x0[5] = x_n0 / p.k_i
    x0[7] = p.omega_rated
    x0[8] = 1e-5
    x0[9] = 1e-5
    x0[12] = 1e-5
    x0[13] = 1e-5
    x0[6] = 0.0
    x0[18] = delta_bus
    return x0


def simulate_ieee9_forward_truth(K, baseMVA, gen_buses, P_inj_pu, delta_ss, P_gen_MW, dt=0.001, T=20.0, torque_scale=1.0, n_substeps=5, n_ref=0.0, disturbances=None, D_kur=2.0, progress_every=5000, verbose=True):
    Nbus = K.shape[0]
    gen_buses = list(gen_buses)
    kuramoto_buses = [b for b in range(3, 9) if b not in gen_buses]
    steps = int(T / dt)
    t_arr = np.linspace(0.0, T, steps + 1)

    rho_, g_, H_, eta_ = 1000.0, 9.81, 595.0, 0.92
    Q_list = [P_gen_MW[gi] * 1e6 / (rho_ * g_ * H_ * eta_) for gi in range(len(gen_buses))]
    p_list = [Parameters(Q_rated=Q_list[gi]) for gi in range(len(gen_buses))]

    P_inj = P_inj_pu.copy()
    disturbances = sorted(disturbances or [], key=lambda d: d["time"])
    applied = [False] * len(disturbances)
    D_vec = np.full(Nbus, D_kur, dtype=float)

    delta = np.zeros((steps + 1, Nbus))
    omega_pu = np.zeros((steps + 1, Nbus))
    x_turb = np.zeros((len(gen_buses), 19, steps + 1))
    delta[0] = delta_ss.copy()

    for gi, bus in enumerate(gen_buses):
        x_turb[gi, :, 0] = initialize_turbine_state(delta_ss[bus], p_list[gi])

    Pe_hist = np.zeros((steps + 1, Nbus))
    Pe_hist[0] = compute_Pe_all(delta[0], K)

    t_wall = time.perf_counter()
    if verbose:
        print(f"[forward] steps={steps}, dt={dt}, T={T}, n_gen={len(gen_buses)}, n_substeps={n_substeps}")

    for k in range(steps):
        _progress_report(k, steps, t_wall, label="forward", every=progress_every)
        tk = t_arr[k]
        for di, dist in enumerate(disturbances):
            if not applied[di] and tk >= dist["time"]:
                P_inj[dist["bus"]] -= dist["MW"] / baseMVA
                applied[di] = True

        Pe_k = Pe_hist[k]
        x_pred_list = []
        for gi, bus in enumerate(gen_buses):
            p = p_list[gi]
            Te = te_from_pe(baseMVA, p, Pe_k[bus], torque_scale=torque_scale)
            x_pred = decomposed_turbine_step_19(
                x19=x_turb[gi, :, k],
                t=tk,
                dt=dt,
                p=p,
                Te_grid=Te,
                n_ref=n_ref,
                n_substeps=1,
                jacobi_iters=2,
            )
            x_pred_list.append(x_pred)

        delta_pred = delta[k].copy()
        for gi, bus in enumerate(gen_buses):
            delta_pred[bus] = x_pred_list[gi][18]

        Pe_for_kur = compute_Pe_all(delta_pred, K)
        for b in kuramoto_buses:
            delta_pred[b] = delta[k, b] + dt * (P_inj[b] + Pe_for_kur[b]) / D_vec[b]

        Pe_pred = compute_Pe_all(delta_pred, K)

        for gi, bus in enumerate(gen_buses):
            p = p_list[gi]
            Te_avg = 0.5 * (
                te_from_pe(baseMVA, p, Pe_k[bus], torque_scale=torque_scale)
                + te_from_pe(baseMVA, p, Pe_pred[bus], torque_scale=torque_scale)
            )
            x_next = decomposed_turbine_step_19(
                x19=x_turb[gi, :, k],
                t=tk,
                dt=dt,
                p=p,
                Te_grid=Te_avg,
                n_ref=n_ref,
                n_substeps=n_substeps,
                jacobi_iters=2,
            )
            x_turb[gi, :, k + 1] = x_next
            delta[k + 1, bus] = x_next[18]
            omega_pu[k + 1, bus] = (x_next[7] - p.omega_rated) / p.omega_rated

        for b in kuramoto_buses:
            f0 = (P_inj[b] + Pe_k[b]) / D_vec[b]
            f1 = (P_inj[b] + Pe_pred[b]) / D_vec[b]
            delta[k + 1, b] = delta[k, b] + 0.5 * dt * (f0 + f1)

        Pe_hist[k + 1] = compute_Pe_all(delta[k + 1], K)

    return t_arr, delta, omega_pu, Pe_hist, x_turb, p_list


def generate_sensor_measurements(x_turb, noise_std, rng=None):
    rng = np.random.default_rng(1) if rng is None else rng
    ngen = x_turb.shape[0]
    steps = x_turb.shape[2]
    sensors = [[None] * steps for _ in range(ngen)]
    for gi in range(ngen):
        for k in range(steps):
            sensors[gi][k] = SensorSample(
                xG=float(x_turb[gi, 8, k] + noise_std["xG"] * rng.standard_normal()),
                yG=float(x_turb[gi, 9, k] + noise_std["yG"] * rng.standard_normal()),
                xR=float(x_turb[gi, 12, k] + noise_std["xR"] * rng.standard_normal()),
                yR=float(x_turb[gi, 13, k] + noise_std["yR"] * rng.standard_normal()),
                q=float(x_turb[gi, 3, k] + noise_std["q"] * rng.standard_normal()),
                omega=float(x_turb[gi, 7, k] + noise_std["omega"] * rng.standard_normal()),
            )
    return sensors


def simulate_ieee9_inverse_with_kf(K, baseMVA, gen_buses, P_inj_pu, delta_ss, P_gen_MW, sensors, dt=0.001, T=20.0, torque_scale=1.0, n_substeps=1, n_ref=0.0, disturbances=None, D_kur=2.0, kf_cfg=None, progress_every=5000, verbose=True):
    kf_cfg = KFConfig() if kf_cfg is None else kf_cfg
    Nbus = K.shape[0]
    gen_buses = list(gen_buses)
    kuramoto_buses = [b for b in range(3, 9) if b not in gen_buses]
    steps = int(T / dt)
    t_arr = np.linspace(0.0, T, steps + 1)

    rho_, g_, H_, eta_ = 1000.0, 9.81, 595.0, 0.92
    Q_list = [P_gen_MW[gi] * 1e6 / (rho_ * g_ * H_ * eta_) for gi in range(len(gen_buses))]
    p_list = [Parameters(Q_rated=Q_list[gi]) for gi in range(len(gen_buses))]

    P_inj = P_inj_pu.copy()
    disturbances = sorted(disturbances or [], key=lambda d: d["time"])
    applied = [False] * len(disturbances)
    D_vec = np.full(Nbus, D_kur, dtype=float)

    delta = np.zeros((steps + 1, Nbus))
    omega_pu = np.zeros((steps + 1, Nbus))
    x_est = np.zeros((len(gen_buses), 19, steps + 1))
    P_gen_list = []
    P_rt_list = []
    delta[0] = delta_ss.copy()

    for gi, bus in enumerate(gen_buses):
        x_est[gi, :, 0] = initialize_turbine_state(delta_ss[bus], p_list[gi])
        mod0 = unpack_19_to_modules_inverse(x_est[gi, :, 0], kf_cfg)
        P_gen_list.append(mod0.P_gen.copy())
        P_rt_list.append(mod0.P_rt.copy())

    Pe_hist = np.zeros((steps + 1, Nbus))
    Pe_hist[0] = compute_Pe_all(delta[0], K)

    t_wall = time.perf_counter()
    if verbose:
        print(f"[inverse] steps={steps}, dt={dt}, T={T}, n_gen={len(gen_buses)}, n_substeps={n_substeps}, jacobi_iters={getattr(kf_cfg, '_jacobi_iters', '?')}")

    for k in range(steps):
        _progress_report(k, steps, t_wall, label="inverse", every=progress_every)
        tk = t_arr[k]
        for di, dist in enumerate(disturbances):
            if not applied[di] and tk >= dist["time"]:
                P_inj[dist["bus"]] -= dist["MW"] / baseMVA
                applied[di] = True

        Pe_k = Pe_hist[k]
        x_pred_list = []
        for gi, bus in enumerate(gen_buses):
            p = p_list[gi]
            Te = te_from_pe(baseMVA, p, Pe_k[bus], torque_scale=torque_scale)
            sensor = sensors[gi][k]
            x_pred, Pg_new, Pr_new = inverse_turbine_step_19(
                x19=x_est[gi, :, k],
                P_gen=P_gen_list[gi],
                P_rt=P_rt_list[gi],
                t=tk,
                dt=dt,
                p=p,
                Te_grid=Te,
                sensor=sensor,
                kf_cfg=kf_cfg,
                n_ref=n_ref,
                n_substeps=1,
                jacobi_iters=2,
                do_update=False,
            )
            x_pred_list.append((x_pred, Pg_new, Pr_new))

        delta_pred = delta[k].copy()
        for gi, bus in enumerate(gen_buses):
            delta_pred[bus] = x_pred_list[gi][0][18]

        Pe_for_kur = compute_Pe_all(delta_pred, K)
        for b in kuramoto_buses:
            delta_pred[b] = delta[k, b] + dt * (P_inj[b] + Pe_for_kur[b]) / D_vec[b]

        Pe_pred = compute_Pe_all(delta_pred, K)

        for gi, bus in enumerate(gen_buses):
            p = p_list[gi]
            Te_avg = 0.5 * (
                te_from_pe(baseMVA, p, Pe_k[bus], torque_scale=torque_scale)
                + te_from_pe(baseMVA, p, Pe_pred[bus], torque_scale=torque_scale)
            )
            sensor = sensors[gi][min(k + 1, len(sensors[gi]) - 1)]
            x_next, Pg_new, Pr_new = inverse_turbine_step_19(
                x19=x_est[gi, :, k],
                P_gen=P_gen_list[gi],
                P_rt=P_rt_list[gi],
                t=tk,
                dt=dt,
                p=p,
                Te_grid=Te_avg,
                sensor=sensor,
                kf_cfg=kf_cfg,
                n_ref=n_ref,
                n_substeps=n_substeps,
                jacobi_iters=2,
                do_update=True,
            )
            x_est[gi, :, k + 1] = x_next
            P_gen_list[gi] = Pg_new
            P_rt_list[gi] = Pr_new
            delta[k + 1, bus] = x_next[18]
            omega_pu[k + 1, bus] = (x_next[7] - p.omega_rated) / p.omega_rated

        for b in kuramoto_buses:
            f0 = (P_inj[b] + Pe_k[b]) / D_vec[b]
            f1 = (P_inj[b] + Pe_pred[b]) / D_vec[b]
            delta[k + 1, b] = delta[k, b] + 0.5 * dt * (f0 + f1)

        Pe_hist[k + 1] = compute_Pe_all(delta[k + 1], K)

    return t_arr, delta, omega_pu, Pe_hist, x_est, p_list

#%%
# ============================================================================
# 9. Example run
# ============================================================================

if __name__ == "__main__":
    K, neighbors, baseMVA, P_gen_MW, gen_buses, load_buses, P_inj_pu, delta_ss = build_ieee9_network()

    disturbances = [
        {"bus": load_buses[2], "MW": +20.0, "time": 5.0, "label": "+20 MW demand increase"},
        {"bus": load_buses[2], "MW": -20.0, "time": 30.0, "label": "-20 MW demand decrease"},
    ]

    dt = 0.001
    T = 60.0

    t, Delta_true, W_true, Pe_true, x_true, p_list = simulate_ieee9_forward_truth(
        K=K,
        baseMVA=baseMVA,
        gen_buses=gen_buses,
        P_inj_pu=P_inj_pu,
        delta_ss=delta_ss,
        P_gen_MW=P_gen_MW,
        dt=dt,
        T=T,
        n_substeps=2,
        disturbances=disturbances,
        D_kur=2.0,
    )

    noise_std = {
        "xG": 2e-6,
        "yG": 2e-6,
        "xR": 2e-6,
        "yR": 2e-6,
        "q": 2e-4,
        "omega": 2e-4,
    }
    sensors = generate_sensor_measurements(x_true, noise_std)

    kf_cfg = KFConfig(
        q_gen=1e-11,
        q_run=2e-12,
        q_torsion=1e-8,
        r_xy_gen=noise_std["xG"] ** 2,
        r_xy_run=100.0 * noise_std["xR"] ** 2,
        p0_gen=1e-7,
        p0_run=1e-7,
        p0_torsion=1e-5,
    )

    t_inv, Delta_est, W_est, Pe_est, x_est, _ = simulate_ieee9_inverse_with_kf(
        K=K,
        baseMVA=baseMVA,
        gen_buses=gen_buses,
        P_inj_pu=P_inj_pu,
        delta_ss=delta_ss,
        P_gen_MW=P_gen_MW,
        sensors=sensors,
        dt=dt,
        T=T,
        n_substeps=2,
        disturbances=disturbances,
        D_kur=2.0,
        kf_cfg=kf_cfg,
    )

    # ------------------------------------------------------------------
    # Build generator and runner state-estimation table at 5 Hz.
    # The table is downsampled from dt=0.001 s to 0.2 s spacing.
    # Variables: positions xG,yG,xR,yR | velocities vxG,vyG,vxR,vyR |
    # torsional alpha, beta, plus smoothed alpha-DC and beta-smooth that
    # match the plots above.
    # ------------------------------------------------------------------
    try:
        import pandas as _pd_csv

        target_fs = 5.0  # Hz
        stride = max(1, int(round(1.0 / (target_fs * dt))))
        idx_ds = np.arange(0, len(t), stride)

        _alpha_dc_window_csv = 500
        _beta_smooth_window_csv = 200

        df_state = _pd_csv.DataFrame({"t_s": t[idx_ds]})
        var_specs_csv = [
            ("xG",    8),
            ("yG",    9),
            ("vxG",  10),
            ("vyG",  11),
            ("xR",   12),
            ("yR",   13),
            ("vxR",  14),
            ("vyR",  15),
            ("alpha", 16),
            ("beta",  17),
        ]
        for gi, bus in enumerate(gen_buses):
            for name, idx_var in var_specs_csv:
                df_state[f"{name}_true_bus{bus+1}"] = x_true[gi, idx_var, idx_ds]
                df_state[f"{name}_est_bus{bus+1}"]  = x_est[gi, idx_var, idx_ds]
            # Smoothed series used in fig_torsion_dc and fig_beta plots
            a_true_full = moving_average(x_true[gi, 16, :], window=_alpha_dc_window_csv)
            a_est_full  = moving_average(x_est[gi, 16, :], window=_alpha_dc_window_csv)
            b_true_full = moving_average(x_true[gi, 17, :], window=_beta_smooth_window_csv)
            b_est_full  = moving_average(x_est[gi, 17, :], window=_beta_smooth_window_csv)
            df_state[f"alpha_dc_true_bus{bus+1}"] = a_true_full[idx_ds]
            df_state[f"alpha_dc_est_bus{bus+1}"]  = a_est_full[idx_ds]
            df_state[f"beta_smooth_true_bus{bus+1}"] = b_true_full[idx_ds]
            df_state[f"beta_smooth_est_bus{bus+1}"]  = b_est_full[idx_ds]

        state_estimation_runner_generator_5hz = df_state
    except Exception as _e_csv:
        print(f"WARNING: could not prepare state-estimation table: {_e_csv}")

    event_times = [dist["time"] for dist in disturbances]

    def add_event_lines(ax):
        for t_ev in event_times:
            ax.axvline(t_ev, color="0.5", ls="--", lw=1.0, alpha=0.7)

    def ensure_2d_axes(axs):
        axs = np.asarray(axs, dtype=object)
        if axs.ndim == 1:
            axs = axs[np.newaxis, :]
        return axs

    disp_specs = [
        ("xG", 8, "Generator x displacement"),
        ("yG", 9, "Generator y displacement"),
        ("xR", 12, "Runner x displacement"),
        ("yR", 13, "Runner y displacement"),
    ]
    vel_specs = [
        ("vxG", 10, "Generator x velocity"),
        ("vyG", 11, "Generator y velocity"),
        ("vxR", 14, "Runner x velocity"),
        ("vyR", 15, "Runner y velocity"),
    ]

    fig_disp, axs_disp = plt.subplots(len(gen_buses), 4, figsize=(18, 3.8 * len(gen_buses)), sharex=True)
    axs_disp = ensure_2d_axes(axs_disp)
    for gi, bus in enumerate(gen_buses):
        for col, (label, idx, title) in enumerate(disp_specs):
            ax = axs_disp[gi, col]
            ax.plot(t, x_true[gi, idx, :], lw=1.0, label=f"true {label}")
            ax.plot(t_inv, x_est[gi, idx, :], "--", lw=1.0, label=f"est {label}")
            if label in {"xR", "yR"}:
                ax.plot(
                    t,
                    moving_average(x_true[gi, idx, :], window=200),
                    color="k",
                    lw=1.2,
                    alpha=0.85,
                    label=f"{label} center",
                )
            if col == 0:
                ax.set_ylabel(f"bus {bus+1}")
            if gi == 0:
                ax.set_title(title)
            ax.grid(True, alpha=0.3)

    axs_disp[0, 0].legend(fontsize=8)
    for ax in axs_disp[-1, :]:
        ax.set_xlabel("t [s]")
    fig_disp.suptitle("Displacement estimates", y=1.02)
    fig_disp.tight_layout()

    fig_vel, axs_vel = plt.subplots(len(gen_buses), 4, figsize=(18, 3.8 * len(gen_buses)), sharex=True)
    axs_vel = ensure_2d_axes(axs_vel)
    for gi, bus in enumerate(gen_buses):
        for col, (label, idx, title) in enumerate(vel_specs):
            ax = axs_vel[gi, col]
            ax.plot(t, x_true[gi, idx, :], lw=1.0, label=f"true {label}")
            ax.plot(t_inv, x_est[gi, idx, :], "--", lw=1.0, label=f"est {label}")
            if label in {"vxR", "vyR"}:
                ax.plot(
                    t_inv,
                    moving_average(x_est[gi, idx, :], window=25),
                    color="k",
                    lw=1.0,
                    alpha=0.8,
                    label=f"smooth {label}",
                )
            if col == 0:
                ax.set_ylabel(f"bus {bus+1}")
            if gi == 0:
                ax.set_title(title)
            ax.grid(True, alpha=0.3)

    axs_vel[0, 0].legend(fontsize=8)
    for ax in axs_vel[-1, :]:
        ax.set_xlabel("t [s]")
    fig_vel.suptitle("Velocity estimates", y=1.02)
    fig_vel.tight_layout()

    fig_torsion, axs_torsion = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs_torsion = np.asarray(axs_torsion, dtype=object)

    ax = axs_torsion[0]
    for gi, bus in enumerate(gen_buses):
        ax.plot(t, x_true[gi, 16, :], lw=1.2, label=f"true bus {bus+1}")
        ax.plot(t_inv, x_est[gi, 16, :], "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("rad")
    ax.set_title("Shaft torsional twist alpha")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    ax.legend(fontsize=8, ncol=2)

    ax = axs_torsion[1]
    for gi, bus in enumerate(gen_buses):
        ax.plot(t, 1e6 * x_true[gi, 16, :], lw=1.2, label=f"true bus {bus+1}")
        ax.plot(t_inv, 1e6 * x_est[gi, 16, :], "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("urad")
    ax.set_xlabel("t [s]")
    ax.set_title("Shaft torsional twist alpha (micro-radians)")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    fig_torsion.suptitle("Torsional twist estimates", y=1.02)
    fig_torsion.tight_layout()

    alpha_dc_window = 500
    fig_torsion_dc, axs_torsion_dc = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs_torsion_dc = np.asarray(axs_torsion_dc, dtype=object)

    ax = axs_torsion_dc[0]
    for gi, bus in enumerate(gen_buses):
        alpha_dc_true = moving_average(x_true[gi, 16, :], window=alpha_dc_window)
        alpha_dc_est = moving_average(x_est[gi, 16, :], window=alpha_dc_window)
        ax.plot(t, alpha_dc_true, lw=1.2, label=f"true bus {bus+1}")
        ax.plot(t_inv, alpha_dc_est, "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("rad")
    ax.set_title(f"Shaft torsional twist alpha DC component ({alpha_dc_window}-sample moving average)")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    ax.legend(fontsize=8, ncol=2)

    ax = axs_torsion_dc[1]
    for gi, bus in enumerate(gen_buses):
        alpha_dc_true = moving_average(x_true[gi, 16, :], window=alpha_dc_window)
        alpha_dc_est = moving_average(x_est[gi, 16, :], window=alpha_dc_window)
        ax.plot(t, 1e6 * alpha_dc_true, lw=1.2, label=f"true bus {bus+1}")
        ax.plot(t_inv, 1e6 * alpha_dc_est, "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("urad")
    ax.set_xlabel("t [s]")
    ax.set_title("Shaft torsional twist alpha DC component (micro-radians)")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    fig_torsion_dc.suptitle("Torsional twist DC estimates", y=1.02)
    fig_torsion_dc.tight_layout()

    beta_smooth_window = 200
    fig_beta, axs_beta = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs_beta = np.asarray(axs_beta, dtype=object)

    ax = axs_beta[0]
    for gi, bus in enumerate(gen_buses):
        ax.plot(t, x_true[gi, 17, :], lw=1.1, label=f"true bus {bus+1}")
        ax.plot(t_inv, x_est[gi, 17, :], "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("rad/s")
    ax.set_title("Shaft torsional rate beta = alpha dot")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    ax.legend(fontsize=8, ncol=2)

    ax = axs_beta[1]
    for gi, bus in enumerate(gen_buses):
        beta_smooth_true = moving_average(x_true[gi, 17, :], window=beta_smooth_window)
        beta_smooth_est = moving_average(x_est[gi, 17, :], window=beta_smooth_window)
        ax.plot(t, beta_smooth_true, lw=1.1, label=f"true bus {bus+1}")
        ax.plot(t_inv, beta_smooth_est, "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("rad/s")
    ax.set_xlabel("t [s]")
    ax.set_title(f"Shaft torsional rate beta (smoothed, {beta_smooth_window}-sample moving average)")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    fig_beta.suptitle("Torsional rate estimates", y=1.02)
    fig_beta.tight_layout()

    delta_coi_true = sum((P_gen_MW[gi] / P_gen_MW.sum()) * Delta_true[:, bus] for gi, bus in enumerate(gen_buses))
    delta_coi_est = sum((P_gen_MW[gi] / P_gen_MW.sum()) * Delta_est[:, bus] for gi, bus in enumerate(gen_buses))
    Delta_rel_true = np.column_stack([Delta_true[:, b] - delta_coi_true for b in range(K.shape[0])])
    Delta_rel_est = np.column_stack([Delta_est[:, b] - delta_coi_est for b in range(K.shape[0])])

    fig_resp, axs_resp = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axs_resp = np.asarray(axs_resp, dtype=object)

    ax = axs_resp[0, 0]
    for gi, bus in enumerate(gen_buses):
        ax.plot(t, np.degrees(Delta_rel_true[:, bus]), lw=1.2, label=f"true bus {bus+1}")
        ax.plot(t_inv, np.degrees(Delta_rel_est[:, bus]), "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("deg")
    ax.set_title("COI-relative rotor angle")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)

    ax = axs_resp[0, 1]
    for gi, bus in enumerate(gen_buses):
        ax.plot(t, 100.0 * W_true[:, bus], lw=1.2, label=f"true bus {bus+1}")
        ax.plot(t_inv, 100.0 * W_est[:, bus], "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("%")
    ax.set_title("Generator speed deviation")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)

    ax = axs_resp[1, 0]
    for gi, bus in enumerate(gen_buses):
        ax.plot(t, -baseMVA * Pe_true[:, bus], lw=1.2, label=f"true bus {bus+1}")
        ax.plot(t_inv, -baseMVA * Pe_est[:, bus], "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("MW")
    ax.set_title("Electrical power")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)

    ax = axs_resp[1, 1]
    for gi, bus in enumerate(gen_buses):
        ax.plot(t, x_true[gi, 3, :], lw=1.2, label=f"true bus {bus+1}")
        ax.plot(t_inv, x_est[gi, 3, :], "--", lw=1.0, label=f"est bus {bus+1}")
    ax.set_ylabel("pu")
    ax.set_title("Flow rate q")
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)

    axs_resp[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs_resp[-1, :]:
        ax.set_xlabel("t [s]")
    fig_resp.suptitle("Disturbance-sensitive states", y=1.02)
    fig_resp.tight_layout()

    def compute_force_histories(x_hist, Delta_hist, p_list_local, t_hist):
        ngen = x_hist.shape[0]
        ns = x_hist.shape[2]
        diag = {
            "gen_unb": np.zeros((ngen, ns)),
            "gen_total": np.zeros((ngen, ns)),
            "run_hyd": np.zeros((ngen, ns)),
            "run_blade": np.zeros((ngen, ns)),
            "run_torque": np.zeros((ngen, ns)),
            "run_dc": np.zeros((ngen, ns)),
            "run_total": np.zeros((ngen, ns)),
        }
        for gi in range(ngen):
            p = p_list_local[gi]
            for k in range(ns):
                omega = x_hist[gi, 7, k]
                q = x_hist[gi, 3, k]
                h = x_hist[gi, 2, k]
                delta_state = x_hist[gi, 18, k]
                phi1 = delta_state + p.omega_rated * t_hist[k]
                xG = x_hist[gi, 8:12, k]
                xR = x_hist[gi, 12:18, k]
                M_t = torque_from_rot_state(
                    np.array([x_hist[gi, 6, k], omega, delta_state, 1.0, 1.0, 1.0], dtype=float),
                    p,
                    q,
                    h,
                )
                gen_comp = generator_force_components(xG, p, omega, phi1, x_hist[gi, 12, k], x_hist[gi, 13, k])
                run_comp = runner_force_components(
                    xR,
                    p,
                    omega,
                    phi1,
                    q,
                    h,
                    M_t,
                    x_hist[gi, 8, k],
                    x_hist[gi, 9, k],
                    t_hist[k],
                )
                diag["gen_unb"][gi, k] = gen_comp["unb_mag"]
                diag["gen_total"][gi, k] = gen_comp["total_mag"]
                diag["run_hyd"][gi, k] = run_comp["hyd_mag"]
                diag["run_blade"][gi, k] = run_comp["blade_mag"]
                diag["run_torque"][gi, k] = run_comp["torque_mag"]
                diag["run_dc"][gi, k] = run_comp["dc_mag"]
                diag["run_total"][gi, k] = run_comp["total_mag"]
        return diag

    force_true = compute_force_histories(x_true, Delta_true, p_list, t)
    force_est = compute_force_histories(x_est, Delta_est, p_list, t_inv)

    force_specs = [
        ("gen_unb", "Generator unbalance |F_unb,G| [N]"),
        ("gen_total", "Generator total lateral |F_G| [N]"),
        ("run_hyd", "Runner hydraulic/vortex |F_hyd| [N]"),
        ("run_blade", "Runner blade-passing |F_bld| [N]"),
        ("run_torque", "Runner torque-reaction |F_tq| [N]"),
        ("run_dc", "Runner DC load force |F_dc| [N]"),
        ("run_total", "Runner total lateral |F_R| [N]"),
    ]

    fig_force, axs_force = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axs_force = np.asarray(axs_force, dtype=object)
    for ax, (key, title) in zip(axs_force.flat, force_specs):
        for gi, bus in enumerate(gen_buses):
            ax.plot(t, force_true[key][gi], lw=1.1, label=f"true bus {bus+1}")
            ax.plot(t_inv, force_est[key][gi], "--", lw=0.9, label=f"est bus {bus+1}")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)
    axs_force[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs_force[-1, :]:
        ax.set_xlabel("t [s]")
    axs_force[-1, -1].axis("off")
    fig_force.suptitle("Lateral forcing diagnostics", y=1.02)
    fig_force.tight_layout()

    zoom_windows = [
        (4.7, 5.3, "Zoom around demand increase"),
        (29.7, 30.3, "Zoom around demand decrease"),
    ]

    fig_disp_zoom, axs_disp_zoom = plt.subplots(2, 4, figsize=(18, 8), sharex=False)
    axs_disp_zoom = np.asarray(axs_disp_zoom, dtype=object)
    for row, (t0, t1, row_title) in enumerate(zoom_windows):
        for col, (label, idx, title) in enumerate(disp_specs):
            ax = axs_disp_zoom[row, col]
            for gi, bus in enumerate(gen_buses):
                ax.plot(t, x_true[gi, idx, :], lw=1.0, label=f"true bus {bus+1}")
                ax.plot(t_inv, x_est[gi, idx, :], "--", lw=1.0, label=f"est bus {bus+1}")
                if label in {"xR", "yR"}:
                    ax.plot(
                        t,
                        moving_average(x_true[gi, idx, :], window=200),
                        color="k",
                        lw=1.1,
                        alpha=0.85,
                        label=f"{label} center" if gi == 0 and row == 0 else None,
                    )
            ax.set_xlim(t0, t1)
            add_event_lines(ax)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(row_title)
    axs_disp_zoom[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs_disp_zoom[-1, :]:
        ax.set_xlabel("t [s]")
    fig_disp_zoom.suptitle("Displacement estimates: event zooms", y=1.02)
    fig_disp_zoom.tight_layout()

    fig_vel_zoom, axs_vel_zoom = plt.subplots(2, 4, figsize=(18, 8), sharex=False)
    axs_vel_zoom = np.asarray(axs_vel_zoom, dtype=object)
    for row, (t0, t1, row_title) in enumerate(zoom_windows):
        for col, (label, idx, title) in enumerate(vel_specs):
            ax = axs_vel_zoom[row, col]
            for gi, bus in enumerate(gen_buses):
                ax.plot(t, x_true[gi, idx, :], lw=1.0, label=f"true bus {bus+1}")
                ax.plot(t_inv, x_est[gi, idx, :], "--", lw=1.0, label=f"est bus {bus+1}")
                if label in {"vxR", "vyR"}:
                    ax.plot(
                        t_inv,
                        moving_average(x_est[gi, idx, :], window=25),
                        color="k",
                        lw=0.9,
                        alpha=0.6,
                    )
            ax.set_xlim(t0, t1)
            add_event_lines(ax)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(row_title)
    axs_vel_zoom[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs_vel_zoom[-1, :]:
        ax.set_xlabel("t [s]")
    fig_vel_zoom.suptitle("Velocity estimates: event zooms", y=1.02)
    fig_vel_zoom.tight_layout()

    fig_torsion_zoom, axs_torsion_zoom = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axs_torsion_zoom = np.asarray(axs_torsion_zoom, dtype=object)
    for row, (t0, t1, row_title) in enumerate(zoom_windows):
        ax = axs_torsion_zoom[row, 0]
        for gi, bus in enumerate(gen_buses):
            ax.plot(t, x_true[gi, 16, :], lw=1.1, label=f"true bus {bus+1}")
            ax.plot(t_inv, x_est[gi, 16, :], "--", lw=1.0, label=f"est bus {bus+1}")
        ax.set_xlim(t0, t1)
        ax.set_ylabel(f"{row_title}\nrad")
        ax.set_title("Shaft torsional twist alpha")
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)

        ax = axs_torsion_zoom[row, 1]
        for gi, bus in enumerate(gen_buses):
            ax.plot(t, 1e6 * x_true[gi, 16, :], lw=1.1, label=f"true bus {bus+1}")
            ax.plot(t_inv, 1e6 * x_est[gi, 16, :], "--", lw=1.0, label=f"est bus {bus+1}")
        ax.set_xlim(t0, t1)
        ax.set_title("Shaft torsional twist alpha (micro-radians)")
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)
    axs_torsion_zoom[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs_torsion_zoom[-1, :]:
        ax.set_xlabel("t [s]")
    fig_torsion_zoom.suptitle("Torsional twist estimates: event zooms", y=1.02)
    fig_torsion_zoom.tight_layout()

    fig_torsion_dc_zoom, axs_torsion_dc_zoom = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axs_torsion_dc_zoom = np.asarray(axs_torsion_dc_zoom, dtype=object)
    for row, (t0, t1, row_title) in enumerate(zoom_windows):
        ax = axs_torsion_dc_zoom[row, 0]
        for gi, bus in enumerate(gen_buses):
            alpha_dc_true = moving_average(x_true[gi, 16, :], window=alpha_dc_window)
            alpha_dc_est = moving_average(x_est[gi, 16, :], window=alpha_dc_window)
            ax.plot(t, alpha_dc_true, lw=1.1, label=f"true bus {bus+1}")
            ax.plot(t_inv, alpha_dc_est, "--", lw=1.0, label=f"est bus {bus+1}")
        ax.set_xlim(t0, t1)
        ax.set_ylabel(f"{row_title}\nrad")
        ax.set_title("Alpha DC component")
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)

        ax = axs_torsion_dc_zoom[row, 1]
        for gi, bus in enumerate(gen_buses):
            alpha_dc_true = moving_average(x_true[gi, 16, :], window=alpha_dc_window)
            alpha_dc_est = moving_average(x_est[gi, 16, :], window=alpha_dc_window)
            ax.plot(t, 1e6 * alpha_dc_true, lw=1.1, label=f"true bus {bus+1}")
            ax.plot(t_inv, 1e6 * alpha_dc_est, "--", lw=1.0, label=f"est bus {bus+1}")
        ax.set_xlim(t0, t1)
        ax.set_title("Alpha DC component (micro-radians)")
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)
    axs_torsion_dc_zoom[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs_torsion_dc_zoom[-1, :]:
        ax.set_xlabel("t [s]")
    fig_torsion_dc_zoom.suptitle("Torsional twist DC estimates: event zooms", y=1.02)
    fig_torsion_dc_zoom.tight_layout()

    fig_beta_zoom, axs_beta_zoom = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axs_beta_zoom = np.asarray(axs_beta_zoom, dtype=object)
    for row, (t0, t1, row_title) in enumerate(zoom_windows):
        ax = axs_beta_zoom[row, 0]
        for gi, bus in enumerate(gen_buses):
            ax.plot(t, x_true[gi, 17, :], lw=1.1, label=f"true bus {bus+1}")
            ax.plot(t_inv, x_est[gi, 17, :], "--", lw=1.0, label=f"est bus {bus+1}")
        ax.set_xlim(t0, t1)
        ax.set_ylabel(f"{row_title}\nrad/s")
        ax.set_title("Shaft torsional rate beta")
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)

        ax = axs_beta_zoom[row, 1]
        for gi, bus in enumerate(gen_buses):
            beta_smooth_true = moving_average(x_true[gi, 17, :], window=beta_smooth_window)
            beta_smooth_est = moving_average(x_est[gi, 17, :], window=beta_smooth_window)
            ax.plot(t, beta_smooth_true, lw=1.1, label=f"true bus {bus+1}")
            ax.plot(t_inv, beta_smooth_est, "--", lw=1.0, label=f"est bus {bus+1}")
        ax.set_xlim(t0, t1)
        ax.set_title(f"Beta smoothed ({beta_smooth_window} samples)")
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)
    axs_beta_zoom[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs_beta_zoom[-1, :]:
        ax.set_xlabel("t [s]")
    fig_beta_zoom.suptitle("Torsional rate estimates: event zooms", y=1.02)
    fig_beta_zoom.tight_layout()

    fig_force_zoom, axs_force_zoom = plt.subplots(2, 4, figsize=(18, 8), sharex=False)
    axs_force_zoom = np.asarray(axs_force_zoom, dtype=object)
    zoom_force_specs = [
        ("gen_unb", "Generator unbalance"),
        ("run_hyd", "Runner hydraulic/vortex"),
        ("run_torque", "Runner torque reaction"),
        ("run_dc", "Runner DC load force"),
    ]
    for row, (t0, t1, row_title) in enumerate(zoom_windows):
        for col, (key, title) in enumerate(zoom_force_specs):
            ax = axs_force_zoom[row, col]
            for gi, bus in enumerate(gen_buses):
                ax.plot(t, force_true[key][gi], lw=1.0, label=f"true bus {bus+1}")
                ax.plot(t_inv, force_est[key][gi], "--", lw=0.9, label=f"est bus {bus+1}")
            ax.set_xlim(t0, t1)
            add_event_lines(ax)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(row_title)
    axs_force_zoom[0, 0].legend(fontsize=8, ncol=2)
    for ax in axs_force_zoom[-1, :]:
        ax.set_xlabel("t [s]")
    fig_force_zoom.suptitle("Lateral forcing diagnostics: event zooms", y=1.02)
    fig_force_zoom.tight_layout()
#%%
# ============================================================================
# 10. Appendix figures (top-level cell; reuses variables from the main run)
#     A1 - cross-scale disturbance propagation
#     A2 - hidden-state recovery (unmeasured states) across the 3 turbines
#     A3 - embedded estimate vs standalone-reference forward simulation
# ============================================================================

import matplotlib as mpl

# Colorblind-safe plotting palette.
CB_BLUE      = (86 / 255, 180 / 255, 233 / 255)
CB_ORANGE    = (230 / 255, 159 / 255,   0 / 255)
CB_GREEN     = (  0 / 255, 158 / 255, 115 / 255)
CB_PURPLE    = (204 / 255, 121 / 255, 167 / 255)
CB_REDORANGE = (213 / 255,  94 / 255,   0 / 255)
CB_YELLOW    = (240 / 255, 228 / 255,  66 / 255)

mpl.rcParams.update({
    "axes.prop_cycle": mpl.cycler(color=[CB_BLUE, CB_ORANGE, CB_GREEN, CB_PURPLE, CB_REDORANGE, CB_YELLOW]),
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "lines.linewidth": 1.1,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def simulate_standalone_turbine(Te_grid_traj, x0, t_arr, p_obj, dt_local, n_substeps=2, n_ref_local=0.0):
    steps = len(t_arr) - 1
    x_hist = np.zeros((19, steps + 1))
    x_hist[:, 0] = x0
    for k in range(steps):
        Te_avg = 0.5 * (Te_grid_traj[k] + Te_grid_traj[k + 1])
        x_hist[:, k + 1] = decomposed_turbine_step_19(
            x19=x_hist[:, k],
            t=t_arr[k],
            dt=dt_local,
            p=p_obj,
            Te_grid=Te_avg,
            n_ref=n_ref_local,
            n_substeps=n_substeps,
            jacobi_iters=2,
        )
    return x_hist

Te_grid_true = np.zeros((len(gen_buses), len(t)))
for gi, bus in enumerate(gen_buses):
    for k in range(len(t)):
        Te_grid_true[gi, k] = te_from_pe(baseMVA, p_list[gi], Pe_true[k, bus])

x_standalone = np.zeros_like(x_true)
for gi, bus in enumerate(gen_buses):
    x0_g = initialize_turbine_state(delta_ss[bus], p_list[gi])
    x_standalone[gi] = simulate_standalone_turbine(
        Te_grid_true[gi], x0_g, t, p_list[gi], dt, n_substeps=2
    )

# ------------------------------------------------------------------
# Figure A1: cross-scale disturbance propagation (single column)
# ------------------------------------------------------------------
gi_focus = 0
bus_focus = gen_buses[gi_focus]

P_load_step = np.zeros(len(t))
for dist in disturbances:
    k_on = int(dist["time"] / dt)
    P_load_step[k_on:] += dist["MW"]

# Compact 4-panel version: load step, coupling torque, rotational n, governor x_n.
# (Network rotor angles, hydraulic flow, and structural twist are shown elsewhere
#  -- e.g. flow q is in Fig A3.)
fig_A1, axs_A1 = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

ax = axs_A1[0]
ax.plot(t, P_load_step, color=CB_REDORANGE, lw=1.5)
ax.set_ylabel("dP_load [MW]")
ax.set_title(f"(a) Load-step disturbance at bus {disturbances[0]['bus']+1}")
ax.grid(True, alpha=0.3)
add_event_lines(ax)

ax = axs_A1[1]
for gi, b in enumerate(gen_buses):
    ax.plot(t, 1e-6 * Te_grid_true[gi], lw=1.1, label=f"gen bus {b+1}")
ax.set_ylabel("Te_grid [MN.m]")
ax.set_title("(b) Coupling into embedded turbines: grid-induced electrical torque")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
add_event_lines(ax)

ax = axs_A1[2]
ax.plot(t, x_true[gi_focus, 6, :], lw=1.1, label="true")
ax.plot(t_inv, x_est[gi_focus, 6, :], "--", lw=1.0, label="est")
ax.set_ylabel("n [pu]")
ax.set_title(f"(c) Rotational subsystem (bus {bus_focus+1}): speed deviation n")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
add_event_lines(ax)

ax = axs_A1[3]
ax.plot(t, x_true[gi_focus, 4, :], lw=1.1, label="true")
ax.plot(t_inv, x_est[gi_focus, 4, :], "--", lw=1.0, label="est")
ax.set_ylabel("x_n [-]")
ax.set_xlabel("t [s]")
ax.set_title(f"(d) Governor (bus {bus_focus+1}): needle position x_n")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
add_event_lines(ax)

fig_A1.tight_layout()
# Build plotted series table.
df_A1 = pd.DataFrame({
    't_s': t,
    'dP_load_MW': P_load_step,
})
for gi, b in enumerate(gen_buses):
    df_A1[f'Te_grid_bus{b+1}_MNm'] = 1e-6 * Te_grid_true[gi]

df_A1[f'n_true_bus{bus_focus+1}'] = x_true[gi_focus, 6, :]
df_A1[f'n_est_bus{bus_focus+1}']  = x_est[gi_focus, 6, :]
df_A1[f'x_n_true_bus{bus_focus+1}'] = x_true[gi_focus, 4, :]
df_A1[f'x_n_est_bus{bus_focus+1}']  = x_est[gi_focus, 4, :]

#%%
# ------------------------------------------------------------------
# Figure A2: hidden-state recovery across the 3 embedded turbines
# rows = generators, cols = vxR, vyR, alpha (DC), beta (smoothed)
# ------------------------------------------------------------------
fig_A2, axs_A2 = plt.subplots(len(gen_buses), 4, figsize=(18, 3.5 * len(gen_buses)), sharex=True)
axs_A2 = ensure_2d_axes(axs_A2)
for gi, bus in enumerate(gen_buses):
    ax = axs_A2[gi, 0]
    ax.plot(t, x_true[gi, 14, :], lw=1.0, label="true")
    ax.plot(t_inv, x_est[gi, 14, :], "--", lw=1.0, label="est")
    rmse = float(np.sqrt(np.mean((x_est[gi, 14, :] - x_true[gi, 14, :]) ** 2)))
    ax.text(
        0.02, 0.95, f"RMSE = {rmse:.2e}", transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.7"),
    )
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    if gi == 0:
        ax.set_title("Runner x velocity vxR [m/s]")
    ax.set_ylabel(f"bus {bus+1}")

    ax = axs_A2[gi, 1]
    ax.plot(t, x_true[gi, 15, :], lw=1.0, label="true")
    ax.plot(t_inv, x_est[gi, 15, :], "--", lw=1.0, label="est")
    rmse = float(np.sqrt(np.mean((x_est[gi, 15, :] - x_true[gi, 15, :]) ** 2)))
    ax.text(
        0.02, 0.95, f"RMSE = {rmse:.2e}", transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.7"),
    )
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    if gi == 0:
        ax.set_title("Runner y velocity vyR [m/s]")

    a_true = moving_average(x_true[gi, 16, :], window=alpha_dc_window)
    a_est = moving_average(x_est[gi, 16, :], window=alpha_dc_window)
    ax = axs_A2[gi, 2]
    ax.plot(t, 1e6 * a_true, lw=1.0, label="true (DC)")
    ax.plot(t_inv, 1e6 * a_est, "--", lw=1.0, label="est (DC)")
    rmse = float(np.sqrt(np.mean((a_est - a_true) ** 2)))
    ax.text(
        0.02, 0.95, f"RMSE = {rmse:.2e}", transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.7"),
    )
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    if gi == 0:
        ax.set_title("Torsional twist alpha (DC) [urad]")

    b_true = moving_average(x_true[gi, 17, :], window=beta_smooth_window)
    b_est = moving_average(x_est[gi, 17, :], window=beta_smooth_window)
    ax = axs_A2[gi, 3]
    ax.plot(t, b_true, lw=1.0, label="true")
    ax.plot(t_inv, b_est, "--", lw=1.0, label="est")
    rmse = float(np.sqrt(np.mean((b_est - b_true) ** 2)))
    ax.text(
        0.02, 0.95, f"RMSE = {rmse:.2e}", transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.7"),
    )
    ax.grid(True, alpha=0.3)
    add_event_lines(ax)
    if gi == 0:
        ax.set_title("Torsional rate beta (smoothed) [rad/s]")

axs_A2[0, 0].legend(fontsize=8)
for ax in axs_A2[-1, :]:
    ax.set_xlabel("t [s]")
# fig_A2.suptitle(
#     "Fig A2. Hidden-state recovery (unmeasured states) across the three embedded turbines",
#     y=1.005,
# )
fig_A2.tight_layout()

# ------------------------------------------------------------------
# Figure A3: embedded estimate vs standalone-reference forward sim
# rows = generators, cols = n, x_n, q
# ------------------------------------------------------------------
compare_specs = [
    (6, "Speed deviation n [pu]"),
    (4, "Needle position x_n [-]"),
    (3, "Flow rate q [pu]"),
]

fig_A3, axs_A3 = plt.subplots(len(gen_buses), 3, figsize=(15, 3.5 * len(gen_buses)), sharex=True)
axs_A3 = ensure_2d_axes(axs_A3)
rmse_components = []
for gi, bus in enumerate(gen_buses):
    for col, (idx, title) in enumerate(compare_specs):
        ax = axs_A3[gi, col]
        ref = x_standalone[gi, idx, :]
        emb = x_est[gi, idx, :]
        ax.plot(t, ref, lw=1.1, color=CB_BLUE, label="standalone reference")
        ax.plot(t_inv, emb, "--", lw=1.0, color=CB_ORANGE, label="embedded estimate")
        rmse = float(np.sqrt(np.mean((emb - ref) ** 2)))
        rmse_components.append((emb - ref) ** 2)
        ax.text(
            0.02, 0.95, f"RMSE = {rmse:.2e}", transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.7"),
        )
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)
        if gi == 0:
            ax.set_title(title)
        if col == 0:
            ax.set_ylabel(f"bus {bus+1}")
axs_A3[0, 0].legend(fontsize=8)
for ax in axs_A3[-1, :]:
    ax.set_xlabel("t [s]")
rmse_agg = float(np.sqrt(np.mean(np.concatenate(rmse_components))))
# fig_A3.suptitle(
#     f"Fig A3. Embedded estimate vs standalone reference (aggregate RMSE = {rmse_agg:.3e})",
#     y=1.005,
# )
fig_A3.tight_layout()

# ------------------------------------------------------------------
# Figure A4: TWO figures (Generator, Runner) of 2 rows x 4 cols
# rows = displacement (top), velocity (bottom)
# cols = x | y | x | y
#        cols 0-1: zoom 1 (4.5-5.5 s); cols 2-3: zoom 2 (29.5-30.5 s)
# Time labels only on the last row; columns share x-axis (same time window).
# ------------------------------------------------------------------
A4_zoom_windows = [
    (4.5, 5.5),
    (29.5, 30.5),
]


def make_zoom_figure(comp_name, idx_x_pos, idx_y_pos, idx_x_vel, idx_y_vel, save_name):
    fig = plt.figure(figsize=(12.0, 6.0))
    gs_outer = fig.add_gridspec(1, 2, wspace=0.36)
    gs_left = gs_outer[0, 0].subgridspec(2, 2, wspace=0.16, hspace=0.15)
    gs_right = gs_outer[0, 1].subgridspec(2, 2, wspace=0.16, hspace=0.15)

    # axs[row][col]; cols 0-1 = zoom 1, cols 2-3 = zoom 2
    axs = [[None] * 4, [None] * 4]
    # Left half (zoom 1): share x within column-pair, share y per row
    axs[0][0] = fig.add_subplot(gs_left[0, 0])
    axs[0][1] = fig.add_subplot(gs_left[0, 1], sharex=axs[0][0], sharey=axs[0][0])
    axs[1][0] = fig.add_subplot(gs_left[1, 0], sharex=axs[0][0])
    axs[1][1] = fig.add_subplot(gs_left[1, 1], sharex=axs[0][0], sharey=axs[1][0])
    # Right half (zoom 2)
    axs[0][2] = fig.add_subplot(gs_right[0, 0])
    axs[0][3] = fig.add_subplot(gs_right[0, 1], sharex=axs[0][2], sharey=axs[0][2])
    axs[1][2] = fig.add_subplot(gs_right[1, 0], sharex=axs[0][2])
    axs[1][3] = fig.add_subplot(gs_right[1, 1], sharex=axs[0][2], sharey=axs[1][2])

    component_letter = comp_name[0]  # G or R
    # (row, col, idx, zoom_window, panel_label)
    panel_specs = [
        (0, 0, idx_x_pos, A4_zoom_windows[0], f"x{component_letter}"),
        (0, 1, idx_y_pos, A4_zoom_windows[0], f"y{component_letter}"),
        (0, 2, idx_x_pos, A4_zoom_windows[1], f"x{component_letter}"),
        (0, 3, idx_y_pos, A4_zoom_windows[1], f"y{component_letter}"),
        (1, 0, idx_x_vel, A4_zoom_windows[0], f"vx{component_letter}"),
        (1, 1, idx_y_vel, A4_zoom_windows[0], f"vy{component_letter}"),
        (1, 2, idx_x_vel, A4_zoom_windows[1], f"vx{component_letter}"),
        (1, 3, idx_y_vel, A4_zoom_windows[1], f"vy{component_letter}"),
    ]

    for r, c, idx, (t0, t1), label in panel_specs:
        ax = axs[r][c]
        k0 = max(0, int(t0 / dt))
        k1 = min(len(t) - 1, int(t1 / dt))
        ax.plot(t[k0:k1 + 1], x_true[gi_focus, idx, k0:k1 + 1], lw=1.0, label="true")
        ax.plot(t_inv[k0:k1 + 1], x_est[gi_focus, idx, k0:k1 + 1], "--", lw=1.0, label="est")
        ax.set_xlim(t0, t1)
        ax.set_xticks([t0, 0.5 * (t0 + t1), t1])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, prune="both"))
        ax.grid(True, alpha=0.3)
        add_event_lines(ax)
        if r == 0:
            ax.set_title(label)
        # Y label: only on cols 0 and 2 (where y-tick numbers are visible).
        # Row 0 = displacement, Row 1 = velocity.
        if c in (0, 2):
            if r == 0:
                ax.set_ylabel(f"{comp_name} displacement [m]", fontsize=9)
            else:
                ax.set_ylabel(f"{comp_name} velocity [m/s]", fontsize=9)
        # Time labels only on the last row
        if r == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("t [s]")
        # Y numbers only on cols 0 and 2
        if c in (1, 3):
            ax.tick_params(labelleft=False)

    axs[0][0].legend(fontsize=8, loc="best")
    return fig, axs


fig_A4_gen, axs_A4_gen = make_zoom_figure(
    "Generator",
    idx_x_pos=8, idx_y_pos=9, idx_x_vel=10, idx_y_vel=11,
    save_name="figA4_generator_zooms.pdf",
)

fig_A4_run, axs_A4_run = make_zoom_figure(
    "Runner",
    idx_x_pos=12, idx_y_pos=13, idx_x_vel=14, idx_y_vel=15,
    save_name="figA4_runner_zooms.pdf",
)

plt.show()
#%%

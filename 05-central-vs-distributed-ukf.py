"""
Generalized N-DOF Distributed UKF for Chain Systems
=====================================================

This benchmark builds chain systems from repeated 2-DOF subsystems and
compares centralized and distributed UKF estimation as the number of
subsystems increases.
"""

import numpy as np
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt
import time
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ============================================================
#  Configuration
# ============================================================
@dataclass
class SystemConfig:
    """Configuration for the N-DOF chain system"""
    n_subsystems: int = 3       # Number of 2-DOF subsystems
    mass: float = 500.0         # Mass per DOF [kg]
    stiffness: float = 50_000.0 # Stiffness per spring [N/m]
    damping: float = 300.0      # Damping per damper [N·s/m]
    dt: float = 1e-3            # Time step [s]
    T: float = 5.0              # Total simulation time [s]
    sigmaF: float = 400.0       # Force input std [N]
    acc_std: float = 1e-3       # Acceleration measurement noise std [m/s²]
    initial_k_guess: float = 30_000.0  # Initial stiffness guess [N/m]
    random_seed: int = 123
    
    @property
    def n_dof(self) -> int:
        return 2 * self.n_subsystems
    
    @property
    def N(self) -> int:
        return int(self.T / self.dt)


# ============================================================
#  UKF Step Function
# ============================================================
from src.filters import unscented_kalman_filter_step_unscaled_columns as _shared_ukf_step


def unscented_kalman_filter_step(x_prev, P_prev, R_mat, Q_mat, gamma_param,
                                 y_meas_k, dt, tf_func, mf_func, u):
    return _shared_ukf_step(
        x_prev, P_prev, R_mat, Q_mat, gamma_param, y_meas_k, dt,
        tf_func, mf_func, u,
    )


# ============================================================
#  Matrix Construction (Generalized via loops)
# ============================================================
def build_MKC_matrices(cfg: SystemConfig):
    """Build mass, stiffness, and damping matrices for n_dof chain system."""
    n = cfg.n_dof
    M = cfg.mass * np.eye(n)
    K = np.zeros((n, n))
    C = np.zeros((n, n))
    
    k_vec = cfg.stiffness * np.ones(n)
    c_vec = cfg.damping * np.ones(n)
    
    for i in range(n):
        if i == 0:
            K[i, i] = k_vec[0] + k_vec[1]
            C[i, i] = c_vec[0] + c_vec[1]
        elif i == n - 1:
            K[i, i] = k_vec[i]
            C[i, i] = c_vec[i]
        else:
            K[i, i] = k_vec[i] + k_vec[i + 1]
            C[i, i] = c_vec[i] + c_vec[i + 1]
        
        if i < n - 1:
            K[i, i + 1] = -k_vec[i + 1]
            K[i + 1, i] = -k_vec[i + 1]
            C[i, i + 1] = -c_vec[i + 1]
            C[i + 1, i] = -c_vec[i + 1]
    
    return M, K, C, k_vec, c_vec


def cont_ss_ndof(M, K, C):
    """Build continuous state-space matrices"""
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    Ac = np.block([[np.zeros((n, n)), np.eye(n)],
                   [-Minv @ K, -Minv @ C]])
    Bc = np.vstack([np.zeros((n, n)), Minv])
    return Ac, Bc


def euler_c2d(Ac, Bc, dt):
    """Euler discretization"""
    Ad = np.eye(Ac.shape[0]) + dt * Ac
    Bd = dt * Bc
    return Ad, Bd


# ============================================================
#  Distributed UKF System Class
# ============================================================
class DistributedUKFSystem:
    """Generalized distributed UKF system with N subsystems."""
    
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.n_sub = cfg.n_subsystems
        self.n_dof = cfg.n_dof
        self.dt = cfg.dt
        
        self.M, self.K, self.C, self.k_vec, self.c_vec = build_MKC_matrices(cfg)
        self.Minv = np.linalg.inv(self.M)
        
        Ac, Bc = cont_ss_ndof(self.M, self.K, self.C)
        self.Ad, self.Bd = euler_c2d(Ac, Bc, cfg.dt)
        
        self.sub_nx = [4] + [5] * (self.n_sub - 1)
        self.unknown_k_indices = [2 * i - 1 for i in range(2, self.n_sub + 1)]
        self.interface_k_indices = [2 * i for i in range(1, self.n_sub)]
        
        self._init_subsystem_states()
        
    def _init_subsystem_states(self):
        cfg = self.cfg
        self.sub_x = []
        self.sub_P = []
        
        x1 = np.array([0.01, 0.0, 0.01, 0.0])
        P1 = 1e-4 * np.eye(4)
        self.sub_x.append(x1)
        self.sub_P.append(P1)
        
        for i in range(1, self.n_sub):
            x_i = np.array([0.0, 0.0, 0.0, 0.0, cfg.initial_k_guess])
            P_i = np.diag([1e-4, 1e-4, 1e-4, 1e-4, cfg.stiffness**2])
            self.sub_x.append(x_i)
            self.sub_P.append(P_i)
        
        self.sub_Q = [1e-8 * np.eye(nx) for nx in self.sub_nx]
        self.sub_R = [1e-3 * np.eye(1) for _ in range(self.n_sub)]
        self.gamma_param = 0.0
    
    def compute_interface_forces(self, sub_means: List[np.ndarray]) -> List[float]:
        n_interfaces = self.n_sub - 1
        interface_forces = []
        
        for j in range(n_interfaces):
            k_idx = self.interface_k_indices[j]
            q_left = sub_means[j][1]
            v_left = sub_means[j][3]
            q_right = sub_means[j + 1][0]
            v_right = sub_means[j + 1][2]
            F = self.k_vec[k_idx] * (q_left - q_right) + self.c_vec[k_idx] * (v_left - v_right)
            interface_forces.append(F)
        
        return interface_forces
    
    def build_subsystem_inputs(self, sub_means: List[np.ndarray], 
                               external_forces: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        interface_forces = self.compute_interface_forces(sub_means)
        
        sub_inputs = []
        for j in range(self.n_sub):
            f1 = external_forces[2 * j]
            f2 = external_forces[2 * j + 1]
            
            if j > 0:
                f1 += interface_forces[j - 1]
            if j < self.n_sub - 1:
                f2 -= interface_forces[j]
            
            sub_inputs.append(np.array([f1, f2]))
        
        return sub_inputs, interface_forces
    
    def create_subsystem_functions(self, sub_idx: int):
        cfg = self.cfg
        dt = cfg.dt
        M1 = M2 = cfg.mass
        
        if sub_idx == 0:
            k1, k2 = self.k_vec[0], self.k_vec[1]
            c1, c2 = self.c_vec[0], self.c_vec[1]
            
            def f_sub(x, u, dt_):
                q1, q2, v1, v2 = x
                u1, u2 = u
                a1 = (u1 - k1*q1 - c1*v1 - k2*(q1-q2) - c2*(v1-v2)) / M1
                a2 = (u2 + k2*(q1-q2) + c2*(v1-v2)) / M2
                x_new = np.array([q1 + dt_*v1, q2 + dt_*v2, v1 + dt_*a1, v2 + dt_*a2])
                return x_new
            
            def h_sub(x, u, dt_):
                q1, q2, v1, v2 = x
                u1, u2 = u
                a1 = (u1 - k1*q1 - c1*v1 - k2*(q1-q2) - c2*(v1-v2)) / M1
                return np.array([a1])
        else:
            c_internal = self.c_vec[2 * sub_idx + 1]
            
            def f_sub(x, u, dt_):
                q1, q2, v1, v2, k_est = x
                u1, u2 = u
                a1 = (u1 - k_est*(q1-q2) - c_internal*(v1-v2)) / M1
                a2 = (u2 + k_est*(q1-q2) + c_internal*(v1-v2)) / M2
                x_new = np.array([q1 + dt_*v1, q2 + dt_*v2, v1 + dt_*a1, v2 + dt_*a2, k_est])
                return x_new
            
            def h_sub(x, u, dt_):
                q1, q2, v1, v2, k_est = x
                u1, u2 = u
                a2 = (u2 + k_est*(q1-q2) + c_internal*(v1-v2)) / M2
                return np.array([a2])
        
        return f_sub, h_sub
    
    def run_distributed_ukf(self, u_hist: np.ndarray, a_meas_hist: np.ndarray):
        N = self.cfg.N
        n_sub = self.n_sub
        n_dof = self.n_dof
        
        Q_est = np.zeros((N + 1, n_dof))
        V_est = np.zeros((N + 1, n_dof))
        k_est_hist = {idx: np.zeros(N + 1) for idx in self.unknown_k_indices}
        F_hist = {j: np.zeros(N + 1) for j in range(n_sub - 1)}
        
        for j in range(n_sub):
            Q_est[0, 2*j] = self.sub_x[j][0]
            Q_est[0, 2*j + 1] = self.sub_x[j][1]
            V_est[0, 2*j] = self.sub_x[j][2]
            V_est[0, 2*j + 1] = self.sub_x[j][3]
            if j > 0:
                k_idx = self.unknown_k_indices[j - 1]
                k_est_hist[k_idx][0] = self.sub_x[j][4]
        
        sub_funcs = [self.create_subsystem_functions(j) for j in range(n_sub)]
        current_x = [x.copy() for x in self.sub_x]
        current_P = [P.copy() for P in self.sub_P]
        
        start_time = time.time()
        
        for k in range(N):
            if k % 200 == 0:
                print(f"\r    Distributed: Step {k}/{N} ({100*k/N:.1f}%)", end="", flush=True)
            
            sub_inputs, interface_forces = self.build_subsystem_inputs(current_x, u_hist[k])
            
            for j, F in enumerate(interface_forces):
                F_hist[j][k + 1] = F
            
            measurements = []
            measurements.append(a_meas_hist[k + 1, 0:1])
            for j in range(1, n_sub):
                meas_idx = 2 * j + 1
                measurements.append(a_meas_hist[k + 1, meas_idx:meas_idx + 1])
            
            for j in range(n_sub):
                f_sub, h_sub = sub_funcs[j]
                current_x[j], current_P[j] = unscented_kalman_filter_step(
                    x_prev=current_x[j], P_prev=current_P[j],
                    R_mat=self.sub_R[j], Q_mat=self.sub_Q[j],
                    gamma_param=self.gamma_param, y_meas_k=measurements[j],
                    dt=self.dt, tf_func=f_sub, mf_func=h_sub, u=sub_inputs[j]
                )
            
            for j in range(n_sub):
                Q_est[k + 1, 2*j] = current_x[j][0]
                Q_est[k + 1, 2*j + 1] = current_x[j][1]
                V_est[k + 1, 2*j] = current_x[j][2]
                V_est[k + 1, 2*j + 1] = current_x[j][3]
                if j > 0:
                    k_idx = self.unknown_k_indices[j - 1]
                    k_est_hist[k_idx][k + 1] = current_x[j][4]
        
        print()  # New line after progress
        dist_time = time.time() - start_time
        
        return Q_est, V_est, k_est_hist, F_hist, dist_time


# ============================================================
#  Centralized UKF System
# ============================================================
class CentralizedUKFSystem:
    """Centralized UKF for the full N-DOF system."""
    
    def __init__(self, cfg: SystemConfig, k_vec: np.ndarray, c_vec: np.ndarray, 
                 unknown_k_indices: List[int], Minv: np.ndarray):
        self.cfg = cfg
        self.n_dof = cfg.n_dof
        self.dt = cfg.dt
        self.k_vec = k_vec
        self.c_vec = c_vec
        self.unknown_k_indices = unknown_k_indices
        self.n_unknown = len(unknown_k_indices)
        self.Minv = Minv
        
        self.nx = 2 * self.n_dof + self.n_unknown
        n_sub = cfg.n_subsystems
        self.meas_indices = [0] + [2 * j + 1 for j in range(1, n_sub)]
        self.ny = len(self.meas_indices)
        
        self._init_state()
    
    def _init_state(self):
        cfg = self.cfg
        self.x = np.zeros(self.nx)
        self.x[0] = 0.01
        self.x[self.n_dof] = 0.01
        
        for i in range(self.n_unknown):
            self.x[2 * self.n_dof + i] = cfg.initial_k_guess
        
        P_diag = np.concatenate([
            1e-4 * np.ones(2 * self.n_dof),
            cfg.stiffness**2 * np.ones(self.n_unknown)
        ])
        self.P = np.diag(P_diag)
        self.Q = 1e-8 * np.eye(self.nx)
        self.R = 1e-3 * np.eye(self.ny)
        self.gamma_param = 0.0
    
    def _build_K_with_estimates(self, k_estimates: np.ndarray) -> np.ndarray:
        k_eff = self.k_vec.copy()
        for i, k_idx in enumerate(self.unknown_k_indices):
            k_eff[k_idx] = k_estimates[i]
        return self._build_K_from_vec(k_eff)
    
    def _build_K_from_vec(self, k_vec: np.ndarray) -> np.ndarray:
        n = self.n_dof
        K = np.zeros((n, n))
        for i in range(n):
            if i == 0:
                K[i, i] = k_vec[0] + k_vec[1]
            elif i == n - 1:
                K[i, i] = k_vec[i]
            else:
                K[i, i] = k_vec[i] + k_vec[i + 1]
            if i < n - 1:
                K[i, i + 1] = -k_vec[i + 1]
                K[i + 1, i] = -k_vec[i + 1]
        return K
    
    def _build_C_from_vec(self, c_vec: np.ndarray) -> np.ndarray:
        n = self.n_dof
        C = np.zeros((n, n))
        for i in range(n):
            if i == 0:
                C[i, i] = c_vec[0] + c_vec[1]
            elif i == n - 1:
                C[i, i] = c_vec[i]
            else:
                C[i, i] = c_vec[i] + c_vec[i + 1]
            if i < n - 1:
                C[i, i + 1] = -c_vec[i + 1]
                C[i + 1, i] = -c_vec[i + 1]
        return C
    
    def accel_with_estimates(self, q, v, u, k_estimates):
        K = self._build_K_with_estimates(k_estimates)
        C = self._build_C_from_vec(self.c_vec)
        return self.Minv @ (u - C @ v - K @ q)
    
    def create_functions(self):
        n = self.n_dof
        dt = self.dt
        
        def f_central(x, u, dt_):
            q = x[:n]
            v = x[n:2*n]
            k_est = x[2*n:]
            a = self.accel_with_estimates(q, v, u, k_est)
            x_new = np.zeros_like(x)
            x_new[:n] = q + dt_ * v
            x_new[n:2*n] = v + dt_ * a
            x_new[2*n:] = k_est
            return x_new
        
        def h_central(x, u, dt_):
            q = x[:n]
            v = x[n:2*n]
            k_est = x[2*n:]
            a = self.accel_with_estimates(q, v, u, k_est)
            return a[self.meas_indices]
        
        return f_central, h_central
    
    def run_centralized_ukf(self, u_hist: np.ndarray, a_meas_hist: np.ndarray):
        N = self.cfg.N
        n = self.n_dof
        
        x_hist = np.zeros((N + 1, self.nx))
        x_hist[0] = self.x.copy()
        
        f_central, h_central = self.create_functions()
        current_x = self.x.copy()
        current_P = self.P.copy()
        
        start_time = time.time()
        
        for k in range(N):
            if k % 200 == 0:
                print(f"\r    Centralized: Step {k}/{N} ({100*k/N:.1f}%)", end="", flush=True)
            
            u = u_hist[k]
            z = a_meas_hist[k + 1, self.meas_indices]
            
            current_x, current_P = unscented_kalman_filter_step(
                x_prev=current_x, P_prev=current_P,
                R_mat=self.R, Q_mat=self.Q,
                gamma_param=self.gamma_param, y_meas_k=z,
                dt=self.dt, tf_func=f_central, mf_func=h_central, u=u
            )
            x_hist[k + 1] = current_x
        
        print()  # New line after progress
        cent_time = time.time() - start_time
        
        Q_est = x_hist[:, :n]
        V_est = x_hist[:, n:2*n]
        k_est_hist = {self.unknown_k_indices[i]: x_hist[:, 2*n + i] 
                      for i in range(self.n_unknown)}
        
        return Q_est, V_est, k_est_hist, cent_time


# ============================================================
#  Simulation and Results Storage
# ============================================================
def simulate_truth(cfg: SystemConfig, M, K, C, Ad, Bd, Minv, u_hist):
    """Simulate the true system response"""
    N = cfg.N
    n = cfg.n_dof
    
    x_true = np.zeros(2 * n)
    Q_true = np.zeros((N + 1, n))
    V_true = np.zeros((N + 1, n))
    A_true = np.zeros((N + 1, n))
    
    Q_true[0] = x_true[:n]
    V_true[0] = x_true[n:]
    A_true[0] = Minv @ (u_hist[0] - C @ V_true[0] - K @ Q_true[0])
    
    for k in range(N):
        x_true = Ad @ x_true + Bd @ u_hist[k]
        Q_true[k + 1] = x_true[:n]
        V_true[k + 1] = x_true[n:]
        A_true[k + 1] = Minv @ (u_hist[k] - C @ V_true[k + 1] - K @ Q_true[k + 1])
    
    return Q_true, V_true, A_true


def load_results(filepath: str) -> Dict:
    """Load results from a pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ============================================================
#  Plotting Functions
# ============================================================
def plot_stiffness_boxplot(all_results: Dict, k_true: float, output_dir: str):
    """
    Create box/candle plots for stiffness estimation errors.
    
    all_results: {n_subsystems: {'k_cent_final': [...], 'k_dist_final': [...], ...}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for box plots
    n_subs_list = sorted(all_results.keys())
    
    # --- Centralized box plot ---
    ax1 = axes[0]
    cent_data = []
    cent_labels = []
    for n_sub in n_subs_list:
        if 'k_cent_final' in all_results[n_sub] and all_results[n_sub]['k_cent_final'] is not None:
            errors = [(k - k_true) / k_true * 100 for k in all_results[n_sub]['k_cent_final']]
            cent_data.append(errors)
            cent_labels.append(f'{n_sub}\n({2*n_sub} DOF)')
    
    if cent_data:
        bp1 = ax1.boxplot(cent_data, labels=cent_labels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('coral')
            patch.set_alpha(0.7)
        ax1.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Number of Subsystems', fontsize=11)
        ax1.set_ylabel('Stiffness Estimation Error [%]', fontsize=11)
        ax1.set_title('Centralized UKF - Stiffness Estimation Errors', fontsize=12)
        ax1.grid(True, alpha=0.3)
    
    # --- Distributed box plot ---
    ax2 = axes[1]
    dist_data = []
    dist_labels = []
    for n_sub in n_subs_list:
        if 'k_dist_final' in all_results[n_sub] and all_results[n_sub]['k_dist_final'] is not None:
            errors = [(k - k_true) / k_true * 100 for k in all_results[n_sub]['k_dist_final']]
            dist_data.append(errors)
            dist_labels.append(f'{n_sub}\n({2*n_sub} DOF)')
    
    if dist_data:
        bp2 = ax2.boxplot(dist_data, labels=dist_labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Number of Subsystems', fontsize=11)
        ax2.set_ylabel('Stiffness Estimation Error [%]', fontsize=11)
        ax2.set_title('Distributed UKF - Stiffness Estimation Errors', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_stiffness_boxplot_combined(all_results: Dict, k_true: float, output_dir: str):
    """
    Create a combined box plot comparing centralized vs distributed side by side.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    n_subs_list = sorted(all_results.keys())
    positions_cent = []
    positions_dist = []
    cent_data = []
    dist_data = []
    x_ticks = []
    x_tick_labels = []
    
    pos = 1
    for i, n_sub in enumerate(n_subs_list):
        has_cent = 'k_cent_final' in all_results[n_sub] and all_results[n_sub]['k_cent_final'] is not None
        has_dist = 'k_dist_final' in all_results[n_sub] and all_results[n_sub]['k_dist_final'] is not None
        
        if has_cent:
            errors_cent = [(k - k_true) / k_true * 100 for k in all_results[n_sub]['k_cent_final']]
            cent_data.append(errors_cent)
            positions_cent.append(pos)
            pos += 1
        
        if has_dist:
            errors_dist = [(k - k_true) / k_true * 100 for k in all_results[n_sub]['k_dist_final']]
            dist_data.append(errors_dist)
            positions_dist.append(pos)
            pos += 1
        
        x_ticks.append((positions_cent[-1] + positions_dist[-1]) / 2 if has_cent and has_dist else pos - 1)
        x_tick_labels.append(f'{n_sub} sub\n({2*n_sub} DOF)')
        pos += 1  # Gap between groups
    
    # Plot
    if cent_data:
        bp1 = ax.boxplot(cent_data, positions=positions_cent, widths=0.6, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('coral')
            patch.set_alpha(0.7)
    
    if dist_data:
        bp2 = ax.boxplot(dist_data, positions=positions_dist, widths=0.6, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
    
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel('Number of Subsystems', fontsize=12)
    ax.set_ylabel('Stiffness Estimation Error [%]', fontsize=12)
    ax.set_title('Stiffness Estimation Errors: Centralized (coral) vs Distributed (blue)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='coral', alpha=0.7, label='Centralized'),
                       Patch(facecolor='steelblue', alpha=0.7, label='Distributed')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_timing_comparison(timing_results: dict, save_path=None):
    """Plot execution time comparison using lines"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Extract and sort data
    n_subs = sorted(list(timing_results.keys()))
    cent_times = [timing_results[n]['centralized'] for n in n_subs]
    dist_times = [timing_results[n]['distributed'] for n in n_subs]
    
    # 2. Plot Distributed UKF (Usually complete data)
    ax.plot(n_subs, dist_times, marker='o', linestyle='-', linewidth=2, 
            label='Distributed UKF', color='steelblue')
    
    # 3. Plot Centralized UKF (Handling potential NaNs/missing values)
    # This filters out NaNs so the line doesn't break or disappear
    valid_n = [n for n, c in zip(n_subs, cent_times) if not np.isnan(c)]
    valid_c = [c for c in cent_times if not np.isnan(c)]
    
    if valid_c:
        ax.plot(valid_n, valid_c, marker='s', linestyle='--', linewidth=2, 
                label='Centralized UKF', color='coral')
    
    # 4. Formatting
    ax.set_xlabel('Number of Subsystems', fontsize=12)
    ax.set_ylabel('Execution Time [s]', fontsize=12)
    ax.set_title('Execution Time Scaling: Centralized vs Distributed UKF', fontsize=14)
    
    # Update X-ticks to show DOF (Degrees of Freedom)
    ax.set_xticks(n_subs)
    ax.set_xticklabels([f'{n}\n({2*n} DOF)' for n in n_subs])
    
    ax.legend(frameon=False, fontsize=10)
    # ax.grid(False, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    plt.show()


def plot_distributed_scaling(timing_results: dict, save_path=None):
    """Plot how distributed UKF scales with system size"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_subs = list(timing_results.keys())
    dofs = [2 * n for n in n_subs]
    dist_times = [timing_results[n]['distributed'] for n in n_subs]
    
    ax.plot(dofs, dist_times, 'o-', color='steelblue', linewidth=2, markersize=10)
    ax.set_xlabel('Number of DOF', fontsize=12)
    ax.set_ylabel('Execution Time [s]', fontsize=12)
    ax.set_title('Distributed UKF Scaling with System Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    for dof, t in zip(dofs, dist_times):
        ax.annotate(f'{t:.2f}s', xy=(dof, t), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()


# ============================================================
#  Main Simulation Function
# ============================================================
def run_simulation(n_subsystems: int, output_dir: str, plot: bool = True, 
                   run_centralized: bool = True, save_data: bool = True):
    """Run complete simulation for given number of subsystems"""
    print(f"\n{'='*60}")
    print(f"Running simulation with {n_subsystems} subsystems ({2*n_subsystems} DOF)")
    print(f"{'='*60}")
    
    cfg = SystemConfig(n_subsystems=n_subsystems)
    
    rng = np.random.default_rng(cfg.random_seed)
    u_hist = rng.normal(0.0, cfg.sigmaF, size=(cfg.N, cfg.n_dof))
    
    dist_sys = DistributedUKFSystem(cfg)
    
    Q_true, V_true, A_true = simulate_truth(
        cfg, dist_sys.M, dist_sys.K, dist_sys.C,
        dist_sys.Ad, dist_sys.Bd, dist_sys.Minv, u_hist
    )
    
    a_meas_hist = A_true + rng.normal(0, cfg.acc_std, size=(cfg.N + 1, cfg.n_dof))
    t = np.linspace(0, cfg.T, cfg.N + 1)
    
    # Run distributed UKF
    print("Running Distributed UKF...")
    Q_dist, V_dist, k_dist_hist, F_hist, dist_time = dist_sys.run_distributed_ukf(u_hist, a_meas_hist)
    print(f"  Distributed UKF time: {dist_time:.3f} s")
    
    # Run centralized UKF
    if run_centralized:
        print("Running Centralized UKF...")
        cent_sys = CentralizedUKFSystem(cfg, dist_sys.k_vec, dist_sys.c_vec,
                                         dist_sys.unknown_k_indices, dist_sys.Minv)
        Q_cent, V_cent, k_cent_hist, cent_time = cent_sys.run_centralized_ukf(u_hist, a_meas_hist)
        print(f"  Centralized UKF time: {cent_time:.3f} s")
    else:
        print("  Skipping Centralized UKF")
        Q_cent, V_cent, k_cent_hist, cent_time = None, None, None, float('nan')
    
    # Extract final stiffness values
    k_dist_final = [k_dist_hist[k_idx][-1] for k_idx in dist_sys.unknown_k_indices]
    k_cent_final = [k_cent_hist[k_idx][-1] for k_idx in dist_sys.unknown_k_indices] if k_cent_hist else None
    
    # Print stiffness table
    print(f"\n{'Index':<8} {'True':<12} {'Central':<12} {'Distributed':<12} {'Cent Err%':<12} {'Dist Err%':<12}")
    print("-" * 68)
    for i, k_idx in enumerate(dist_sys.unknown_k_indices):
        k_true = cfg.stiffness
        k_dist = k_dist_final[i]
        err_dist = 100 * abs(k_dist - k_true) / k_true
        if k_cent_final:
            k_cent = k_cent_final[i]
            err_cent = 100 * abs(k_cent - k_true) / k_true
            print(f"k_{k_idx+1:<6} {k_true:<12.0f} {k_cent:<12.0f} {k_dist:<12.0f} {err_cent:<12.1f} {err_dist:<12.1f}")
        else:
            print(f"k_{k_idx+1:<6} {k_true:<12.0f} {'N/A':<12} {k_dist:<12.0f} {'N/A':<12} {err_dist:<12.1f}")
    
    # Prepare results dictionary
    results = {
        'n_subsystems': n_subsystems,
        'n_dof': cfg.n_dof,
        'cfg': cfg,
        't': t,
        'Q_true': Q_true,
        'V_true': V_true,
        'A_true': A_true,
        'Q_dist': Q_dist,
        'V_dist': V_dist,
        'k_dist_hist': k_dist_hist,
        'k_dist_final': k_dist_final,
        'F_hist': F_hist,
        'dist_time': dist_time,
        'Q_cent': Q_cent,
        'V_cent': V_cent,
        'k_cent_hist': k_cent_hist,
        'k_cent_final': k_cent_final,
        'cent_time': cent_time,
        'unknown_k_indices': dist_sys.unknown_k_indices,
        'k_true': cfg.stiffness,
    }
    
    return {
        'centralized': cent_time,
        'distributed': dist_time,
        'k_cent_final': k_cent_final,
        'k_dist_final': k_dist_final,
        'results': results
    }


def main():
    """Main function to run simulations"""
    print("="*70)
    print("GENERALIZED N-DOF DISTRIBUTED UKF SIMULATION")
    print("="*70)
    
    output_dir = "."
    
    # Subsystem counts to test
    subsystem_counts = [4, 8, 12, 16, 20, 24, 28, 32]
    
    all_results = {}
    timing_results = {}
    
    for n_sub in subsystem_counts:
        run_cent = True  # Set to True to get centralized timing
        results = run_simulation(n_sub, output_dir, plot=False,
                                run_centralized=run_cent, save_data=False)
        all_results[n_sub] = results
        timing_results[n_sub] = {
            'centralized': results['centralized'],
            'distributed': results['distributed']
        }
    
    # Print timing summary
    print("\n" + "="*70)
    print("TIMING COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Subsystems':<12} {'DOF':<8} {'Centralized [s]':<18} {'Distributed [s]':<18} {'Speedup':<10}")
    print("-" * 66)
    for n_sub in subsystem_counts:
        cent_t = timing_results[n_sub]['centralized']
        dist_t = timing_results[n_sub]['distributed']
        if not np.isnan(cent_t):
            speedup = cent_t / dist_t if dist_t > 0 else float('inf')
            print(f"{n_sub:<12} {2*n_sub:<8} {cent_t:<18.3f} {dist_t:<18.3f} {speedup:<10.2f}x")
        else:
            print(f"{n_sub:<12} {2*n_sub:<8} {'N/A':<18} {dist_t:<18.3f} {'N/A':<10}")
    
    # Create plots
    print("\nGenerating plots...")
    k_true = 50_000.0  # True stiffness value
    
    plot_stiffness_boxplot(all_results, k_true, output_dir)
    plot_stiffness_boxplot_combined(all_results, k_true, output_dir)
    plot_timing_comparison(timing_results)
    plot_distributed_scaling(timing_results)

    print("\nDone!")
    return all_results, timing_results

def empirical_coverage_from_saved_results(summary_path: str,
                                          n_subsystems: int,
                                          nominal: float = 0.95,
                                          true_value: float = 50_000.0):
    """
    Compute a coverage proxy from the saved final stiffness estimates only.

    Important:
    The saved pickle files do not contain the UKF posterior covariance history,
    so this is not the exact interval coverage of the filter. Instead, it uses
    the empirical spread of the saved final estimates for each estimator:

        coverage = mean(|estimate - true_value| <= z * std(estimates))

    This lets you post-process the existing saved results without rerunning the
    simulation.
    """
    class PickleSystemConfig:
        pass

    import __main__

    __main__.SystemConfig = PickleSystemConfig

    with open(summary_path, "rb") as f:
        saved = pickle.load(f)

    result_block = saved["all_results"][n_subsystems]
    z_value = 1.959963984540054 if nominal == 0.95 else scipy.stats.norm.ppf(0.5 + nominal / 2.0)

    out = {"n_subsystems": n_subsystems, "nominal": nominal, "true_value": true_value}
    for label, key in [("distributed", "k_dist_final"), ("centralized", "k_cent_final")]:
        estimates = np.asarray(result_block[key], dtype=float)
        std_est = np.std(estimates, ddof=1)
        half_width = z_value * std_est
        coverage = np.mean(np.abs(estimates - true_value) <= half_width)
        out[label] = {
            "n_estimates": int(estimates.size),
            "mean_estimate": float(np.mean(estimates)),
            "std_estimate": float(std_est),
            "interval_half_width": float(half_width),
            "coverage": float(coverage),
        }

    return out

#%%
if __name__ == "__main__":
    RUN_SIMULATIONS = True
    COMPUTE_COVERAGE_FROM_SAVED = False

    if RUN_SIMULATIONS:
        all_results, timing_results = main()

    if COMPUTE_COVERAGE_FROM_SAVED:
        coverage_summary = empirical_coverage_from_saved_results(
            summary_path="results/all-results-summary.pkl",
            n_subsystems=32,
            nominal=0.95,
            true_value=50_000.0,
        )

        print("\n95% empirical coverage proxy from saved final stiffness estimates")
        print(f"Network size: {coverage_summary['n_subsystems']} subsystems ({2 * coverage_summary['n_subsystems']} DOF)")
        print(f"Distributed : {coverage_summary['distributed']['coverage']:.6f}")
        print(f"Centralized : {coverage_summary['centralized']['coverage']:.6f}")
    
#%%

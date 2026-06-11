"""
Four-DOF inverse example.

The script simulates a coupled 4-DOF mechanical system, estimates the
interface stiffness with centralized and distributed UKF formulations, and
compares deterministic, probabilistic, and SINDy-based message passing.
"""

import numpy as np
import numpy.linalg as linalg
import scipy.linalg
import matplotlib.pyplot as plt

#%% Measurements are available for all four DOFs.
# ============================================================
#  UKF Function Definition
# ============================================================
from src.filters import unscented_kalman_filter_step as _shared_ukf_step


def unscented_kalman_filter_step(x_prev, P_prev, R_mat, Q_mat, gamma_param,
                                y_meas_k, t_idx, dt, force_arr,
                                tf_func, mf_func, u):
    return _shared_ukf_step(
        x_prev, P_prev, R_mat, Q_mat, gamma_param, y_meas_k, dt,
        tf_func, mf_func, u, t_idx=t_idx, return_prediction=True,
        fallback_eigendecomp=False, scaled_cholesky=True,
    )


# ============================================================
#  Parameters and simulation
# ============================================================
M1 = M2 = M3 = M4 = 500.0
k1 = k2 = k3 = k4 = 50_000.0
c1 = c2 = c3 = c4 = 300.0

dt = 1e-3
T  = 10.0
N  = int(T/dt)

sigmaF = 400.0       # std of input forces
acc_std = 1e-3    # accel measurement noise std

k4_ref = k4         # reference scale for parameter state θ_k (still used for central, but not dist)

# Structural helpers
def S1_MKC_self():
    M  = np.diag([M1, M2])
    K  = np.array([[k1+k2, -k2],
                   [   -k2,   k2]], float)
    C  = np.array([[c1+c2, -c2],
                   [   -c2,   c2]], float)
    return M, K, C

def S2_MKC_self():
    M  = np.diag([M3, M4])
    K  = np.array([[k4, -k4],
                   [-k4,  k4]], float)
    C  = np.array([[c4, -c4],
                   [-c4,  c4]], float)
    return M, K, C

def cont_ss_2dof(M, K, C):
    Minv = np.linalg.inv(M)
    Ac = np.block([[np.zeros((2,2)), np.eye(2)],
                   [-Minv@K,        -Minv@C]])
    Bc = np.vstack([np.zeros((2,2)), Minv])
    return Ac, Bc

def mono_MKC():
    M = np.diag([M1, M2, M3, M4])
    K = np.array([[k1+k2,   -k2,      0,      0],
                  [  -k2, k2+k3,    -k3,     0],
                  [   0,   -k3,  k3+k4,    -k4],
                  [   0,     0,    -k4,     k4]], float)
    C = np.array([[c1+c2,   -c2,      0,      0],
                  [  -c2, c2+c3,    -c3,     0],
                  [   0,   -c3,  c3+c4,    -c4],
                  [   0,     0,    -c4,     c4]], float)
    return M, K, C

def cont_ss_4dof(M, K, C):
    Minv = np.linalg.inv(M)
    Ac = np.block([[np.zeros((4,4)), np.eye(4)],
                   [-Minv@K,         -Minv@C]])
    Bc = np.vstack([np.zeros((4,4)), Minv])
    return Ac, Bc

def euler_c2d(Ac, Bc, dt):
    Ad = np.eye(Ac.shape[0]) + dt*Ac
    Bd = dt*Bc
    return Ad, Bd

# Monolithic model and truth simulation
Mmono, Kmono, Cmono = mono_MKC()
AcM, BcM = cont_ss_4dof(Mmono, Kmono, Cmono)
AdM, BdM = euler_c2d(AcM, BcM, dt)

Minv_mono = np.linalg.inv(Mmono)

def accel_mono(q, v, u):
    """True acceleration with physical k4."""
    return Minv_mono @ (u - Cmono@v - Kmono@q)

# random force inputs
rng = np.random.default_rng(123)
u_hist = rng.normal(0.0, sigmaF, size=(N,4))

# simulate truth
x_true = np.zeros(8)          # [q1..q4, v1..v4]
Q_true = np.zeros((N+1,4))
V_true = np.zeros((N+1,4))
A_true = np.zeros((N+1,4))

Q_true[0] = x_true[:4]
V_true[0] = x_true[4:]
A_true[0] = accel_mono(Q_true[0], V_true[0], u_hist[0])

for k in range(N):
    x_true = AdM @ x_true + BdM @ u_hist[k]
    Q_true[k+1] = x_true[:4]
    V_true[k+1] = x_true[4:]
    A_true[k+1] = accel_mono(Q_true[k+1], V_true[k+1], u_hist[k])

# measurements (a1_meas and a4_meas were in the original code)
a1_meas = A_true[:,0] + rng.normal(0, acc_std, size=N+1)
a2_meas = A_true[:,1] + rng.normal(0, acc_std, size=N+1) 
a3_meas = A_true[:,2] + rng.normal(0, acc_std, size=N+1) 
a4_meas = A_true[:,3] + rng.normal(0, acc_std, size=N+1)
a_meas_hist = np.vstack([a1_meas, a2_meas, a3_meas, a4_meas]).T

t = np.linspace(0, T, N+1)
rmse_window_sec = 6.0
param_rmse_window_sec = 5.0


# ============================================================
#  Distributed model: WRAPPER FUNCTIONS for new UKF signature (UNNORMALIZED)
# ============================================================
def build_sub_inputs_jacobi(s1_mean, s2_mean, f1, f2, f3, f4, mp_params=None):
    """Message passing between S1 and S2."""
    if mp_params is None:
        k3_eff, c3_eff = k3, c3
    else:
        k3_eff, c3_eff = mp_params

    x2, v2 = s1_mean[1], s1_mean[3]
    x3, v3 = s2_mean[0], s2_mean[2]

    Fb = k3_eff*(x2 - x3) + c3_eff*(v2 - v3)   # + on DOF3, - on DOF2

    u1 = np.array([f1,        f2 - Fb])       # S1 inputs (DOF1, DOF2)
    u2 = np.array([f3 + Fb,   f4     ])       # S2 inputs (DOF3, DOF4)
    return u1, u2

def build_sub_inputs_jacobi_prob(s1_mean, P1, s2_mean, P2, f1, f2, f3, f4):
    """
    Probabilistic Jacobi message passing.
    Returns:
      u1, u2        : mean inputs
      var_Fb        : variance of interface force
    """

    # Extract interface states
    x2, v2 = s1_mean[1], s1_mean[3]
    x3, v3 = s2_mean[0], s2_mean[2]

    # Mean interface force
    Fb_mean = k3*(x2 - x3) + c3*(v2 - v3)

    # Interface covariance blocks
    P2_s1 = P1[np.ix_([1,3],[1,3])]   # Cov[x2, v2]
    P2_s2 = P2[np.ix_([0,2],[0,2])]   # Cov[x3, v3]

    a = np.array([k3, c3])

    # Variance of Fb
    var_Fb = a @ P2_s1 @ a + a @ P2_s2 @ a

    # Mean inputs
    u1 = np.array([f1, f2 - Fb_mean])
    u2 = np.array([f3 + Fb_mean, f4])

    return u1, u2, Fb_mean, var_Fb

# ============================================================
#  Surrogate message passing (SINDy) — use ALL identified terms
# ============================================================
# SINDy coefficients
xi0 = +5.613034e+04   # Δx
xi1 = +3.308257e+02   # Δv
xi2 = -2.863096e+08   # (Δx)^3
xi3 = +6.696752e+03   # |Δv|Δv
# xi4 = 0             # ΔxΔv (ignored)
# xi5 = 0             # bias   (ignored)

def build_sub_inputs_jacobi_sindy(s1_mean, s2_mean, f1, f2, f3, f4):
    """
    Message passing using learned SINDy surrogate:
      Fb_hat = xi0*dx + xi1*dv + xi2*dx^3 + xi3*|dv|dv
    """
    x2, v2 = s1_mean[1], s1_mean[3]
    x3, v3 = s2_mean[0], s2_mean[2]
    dx = (x2 - x3)
    dv = (v2 - v3)

    Fb_hat = xi0*dx + xi1*dv + xi2*(dx**3) + xi3*(np.abs(dv)*dv)

    u1 = np.array([f1,        f2 - Fb_hat])
    u2 = np.array([f3 + Fb_hat, f4])
    return u1, u2, Fb_hat


def build_full_state_from_subsystems(s1_state, s2_state):
    return np.array([
        s1_state[0], s1_state[1], s2_state[0], s2_state[1],
        s1_state[2], s1_state[3], s2_state[2], s2_state[3]
    ], dtype=float)


def build_full_cov_from_subsystems(P1_sub, P2_sub):
    P_full = np.zeros((8, 8), dtype=float)
    idx_s1 = np.array([0, 1, 4, 5])
    idx_s2 = np.array([2, 3, 6, 7])
    P_full[np.ix_(idx_s1, idx_s1)] = P1_sub
    P_full[np.ix_(idx_s2, idx_s2)] = P2_sub[:4, :4]
    return P_full


def build_interface_state_stats(s1_state, P1_sub, s2_state, P2_sub):
    mean = np.array([s1_state[1], s1_state[3], s2_state[0], s2_state[2]], dtype=float)
    cov = np.zeros((4, 4), dtype=float)
    cov[:2, :2] = P1_sub[np.ix_([1, 3], [1, 3])]
    cov[2:, 2:] = P2_sub[np.ix_([0, 2], [0, 2])]
    return mean, cov


def surrogate_force_from_interface(z):
    dx = z[0] - z[2]
    dv = z[1] - z[3]
    return xi0 * dx + xi1 * dv + xi2 * (dx ** 3) + xi3 * (np.abs(dv) * dv)


def gaussian_sigma_points(mean, cov, gamma_param=0.0):
    n = mean.shape[0]
    c = n + gamma_param
    W = np.zeros(2 * n + 1)
    sigma = np.zeros((n, 2 * n + 1))
    sigma[:, 0] = mean

    P_sym = 0.5 * (cov + cov.T)
    chol_arg = c * P_sym
    jitter = 0.0
    max_tries = 10
    for i in range(max_tries):
        try:
            S_mat = scipy.linalg.cholesky(chol_arg + jitter * np.eye(n), lower=False)
            break
        except scipy.linalg.LinAlgError:
            jitter = 1e-9 if jitter == 0.0 else 10.0 * jitter
            if i == max_tries - 1:
                raise

    W[0] = gamma_param / c
    for k in range(1, n + 1):
        sigma[:, k] = mean + S_mat[k - 1]
        sigma[:, n + k] = mean - S_mat[k - 1]
        W[k] = 1.0 / (2 * c)
        W[n + k] = 1.0 / (2 * c)
    return sigma, W


def sigma_point_scalar_moments(mean, cov, func, gamma_param=0.0):
    sigma, W = gaussian_sigma_points(np.asarray(mean, dtype=float), np.asarray(cov, dtype=float), gamma_param=gamma_param)
    values = np.array([func(sigma[:, i]) for i in range(sigma.shape[1])], dtype=float)
    mean_val = float(np.sum(W * values))
    var_val = float(np.sum(W * (values - mean_val) ** 2))
    return mean_val, var_val


def gaussian_nll_scalar(truth, mean, variance, min_variance=1e-12):
    var = max(float(variance), min_variance)
    err = float(truth) - float(mean)
    return 0.5 * ((err * err) / var + np.log(var) + np.log(2.0 * np.pi))


def gaussian_nll_vector(truth, mean, cov, min_jitter=1e-12, max_tries=12):
    truth = np.asarray(truth, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    dim = truth.shape[0]

    jitter = min_jitter
    eye = np.eye(dim)
    for _ in range(max_tries):
        try:
            L = scipy.linalg.cholesky(cov + jitter * eye, lower=True)
            break
        except scipy.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise scipy.linalg.LinAlgError("Unable to stabilize covariance for Gaussian NLL.")

    diff = truth - mean
    solved = scipy.linalg.solve_triangular(L, diff, lower=True)
    mahal = solved @ solved
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * (mahal + logdet + dim * np.log(2.0 * np.pi))


def build_full_state_var_from_subsystems(P1_sub, P2_sub):
    return np.array([
        P1_sub[0, 0], P1_sub[1, 1], P2_sub[0, 0], P2_sub[1, 1],
        P1_sub[2, 2], P1_sub[3, 3], P2_sub[2, 2], P2_sub[3, 3]
    ], dtype=float)


# S1 state: [x1, x2, v1, v2]
def f_s1(x, u, dt_dummy, t_idx_dummy):
    """Wrapper for S1 state transition, compatible with the shared UKF step."""
    q1, q2, v1, v2 = x
    u1, u2 = u

    a1 = (u1 - k1*q1 - c1*v1 - k2*(q1 - q2) - c2*(v1 - v2)) / M1
    a2 = (u2 + k2*(q1 - q2) + c2*(v1 - v2)) / M2

    x_new = np.zeros_like(x)
    x_new[0] = q1 + dt*v1
    x_new[1] = q2 + dt*v2
    x_new[2] = v1 + dt*a1
    x_new[3] = v2 + dt*a2
    return x_new

def h_s1(x, u, dt_dummy, t_idx_dummy):
    """Wrapper for S1 measurement (a1, a2), compatible with the shared UKF step."""
    q1, q2, v1, v2 = x
    u1, u2 = u

    a1 = (u1 - k1*q1 - c1*v1 - k2*(q1 - q2) - c2*(v1 - v2)) / M1
    a2 = (u2 + k2*(q1 - q2) + c2*(v1 - v2)) / M2 
    return np.array([a1, a2])

# S2 state: [x3, x4, v3, v4, k4_est]
def f_s2(x, u, dt_dummy, t_idx_dummy):
    """Wrapper for S2 state transition, estimating k4 directly."""
    q3, q4, v3, v4, k4_est = x # <--- k4_est is the 5th state
    u3, u4 = u

    # k4_est is used directly, no k4_ref multiplication
    a3 = (u3 - k4_est*(q3 - q4) - c4*(v3 - v4)) / M3
    a4 = (u4 + k4_est*(q3 - q4) + c4*(v3 - v4)) / M4

    x_new = np.zeros_like(x)
    x_new[0] = q3 + dt*v3
    x_new[1] = q4 + dt*v4
    x_new[2] = v3 + dt*a3
    x_new[3] = v4 + dt*a4
    x_new[4] = k4_est # Random walk for k4_est
    return x_new

def h_s2(x, u, dt_dummy, t_idx_dummy):
    """Wrapper for S2 measurement (a3, a4), estimating k4 directly."""
    q3, q4, v3, v4, k4_est = x # <--- k4_est is the 5th state
    u3, u4 = u

    # k4_est is used directly
    a3 = (u3 - k4_est*(q3 - q4) - c4*(v3 - v4)) / M3
    a4 = (u4 + k4_est*(q3 - q4) + c4*(v3 - v4)) / M4
    return np.array([a3, a4])


# ============================================================
#  Noise covariances for distributed filters (UNNORMALIZED)
# ============================================================
Q1 = 1e-18 * np.eye(4)   # (4x4)

# Process noise for k4_est.
Q2 = 1e-18 *np.eye(5) 

# Measurement noise for local acceleration pairs.
R1 = 1e-2 * np.eye(2)
R2 = 1e-2 * np.eye(2)
gamma_param = 0.0 

# ============================================================
#  Distributed UKF run
# ============================================================
s1 = np.array([0.01, 0.0, 0.01, 0.0])   # [x1,x2,v1,v2]

k4_init_dist = 30_000.0
# s2 state is now [x3, x4, v3, v4, k4_est]
s2 = np.array([0.0, 0.0, 0.0, 0.0, k4_init_dist]) 

P1 = 1e-4 * np.eye(4)
# P2 stores the initial uncertainty for k4_est.
P2_k4_std =  k4_ref # ~25000.0, half the expected value
# Initial variance for k4_est.
P2 = np.diag([1e-4, 1e-4, 1e-4, 1e-4, P2_k4_std**2]) 
# ============================================================
#  Distributed UKF run (BOTH scenarios in the SAME loop)
#   A) Deterministic Jacobi
#   B) Probabilistic Jacobi
# ============================================================

# -------- init (deterministic) --------
x1_det = np.array([0.01, 0.0, 0.01, 0.0])
x2_det = np.array([0.0, 0.0, 0.0, 0.0, 30_000.0])

P1_det = 1e-4 * np.eye(4)
P2_det = np.diag([1e-4, 1e-4, 1e-4, 1e-4, (k4_ref**2)])

Q_est_det = np.zeros((N+1, 4))
V_est_det = np.zeros((N+1, 4))
k4_est_det = np.zeros(N+1)
state_var_det = np.zeros((N+1, 8))
k4_var_det = np.zeros(N+1)
state_pred_nll_det = np.full(N+1, np.nan)

Fb_mean_det = np.zeros(N+1)
Fb_var_used_det = np.zeros(N+1)  # always 0 in deterministic

Q_est_det[0] = [x1_det[0], x1_det[1], x2_det[0], x2_det[1]]
V_est_det[0] = [x1_det[2], x1_det[3], x2_det[2], x2_det[3]]
k4_est_det[0] = x2_det[4]
state_var_det[0] = build_full_state_var_from_subsystems(P1_det, P2_det)
k4_var_det[0] = P2_det[4, 4]

# -------- init (probabilistic) --------
x1_prob = np.array([0.01, 0.0, 0.01, 0.0])
x2_prob = np.array([0.0, 0.0, 0.0, 0.0, 30_000.0])

P1_prob = 1e-4 * np.eye(4)
P2_prob = np.diag([1e-4, 1e-4, 1e-4, 1e-4, (k4_ref**2)])

Q_est_prob = np.zeros((N+1, 4))
V_est_prob = np.zeros((N+1, 4))
k4_est_prob = np.zeros(N+1)
state_var_prob = np.zeros((N+1, 8))

Fb_mean_prob = np.zeros(N+1)
Fb_var_used_prob = np.zeros(N+1)

state_pred_nll_prob = np.full(N+1, np.nan)
force_pred_nll_prob = np.full(N+1, np.nan)
k4_pred_nll_prob = np.full(N+1, np.nan)

Q_est_prob[0] = [x1_prob[0], x1_prob[1], x2_prob[0], x2_prob[1]]
V_est_prob[0] = [x1_prob[2], x1_prob[3], x2_prob[2], x2_prob[3]]
k4_est_prob[0] = x2_prob[4]
state_var_prob[0] = build_full_state_var_from_subsystems(P1_prob, P2_prob)
P2_var_prob = np.zeros((N+1, 5))
P2_var_prob[0] = np.diag(P2_prob)


var_Fb_prev = 0.0  # only for probabilistic incremental variance

# -------- init (SINDy surrogate message passing) --------
x1_sindy = np.array([0.01, 0.0, 0.01, 0.0])
x2_sindy = np.array([0.0, 0.0, 0.0, 0.0, 30_000.0])

P1_sindy = 1e-4 * np.eye(4)
P2_sindy = np.diag([1e-4, 1e-4, 1e-4, 1e-4, (k4_ref**2)])

Q_est_sindy = np.zeros((N+1, 4))
V_est_sindy = np.zeros((N+1, 4))
k4_est_sindy = np.zeros(N+1)

Fb_mean_sindy = np.zeros(N+1)

state_pred_nll_sindy = np.full(N+1, np.nan)
force_pred_nll_sindy = np.full(N+1, np.nan)
k4_pred_nll_sindy = np.full(N+1, np.nan)

Q_est_sindy[0] = [x1_sindy[0], x1_sindy[1], x2_sindy[0], x2_sindy[1]]
V_est_sindy[0] = [x1_sindy[2], x1_sindy[3], x2_sindy[2], x2_sindy[3]]
k4_est_sindy[0] = x2_sindy[4]


for k in range(N):
    f1, f2, f3, f4 = u_hist[k]

    # measurements at k+1 (shared)
    z1 = a_meas_hist[k+1, 0:2]  # [a1, a2]
    z2 = a_meas_hist[k+1, 2:4]  # [a3, a4]

    # =========================================================
    # A) Deterministic Jacobi
    # =========================================================
    u1_det, u2_det = build_sub_inputs_jacobi(x1_det, x2_det, f1, f2, f3, f4)

    # deterministic Fb mean for plotting consistency
    x2i, v2i = x1_det[1], x1_det[3]
    x3i, v3i = x2_det[0], x2_det[2]
    Fb_det = k3*(x2i - x3i) + c3*(v2i - v3i)

    Fb_mean_det[k] = Fb_det
    Fb_var_used_det[k] = 0.0  # no uncertainty in deterministic case

    # UKF S1 det
    Q1_eff_det = Q1.copy()
    x1_det, P1_det, x1_det_pred, P1_det_pred, _, _ = unscented_kalman_filter_step(
        x_prev=x1_det, P_prev=P1_det,
        R_mat=R1, Q_mat=Q1_eff_det,
        gamma_param=gamma_param, y_meas_k=z1,
        t_idx=k+1, dt=dt,
        force_arr=None,
        tf_func=f_s1, mf_func=h_s1, u=u1_det
    )

    # UKF S2 det
    Q2_eff_det = Q2.copy()
    x2_det, P2_det, x2_det_pred, P2_det_pred, _, _ = unscented_kalman_filter_step(
        x_prev=x2_det, P_prev=P2_det,
        R_mat=R2, Q_mat=Q2_eff_det,
        gamma_param=gamma_param, y_meas_k=z2,
        t_idx=k+1, dt=dt,
        force_arr=None,
        tf_func=f_s2, mf_func=h_s2, u=u2_det
    )

    Q_est_det[k+1] = [x1_det[0], x1_det[1], x2_det[0], x2_det[1]]
    V_est_det[k+1] = [x1_det[2], x1_det[3], x2_det[2], x2_det[3]]
    k4_est_det[k+1] = x2_det[4]
    state_var_det[k+1] = build_full_state_var_from_subsystems(P1_det, P2_det)
    k4_var_det[k+1] = P2_det[4, 4]

    x_det_pred_full = build_full_state_from_subsystems(x1_det_pred, x2_det_pred)
    P_det_pred_full = build_full_cov_from_subsystems(P1_det_pred, P2_det_pred)
    x_true_k = np.hstack([Q_true[k+1], V_true[k+1]])
    state_pred_nll_det[k+1] = gaussian_nll_vector(x_true_k, x_det_pred_full, P_det_pred_full)

    # =========================================================
    # B) Probabilistic Jacobi
    # =========================================================
    u1_prob, u2_prob, mean_Fb, var_Fb = build_sub_inputs_jacobi_prob(
        x1_prob, P1_prob, x2_prob, P2_prob, f1, f2, f3, f4
    )
    # alpha = 0.99  # smoothing (0.01–0.2 typical)
    # var_Fb_used = (1 - alpha) * var_Fb_used + alpha * max(0.0, var_Fb)

    var_Fb_used = max(0.0, var_Fb - var_Fb_prev)
    var_Fb_prev = var_Fb

    Fb_mean_prob[k] = mean_Fb
    Fb_var_used_prob[k] = var_Fb_used

    # UKF S1 prob (inflate v2 noise)
    Q1_eff_prob = Q1.copy()
    Q1_eff_prob[3,3] += (dt / M2)**2 * var_Fb_used

    x1_prob, P1_prob, x1_prob_pred, P1_prob_pred, _, _ = unscented_kalman_filter_step(
        x_prev=x1_prob, P_prev=P1_prob,
        R_mat=R1, Q_mat=Q1_eff_prob,
        gamma_param=gamma_param, y_meas_k=z1,
        t_idx=k+1, dt=dt,
        force_arr=None,
        tf_func=f_s1, mf_func=h_s1, u=u1_prob
    )

    # UKF S2 prob (inflate v3 noise)
    Q2_eff_prob = Q2.copy()
    Q2_eff_prob[2,2] += (dt / M3)**2 * var_Fb_used

    x2_prob, P2_prob, x2_prob_pred, P2_prob_pred, _, _ = unscented_kalman_filter_step(
        x_prev=x2_prob, P_prev=P2_prob,
        R_mat=R2, Q_mat=Q2_eff_prob,
        gamma_param=gamma_param, y_meas_k=z2,
        t_idx=k+1, dt=dt,
        force_arr=None,
        tf_func=f_s2, mf_func=h_s2, u=u2_prob
    )

    Q_est_prob[k+1] = [x1_prob[0], x1_prob[1], x2_prob[0], x2_prob[1]]
    V_est_prob[k+1] = [x1_prob[2], x1_prob[3], x2_prob[2], x2_prob[3]]
    k4_est_prob[k+1] = x2_prob[4]
    state_var_prob[k+1] = build_full_state_var_from_subsystems(P1_prob, P2_prob)
    P2_var_prob[k+1] = np.diag(P2_prob)  # shape (5,)

    x_prob_pred_full = build_full_state_from_subsystems(x1_prob_pred, x2_prob_pred)
    P_prob_pred_full = build_full_cov_from_subsystems(P1_prob_pred, P2_prob_pred)
    Fb_prob_pred_mean = (
        k3 * (x_prob_pred_full[1] - x_prob_pred_full[2]) +
        c3 * (x_prob_pred_full[5] - x_prob_pred_full[6])
    )
    force_map = np.array([0.0, k3, -k3, 0.0, 0.0, c3, -c3, 0.0], dtype=float)
    Fb_prob_pred_var = float(force_map @ P_prob_pred_full @ force_map)
    Fb_true_k = k3 * (Q_true[k+1, 1] - Q_true[k+1, 2]) + c3 * (V_true[k+1, 1] - V_true[k+1, 2])
    x_true_k = np.hstack([Q_true[k+1], V_true[k+1]])

    state_pred_nll_prob[k+1] = gaussian_nll_vector(x_true_k, x_prob_pred_full, P_prob_pred_full)
    force_pred_nll_prob[k+1] = gaussian_nll_scalar(Fb_true_k, Fb_prob_pred_mean, Fb_prob_pred_var)
    k4_pred_nll_prob[k+1] = gaussian_nll_scalar(k4, x2_prob_pred[4], P2_prob_pred[4, 4])

        # =========================================================
    # C) SINDy surrogate message passing (deterministic)
    # =========================================================
    u1_sindy, u2_sindy, Fb_hat = build_sub_inputs_jacobi_sindy(
        x1_sindy, x2_sindy, f1, f2, f3, f4
    )
    Fb_mean_sindy[k] = Fb_hat

    # UKF S1 (SINDy)
    Q1_eff_sindy = Q1.copy()
    x1_sindy, P1_sindy, x1_sindy_pred, P1_sindy_pred, _, _ = unscented_kalman_filter_step(
        x_prev=x1_sindy, P_prev=P1_sindy,
        R_mat=R1, Q_mat=Q1_eff_sindy,
        gamma_param=gamma_param, y_meas_k=z1,
        t_idx=k+1, dt=dt,
        force_arr=None,
        tf_func=f_s1, mf_func=h_s1, u=u1_sindy
    )

    # UKF S2 (SINDy)
    Q2_eff_sindy = Q2.copy()
    x2_sindy, P2_sindy, x2_sindy_pred, P2_sindy_pred, _, _ = unscented_kalman_filter_step(
        x_prev=x2_sindy, P_prev=P2_sindy,
        R_mat=R2, Q_mat=Q2_eff_sindy,
        gamma_param=gamma_param, y_meas_k=z2,
        t_idx=k+1, dt=dt,
        force_arr=None,
        tf_func=f_s2, mf_func=h_s2, u=u2_sindy
    )

    Q_est_sindy[k+1] = [x1_sindy[0], x1_sindy[1], x2_sindy[0], x2_sindy[1]]
    V_est_sindy[k+1] = [x1_sindy[2], x1_sindy[3], x2_sindy[2], x2_sindy[3]]
    k4_est_sindy[k+1] = x2_sindy[4]

    x_sindy_pred_full = build_full_state_from_subsystems(x1_sindy_pred, x2_sindy_pred)
    P_sindy_pred_full = build_full_cov_from_subsystems(P1_sindy_pred, P2_sindy_pred)
    interface_mean_sindy, interface_cov_sindy = build_interface_state_stats(
        x1_sindy_pred, P1_sindy_pred, x2_sindy_pred, P2_sindy_pred
    )
    Fb_sindy_pred_mean, Fb_sindy_pred_var = sigma_point_scalar_moments(
        interface_mean_sindy, interface_cov_sindy, surrogate_force_from_interface, gamma_param=gamma_param
    )

    state_pred_nll_sindy[k+1] = gaussian_nll_vector(x_true_k, x_sindy_pred_full, P_sindy_pred_full)
    force_pred_nll_sindy[k+1] = gaussian_nll_scalar(Fb_true_k, Fb_sindy_pred_mean, Fb_sindy_pred_var)
    k4_pred_nll_sindy[k+1] = gaussian_nll_scalar(k4, x2_sindy_pred[4], P2_sindy_pred[4, 4])



# ============================================================
#  Central 4-DOF + k4 model and UKF
# ============================================================
# Central system state: [q1..q4, v1..v4, k4_est] (9 states)
# Central measurement: [a1, a2, a3, a4] (4 measurements)

def accel_central_param(q, v, u, k4_est):
    """Monolithic acceleration with k4_est used directly."""
    # k4_est is used directly, no k4_ref division/multiplication
    K = np.array([[k1+k2,   -k2,        0,        0],
                  [  -k2, k2+k3,      -k3,       0],
                  [   0,   -k3,  k3+k4_est, -k4_est],
                  [   0,     0,   -k4_est,  k4_est]], float)
    return Minv_mono @ (u - Cmono@v - K@q)

def f_central(x, u, dt_dummy, t_idx_dummy):
    """Wrapper for Central state transition, estimating k4 directly."""
    q = x[:4]
    v = x[4:8]
    k4_est = x[8] # <--- k4_est is the 9th state

    a = accel_central_param(q, v, u, k4_est)

    x_new = np.zeros_like(x)
    x_new[:4]  = q + dt*v
    x_new[4:8] = v + dt*a
    x_new[8]   = k4_est # Random walk for k4_est
    return x_new

def h_central(x, u, dt_dummy, t_idx_dummy):
    """Wrapper for Central measurement (a1, a2, a3, a4), estimating k4 directly."""
    q = x[:4]
    v = x[4:8]
    k4_est = x[8] # <--- k4_est is the 9th state
    a = accel_central_param(q, v, u, k4_est)
    return a # Return all 4 accelerations

# Central Noise Covariances (UNNORMALIZED)
R_c = 1e-1 * np.eye(4) 
Q_c = 1e-8*np.eye(9)

# initial central state
x_c = np.zeros(9)
x_c[0] = 0.01
x_c[4] = 0.01
k4_init_c = 30_000.0
x_c[8] = k4_init_c # <--- Initialize k4_est directly

P_c_k4_std =   k4_ref
P_c = np.diag(np.concatenate([1e-4*np.ones(8), (P_c_k4_std**2,)])) # Set parameter initial variance

x_c_hist  = np.zeros((N+1,9))
x_c_hist[0] = x_c
k4_c_hist = np.zeros(N+1)
k4_c_hist[0] = x_c[8] # k4 is saved directly

state_pred_nll_cent = np.full(N+1, np.nan)
force_pred_nll_cent = np.full(N+1, np.nan)
k4_pred_nll_cent = np.full(N+1, np.nan)
state_var_cent = np.zeros((N+1, 8))

current_xc = x_c.copy()
current_Pc = P_c.copy()
Pc_saved = np.zeros((N+1, 9))
Pc_saved[0] = np.diag(P_c)
state_var_cent[0] = np.diag(P_c[:8, :8])

for k in range(N):
    u = u_hist[k]
    z_c = a_meas_hist[k+1, :] # [a1, a2, a3, a4]

    current_xc, current_Pc, x_c_pred, P_c_pred, _, _ = unscented_kalman_filter_step(
        x_prev=current_xc, P_prev=current_Pc, R_mat=R_c, Q_mat=Q_c,
        gamma_param=gamma_param, y_meas_k=z_c, t_idx=k+1, dt=dt,
        force_arr=None, # Dummy
        tf_func=f_central, mf_func=h_central, u=u # u is passed explicitly
    )

    x_c_hist[k+1] = current_xc
    k4_c_hist[k+1] = current_xc[8] # k4 is saved directly
    Pc_saved[k+1] = np.diag(current_Pc)
    state_var_cent[k+1] = np.diag(current_Pc[:8, :8])

    x_true_k = np.hstack([Q_true[k+1], V_true[k+1]])
    Fb_true_k = k3 * (Q_true[k+1, 1] - Q_true[k+1, 2]) + c3 * (V_true[k+1, 1] - V_true[k+1, 2])
    Fb_cent_pred_mean = k3 * (x_c_pred[1] - x_c_pred[2]) + c3 * (x_c_pred[5] - x_c_pred[6])
    force_map = np.array([0.0, k3, -k3, 0.0, 0.0, c3, -c3, 0.0], dtype=float)
    Fb_cent_pred_var = float(force_map @ P_c_pred[:8, :8] @ force_map)

    state_pred_nll_cent[k+1] = gaussian_nll_vector(x_true_k, x_c_pred[:8], P_c_pred[:8, :8])
    force_pred_nll_cent[k+1] = gaussian_nll_scalar(Fb_true_k, Fb_cent_pred_mean, Fb_cent_pred_var)
    k4_pred_nll_cent[k+1] = gaussian_nll_scalar(k4, x_c_pred[8], P_c_pred[8, 8])

Q_est_cent = x_c_hist[:, :4]
V_est_cent = x_c_hist[:, 4:8]

# ============================================================
#  Plots: states
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

for i in range(4):
    axL = axes[i, 0]
    axL.plot(t, Q_true[:, i], label='Truth')
    axL.plot(t, Q_est_cent[:, i], '--', label='Central UKF')
    axL.plot(t, Q_est_det[:, i], ':', label='Distributed UKF')
    axL.set_ylabel(f'x{i+1} [m]')
    if i == 0:
        axL.legend(ncol=3, frameon=False, loc='upper right')
    axL.grid(True, alpha=0.2)

    axR = axes[i, 1]
    axR.plot(t, V_true[:, i], label='Truth')
    axR.plot(t, V_est_cent[:, i], '--', label='Central UKF')
    axR.plot(t, V_est_prob[:, i], ':', label='Distributed UKF')
    axR.set_ylabel(f'v{i+1} [m/s]')
    if i == 0:
        axR.legend(ncol=3, frameon=False, loc='upper right')
    axR.grid(True, alpha=0.2)

axes[-1,0].set_xlabel('time [s]')
axes[-1,1].set_xlabel('time [s]')
axes[0,0].set_title('Displacement')
axes[0,1].set_title('Velocity')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------

# ============================================================
#  Plots: k4 estimates
# ============================================================
plt.figure(figsize=(8,4))
plt.plot(t, k4_c_hist, label=r'Central UKF $\hat{k}_4$')
plt.plot(t, k4_est_prob, label=r'Distributed UKF $\hat{k}_4$')
plt.axhline(k4, color='k', linestyle='--', label=r'$k_4$ true')
plt.xlabel('time [s]')
plt.ylabel('$k_4$ [N/m]')
plt.grid(True, alpha=0.2)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns

def build_4dof_summary_tables(
    t, N,
    t_max,
    k4,                      # true scalar
    k4_c_hist, k4_est_prob, k4_est_sindy,
    P2_var_prob, Pc_saved,   # arrays used for k4 sigmas
    Fb_true,
    Fb_mean_prob, Fb_var_used_prob,
    Fb_mean_sindy,
):
    # Plot indices.
    idx_max = np.searchsorted(t, t_max, side="right")
    t8 = t[:idx_max]

    t_plot = t[:N]
    tFb8 = t_plot[:min(N, idx_max)]
    Lfb = len(tFb8)

    # k4 1-sigma band.
    k4_std_prob = np.sqrt(P2_var_prob[:idx_max, 4])
    k4_c_std    = np.sqrt(Pc_saved[:idx_max, 8])

    df_k4 = pd.DataFrame({
        "t_s": t8,

        "k4_true_const": np.full_like(t8, float(k4), dtype=float),

        "k4_central_est": k4_c_hist[:idx_max],
        "k4_central_std": k4_c_std,
        "k4_central_lo":  k4_c_hist[:idx_max] - 1.0 * k4_c_std,
        "k4_central_hi":  k4_c_hist[:idx_max] + 1.0 * k4_c_std,

        "k4_prob_est": k4_est_prob[:idx_max],
        "k4_prob_std": k4_std_prob,
        "k4_prob_lo":  k4_est_prob[:idx_max] - 1.0 * k4_std_prob,
        "k4_prob_hi":  k4_est_prob[:idx_max] + 1.0 * k4_std_prob,

        "k4_sindy_est": k4_est_sindy[:idx_max],
    })

    # Interface-force mean and 1-sigma band.
    Fb_std_prob = np.sqrt(Fb_var_used_prob)

    df_fb = pd.DataFrame({
        "t_s": tFb8,

        "Fb_true":      Fb_true[:Lfb],

        "Fb_prob_mean": Fb_mean_prob[:Lfb],
        "Fb_prob_std":  Fb_std_prob[:Lfb],
        "Fb_prob_lo":   Fb_mean_prob[:Lfb] - 1.0 * Fb_std_prob[:Lfb],
        "Fb_prob_hi":   Fb_mean_prob[:Lfb] + 1.0 * Fb_std_prob[:Lfb],

        "Fb_sindy":     Fb_mean_sindy[:Lfb],
    })

    return df_k4, df_fb


colors  = sns.color_palette("crest", n_colors=8)
colors1 = sns.color_palette("colorblind", n_colors=8)
# ============================================================
#  Final comparison plot (0–8 s): k4 + Fb for THREE scenarios
#   Central UKF, probabilistic Jacobi, and SINDy message passing
# ============================================================

Fb_true = (
    k3 * (Q_true[:,1] - Q_true[:,2]) +
    c3 * (V_true[:,1] - V_true[:,2])
)

x_est_cent = np.hstack([Q_est_cent, V_est_cent])
x_est_det = np.hstack([Q_est_det, V_est_det])
x_est_prob = np.hstack([Q_est_prob, V_est_prob])
x_est_sindy = np.hstack([Q_est_sindy, V_est_sindy])
x_true_full = np.hstack([Q_true, V_true])

Fb_central = (
    k3 * (Q_est_cent[:,1] - Q_est_cent[:,2]) +
    c3 * (V_est_cent[:,1] - V_est_cent[:,2])
)

# Fill the final force samples so the RMSE window uses valid end-of-run values.
Fb_mean_prob[-1] = (
    k3 * (x1_prob[1] - x2_prob[0]) +
    c3 * (x1_prob[3] - x2_prob[2])
)
Fb_mean_sindy[-1] = (
    xi0 * (x1_sindy[1] - x2_sindy[0]) +
    xi1 * (x1_sindy[3] - x2_sindy[2]) +
    xi2 * ((x1_sindy[1] - x2_sindy[0]) ** 3) +
    xi3 * (np.abs(x1_sindy[3] - x2_sindy[2]) * (x1_sindy[3] - x2_sindy[2]))
)

def rmse_window(truth, estimate, mask):
    diff = np.asarray(estimate)[mask] - np.asarray(truth)[mask]
    return np.sqrt(np.mean(diff**2, axis=0))

def nrmse_window(truth, estimate, mask):
    truth_window = np.asarray(truth)[mask]
    rmse_val = rmse_window(truth, estimate, mask)
    truth_range = np.max(truth_window, axis=0) - np.min(truth_window, axis=0)
    truth_range = np.where(truth_range > 0.0, truth_range, 1.0)
    return rmse_val / truth_range

def normalized_parameter_rmse_window(true_value, estimate, mask):
    rmse_val = rmse_window(
        np.full_like(np.asarray(estimate), float(true_value), dtype=float),
        estimate,
        mask
    )
    scale = abs(float(true_value))
    if scale == 0.0:
        scale = 1.0
    return rmse_val / scale


def empirical_coverage_window(truth, estimate, variance, mask, sigma_multiplier):
    truth_window = np.asarray(truth)[mask]
    estimate_window = np.asarray(estimate)[mask]
    std_window = np.sqrt(np.maximum(np.asarray(variance)[mask], 0.0))
    covered = np.abs(estimate_window - truth_window) <= sigma_multiplier * std_window
    return float(np.mean(covered))

window_start_time = max(0.0, T - rmse_window_sec)
rmse_mask = t >= window_start_time
param_window_start_time = max(0.0, T - param_rmse_window_sec)
param_rmse_mask = t >= param_window_start_time
state_labels = ['q1', 'q2', 'q3', 'q4', 'v1', 'v2', 'v3', 'v4']

state_rmse_central_each = rmse_window(x_true_full, x_est_cent, rmse_mask)
state_rmse_det_each = rmse_window(x_true_full, x_est_det, rmse_mask)
state_rmse_prob_each = rmse_window(x_true_full, x_est_prob, rmse_mask)
state_rmse_sindy_each = rmse_window(x_true_full, x_est_sindy, rmse_mask)

state_nrmse_central_each = nrmse_window(x_true_full, x_est_cent, rmse_mask)
state_nrmse_det_each = nrmse_window(x_true_full, x_est_det, rmse_mask)
state_nrmse_prob_each = nrmse_window(x_true_full, x_est_prob, rmse_mask)
state_nrmse_sindy_each = nrmse_window(x_true_full, x_est_sindy, rmse_mask)

state_rmse_central = float(np.sqrt(np.mean((x_est_cent[rmse_mask] - x_true_full[rmse_mask])**2)))
state_rmse_det = float(np.sqrt(np.mean((x_est_det[rmse_mask] - x_true_full[rmse_mask])**2)))
state_rmse_prob = float(np.sqrt(np.mean((x_est_prob[rmse_mask] - x_true_full[rmse_mask])**2)))
state_rmse_sindy = float(np.sqrt(np.mean((x_est_sindy[rmse_mask] - x_true_full[rmse_mask])**2)))

state_nrmse_central = float(np.mean(state_nrmse_central_each))
state_nrmse_det = float(np.mean(state_nrmse_det_each))
state_nrmse_prob = float(np.mean(state_nrmse_prob_each))
state_nrmse_sindy = float(np.mean(state_nrmse_sindy_each))

force_rmse_central = float(rmse_window(Fb_true, Fb_central, rmse_mask))
force_rmse_prob = float(rmse_window(Fb_true, Fb_mean_prob, rmse_mask))
force_rmse_sindy = float(rmse_window(Fb_true, Fb_mean_sindy, rmse_mask))

force_nrmse_central = float(nrmse_window(Fb_true, Fb_central, rmse_mask))
force_nrmse_prob = float(nrmse_window(Fb_true, Fb_mean_prob, rmse_mask))
force_nrmse_sindy = float(nrmse_window(Fb_true, Fb_mean_sindy, rmse_mask))

k4_truth = np.full_like(k4_c_hist, float(k4), dtype=float)
k4_rmse_central = float(rmse_window(k4_truth, k4_c_hist, param_rmse_mask))
k4_rmse_det = float(rmse_window(k4_truth, k4_est_det, param_rmse_mask))
k4_rmse_prob = float(rmse_window(k4_truth, k4_est_prob, param_rmse_mask))
k4_rmse_sindy = float(rmse_window(k4_truth, k4_est_sindy, param_rmse_mask))

k4_nrmse_central = float(normalized_parameter_rmse_window(k4, k4_c_hist, param_rmse_mask))
k4_nrmse_det = float(normalized_parameter_rmse_window(k4, k4_est_det, param_rmse_mask))
k4_nrmse_prob = float(normalized_parameter_rmse_window(k4, k4_est_prob, param_rmse_mask))
k4_nrmse_sindy = float(normalized_parameter_rmse_window(k4, k4_est_sindy, param_rmse_mask))

state_nll_mask = rmse_mask & np.isfinite(state_pred_nll_cent) & np.isfinite(state_pred_nll_det) & np.isfinite(state_pred_nll_prob) & np.isfinite(state_pred_nll_sindy)
force_nll_mask = rmse_mask & np.isfinite(force_pred_nll_cent) & np.isfinite(force_pred_nll_prob) & np.isfinite(force_pred_nll_sindy)
k4_nll_mask = param_rmse_mask & np.isfinite(k4_pred_nll_cent) & np.isfinite(k4_pred_nll_prob) & np.isfinite(k4_pred_nll_sindy)

state_pred_nll_avg_cent = float(np.mean(state_pred_nll_cent[state_nll_mask]))
state_pred_nll_avg_det = float(np.mean(state_pred_nll_det[state_nll_mask]))
state_pred_nll_avg_prob = float(np.mean(state_pred_nll_prob[state_nll_mask]))
state_pred_nll_avg_sindy = float(np.mean(state_pred_nll_sindy[state_nll_mask]))

force_pred_nll_avg_cent = float(np.mean(force_pred_nll_cent[force_nll_mask]))
force_pred_nll_avg_prob = float(np.mean(force_pred_nll_prob[force_nll_mask]))
force_pred_nll_avg_sindy = float(np.mean(force_pred_nll_sindy[force_nll_mask]))

k4_pred_nll_avg_cent = float(np.mean(k4_pred_nll_cent[k4_nll_mask]))
k4_pred_nll_avg_prob = float(np.mean(k4_pred_nll_prob[k4_nll_mask]))
k4_pred_nll_avg_sindy = float(np.mean(k4_pred_nll_sindy[k4_nll_mask]))

coverage_68_cent = empirical_coverage_window(x_true_full, x_est_cent, state_var_cent, rmse_mask, 1.0)
coverage_68_det = empirical_coverage_window(x_true_full, x_est_det, state_var_det, rmse_mask, 1.0)
coverage_68_prob = empirical_coverage_window(x_true_full, x_est_prob, state_var_prob, rmse_mask, 1.0)

coverage_95_cent = empirical_coverage_window(x_true_full, x_est_cent, state_var_cent, rmse_mask, 1.959963984540054)
coverage_95_det = empirical_coverage_window(x_true_full, x_est_det, state_var_det, rmse_mask, 1.959963984540054)
coverage_95_prob = empirical_coverage_window(x_true_full, x_est_prob, state_var_prob, rmse_mask, 1.959963984540054)

print("\n" + "=" * 72)
print(f"RMSE and NRMSE over last {rmse_window_sec:.3f} s window (t >= {window_start_time:.3f} s)")
print("NRMSE is normalized by the ground-truth range over the same window.")
print("=" * 72)
print(f"State, Central      : RMSE = {state_rmse_central:.6e}, NRMSE = {state_nrmse_central:.6e}")
print(f"State, Det-Jacobi   : RMSE = {state_rmse_det:.6e}, NRMSE = {state_nrmse_det:.6e}")
print(f"State, Prob-Jacobi  : RMSE = {state_rmse_prob:.6e}, NRMSE = {state_nrmse_prob:.6e}")
print(f"State, Surrogate    : RMSE = {state_rmse_sindy:.6e}, NRMSE = {state_nrmse_sindy:.6e}")
print(f"Force, Central      : RMSE = {force_rmse_central:.6e}, NRMSE = {force_nrmse_central:.6e}")
print(f"Force, Prob-Jacobi  : RMSE = {force_rmse_prob:.6e}, NRMSE = {force_nrmse_prob:.6e}")
print(f"Force, Surrogate    : RMSE = {force_rmse_sindy:.6e}, NRMSE = {force_nrmse_sindy:.6e}")
print("-" * 72)
print("Per-state RMSE and NRMSE over the same window:")
for label, rmse_c, rmse_p, rmse_s, nrmse_c, nrmse_p, nrmse_s in zip(
    state_labels,
    state_rmse_central_each,
    state_rmse_prob_each,
    state_rmse_sindy_each,
    state_nrmse_central_each,
    state_nrmse_prob_each,
    state_nrmse_sindy_each
):
    print(
        f"{label:>2} | Central: RMSE={rmse_c:.6e}, NRMSE={nrmse_c:.6e} | "
        f"Prob-Jacobi: RMSE={rmse_p:.6e}, NRMSE={nrmse_p:.6e} | "
        f"Surrogate: RMSE={rmse_s:.6e}, NRMSE={nrmse_s:.6e}"
    )
print("-" * 72)
print(
    f"k4 parameter RMSE and NRMSE over last {param_rmse_window_sec:.3f} s "
    f"window (t >= {param_window_start_time:.3f} s)"
)
print("Parameter NRMSE is normalized by the true parameter value |k4|.")
print(f"k4, Central      : RMSE = {k4_rmse_central:.6e}, NRMSE = {k4_nrmse_central:.6e}")
print(f"k4, Det-Jacobi   : RMSE = {k4_rmse_det:.6e}, NRMSE = {k4_nrmse_det:.6e}")
print(f"k4, Prob-Jacobi  : RMSE = {k4_rmse_prob:.6e}, NRMSE = {k4_nrmse_prob:.6e}")
print(f"k4, Surrogate    : RMSE = {k4_rmse_sindy:.6e}, NRMSE = {k4_nrmse_sindy:.6e}")
print("-" * 72)
print(f"Empirical state coverage over last {rmse_window_sec:.3f} s window")
print("Coverage is computed marginally across all 8 state components.")
print(
    f"68% nominal, Central      : {coverage_68_cent:.6f}\n"
    f"68% nominal, Det-Jacobi   : {coverage_68_det:.6f}\n"
    f"68% nominal, Prob-Jacobi  : {coverage_68_prob:.6f}"
)
print(
    f"95% nominal, Central      : {coverage_95_cent:.6f}\n"
    f"95% nominal, Det-Jacobi   : {coverage_95_det:.6f}\n"
    f"95% nominal, Prob-Jacobi  : {coverage_95_prob:.6f}"
)
print("-" * 72)
print(
    f"Time-averaged predictive NLL over last {rmse_window_sec:.3f} s "
    f"for states/force and last {param_rmse_window_sec:.3f} s for k4"
)
print("Lower predictive NLL is better.")
print(
    f"State predictive NLL, Central      : {state_pred_nll_avg_cent:.6e}\n"
    f"State predictive NLL, Det-Jacobi   : {state_pred_nll_avg_det:.6e}\n"
    f"State predictive NLL, Prob-Jacobi  : {state_pred_nll_avg_prob:.6e}\n"
    f"State predictive NLL, Surrogate    : {state_pred_nll_avg_sindy:.6e}"
)
print(
    f"Force predictive NLL, Central      : {force_pred_nll_avg_cent:.6e}\n"
    f"Force predictive NLL, Prob-Jacobi  : {force_pred_nll_avg_prob:.6e}\n"
    f"Force predictive NLL, Surrogate    : {force_pred_nll_avg_sindy:.6e}"
)
print(
    f"k4 predictive NLL, Central         : {k4_pred_nll_avg_cent:.6e}\n"
    f"k4 predictive NLL, Prob-Jacobi     : {k4_pred_nll_avg_prob:.6e}\n"
    f"k4 predictive NLL, Surrogate       : {k4_pred_nll_avg_sindy:.6e}"
)
print("=" * 72)

t_max = 8.0
idx_max = np.searchsorted(t, t_max, side="right")

t8 = t[:idx_max]
t_plot = t[:N]
tFb8 = t_plot[:min(N, idx_max)]

Fb_std_prob = np.sqrt(Fb_var_used_prob)

fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True)

# --- (1) k4 estimates ---
ax = axes[0]
ax.plot(t8, k4_c_hist[:idx_max], color=colors[0], label='Central UKF $\\hat{k}_4$')
ax.plot(t8, k4_est_prob[:idx_max], '-', color=colors1[0], label='Prob-Jacobi $\\hat{k}_4$')
ax.plot(t8, k4_est_sindy[:idx_max], ':', color=colors1[3], linewidth=1.52, label='Surrogate-MP $\\hat{k}_4$')
ax.axhline(k4, color='k', linestyle='--', label=r'$k_4$ true')

# 1-sigma bands.
k4_std_prob = np.sqrt(P2_var_prob[:idx_max, 4])
ax.fill_between(
    t8,
    k4_est_prob[:idx_max] - 1.0 * k4_std_prob,
    k4_est_prob[:idx_max] + 1.0 * k4_std_prob,
    color=colors1[0],
    alpha=0.45,
    label=r'Prob-Jacobi: $\pm 1\sigma_{\hat{k}_4}$'
)

k4_c_std = np.sqrt(Pc_saved[:idx_max, 8])
ax.fill_between(
    t8,
    k4_c_hist[:idx_max] - 1.0 * k4_c_std,
    k4_c_hist[:idx_max] + 1.0 * k4_c_std,
    color=colors[0],
    alpha=0.18,
    label=r'Central: $\pm 1\sigma_{\hat{k}_4}$'
)

ax.set_ylabel('$k_4$ [N/m]')
ax.grid(False, alpha=0.2)
ax.legend(loc='upper right', frameon=False)
ax.set_ylim(25000, 75000)

# --- (2) Interface force Fb ---
ax = axes[1]
ax.plot(tFb8, Fb_true[:len(tFb8)], 'k', linestyle='--', label='True $e_{12}$')

# Probabilistic mean + band
ax.plot(tFb8, Fb_mean_prob[:len(tFb8)], ':', linewidth=2, color=colors[2], label='Prob-Jacobi $\\mu_{e_{12}}$')
ax.fill_between(
    tFb8,
    Fb_mean_prob[:len(tFb8)] - 1*Fb_std_prob[:len(tFb8)],
    Fb_mean_prob[:len(tFb8)] + 1*Fb_std_prob[:len(tFb8)],
    color=colors1[5],
    alpha=0.55,
    label=r'Prob-Jacobi $\pm 1\sigma_{e_{12}}$'
)

# SINDy MP force (deterministic)
ax.plot(tFb8, Fb_mean_sindy[:len(tFb8)], '-', linewidth=1.3, color=colors1[3], label='Surrogate $e_{12}$')

ax.set_xlabel('time [s]')
ax.set_ylabel('Interface force $e_{12}$ [N]')
ax.grid(False, alpha=0.2)
ax.legend(loc='upper right', frameon=False)
ax.set_ylim(-350, 350)

plt.tight_layout()
plt.show()

build_4dof_summary_tables(
    t=t, N=N,
    t_max=8.0,
    k4=k4,
    k4_c_hist=k4_c_hist,
    k4_est_prob=k4_est_prob,
    k4_est_sindy=k4_est_sindy,
    P2_var_prob=P2_var_prob,
    Pc_saved=Pc_saved,
    Fb_true=Fb_true,
    Fb_mean_prob=Fb_mean_prob,
    Fb_var_used_prob=Fb_var_used_prob,
    Fb_mean_sindy=Fb_mean_sindy,
)

#%%

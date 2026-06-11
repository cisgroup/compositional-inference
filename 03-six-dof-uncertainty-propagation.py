"""
Six-DOF uncertainty-propagation example.

The script extends the distributed UKF calculation by propagating interface
force uncertainty between neighboring subsystems and plotting the resulting
state and parameter estimates.
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import resample

# ============================================================
# TRUE PHYSICAL SYSTEM: 6-DOF (S1, S3, S2)
# ============================================================

# --- masses (true) ---
m1 = m2 = m3 = m4 = m5 = m6 = 500.0
m_star_true = 100.0            # unknown added mass on m3 (m*)
m3_tot_true = m3 + m_star_true

# --- springs (true) ---
k1 = 50000.0
k2 = 40000.0
k3_true = 45000.0              # unknown (S1)  m1–m2
k4 = 30000.0                   # coupling 2–3
k5 = 25000.0
k34_true = 20000.0                  # extra (parallel) spring between m3–m4, known
k6_true = 55000.0              # unknown (S3)  m4–ground
k7 = 30000.0                   # coupling 4–5
k8 = 45000.0
k9_true = 40000.0              # unknown (S2)  m5–m6

# --- dampers (true) ---
c1 = 300.0
c2 = 350.0
c3_true = 320.0                # unknown (S1)  m1–m2
c4 = 400.0                     # coupling 2–3
c5 = 480.0
c6_true = 700.0                # unknown (S3)  m4–ground
c7 = 350.0                     # coupling 4–5
c8 = 460.0
c9_true = 480.0                # unknown (S2)  m5–m6

# --- time & excitation ---
dt = 2e-3
T  = 50.0
N  = int(T/dt)

sigmaF = 1200.0     # std of random excitation force on m4
acc_std = 1e-3     # accel measurement noise std

# ============================================================
# BUILD TRUE MONOLITHIC M, K, C
# ============================================================

def build_MKC_true():
    M = np.diag([m1, m2, m3_tot_true, m4, m5, m6])
    K = np.zeros((6, 6))
    C = np.zeros((6, 6))

    # --- internal S1: m1,m2; k1,c1 to ground; k2,c2 to ground; k3 between 1–2 ---
    K[0,0] += k1 + k3_true
    K[0,1] += -k3_true
    K[1,0] += -k3_true
    K[1,1] += k2 + k3_true

    C[0,0] += c1 + c3_true
    C[0,1] += -c3_true
    C[1,0] += -c3_true
    C[1,1] += c2 + c3_true

    # --- internal S3: m3,m4; parallel (k5+k34,c5) between 3–4; k6,c6 from 4–ground ---
    k_34tot = k5 + k34_true
    K[2,2] += k_34tot
    K[2,3] += -k_34tot
    K[3,2] += -k_34tot
    K[3,3] += k_34tot + k6_true

    C[2,2] += c5
    C[2,3] += -c5
    C[3,2] += -c5
    C[3,3] += c5 + c6_true

    # --- internal S2: m5,m6; k8,c8 to ground at m5; k9,c9 between 5–6 ---
    K[4,4] += k8 + k9_true
    K[4,5] += -k9_true
    K[5,4] += -k9_true
    K[5,5] += k9_true

    C[4,4] += c8 + c9_true
    C[4,5] += -c9_true
    C[5,4] += -c9_true
    C[5,5] += c9_true

    # --- coupling 2–3: k4,c4 ---
    K[1,1] += k4
    K[1,2] += -k4
    K[2,1] += -k4
    K[2,2] += k4

    C[1,1] += c4
    C[1,2] += -c4
    C[2,1] += -c4
    C[2,2] += c4

    # --- coupling 4–5: k7,c7 ---
    K[3,3] += k7
    K[3,4] += -k7
    K[4,3] += -k7
    K[4,4] += k7

    C[3,3] += c7
    C[3,4] += -c7
    C[4,3] += -c7
    C[4,4] += c7

    return M, K, C

M_true, K_true, C_true = build_MKC_true()
Minv_true = np.diag(1.0 / np.diag(M_true))

def accel_true(q, v, f4_scalar):
    """True acceleration of 6-DOF system given excitation on DOF 4."""
    F = np.zeros(6)
    F[3] = f4_scalar
    return Minv_true @ (F - C_true @ v - K_true @ q)

# ============================================================
# TRUE SIMULATION
# ============================================================

rng = np.random.default_rng(123)
# forces = rng.normal(0.0, sigmaF, size=N)  # excitation on m4
mat = scipy.io.loadmat('data/elcentro.mat')
Forc = mat['e'][1]
freq = 1 / dt       # target frequency (Hz)
samples = T * freq
t = np.linspace(0, T, int(samples))
Forc_1 = resample(Forc, int(samples))
scaling_factor = 500.0 # Force scaling.
forces = Forc_1 * scaling_factor

Q_true = np.zeros((N+1, 6))
V_true = np.zeros((N+1, 6))
A_true = np.zeros((N+1, 6))

q = np.zeros(6)
v = np.zeros(6)
Q_true[0] = q
V_true[0] = v
A_true[0] = accel_true(q, v, forces[0])

for k in range(N):
    a = accel_true(q, v, forces[k])
    v = v + dt * a
    q = q + dt * v
    Q_true[k+1] = q
    V_true[k+1] = v
    A_true[k+1] = accel_true(q, v, forces[k])

# ============================================================
# MEASUREMENTS: accelerations of masses 4,2,5,3
# The local filters use a2, a3, a4, and a5.
# ============================================================

a1_meas = A_true[:,0] + rng.normal(0, acc_std, size=N+1)
a2_meas = A_true[:,1] + rng.normal(0, acc_std, size=N+1)  # S1
a3_meas = A_true[:,2] + rng.normal(0, acc_std, size=N+1)  # S3
a4_meas = A_true[:,3] + rng.normal(0, acc_std, size=N+1)  # S3
a5_meas = A_true[:,4] + rng.normal(0, acc_std, size=N+1)  # S2
a6_meas = A_true[:,5] + rng.normal(0, acc_std, size=N+1)

#%%
# #============================================================
# UNSCENTED KALMAN FILTER STEP
# ============================================================
from src.filters import unscented_kalman_filter_step as _shared_ukf_step


def unscented_kalman_filter_step(x_prev, P_prev, R_mat, Q_mat, gamma_param,
                                 y_meas_k, t_idx, dt, tf_func, mf_func, u):
    return _shared_ukf_step(
        x_prev, P_prev, R_mat, Q_mat, gamma_param, y_meas_k, dt,
        tf_func, mf_func, u, t_idx=t_idx, return_prediction=False,
        fallback_eigendecomp=True, scaled_cholesky=True,
    )

gamma_param = 0.0

# ============================================================
# SUBSYSTEM DYNAMICS & MEASUREMENTS
# ============================================================

# ---------- S1: [x1,x2,v1,v2,k3,c3] ----------

def accel_S1_state(x, u):
    """
    x = [x1,x2,v1,v2,k3_est,c3_est]
    u = [u1,u2] external forces on m1,m2
    """
    x1,x2,v1,v2,k3_est,c3_est = x
    u1,u2 = u
    k3_eff = max(k3_est, 1.0)
    c3_eff = max(c3_est, 1.0)

    a1 = (u1
          - k1*x1 - c1*v1
          - k3_eff*(x1 - x2)
          - c3_eff*(v1 - v2)) / m1

    a2 = (u2
          - k2*x2 - c2*v2
          + k3_eff*(x1 - x2)
          + c3_eff*(v1 - v2)) / m2

    return np.array([a1, a2])

def f_S1(x, u, dt, t_idx):
    x1,x2,v1,v2,k3_est,c3_est = x
    a1,a2 = accel_S1_state(x, u)
    x_new = np.zeros_like(x)
    x_new[0] = x1 + dt*v1
    x_new[1] = x2 + dt*v2
    x_new[2] = v1 + dt*a1
    x_new[3] = v2 + dt*a2
    x_new[4] = k3_est   # random walk
    x_new[5] = c3_est
    return x_new

def f_S1_heun(x, u, dt, t_idx):
    # Separate states and parameters
    x_dyn = x[:4]  # [x1, x2, v1, v2]
    p_est = x[4:]  # [k3_est, c3_est]
    
    # 1. Calculate f(x_k, t_k) = [v_k, a_k]
    v1, v2 = x_dyn[2], x_dyn[3]
    a1, a2 = accel_S1_state(x, u)
    f_k = np.array([v1, v2, a1, a2])

    # 2. Predictor Step: x_tilde = x_k + dt * f_k
    x_dyn_pred = x_dyn + dt * f_k
    
    # Construct predicted state vector for acceleration calculation
    x_pred = np.hstack((x_dyn_pred, p_est))
    
    # 3. Calculate f(x_tilde, t_k+1) = [v_tilde, a_tilde]
    v1_pred, v2_pred = x_dyn_pred[2], x_dyn_pred[3]
    a1_pred, a2_pred = accel_S1_state(x_pred, u)
    f_pred = np.array([v1_pred, v2_pred, a1_pred, a2_pred])

    # 4. Corrector Step: x_k+1 = x_k + (dt/2) * (f_k + f_pred)
    x_dyn_new = x_dyn + 0.5 * dt * (f_k + f_pred)
    
    # 5. Parameter update (Random Walk)
    x_new = np.hstack((x_dyn_new, p_est))
    return x_new

def h_S1(x, u, dt, t_idx):
    _, a2 = accel_S1_state(x, u)
    return np.array([a2])      # accel of mass 2


# ---------- S3: [x3,x4,v3,v4,m*,k6,c6] ----------

def accel_S3_state(x, u):
    """
    x = [x3,x4,v3,v4,m_star_est,k6_est,c6_est,k34_est]
    u = [u3,u4]  (F23,  f4 - F45)
    """
    x3,x4,v3,v4,m_star_est,k6_est,c6_est,k34_est = x
    u3,u4 = u

    m_star_eff = max(m_star_est, 1.0)
    M3_eff = m3 + m_star_eff

    k6_eff = max(k6_est, 1.0)
    c6_eff = max(c6_est, 1.0)

    k34_eff  = max(k34_est, 1.0)
    k_34tot  = k5 + k34_eff

    a3 = (u3
          - k_34tot*(x3 - x4)
          - c5*(v3 - v4)) / M3_eff

    a4 = (u4
          + k_34tot*(x3 - x4)
          + c5*(v3 - v4)
          - k6_eff*x4
          - c6_eff*v4) / m4

    return np.array([a3, a4])

def f_S3(x, u, dt, t_idx):
    x3,x4,v3,v4,m_star_est,k6_est,c6_est,k34_est = x
    a3,a4 = accel_S3_state(x, u)
    x_new = np.zeros_like(x)
    x_new[0] = x3 + dt*v3
    x_new[1] = x4 + dt*v4
    x_new[2] = v3 + dt*a3
    x_new[3] = v4 + dt*a4
    x_new[4] = m_star_est  # random walk
    x_new[5] = k6_est
    x_new[6] = c6_est
    x_new[7] = k34_est
    return x_new

def f_S3_heun(x, u, dt, t_idx):
    # Separate states and parameters
    x_dyn = x[:4]  # [x3, x4, v3, v4]
    p_est = x[4:]  # [m*_est, k6_est, c6_est, k34_est]

    # 1. Calculate f(x_k, t_k) = [v_k, a_k]
    v3, v4 = x_dyn[2], x_dyn[3]
    a3, a4 = accel_S3_state(x, u)
    f_k = np.array([v3, v4, a3, a4])

    # 2. Predictor Step: x_tilde = x_k + dt * f_k
    x_dyn_pred = x_dyn + dt * f_k
    
    # Construct predicted state vector for acceleration calculation
    x_pred = np.hstack((x_dyn_pred, p_est))
    
    # 3. Calculate f(x_tilde, t_k+1) = [v_tilde, a_tilde]
    v3_pred, v4_pred = x_dyn_pred[2], x_dyn_pred[3]
    a3_pred, a4_pred = accel_S3_state(x_pred, u)
    f_pred = np.array([v3_pred, v4_pred, a3_pred, a4_pred])

    # 4. Corrector Step: x_k+1 = x_k + (dt/2) * (f_k + f_pred)
    x_dyn_new = x_dyn + 0.5 * dt * (f_k + f_pred)

    # 5. Parameter update (Random Walk)
    x_new = np.hstack((x_dyn_new, p_est))
    return x_new

def h_S3(x, u, dt, t_idx):
    a3,a4 = accel_S3_state(x, u)
    return np.array([a3, a4])  # accels of masses 3 & 4


# ---------- S2: [x5,x6,v5,v6,k9,c9] ----------

def accel_S2_state(x, u):
    """
    x = [x5,x6,v5,v6,k9_est,c9_est]
    u = [u5,u6]  (F45, 0)
    """
    x5,x6,v5,v6,k9_est,c9_est = x
    u5,u6 = u

    k9_eff = max(k9_est, 1.0)
    c9_eff = max(c9_est, 1.0)

    a5 = (u5
          - k8*x5 - c8*v5
          - k9_eff*(x5 - x6)
          - c9_eff*(v5 - v6)) / m5

    a6 = (u6
          + k9_eff*(x5 - x6)
          + c9_eff*(v5 - v6)) / m6

    return np.array([a5, a6])

def f_S2(x, u, dt, t_idx):
    x5,x6,v5,v6,k9_est,c9_est = x
    a5,a6 = accel_S2_state(x, u)
    x_new = np.zeros_like(x)
    x_new[0] = x5 + dt*v5
    x_new[1] = x6 + dt*v6
    x_new[2] = v5 + dt*a5
    x_new[3] = v6 + dt*a6
    x_new[4] = k9_est
    x_new[5] = c9_est
    return x_new

def f_S2_heun(x, u, dt, t_idx):
    # Separate states and parameters
    x_dyn = x[:4]  # [x5, x6, v5, v6]
    p_est = x[4:]  # [k9_est, c9_est]

    # 1. Calculate f(x_k, t_k) = [v_k, a_k]
    v5, v6 = x_dyn[2], x_dyn[3]
    a5, a6 = accel_S2_state(x, u)
    f_k = np.array([v5, v6, a5, a6])

    # 2. Predictor Step: x_tilde = x_k + dt * f_k
    x_dyn_pred = x_dyn + dt * f_k
    
    # Construct predicted state vector for acceleration calculation
    x_pred = np.hstack((x_dyn_pred, p_est))
    
    # 3. Calculate f(x_tilde, t_k+1) = [v_tilde, a_tilde]
    v5_pred, v6_pred = x_dyn_pred[2], x_dyn_pred[3]
    a5_pred, a6_pred = accel_S2_state(x_pred, u)
    f_pred = np.array([v5_pred, v6_pred, a5_pred, a6_pred])

    # 4. Corrector Step: x_k+1 = x_k + (dt/2) * (f_k + f_pred)
    x_dyn_new = x_dyn + 0.5 * dt * (f_k + f_pred)
    
    # 5. Parameter update (Random Walk)
    x_new = np.hstack((x_dyn_new, p_est))
    return x_new

def h_S2(x, u, dt, t_idx):
    a5,_ = accel_S2_state(x, u)
    return np.array([a5])      # accel of mass 5


# ============================================================
# MESSAGE PASSING (JACOBI)  S1 ↔ S3 ↔ S2
# ============================================================

import numpy as np

def build_sub_inputs_jacobi_prob(x1_mean, P1, x3_mean, P3, x2_mean, P2, f4_scalar):
    """
    Probabilistic Jacobi message passing for 3 subsystems with two interfaces.

    Subsystems:
      S1 state mean x1_mean: [x1, x2, v1, v2, ...]
      S3 state mean x3_mean: [x3, x4, v3, v4, ...]
      S2 state mean x2_mean: [x5, x6, v5, v6, ...]

    Interfaces (from the schematic, page 1):
      e13: between y2 and y3  -> here: (x2,v2) - (x3,v3) using (k4,c4)
      e23: between y5 and y4  -> here: (x4,v4) - (x5,v5) using (k7,c7)

    Returns:
      u1_mean, u3_mean, u2_mean : mean inputs to S1, S3, S2
      F23_mean, F23_var         : Gaussian message for interface (x2,v2)-(x3,v3)
      F45_mean, F45_var         : Gaussian message for interface (x4,v4)-(x5,v5)
    """

    # --- Extract boundary DOFs used in coupling ---
    # From S1
    x2, v2 = x1_mean[1], x1_mean[3]

    # From S3
    x3, v3 = x3_mean[0], x3_mean[2]
    x4, v4 = x3_mean[1], x3_mean[3]

    # From S2
    x5, v5 = x2_mean[0], x2_mean[2]

    # ============================================================
    # Interface force F23 = k4(x2-x3) + c4(v2-v3)
    # ============================================================
    F23_mean = k4*(x2 - x3) + c4*(v2 - v3)

    # Cov blocks for (x2,v2) from S1 and (x3,v3) from S3
    P_s1_x2v2 = P1[np.ix_([1, 3], [1, 3])]
    P_s3_x3v3 = P3[np.ix_([0, 2], [0, 2])]

    a23 = np.array([k4, c4])  # [k, c]
    # Cross-covariances between subsystems are set to zero.
    F23_var = a23 @ P_s1_x2v2 @ a23 + a23 @ P_s3_x3v3 @ a23

    # ============================================================
    # Interface force F45 = k7(x4-x5) + c7(v4-v5)
    # ============================================================
    F45_mean = k7*(x4 - x5) + c7*(v4 - v5)

    # Cov blocks for (x4,v4) from S3 and (x5,v5) from S2
    P_s3_x4v4 = P3[np.ix_([1, 3], [1, 3])]
    P_s2_x5v5 = P2[np.ix_([0, 2], [0, 2])]

    a45 = np.array([k7, c7])  # [k, c]
    # Cross-covariances between subsystems are set to zero.
    F45_var = a45 @ P_s3_x4v4 @ a45 + a45 @ P_s2_x5v5 @ a45

    # ============================================================
    # Mean inputs use the same sign convention as the force model.
    # ============================================================
    # Use mean forces for the deterministic input.
    u1_mean = np.array([0.0,           -F23_mean])            # S1 inputs
    u3_mean = np.array([+F23_mean,     f4_scalar - F45_mean]) # S3 inputs
    u2_mean = np.array([+F45_mean,     0.0])                  # S2 inputs

    return u1_mean, u3_mean, u2_mean, F23_mean, F23_var, F45_mean, F45_var


# ============================================================
# UKF NOISES & INITIAL GUESSES
# ============================================================

# initial parameter guesses (intentionally biased)
int_coef = 0.7
k3_init    = int_coef * k3_true
c3_init    = int_coef * c3_true
m_star_init= int_coef * m_star_true
k6_init    = int_coef * k6_true
c6_init    = int_coef * c6_true
k9_init    = int_coef * k9_true
c9_init    = int_coef  * c9_true
k34_init   = int_coef  * k34_true

# initial states (all DOF at rest)
x1 = np.array([0.0, 0.0, 0.0, 0.0, k3_init,    c3_init])
x3 = np.array([0.0, 0.0, 0.0, 0.0, m_star_init,k6_init,c6_init,k34_init])
x2 = np.array([0.0, 0.0, 0.0, 0.0, k9_init,    c9_init])

# initial covariances
P1 = np.diag([1e-4,1e-4,1e-4,1e-4, 1e8, 1e5])
P3 = np.diag([1e-4,1e-4,1e-4,1e-4, 1e4, 1e8, 1e3, 1e8])
P2 = np.diag([1e-4,1e-4,1e-4,1e-4, 1e8, 1e5])

# process noise: small on states, larger on parameters (random walk)
Q1 = 1e-11 * np.eye(6) 
Q3 = 1e-12 * np.eye(8) 
Q2 = 1e-11 * np.eye(6) 

# measurement noise covariances
R1 = np.array([[1e-3]])      # a2
R3 = 1e-2* np.eye(2)           # a3,a4
R2 = np.array([[1e-3]])       # a5

# storage
Q_est = np.zeros((N+1, 6))
V_est = np.zeros((N+1, 6))
params_hist = np.zeros((N+1, 8))  # [k3,c3,m*,k6,c6,k34,k9,c9]

Q_est[0] = [x1[0],x1[1], x3[0],x3[1], x2[0],x2[1]]
V_est[0] = [x1[2],x1[3], x3[2],x3[3], x2[2],x2[3]]
params_hist[0] = [x1[4],x1[5], x3[4],x3[5],x3[6], x3[7], x2[4],x2[5]]

# --- before loop ---
F23_var_prev = 0.0
F45_var_prev = 0.0

F23_mean_hist = np.zeros(N)
F45_mean_hist = np.zeros(N)
F23_var_used_hist = np.zeros(N)
F45_var_used_hist = np.zeros(N)

for k in range(N):
    f4 = forces[k]

    # 1) Probabilistic message passing: mean inputs + (mean,var) of interface forces
    u1, u3, u2, F23_mean, F23_var, F45_mean, F45_var = build_sub_inputs_jacobi_prob(
        x1, P1, x3, P3, x2, P2, f4
    )

    # 1b) Information-reuse protection (separately for each interface)
    F23_var_used = max(0.0, F23_var - F23_var_prev)
    F45_var_used = max(0.0, F45_var - F45_var_prev)
    F23_var_prev = F23_var
    F45_var_prev = F45_var

    F23_mean_hist[k] = F23_mean
    F45_mean_hist[k] = F45_mean
    F23_var_used_hist[k] = F23_var_used
    F45_var_used_hist[k] = F45_var_used

    # 2) Local measurements at time k+1
    z1 = np.array([a2_meas[k+1]])                # S1: a2
    z3 = np.array([a3_meas[k+1], a4_meas[k+1]])  # S3: a3,a4
    z2 = np.array([a5_meas[k+1]])                # S2: a5

    # 3) Build effective Q matrices (inject interface-force uncertainty)
    Q1_eff = Q1.copy()
    Q3_eff = Q3.copy()
    Q2_eff = Q2.copy()

    # ---- Interface F23 affects:
    # S1 DOF2 -> v2 (index 3 in x1: [x1,x2,v1,v2,...])
    Q1_eff[3, 3] += (dt / m2)**2 * F23_var_used   # <-- uses mass of DOF2

    # S3 DOF3 -> v3 (index 2 in x3: [x3,x4,v3,v4,...])
    Q3_eff[2, 2] += (dt / m3)**2 * F23_var_used   # <-- uses mass of DOF3

    # ---- Interface F45 affects:
    # S3 DOF4 -> v4 (index 3 in x3)
    Q3_eff[3, 3] += (dt / m4)**2 * F45_var_used   # <-- uses mass of DOF4

    # S2 DOF5 -> v5 (index 2 in x2: [x5,x6,v5,v6,...])
    Q2_eff[2, 2] += (dt / m5)**2 * F45_var_used   # <-- uses mass of DOF5

    # 4) UKF steps for each subsystem using Q*_eff
    x1, P1 = unscented_kalman_filter_step(
        x1, P1, R1, Q1_eff, gamma_param,
        z1, k+1, dt, f_S1_heun, h_S1, u1
    )

    x3, P3 = unscented_kalman_filter_step(
        x3, P3, R3, Q3_eff, gamma_param,
        z3, k+1, dt, f_S3_heun, h_S3, u3
    )

    x2, P2 = unscented_kalman_filter_step(
        x2, P2, R2, Q2_eff, gamma_param,
        z2, k+1, dt, f_S2_heun, h_S2, u2
    )

    # 5) Store histories.
    Q_est[k+1] = [x1[0],x1[1], x3[0],x3[1], x2[0],x2[1]]
    V_est[k+1] = [x1[2],x1[3], x3[2],x3[3], x2[2],x2[3]]
    params_hist[k+1] = [x1[4],x1[5], x3[4],x3[5],x3[6],x3[7], x2[4],x2[5]]


#%%
# ============================================================
# UPDATED STATE COMPARISON PLOTS
# ============================================================

t = np.linspace(0, T, N+1)

fig, axes = plt.subplots(6, 2, figsize=(13, 16), sharex=True)


for i in range(6):

    # -------- displacement --------
    axL = axes[i, 0]
    axL.plot(t, Q_true[:, i], label='Monolithic (truth)')
    axL.plot(t, Q_est[:, i], '--', label='Distributed UKF')
    axL.set_ylabel(f'x{i+1}')
    axL.grid(True, alpha=0.3)

    if i == 0:
        axL.legend(loc='upper right', frameon=False)
        axL.set_title('Displacements')

    # -------- velocity --------
    axR = axes[i, 1]
    axR.plot(t, V_true[:, i], label='Monolithic (truth)')
    axR.plot(t, V_est[:, i], '--', label='Distributed UKF')
    axR.set_ylabel(f'v{i+1}')
    axR.grid(True, alpha=0.3)

    if i == 0:
        axR.legend(loc='upper right', frameon=False)
        axR.set_title('Velocities')

axes[-1, 0].set_xlabel('time [s]')
axes[-1, 1].set_xlabel('time [s]')

plt.tight_layout()
plt.show()


# ============================================================
# UPDATED PARAMETER ESTIMATE PLOTS
# ============================================================

# params_hist columns:
# [k3_est, c3_est, m*_est, k6_est, c6_est,k34_est,  k9_est, c9_est]

# ------------------------------------------------------------
# 1) ALL STIFFNESSES: k3, k6, k9, k34
# ------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(t, params_hist[:,0], label='k3_est')
plt.axhline(k3_true, color='k', linestyle='--', label='k3 true')

plt.plot(t, params_hist[:,3], label='k6_est')
plt.axhline(k6_true, color='r', linestyle='--', label='k6 true')

plt.plot(t, params_hist[:,6], label='k9_est')
plt.axhline(k9_true, color='g', linestyle='--', label='k9 true')

plt.plot(t, params_hist[:,5], label='k34_est')
plt.axhline(k34_true, linestyle='--', label='k34 true')


plt.title('Estimated Stiffness Parameters')
plt.ylabel('Stiffness [N/m]')
plt.xlabel('time [s]')
plt.grid(True, alpha=0.3)
plt.legend(ncol=3, frameon=False)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 2) ALL DAMPINGS: c3, c6, c9
# ------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(t, params_hist[:,1], label='c3_est')
plt.axhline(c3_true, color='k', linestyle='--', label='c3 true')

plt.plot(t, params_hist[:,4], label='c6_est')
plt.axhline(c6_true, color='r', linestyle='--', label='c6 true')

plt.plot(t, params_hist[:,7], label='c9_est')
plt.axhline(c9_true, color='g', linestyle='--', label='c9 true')

plt.title('Estimated Damping Parameters')
plt.ylabel('Damping [N·s/m]')
plt.xlabel('time [s]')
plt.grid(True, alpha=0.3)
plt.legend(ncol=3, frameon=False)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3) MASS PARAMETER m*
# ------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(t, params_hist[:,2], label='m*_est')
plt.axhline(m_star_true, color='k', linestyle='--', label='m* true')

plt.title('Estimated Added Mass Parameter')
plt.ylabel('Mass [kg]')
plt.xlabel('time [s]')
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

#%%
# time base for interface forces
t_force = t[:-1]

idx = t_force <= 20.0

plt.figure(figsize=(6.5,3))
plt.plot(t_force[idx], F23_mean_hist[idx],
         label=r'$F_{23}$ (S1$\leftrightarrow$S3)')
plt.plot(t_force[idx], F45_mean_hist[idx],
         label=r'$F_{45}$ (S3$\leftrightarrow$S2)')
plt.xlabel('time [s]')
plt.ylabel('interface force [N]')
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns

# -----------------------------
# Plot style.
# -----------------------------
colors  = sns.color_palette("crest", n_colors=8)
colors1 = sns.color_palette("colorblind", n_colors=8)

font = {'family': 'Times New Roman',
        'size': 11}
plt.rc('font', **font)

# -----------------------------
# Time window: first 20 seconds
# -----------------------------
t_max = 20.0
idx   = t <= t_max

t_plot = t[idx]

# -----------------------------
# Figure layout: 2 rows
# Row 1 → 2 plots
# Row 2 → 3 plots
# -----------------------------
fig = plt.figure(figsize=(9.4, 5.2))
gs  = gridspec.GridSpec(
    2, 6,
    # height_ratios=[1.0, 1.0],
    hspace=0.25,
    wspace=0.95
)

# ============================================================
# Row 1 — Displacement (mass 6)
# ============================================================
ax00 = plt.subplot(gs[0, 0:3])   # spans columns 1–2 (centered)

ax00.plot(t_plot, Q_true[idx, 5],
          label='Truth',
          color=colors1[0],
          linewidth=1)

ax00.plot(t_plot, Q_est[idx, 5],
          label='Distributed UKF',
          color=colors1[1],
          linewidth=1,
          linestyle='--')

ax00.set_ylabel(r'$x_6$ [m]')
# ax00.set_xticks([])
ax00.grid(alpha=0.3)

# ============================================================
# Row 1 — Velocity (mass 6)
# ============================================================
ax01 = plt.subplot(gs[0, 3:])   # spans columns 2–3 (right)

ax01.plot(t_plot, V_true[idx, 5],
          label='Truth',
          color=colors1[0],
          linewidth=1)

ax01.plot(t_plot, V_est[idx, 5],
          label='Distributed UKF',
          color=colors1[1],
          linewidth=1,
          linestyle='--')

ax01.set_ylabel(r'$v_6$ [m/s]')
# ax01.set_xticks([])
ax01.grid(alpha=0.3)

# Add one shared legend.
ax01.legend(loc='upper right', frameon=False)

# # turn off unused panel (row 1, col 3)
# ax02 = plt.subplot(gs[0, 2])
# ax02.axis('off')

# ============================================================
# Row 2 — Stiffnesses
# ============================================================
ax10 = plt.subplot(gs[1, 0:2])

ax10.plot(t_plot, params_hist[idx, 0], color=colors[2], label=r'$k_3$')
ax10.plot(t_plot, params_hist[idx, 3], color=colors[4], label=r'$k_6$')
ax10.plot(t_plot, params_hist[idx, 6], color=colors[0], label=r'$k_9$')
ax10.plot(t_plot, params_hist[idx, 5], color=colors1[0], label=r'$k^{*}$')

ax10.axhline(k3_true, color=colors[2], linestyle='--', linewidth=1)
ax10.axhline(k6_true, color=colors[4], linestyle='--', linewidth=1)
ax10.axhline(k9_true, color=colors[0], linestyle='--', linewidth=1)
ax10.axhline(k34_true, color=colors1[0], linestyle='--', linewidth=1)

ax10.set_ylabel(r'$k$ [N/m]')
ax10.set_xlabel('time [s]')
# ax10.set_title('Stiffness')
ax10.grid(alpha=0.3)
ax10.legend(frameon=False, ncol=1)

# ============================================================
# Row 2 — Dampings
# ============================================================
ax11 = plt.subplot(gs[1, 2:4])

ax11.plot(t_plot, params_hist[idx, 1], color=colors[2], label=r'$c_3$')
ax11.plot(t_plot, params_hist[idx, 4], color=colors[4], label=r'$c_6$')
ax11.plot(t_plot, params_hist[idx, 7], color=colors[6], label=r'$c_9$')

ax11.axhline(c3_true, color=colors[2], linestyle='--', linewidth=1)
ax11.axhline(c6_true, color=colors[4], linestyle='--', linewidth=1)
ax11.axhline(c9_true, color=colors[6], linestyle='--', linewidth=1)

ax11.set_ylabel(r'$c$ [N·s/m]')
ax11.set_xlabel('time [s]')
# ax11.set_title('Damping')
ax11.grid(alpha=0.3)
ax11.legend(frameon=False, ncol=1)

# ============================================================
# Row 2 — Added mass
# ============================================================
ax12 = plt.subplot(gs[1, 4:6])

ax12.plot(t_plot, params_hist[idx, 2],
          label=r'$m^{*}$ (est.)',
          color=colors[4],
          linewidth=1)


ax12.axhline(m_star_true,
             color='k',
             linestyle='--',
             linewidth=1,
             label=r'$m^{*}$ (true)')


ax12.set_ylabel(r'$m^{*}\,[\mathrm{kg}]$')
ax12.set_xlabel('time [s]')
# ax12.set_title('Added mass')
ax12.grid(alpha=0.3)
ax12.legend(frameon=False)

plt.show()

#%%
# ============================================================
#  Plot: Interface forces (mean ± std vs ground truth)
# ============================================================

# --- time alignment ---
t_plot = t[:N]  # Histories are stored at each time step.

# -----------------------------
# Interface force F23 (between DOF2 and DOF3)
# -----------------------------
F23_true = (
    k4 * (Q_true[:, 1] - Q_true[:, 2]) +
    c4 * (V_true[:, 1] - V_true[:, 2])
)

F23_mean = F23_mean_hist[:N]
F23_std  = 1*np.sqrt(F23_var_used_hist[:N])

plt.figure(figsize=(9, 4))
# plt.plot(t_plot, F23_true[:N], 'k', linewidth=2, label='True $F_{23}$')
# plt.plot(t_plot, F23_mean, 'b--', linewidth=2, label='Estimated mean $\\mu_{F_{23}}$')
plt.fill_between(
    t_plot,
    F23_mean - 3*F23_std,
    F23_mean + 3*F23_std,
    color='blue',
    alpha=0.25,
    label=r'$\pm 3\sigma_{F_{23}}$'
)
plt.xlabel('time [s]')
plt.ylabel('Interface force $F_{23}$ [N]')
plt.title('Interface force $F_{23}$ message: mean and uncertainty')
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


# -----------------------------
# Interface force F45 (between DOF4 and DOF5)
# -----------------------------
F45_true = (
    k7 * (Q_true[:, 3] - Q_true[:, 4]) +
    c7 * (V_true[:, 3] - V_true[:, 4])
)

F45_mean = F45_mean_hist[:N]
F45_std  = np.sqrt(F45_var_used_hist[:N])

plt.figure(figsize=(9, 4))
# plt.plot(t_plot, F45_true[:N], 'k', linewidth=2, label='True $F_{45}$')
# plt.plot(t_plot, F45_mean, 'b--', linewidth=2, label='Estimated mean $\\mu_{F_{45}}$')
plt.fill_between(
    t_plot,
    F45_mean - 3*F45_std,
    F45_mean + 3*F45_std,
    color='blue',
    alpha=0.25,
    label=r'$\pm 3\sigma_{F_{45}}$'
)
plt.xlabel('time [s]')
plt.ylabel('Interface force $F_{45}$ [N]')
plt.title('Interface force $F_{45}$ message: mean and uncertainty')
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()


#%%

"""Reusable filtering and weighted least-squares helpers.

The functions in this file are factored from the paper scripts without
changing the numerical formulas, default tolerances, or update equations.
Model-specific dynamics and parameters remain in the individual examples.
"""

import numpy as np
import numpy.linalg as npl
import scipy.linalg


# Numerical derivatives and stable linear algebra used by several estimators.
def numerical_jacobian(fun, x, eps=1e-6):
    """Finite-difference Jacobian of a vector-valued function."""
    x = np.asarray(x, dtype=float).copy()
    f0 = np.asarray(fun(x), dtype=float)
    J = np.zeros((f0.size, x.size), dtype=float)

    for j in range(x.size):
        dx = np.zeros(x.size, dtype=float)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        fp = np.asarray(fun(x + dx), dtype=float)
        fm = np.asarray(fun(x - dx), dtype=float)
        J[:, j] = (fp - fm) / (2.0 * step)

    return J


def robust_inverse(A, base_jitter=1e-10, max_tries=12):
    """Stable inverse with diagonal jitter escalation."""
    A_sym = 0.5 * (A + A.T)
    jitter = 0.0
    I = np.eye(A.shape[0])
    for _ in range(max_tries):
        try:
            return npl.inv(A_sym + jitter * I)
        except npl.LinAlgError:
            jitter = base_jitter if jitter == 0.0 else 10.0 * jitter
    raise RuntimeError("Matrix inversion failed even after jitter escalation.")


# UKF routines for the structural dynamics examples.
def unscented_kalman_filter_step(
    x_prev,
    P_prev,
    R_mat,
    Q_mat,
    gamma_param,
    y_meas_k,
    dt,
    tf_func,
    mf_func,
    u,
    *,
    t_idx=None,
    return_prediction=False,
    fallback_eigendecomp=False,
    scaled_cholesky=True,
):
    """One-step UKF used by the structural examples.

    If ``t_idx`` is set, transition and measurement functions are called as
    ``f(x, u, dt, t_idx)``. Otherwise they are called as ``f(x, u, dt)``.
    """
    x_prev = np.asarray(x_prev, dtype=float).reshape(-1)
    nx = x_prev.shape[0]
    ny = R_mat.shape[0]
    c = nx + gamma_param

    W = np.zeros(2 * nx + 1)
    x_sigma = np.zeros((nx, 2 * nx + 1))
    x_sigma[:, 0] = x_prev

    if c != 0.0:
        P_sym = 0.5 * (P_prev + P_prev.T)
        P_chol = c * P_sym if scaled_cholesky else P_sym

        S_mat = None
        jitter = 0.0
        max_tries = 10
        for _ in range(max_tries):
            try:
                S_mat = scipy.linalg.cholesky(P_chol + jitter * np.eye(nx), lower=False)
                break
            except scipy.linalg.LinAlgError:
                jitter = 1e-9 if jitter == 0.0 else jitter * 10.0

        if S_mat is None:
            if not fallback_eigendecomp:
                raise RuntimeError("Cholesky failed in UKF")
            w, V = np.linalg.eigh(P_chol)
            w_clipped = np.clip(w, 1e-12, None)
            S_mat = V @ np.diag(np.sqrt(w_clipped)) @ V.T

        if not scaled_cholesky:
            S_mat = np.sqrt(c) * S_mat

        W[0] = gamma_param / c
        for k in range(1, nx + 1):
            col = S_mat[k - 1].reshape(nx, 1)
            x_sigma[:, k:k + 1] = x_prev.reshape(nx, 1) + col
            x_sigma[:, nx + k:nx + k + 1] = x_prev.reshape(nx, 1) - col
            W[k] = 1.0 / (2.0 * c)
            W[nx + k] = 1.0 / (2.0 * c)

    def call_model(fun, x):
        if t_idx is None:
            return fun(x, u, dt)
        return fun(x, u, dt, t_idx)

    X_P = np.zeros_like(x_sigma)
    Y_P = np.zeros((ny, 2 * nx + 1))

    for i in range(2 * nx + 1):
        xp = call_model(tf_func, x_sigma[:, i])
        yp = call_model(mf_func, xp)
        X_P[:, i] = xp
        Y_P[:, i] = yp

    XP = (W * X_P).sum(axis=1).reshape(nx, 1)
    YP = (W * Y_P).sum(axis=1).reshape(ny, 1)

    PXX_cov = np.zeros((nx, nx))
    PYY_cov = np.zeros((ny, ny))
    PXY = np.zeros((nx, ny))
    for i in range(2 * nx + 1):
        dx = X_P[:, i:i + 1] - XP
        dy = Y_P[:, i:i + 1] - YP
        PXX_cov += W[i] * (dx @ dx.T)
        PYY_cov += W[i] * (dy @ dy.T)
        PXY += W[i] * (dx @ dy.T)

    PXX = PXX_cov + Q_mat
    PYY = PYY_cov + R_mat
    K = PXY @ np.linalg.inv(PYY)
    x_new = XP + K @ (y_meas_k.reshape(ny, 1) - YP)
    P_new = PXX - K @ PYY @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    if return_prediction:
        return x_new.ravel(), P_new, XP.ravel(), PXX, YP.ravel(), PYY
    return x_new.ravel(), P_new


# Variant matching the column-oriented sigma-point convention in the scaling benchmark.
def unscented_kalman_filter_step_unscaled_columns(
    x_prev,
    P_prev,
    R_mat,
    Q_mat,
    gamma_param,
    y_meas_k,
    dt,
    tf_func,
    mf_func,
    u,
):
    """UKF variant used by the generalized central/distributed benchmark."""
    nx = x_prev.shape[0]
    ny = R_mat.shape[0]
    c = nx + gamma_param

    W = np.zeros(2 * nx + 1)
    x_sigma = np.zeros((nx, 2 * nx + 1))
    x_sigma[:, 0] = x_prev

    if c != 0:
        P_sym = 0.5 * (P_prev + P_prev.T)

        jitter = 0.0
        max_tries = 10
        for i in range(max_tries):
            try:
                S = scipy.linalg.cholesky(P_sym + jitter * np.eye(nx), lower=True)
                break
            except scipy.linalg.LinAlgError:
                jitter = 1e-9 if jitter == 0.0 else jitter * 10.0
                if i == max_tries - 1:
                    raise RuntimeError("Cholesky failed in UKF")

        S = np.sqrt(c) * S
        W[0] = gamma_param / c
        W[1:] = 1.0 / (2.0 * c)

        for k in range(nx):
            col = S[:, k]
            x_sigma[:, 1 + k] = x_prev + col
            x_sigma[:, 1 + nx + k] = x_prev - col

    X_P_temp = np.zeros((nx, 2 * nx + 1))
    Y_P_temp = np.zeros((ny, 2 * nx + 1))

    for i in range(2 * nx + 1):
        A = tf_func(x_sigma[:, i], u, dt)
        X_P_temp[:, i:i + 1] = A.reshape(nx, 1)
        B = mf_func(A, u, dt)
        Y_P_temp[:, i:i + 1] = B.reshape(ny, 1)

    XP = np.sum(W * X_P_temp, axis=1).reshape(nx, 1)
    YP = np.sum(W * Y_P_temp, axis=1).reshape(ny, 1)

    PXX_cov = np.zeros((nx, nx))
    PYY_cov = np.zeros((ny, ny))
    PXY = np.zeros((nx, ny))
    for i in range(2 * nx + 1):
        dx = X_P_temp[:, i:i + 1] - XP
        dy = Y_P_temp[:, i:i + 1] - YP
        PXX_cov += W[i] * (dx @ dx.T)
        PYY_cov += W[i] * (dy @ dy.T)
        PXY += W[i] * (dx @ dy.T)

    PXX = PXX_cov + Q_mat
    PYY = PYY_cov + R_mat

    K = PXY @ np.linalg.inv(PYY)
    innovation = y_meas_k.reshape(ny, 1) - YP
    x_new = XP + K @ innovation
    P_new = PXX - K @ PYY @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    return x_new.ravel(), P_new


# UKF, WLS, and WNLS routines for the Kuramoto/PYPOWER benchmark.
def ukf_step(x_prev, P_prev, R, Q, gamma, y, dt, tf, mf, u):
    """UKF step used by the Kuramoto benchmark."""
    nx_ = x_prev.shape[0]
    ny = R.shape[0]
    c = nx_ + gamma
    P_sym = 0.5 * (P_prev + P_prev.T)

    jitter = 0.0
    S = None
    for _ in range(12):
        try:
            S = scipy.linalg.cholesky(c * P_sym + jitter * np.eye(nx_), lower=False)
            break
        except scipy.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
    if S is None:
        raise RuntimeError("UKF Cholesky failed even after jitter escalation.")

    W = np.empty(2 * nx_ + 1)
    W[0] = gamma / c
    W[1:] = 0.5 / c

    xs = np.empty((nx_, 2 * nx_ + 1))
    xs[:, 0] = x_prev
    for k in range(1, nx_ + 1):
        xs[:, k] = x_prev + S[k - 1]
        xs[:, nx_ + k] = x_prev - S[k - 1]

    XP = np.empty((nx_, 2 * nx_ + 1))
    YP = np.empty((ny, 2 * nx_ + 1))
    for i in range(2 * nx_ + 1):
        XP[:, i] = tf(xs[:, i], u, dt)
        YP[:, i] = mf(XP[:, i], u, dt)

    xp = (W * XP).sum(1, keepdims=True)
    yp = (W * YP).sum(1, keepdims=True)

    Pxx = Q.copy()
    Pyy = R.copy()
    Pxy = np.zeros((nx_, ny))
    for i in range(2 * nx_ + 1):
        dx = XP[:, i:i + 1] - xp
        dy = YP[:, i:i + 1] - yp
        Pxx += W[i] * (dx @ dx.T)
        Pyy += W[i] * (dy @ dy.T)
        Pxy += W[i] * (dx @ dy.T)

    K = Pxy @ npl.inv(Pyy)
    x_new = (xp + K @ (y.reshape(ny, 1) - yp)).ravel()
    P_new = Pxx - K @ Pyy @ K.T
    return x_new, 0.5 * (P_new + P_new.T)


def dynamic_wls_step(x_prev, Q, R, y, dt, tf, mf, u, jac_eps=1e-6):
    """One-shot linearized weighted least-squares step."""
    x_bar = np.asarray(tf(x_prev, u, dt), dtype=float)
    y_bar = np.asarray(mf(x_bar, u, dt), dtype=float)

    H = numerical_jacobian(lambda x: mf(x, u, dt), x_bar, eps=jac_eps)
    Qinv = robust_inverse(Q)
    Rinv = robust_inverse(R)

    lhs = Qinv + H.T @ Rinv @ H
    rhs = H.T @ Rinv @ (y - y_bar)
    dx = robust_inverse(lhs) @ rhs
    x_new = x_bar + dx
    P_new = robust_inverse(lhs)
    return x_new, 0.5 * (P_new + P_new.T)


def dynamic_wnls_step(
    x_prev,
    Q,
    R,
    y,
    dt,
    tf,
    mf,
    u,
    *,
    max_iter=5,
    tol=1e-7,
    jac_eps=1e-6,
    lm_damping=1e-8,
):
    """Iterative nonlinear weighted least-squares step."""
    x_bar = np.asarray(tf(x_prev, u, dt), dtype=float)
    x = x_bar.copy()
    Qinv = robust_inverse(Q)
    Rinv = robust_inverse(R)
    I = np.eye(x.size)

    for _ in range(max_iter):
        hx = np.asarray(mf(x, u, dt), dtype=float)
        J = numerical_jacobian(lambda z: mf(z, u, dt), x, eps=jac_eps)

        lhs = Qinv + J.T @ Rinv @ J + lm_damping * I
        rhs = J.T @ Rinv @ (y - hx) - Qinv @ (x - x_bar)

        dx = robust_inverse(lhs) @ rhs
        x = x + dx

        if npl.norm(dx) < tol:
            break

    P_new = robust_inverse(lhs)
    return x, 0.5 * (P_new + P_new.T)


# Linearized Kalman-filter helpers used by the IEEE-9 turbine example.
def heun_step_cont(f_cont, x, u, t, dt):
    """Continuous-time Heun step."""
    k1 = f_cont(x, u, t)
    x_pred = x + dt * k1
    k2 = f_cont(x_pred, u, t + dt)
    return x + 0.5 * dt * (k1 + k2)


def heun_transition_matrix(f_cont, x, u, t, dt):
    """Numerical transition matrix for Heun propagation."""
    A0 = numerical_jacobian(lambda z: f_cont(z, u, t), x)
    x_euler = x + dt * f_cont(x, u, t)
    A1 = numerical_jacobian(lambda z: f_cont(z, u, t + dt), x_euler)
    A_bar = 0.5 * (A0 + A1)
    I = np.eye(len(x))
    return I + dt * A_bar + 0.5 * (dt**2) * (A_bar @ A_bar)


def linear_kf_predict_heun(f_cont, x, P, u, t, dt, Q):
    """Linearized KF predict step with Heun mean propagation."""
    x_pred = heun_step_cont(f_cont, x, u, t, dt)
    Ad = heun_transition_matrix(f_cont, x, u, t, dt)
    P_pred = Ad @ P @ Ad.T + Q
    return x_pred, P_pred


def linear_kf_update(x_pred, P_pred, z, H, R):
    """Joseph-form linear Kalman update."""
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    Kgain = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + Kgain @ y
    I = np.eye(len(x_pred))
    P_upd = (I - Kgain @ H) @ P_pred @ (I - Kgain @ H).T + Kgain @ R @ Kgain.T
    return x_upd, P_upd

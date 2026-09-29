"""Cost-aware optimal style rotation under an AR(1) model of the spread.

Model (per rebalancing period of a given frequency)
    x_{k+1} = rho * x_k + sqrt(1 - rho^2) * sigma * eps_{k+1},  eps ~ N(0, 1)
where x_k is the value-minus-growth spread over period k.  The investor holds
either value (d = +1) or growth (d = -1); relative to the 50/50 mix the period
return is d * x_{k+1} / 2.  Switching costs 2c (two one-way trades of the whole
portfolio), i.e. c * |d' - d|.  Because the growth and value legs have almost
identical variance, the variance of the portfolio does not depend on d, and the
mean-variance (or, to second order, log-utility) objective reduces to
maximizing the long-run average net return.

In units of sigma (z = x / sigma, kappa = c / sigma) the average-reward
Bellman equation is
    h(z, d) + g = max_{d'} { d' rho z / 2 - kappa |d' - d| + E[h(z', d') | z] }.
We solve it by relative value iteration on a grid and read off the optimal
policy, which is a no-trade band: switch to d' = sign(rho z) iff |z| > b*.
"""
from __future__ import annotations

from functools import lru_cache
from math import asin, pi, sqrt

import numpy as np
from scipy.stats import norm

ZMAX, NZ = 5.0, 401
ZG = np.linspace(-ZMAX, ZMAX, NZ)


def _transition(rho: float) -> np.ndarray:
    s = sqrt(max(1 - rho ** 2, 1e-12))
    dz = ZG[1] - ZG[0]
    edges = np.concatenate([[-np.inf], (ZG[:-1] + ZG[1:]) / 2, [np.inf]])
    mu = rho * ZG[:, None]
    P = norm.cdf((edges[None, 1:] - mu) / s) - norm.cdf((edges[None, :-1] - mu) / s)
    return P / P.sum(1, keepdims=True)


def _stationary_weights() -> np.ndarray:
    w = norm.pdf(ZG)
    return w / w.sum()


def solve(rho: float, kappa: float, tol: float = 1e-11, maxit: int = 20000):
    """Return (g, band, is_threshold) for the average-reward problem.

    g    : optimal long-run average net gain per period, in units of sigma
    band : optimal no-trade half-width in units of sigma (|z| <= band: keep)
    """
    a = 1.0 if rho >= 0 else -1.0
    r = abs(rho)
    if r < 1e-9:
        return 0.0, np.inf, True
    P = _transition(r)
    h = np.zeros((NZ, 2))                  # columns: d = -1, d = +1
    D = np.array([-1.0, 1.0])
    g = 0.0
    for _ in range(maxit):
        Eh = P @ h                         # E[h(z', d') | z] for each d'
        # q[z, d, d'] = d' r z / 2 - kappa |d' - d| + Eh[z, d']
        q = (D[None, None, :] * r * ZG[:, None, None] / 2
             - kappa * np.abs(D[None, None, :] - D[None, :, None])
             + Eh[:, None, :])
        hn = q.max(2)
        g_new = hn[NZ // 2, 1]
        hn = hn - g_new
        if np.max(np.abs(hn - h)) < tol:
            h, g = hn, g_new
            break
        h, g = hn, g_new
    pol = q.argmax(2)                      # index of d' given (z, d)
    # band: from d = -1 (index 0), smallest z at which we switch to +1
    sw = ZG[pol[:, 0] == 1]
    band = float(sw.min()) if len(sw) else np.inf
    band = max(band, 0.0)
    # verify threshold structure: switch set from d=-1 is {z > band}
    is_thr = bool(np.all(pol[ZG > band + 1e-9, 0] == 1) and np.all(pol[ZG < band - 1e-9, 0] == 0)) if np.isfinite(band) else True
    return float(g), band, is_thr


def gain_band(rho: float, kappa: float, band: float, n_sim: int = 0) -> tuple[float, float]:
    """Exact average net gain and switching probability of a given band policy
    (sign a = sign(rho)), via the stationary distribution of the (z, d) chain."""
    r = abs(rho)
    P = _transition(r)
    # chain on (z, d); from state (z_k, d) the new position d' is set by z_k
    up = ZG > band
    dn = ZG < -band
    # transition of d given z: d' = +1 if up, -1 if dn, else d
    # build 2NZ x 2NZ matrix: state (i, d) -> (j, d') where d' decided by z_i
    n = NZ
    T = np.zeros((2 * n, 2 * n))
    for di in range(2):
        dprime = np.where(up, 1, np.where(dn, 0, di))
        for i in range(n):
            T[di * n + i, dprime[i] * n:(dprime[i] + 1) * n] = P[i]
    # stationary distribution
    w, v = np.linalg.eig(T.T)
    pi_ = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi_ = pi_ / pi_.sum()
    gain, sw = 0.0, 0.0
    for di, dval in enumerate([-1.0, 1.0]):
        p = pi_[di * n:(di + 1) * n]
        dprime = np.where(up, 1.0, np.where(dn, -1.0, dval))
        gain += np.sum(p * (dprime * r * ZG / 2 - kappa * np.abs(dprime - dval)))
        sw += np.sum(p * (dprime != dval))
    return float(gain), float(sw)


def closed_form(rho: float, kappa: float) -> tuple[float, float, float]:
    """Zero-band sign rule with a = sign(rho): (gross gain, switch prob, net gain)
    per period in units of sigma."""
    r = abs(rho)
    gross = r / sqrt(2 * pi)
    sw = 0.5 - asin(r) / pi
    return gross, sw, gross - 2 * kappa * sw


# ------------------------------------------------------------ lookup table
RHO_GRID = np.round(np.arange(0.0, 0.601, 0.01), 3)
KAPPA_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 2.0, 60)])


CACHE = __import__("pathlib").Path(__file__).resolve().parent.parent / "results" / "dp_table.npz"


@lru_cache(maxsize=1)
def table():
    if CACHE.exists():
        z = np.load(CACHE)
        if z["G"].shape == (len(RHO_GRID), len(KAPPA_GRID)):
            return z["G"], z["B"]
    G = np.zeros((len(RHO_GRID), len(KAPPA_GRID)))
    B = np.zeros_like(G)
    for i, r in enumerate(RHO_GRID):
        for j, k in enumerate(KAPPA_GRID):
            if r == 0:
                G[i, j], B[i, j] = 0.0, np.inf
                continue
            g, b, _ = solve(r, k)
            G[i, j], B[i, j] = g, b
    np.savez(CACHE, G=G, B=B, rho=RHO_GRID, kappa=KAPPA_GRID)
    return G, B


def lookup(rho: float, kappa: float) -> tuple[float, float]:
    """Bilinear interpolation of (g*, b*) for |rho|, kappa."""
    G, B = table()
    r = min(abs(rho), RHO_GRID[-1])
    k = min(kappa, KAPPA_GRID[-1])
    i = np.clip(np.searchsorted(RHO_GRID, r) - 1, 0, len(RHO_GRID) - 2)
    j = np.clip(np.searchsorted(KAPPA_GRID, k) - 1, 0, len(KAPPA_GRID) - 2)
    tr = (r - RHO_GRID[i]) / (RHO_GRID[i + 1] - RHO_GRID[i])
    tk = (k - KAPPA_GRID[j]) / (KAPPA_GRID[j + 1] - KAPPA_GRID[j])
    def bil(M):
        return ((1 - tr) * (1 - tk) * M[i, j] + tr * (1 - tk) * M[i + 1, j]
                + (1 - tr) * tk * M[i, j + 1] + tr * tk * M[i + 1, j + 1])
    g = bil(G)
    Bf = np.where(np.isfinite(B), B, 50.0)
    b = bil(Bf)
    if r < RHO_GRID[1] and i == 0:
        b = max(b, Bf[1, j])
    return float(g), float(b)


if __name__ == "__main__":
    import time
    for r, k in [(0.2, 0.0), (0.2, 0.05), (0.2, 0.2), (0.05, 0.02), (0.3, 0.5)]:
        t = time.time()
        g, b, thr = solve(r, k)
        cf = closed_form(r, k)
        gb, sw = gain_band(r, k, b)
        print(f"rho={r} kappa={k}: g*={g:.5f} band={b:.3f} thr={thr} | band-chain g={gb:.5f} sw={sw:.3f} "
              f"| zero-band closed form net={cf[2]:.5f} gross={cf[0]:.5f} sw={cf[1]:.3f}  ({time.time()-t:.2f}s)")

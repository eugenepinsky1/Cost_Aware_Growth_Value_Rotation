"""Optimization studies for growth-value rotation (special-issue revision).

1. Model validation: AR(1) closed-form gain, switching rate and break-even cost
   of the sign rule at each frequency versus realized values.
2. Out-of-sample cost-aware optimal rotation: at each month end, estimate the
   spread autocorrelation and volatility at each frequency on an expanding
   window, solve the average-reward DP (dp_policy) for the optimal no-trade
   band, and pick the frequency with the highest (robust) expected net gain;
   rotate only if that gain is positive.
3. Mean-variance / Kelly weights with a long-only constraint (monthly).
4. Risk measures (expected shortfall, drawdown, Sortino) over the OOS period.

Writes CSV files to ../results.
"""
from __future__ import annotations

from math import asin, pi, sqrt

import numpy as np
import pandas as pd

import analysis as A
import dp_policy as DP

RES = A.RES
N = A.N
FREQS = ["D", "W", "M", "Q"]
NPER = {"D": 252.0, "W": 52.0, "M": 12.0, "Q": 4.0}
COSTS_OOS = [0.0, 1.0, 2.0, 5.0, 10.0]
Z_ROBUST = 1.645          # one-sided 95% lower confidence bound for |rho|
SE_TYPE = "white"         # "white" (main) or "iid" (1/sqrt(n), robustness)
MIN_MONTHS = 60           # training window before the first OOS decision
GAMMAS = [1, 2, 5, 10, 25, 50, 100]


# ------------------------------------------------------------------ data
def prep(name):
    grp, _, b, gt, vt = [t for t in A.TRIPLETS if t[1] == name][0]
    R = {t: A.returns(t) for t in (b, gt, vt)}
    idx = R[b].index.intersection(R[gt].index).intersection(R[vt].index)
    idx = idx[(idx >= A.START[grp]) & (idx <= A.END)]
    R = {t: R[t].loc[idx] for t in R}
    ok = pd.concat([R[t] for t in R], axis=1).notna().all(axis=1)
    idx = idx[ok.values]
    R = {t: R[t].loc[idx] for t in R}
    irx = A.load("^IRX")["Close"].replace(0, np.nan).ffill()
    rf = (irx.reindex(idx, method="ffill") / 100 / N).fillna(0)
    return grp, R[gt]["cc"], R[vt]["cc"], R[b]["cc"], rf, idx


def spreads(g, v, idx):
    out = {}
    for f in FREQS:
        reb = A.rebal_dates(idx, f)
        s = A.period_ret(v, reb) - A.period_ret(g, reb)
        gg = A.period_ret(g, reb)
        out[f] = (reb, s.iloc[1:], gg.iloc[1:])     # drop the partial first period
    return out


# ------------------------------------------------------------------ 1. validation
def validation(name):
    grp, g, v, b, rf, idx = prep(name)
    sp = spreads(g, v, idx)
    years = len(idx) / N
    rows = []
    for f in FREQS:
        _, s, _ = sp[f]
        s = s.iloc[:-1] if f != "D" else s          # drop partial last period
        rho, sig, n = s.autocorr(1), s.std(), len(s)
        nf = n / years
        gross_m = rho * sig / sqrt(2 * pi) * nf
        sw_m = (0.5 - asin(rho) / pi) * nf
        d = np.sign(s).replace(0, np.nan).ffill()   # inertia position after period k
        gain = (d.shift(1) * s / 2).iloc[1:]
        sw = (d != d.shift(1)).iloc[1:]
        gross_r = gain.mean() * nf
        sw_r = sw.mean() * nf
        rows.append({"triplet": name, "freq": f, "n": n, "rho": rho, "sigma": sig,
                     "gross_model": gross_m, "gross_real": gross_r,
                     "gross_real_t": gain.mean() / gain.std() * sqrt(len(gain)),
                     "sw_model": sw_m, "sw_real": sw_r,
                     "be_model_bps": gross_m / (2 * sw_m) * 1e4,
                     "be_real_bps": gross_r / (2 * sw_r) * 1e4})
    return rows


# ------------------------------------------------------------------ 2. OOS optimizer
def estimates(s: pd.Series):
    """Lag-1 autocorrelation, volatility and heteroskedasticity-robust (White)
    standard error of the autocorrelation."""
    x = s.values
    n = len(x)
    if n < 8:
        return np.nan, np.nan, np.nan
    rho = np.corrcoef(x[:-1], x[1:])[0, 1]
    e = x - x.mean()
    se = np.sqrt(np.sum(e[:-1] ** 2 * e[1:] ** 2)) / np.sum(e[:-1] ** 2)
    if SE_TYPE == "iid":
        se = 1 / sqrt(n)
    return rho, x.std(ddof=1), se


def optimize_choice(sp, t, cost, z):
    """Choose frequency, sign and band at decision date t using data up to t."""
    best = (None, 0.0, 0.0, np.inf, np.nan, {})
    info = {}
    for f in FREQS:
        _, s, _ = sp[f]
        s = s[s.index <= t]
        rho, sig, se = estimates(s)
        if not np.isfinite(rho):
            continue
        r_rob = max(abs(rho) - z * se, 0.0)
        g, band = DP.lookup(r_rob, cost / sig)
        ann = NPER[f] * sig * g
        info[f] = (rho, sig, se, r_rob, band, ann)
        if ann > best[1] + 1e-12:
            best = (f, ann, np.sign(rho), band, sig, info)
    return best[0], best[2], best[3], best[4], info


def run_oos(name, cost_bps, z, force_freq=None, zero_band=False):
    """Daily weight path of the optimized rotation (value weight in {0, .5, 1})."""
    grp, g, v, b, rf, idx = prep(name)
    sp = spreads(g, v, idx)
    c = cost_bps / 1e4
    month_ends = A.rebal_dates(idx, "M")
    rebsets = {f: set(sp[f][0]) for f in FREQS}
    start = month_ends[MIN_MONTHS]
    d = 0.0
    f_cur, a_cur, band_cur, sig_cur = None, 0.0, np.inf, np.nan
    W = pd.Series(np.nan, index=idx)
    choice = pd.Series(None, index=month_ends, dtype=object)
    me_set = set(month_ends)
    for t in idx[idx >= start]:
        new_decision = t in me_set
        if new_decision:
            if force_freq is None:
                f_cur, a_cur, band_cur, sig_cur, _ = optimize_choice(sp, t, c, z)
            else:
                _, s, _ = sp[force_freq]
                rho, sig, se = estimates(s[s.index <= t])
                r_rob = max(abs(rho) - z * se, 0.0)
                gg, band = DP.lookup(r_rob, c / sig)
                f_cur, a_cur, band_cur, sig_cur = (force_freq, np.sign(rho), band, sig) if r_rob > 0 and gg > 0 else (None, 0, np.inf, np.nan)
            if zero_band:
                band_cur = 0.0
            choice[t] = f_cur or "none"
            if f_cur is None:
                d = 0.0
        if f_cur is not None and (new_decision or t in rebsets[f_cur]):
            _, s, _ = sp[f_cur]
            k = s.index.searchsorted(t, side="right")
            if k > 0:
                zt = s.iloc[k - 1] / sig_cur
                target = a_cur * np.sign(zt)
                if d == 0.0:
                    d = target if target != 0 else d
                elif abs(zt) > band_cur and target != d:
                    d = target
        W[t] = (1 + d) / 2
    W = W.dropna()
    w = W.shift(1).dropna()
    gg, vv, bb, rff = g.loc[w.index], v.loc[w.index], b.loc[w.index], rf.loc[w.index]
    to = 2 * w.diff().abs().fillna(0)
    gross = w * vv + (1 - w) * gg
    net = gross - c * to
    return net, gross, to, bb, rff, choice.dropna()


def fixed_monthly(name, cost_bps, start):
    grp, g, v, b, rf, idx = prep(name)
    net, gross, to = A.strat_freq(g, v, "M", "Inertia", cost_bps / 1e4)
    return net.loc[start:], to.loc[start:]


def risk_stats(r: pd.Series, rf: pd.Series, bench: pd.Series):
    p = A.perf(r, rf)
    ex = r - rf
    down = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(N)
    m = (1 + r).resample("ME").prod() - 1
    es = m[m <= m.quantile(0.05)].mean()
    al, at, be = A.alpha_nw(ex.values, (bench - rf).values)
    p.update({"Sortino": ex.mean() * N / down, "ES95_m": es,
              "Calmar": p["CAGR"] / abs(p["MaxDD"]), "alpha": al, "alpha_t": at, "beta": be,
              "p_LW": A.lw_sharpe_p(ex.values, (bench - rf).values) if r.std() > 0 and not np.allclose(r, bench) else np.nan})
    return p


def oos_study(names):
    rows, usage, paths = [], [], {}
    for name in names:
        for c in COSTS_OOS:
            specs = {"Robust optimal": dict(z=Z_ROBUST),
                     "Plug-in optimal": dict(z=0.0),
                     "Robust monthly band": dict(z=Z_ROBUST, force_freq="M")}
            base = None
            for lab, kw in specs.items():
                net, gross, to, bb, rff, ch = run_oos(name, c, **kw)
                if base is None:
                    base = (bb, rff)
                    st = risk_stats(bb, rff, bb)
                    rows.append({"triplet": name, "cost_bps": c, "strategy": "Benchmark",
                                 "switches_yr": 0.0, "start": net.index[0].date(), **st})
                    fm, fto = fixed_monthly(name, c, net.index[0])
                    fm = fm.loc[net.index[0]:]
                    st = risk_stats(fm, rff.loc[fm.index], bb.loc[fm.index])
                    rows.append({"triplet": name, "cost_bps": c, "strategy": "Monthly inertia (fixed)",
                                 "switches_yr": fto.sum() / 2 / len(fto) * N, "start": fm.index[0].date(), **st})
                st = risk_stats(net, rff, bb)
                rows.append({"triplet": name, "cost_bps": c, "strategy": lab,
                             "switches_yr": to.sum() / 2 / len(to) * N, "start": net.index[0].date(), **st})
                u = ch.value_counts(normalize=True)
                usage.append({"triplet": name, "cost_bps": c, "strategy": lab,
                              **{f: u.get(f, 0.0) for f in FREQS + ["none"]}})
                paths[(name, c, lab)] = (net, ch)
            print(name, c, "done")
    return pd.DataFrame(rows), pd.DataFrame(usage), paths


# ------------------------------------------------------------------ 3. mean-variance / Kelly
def mv_study(names, cost_bps=2.0, z=0.0):
    rows = []
    for name in names:
        grp, g, v, b, rf, idx = prep(name)
        reb = A.rebal_dates(idx, "M")
        s = (A.period_ret(v, reb) - A.period_ret(g, reb)).iloc[1:]
        rg = A.period_ret(g, reb).iloc[1:]
        start = reb[MIN_MONTHS]
        dec = s.index[s.index >= start]
        for gam in GAMMAS:
            wv = {}
            at_bound = []
            for t in dec:
                ss, gg_ = s[s.index <= t], rg[rg.index <= t]
                rho, sig, se = estimates(ss)
                r_use = np.sign(rho) * max(abs(rho) - z * se, 0.0)
                var = ss.var()
                # benchmark-relative mean-variance: max (w-1/2)E[s] - gam/2 (w-1/2)^2 Var(s)
                w_raw = 0.5 + r_use * ss.iloc[-1] / (gam * var)
                wv[t] = float(np.clip(w_raw, 0, 1))
                at_bound.append(w_raw <= 0 or w_raw >= 1)
            w = pd.Series(wv).reindex(idx).ffill().shift(1).dropna()
            to = 2 * w.diff().abs().fillna(0)
            gross = w * v.loc[w.index] + (1 - w) * g.loc[w.index]
            net = gross - cost_bps / 1e4 * to
            st = risk_stats(net, rf.loc[w.index], b.loc[w.index])
            rows.append({"triplet": name, "gamma": gam, "cost_bps": cost_bps,
                         "share_at_bound": np.mean(at_bound),
                         "turnover_yr": to.sum() / len(to) * N / 2, "mean_abs_tilt": (w - 0.5).abs().mean(),
                         **st})
    return pd.DataFrame(rows)


def main():
    names = [t[1] for t in A.TRIPLETS]
    val = pd.DataFrame([r for n in names for r in validation(n)])
    val.to_csv(RES / "model_validation.csv", index=False)
    print(val.round(4).to_string())
    oos, use, paths = oos_study(names)
    oos.to_csv(RES / "oos.csv", index=False)
    use.to_csv(RES / "oos_usage.csv", index=False)
    pd.to_pickle(paths, RES / "oos_paths.pkl")
    mv = mv_study(names)
    mv.to_csv(RES / "mv_kelly.csv", index=False)
    print(mv.round(3).to_string())


if __name__ == "__main__":
    main()


def z_sensitivity(names, cost_bps=2.0, zs=(0.0, 1.0, 1.645, 2.326)):
    rows = []
    for name in names:
        for z in zs:
            net, gross, to, bb, rff, ch = run_oos(name, cost_bps, z=z)
            st = risk_stats(net, rff, bb)
            u = ch.value_counts(normalize=True)
            rows.append({"triplet": name, "z": z, "cost_bps": cost_bps,
                         "switches_yr": to.sum() / 2 / len(to) * N,
                         "share_M": u.get("M", 0.0), "share_D": u.get("D", 0.0),
                         "share_none": u.get("none", 0.0), **st})
    return pd.DataFrame(rows)


def iid_se_check(names, cost_bps=2.0):
    global SE_TYPE
    rows = []
    SE_TYPE = "iid"
    try:
        for name in names:
            net, gross, to, bb, rff, ch = run_oos(name, cost_bps, z=Z_ROBUST)
            u = ch.value_counts(normalize=True)
            rows.append({"triplet": name, "se": "iid", "switches_yr": to.sum() / 2 / len(to) * N,
                         "share_D": u.get("D", 0.0), "share_M": u.get("M", 0.0), **risk_stats(net, rff, bb)})
    finally:
        SE_TYPE = "white"
    return pd.DataFrame(rows)

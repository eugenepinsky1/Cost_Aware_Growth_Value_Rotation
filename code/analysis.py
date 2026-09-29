"""Growth-value rotation: inertia vs reversal across rebalancing frequencies.

Reads Yahoo Finance daily files in ../data and writes CSV results and LaTeX
tables to ../results and ../tables, and figures to ../figures.

Returns
  cc = AdjClose_t / AdjClose_{t-1} - 1       (total close-to-close)
  id = Close_t / Open_t - 1                  (intraday, open-to-close)
  on = (1 + cc) / (1 + id) - 1               (overnight, incl. dividends)

Strategies (long-only, 100% in one ETF)
  Inertia : hold the ETF (growth or value) that won the previous period
  Reversal: hold the ETF that lost the previous period
  Frequencies: D, W, M, Q (close-to-close) and 2xD (overnight/intraday sessions)
  2xD variants:
    'prev'  - hold the winner/loser of the immediately preceding session
              (intraday uses this morning's overnight; overnight uses
              today's intraday)
    'same'  - hold the winner/loser of the same session on the previous day
Costs: 0, 1, 2 bps per one-way trade; a switch = 2 one-way trades.
"""
from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, RES, TAB = ROOT / "data", ROOT / "results", ROOT / "tables"
N = 252
COSTS = [0.0, 1.0, 2.0]
FREQS = ["2xD", "D", "W", "M", "Q"]
BOOT, BLOCK, SEED = 2000, 20, 20260927

TRIPLETS = [
    ("Main", "Russell 1000", "IWB", "IWF", "IWD"),
    ("Main", "Russell Midcap", "IWR", "IWP", "IWS"),
    ("Main", "Russell 2000", "IWM", "IWO", "IWN"),
    ("Robustness", "S&P 500", "IVV", "IVW", "IVE"),
    ("Robustness", "S&P MidCap 400", "IJH", "IJK", "IJJ"),
    ("Robustness", "S&P SmallCap 600", "IJR", "IJT", "IJS"),
    ("International", "MSCI EAFE", "EFA", "EFG", "EFV"),
]
START = {"Main": "2001-09-01", "Robustness": "2001-09-01", "International": "2005-09-01"}
END = "2026-09-25"


# ---------------------------------------------------------------- data
def load(t: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / f"{t.replace('^', 'IDX_')}.csv", index_col=0, parse_dates=True).sort_index()
    d = d[~d.index.duplicated(keep="last")]
    return d


def returns(t: str) -> pd.DataFrame:
    d = load(t)
    d = d[(d["Open"] > 0) & (d["Close"] > 0)]
    r = pd.DataFrame(index=d.index)
    r["cc"] = d["Adj Close"].pct_change()
    r["id"] = d["Close"] / d["Open"] - 1
    r["on"] = (1 + r["cc"]) / (1 + r["id"]) - 1
    return r


def rebal_dates(idx: pd.DatetimeIndex, f: str) -> pd.DatetimeIndex:
    if f == "D":
        return idx
    per = {"W": "W-FRI", "M": "M", "Q": "Q"}[f]
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period(per)).max().values)


def period_ret(r: pd.Series, reb: pd.DatetimeIndex) -> pd.Series:
    g = np.searchsorted(reb.values, r.index.values, side="left")
    pr = (1 + r).groupby(g).prod() - 1
    pr = pr[pr.index < len(reb)]
    return pd.Series(pr.values, index=reb[pr.index])


# ---------------------------------------------------------------- stats
def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def nw_cov(u, lags):
    u = u - u.mean(0)
    n = len(u)
    S = u.T @ u / n
    for j in range(1, lags + 1):
        G = u[j:].T @ u[:-j] / n
        S = S + (1 - j / (lags + 1)) * (G + G.T)
    return S


def lw_sharpe_p(x1, x2):
    """Ledoit-Wolf (2008) HAC test of equal Sharpe ratios on excess returns."""
    x = np.column_stack([x1, x2])
    n = len(x)
    m, m2 = x.mean(0), (x ** 2).mean(0)
    u = np.column_stack([x, x ** 2])
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    psi = nw_cov(u, lags)
    v = m2 - m ** 2
    grad = np.array([m2[0] / v[0] ** 1.5, -m2[1] / v[1] ** 1.5,
                     -0.5 * m[0] / v[0] ** 1.5, 0.5 * m[1] / v[1] ** 1.5])
    se = np.sqrt(grad @ psi @ grad / n)
    z = (m[0] / np.sqrt(v[0]) - m[1] / np.sqrt(v[1])) / se
    return 2 * (1 - ncdf(abs(z)))


def alpha_nw(y, x):
    """Annualized CAPM-style alpha of y on x (excess returns), NW t-stat."""
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    lags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    Xe = X * e[:, None]
    S = nw_cov(Xe, lags) * len(y)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return b[0] * N, b[0] / np.sqrt(V[0, 0]), b[1]


def perf(r: pd.Series, rf: pd.Series) -> dict:
    w = (1 + r).cumprod()
    ex = r - rf
    return {"CAGR": w.iloc[-1] ** (N / len(r)) - 1,
            "Vol": r.std() * np.sqrt(N),
            "Sharpe": ex.mean() / ex.std() * np.sqrt(N),
            "MaxDD": (w / w.cummax() - 1).min()}


def sb_idx(n, block, rng):
    new = rng.random(n) < 1 / block
    new[0] = True
    pos = np.flatnonzero(new)
    bid = np.cumsum(new) - 1
    starts = rng.integers(n, size=len(pos))
    return (starts[bid] + np.arange(n) - pos[bid]) % n


def spa(d: np.ndarray, rng) -> float:
    """Hansen (2005) SPA consistent p-value; d = strategy minus benchmark."""
    n, k = d.shape
    mean = d.mean(0)
    bm = np.empty((BOOT, k))
    for b in range(BOOT):
        bm[b] = d[sb_idx(n, BLOCK, rng)].mean(0)
    om = np.sqrt(n) * bm.std(0)
    tstat = max(0.0, np.max(np.sqrt(n) * mean / om))
    mu_c = mean * (np.sqrt(n) * mean / om >= -np.sqrt(2 * np.log(np.log(n))))
    z = np.sqrt(n) * (bm - mu_c) / om
    return float((np.maximum(0, z.max(1)) > tstat).mean())


# ---------------------------------------------------------------- strategies
def switch_daily(g, v, w_reb, cost):
    """w_reb: 1 = value, 0 = growth, decided at rebalance close."""
    w = w_reb.reindex(g.index).ffill().shift(1)
    first = w.first_valid_index()
    w = w.loc[first:]
    g, v = g.loc[first:], v.loc[first:]
    to = 2 * w.diff().abs().fillna(0)
    gross = w * v + (1 - w) * g
    return gross - cost * to, gross, to


def strat_freq(g, v, f, rule, cost):
    reb = rebal_dates(g.index, f)
    s = period_ret(v, reb) - period_ret(g, reb)
    w = (s > 0).astype(float) if rule == "Inertia" else (s <= 0).astype(float)
    w[s == 0] = np.nan          # ties: keep previous holding
    w = w.ffill()
    return switch_daily(g, v, w, cost)


def strat_2xd(G, V, rule, variant, cost):
    s_on, s_id = V["on"] - G["on"], V["id"] - G["id"]
    if variant == "prev":
        sig_on = s_id.shift(1)      # overnight t follows intraday t-1
        sig_id = s_on               # intraday t follows overnight t
    else:
        sig_on = s_on.shift(1)
        sig_id = s_id.shift(1)
    win = lambda s: (s > 0).astype(float).where(s != 0)
    w_on, w_id = win(sig_on), win(sig_id)
    if rule == "Reversal":
        w_on, w_id = 1 - w_on, 1 - w_id
    seq = pd.concat([w_on.rename("on"), w_id.rename("id")], axis=1).stack(future_stack=True).ffill()
    wide = seq.unstack()
    w_on, w_id = wide["on"], wide["id"]
    to = (2 * seq.diff().abs().fillna(0)).unstack()
    first = w_on.first_valid_index()
    sl = slice(first, None)
    r_on = (w_on * V["on"] + (1 - w_on) * G["on"]).loc[sl]
    r_id = (w_id * V["id"] + (1 - w_id) * G["id"]).loc[sl]
    to_on, to_id = to["on"].loc[sl].copy(), to["id"].loc[sl]
    to_on.iloc[0] = 0.0
    gross = (1 + r_on) * (1 + r_id) - 1
    net = (1 + r_on - cost * to_on) * (1 + r_id - cost * to_id) - 1
    return net, gross, to_on + to_id


# ---------------------------------------------------------------- main
def main():
    RES.mkdir(exist_ok=True)
    irx = load("^IRX")["Close"].replace(0, np.nan).ffill()
    rows, spa_rows, desc_rows, ac_rows, series = [], [], [], [], {}
    rng = np.random.default_rng(SEED)

    for grp, name, b, gt, vt in TRIPLETS:
        R = {t: returns(t) for t in (b, gt, vt)}
        idx = R[b].index.intersection(R[gt].index).intersection(R[vt].index)
        idx = idx[(idx >= START[grp]) & (idx <= END)]
        R = {t: R[t].loc[idx] for t in R}
        ok = pd.concat([R[t] for t in R], axis=1).notna().all(axis=1)
        idx = idx[ok.values]
        R = {t: R[t].loc[idx] for t in R}
        G, V, B = R[gt], R[vt], R[b]
        rf = (irx.reindex(idx, method="ffill") / 100 / N).fillna(0)

        # descriptive: value-minus-growth spread by session
        for leg, lab in [("cc", "Close-to-close"), ("on", "Overnight"), ("id", "Intraday")]:
            s = V[leg] - G[leg]
            desc_rows.append({"group": grp, "triplet": name, "leg": lab,
                              "ann_mean": s.mean() * N, "ann_vol": s.std() * np.sqrt(N),
                              "t": s.mean() / s.std() * np.sqrt(len(s)),
                              "share_pos": (s > 0).mean()})
        for f in ["D", "W", "M", "Q"]:
            reb = rebal_dates(idx, f)
            s = period_ret(V["cc"], reb) - period_ret(G["cc"], reb)
            s = s.iloc[1:-1] if f != "D" else s.iloc[1:]
            ac = s.autocorr(1)
            ac_rows.append({"group": grp, "triplet": name, "freq": f, "n": len(s),
                            "ac1": ac, "z": ac * np.sqrt(len(s)),
                            "p_inertia_hit": ((s.shift(1) * s) > 0).iloc[1:].mean()})
        s_on, s_id = V["on"] - G["on"], V["id"] - G["id"]
        for lab, a, bb in [("Overnight(t) on intraday(t-1)", s_on, s_id.shift(1)),
                           ("Intraday(t) on overnight(t)", s_id, s_on),
                           ("Overnight(t) on overnight(t-1)", s_on, s_on.shift(1)),
                           ("Intraday(t) on intraday(t-1)", s_id, s_id.shift(1))]:
            c = pd.concat([a, bb], axis=1).dropna().corr().iloc[0, 1]
            n = pd.concat([a, bb], axis=1).dropna().shape[0]
            ac_rows.append({"group": grp, "triplet": name, "freq": "2xD: " + lab,
                            "n": n, "ac1": c, "z": c * np.sqrt(n), "p_inertia_hit": np.nan})

        # static comparisons
        statics = {"Benchmark": B["cc"], "Growth": G["cc"], "Value": V["cc"]}
        for sname, r in statics.items():
            p = perf(r, rf)
            for c in COSTS:
                rows.append({"group": grp, "triplet": name, "freq": "B&H", "rule": sname,
                             "variant": "", "cost_bps": c, **p, "switches_yr": 0.0,
                             "alpha": np.nan, "alpha_t": np.nan, "beta": np.nan,
                             "p_LW": np.nan, "breakeven_bps": np.nan,
                             "start": idx[0].date(), "end": idx[-1].date()})
            series[(name, "B&H", sname, "", 0.0)] = r

        for c in COSTS:
            fam = []
            for f in FREQS:
                variants = ["prev", "same"] if f == "2xD" else [""]
                if f == "2xD" and grp == "International":
                    continue
                for var in variants:
                    for rule in ["Inertia", "Reversal"]:
                        if f == "2xD":
                            net, gross, to = strat_2xd(G, V, rule, var, c / 1e4)
                        else:
                            net, gross, to = strat_freq(G["cc"], V["cc"], f, rule, c / 1e4)
                        rfx, bx = rf.loc[net.index], B["cc"].loc[net.index]
                        p = perf(net, rfx)
                        al, at, be = alpha_nw((net - rfx).values, (bx - rfx).values)
                        pl = lw_sharpe_p((net - rfx).values, (bx - rfx).values)
                        d_gross = (gross - bx).mean()
                        be_bps = d_gross / to.mean() * 1e4 if to.mean() > 0 else np.nan
                        rows.append({"group": grp, "triplet": name, "freq": f, "rule": rule,
                                     "variant": var, "cost_bps": c, **p,
                                     "switches_yr": to.sum() / 2 / len(to) * N,
                                     "alpha": al, "alpha_t": at, "beta": be, "p_LW": pl,
                                     "breakeven_bps": be_bps,
                                     "start": net.index[0].date(), "end": net.index[-1].date()})
                        series[(name, f, rule, var, c)] = net
                        fam.append((net - bx).rename(f"{f}{var}:{rule}"))
            F = pd.concat(fam, axis=1).dropna()
            cc_cols = [k for k in F.columns if not k.startswith("2xD")]
            spa_rows.append({"group": grp, "triplet": name, "cost_bps": c,
                             "n_all": F.shape[1], "spa_p_all": spa(F.values, rng),
                             "n_cc": len(cc_cols), "spa_p_cc": spa(F[cc_cols].values, rng)})
        print(name, idx[0].date(), idx[-1].date(), len(idx))

    res = pd.DataFrame(rows)
    res.to_csv(RES / "results.csv", index=False)
    pd.DataFrame(spa_rows).to_csv(RES / "spa.csv", index=False)
    pd.DataFrame(desc_rows).to_csv(RES / "spread_by_session.csv", index=False)
    pd.DataFrame(ac_rows).to_csv(RES / "autocorr.csv", index=False)

    # subperiods (0 bps and 2 bps), main + robustness
    sub = []
    periods = [("2001--2007", "2001-09-01", "2007-12-31"),
               ("2008--2019", "2008-01-01", "2019-12-31"),
               ("2020--2026", "2020-01-01", END)]
    for (name, f, rule, var, c), r in series.items():
        if c not in (0.0, 2.0) and f != "B&H":
            continue
        for lab, a, z in periods:
            x = r.loc[a:z]
            if len(x) < 100:
                continue
            rfx = (irx.reindex(x.index, method="ffill") / 100 / N).fillna(0)
            sub.append({"triplet": name, "freq": f, "rule": rule, "variant": var,
                        "cost_bps": c, "period": lab, **perf(x, rfx)})
    pd.DataFrame(sub).to_csv(RES / "subperiods.csv", index=False)
    pd.to_pickle(series, RES / "series.pkl")


if __name__ == "__main__":
    main()

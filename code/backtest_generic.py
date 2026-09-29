"""Growth-vs-value rotation backtest across rebalancing frequencies.

Question: is it profitable to rotate between a growth ETF and a value ETF, and
at which rebalancing frequency (twice-daily, daily, weekly, monthly, quarterly)?

Design
------
* Long-only switching: at each rebalance date the strategy holds 100% growth or
  100% value, decided from information available at that time (no look-ahead).
* Signals, fixed in advance:
    mom_L  - value if the value-minus-growth spread over the last L periods > 0
    rev_L  - the opposite of mom_L
    rates_L- value if the 10-year yield (^TNX) rose over the last L periods
  Twice-daily adds session-specific signals (see run_twice_daily).
* Comparisons: buy-and-hold benchmark ETF, static 50/50 growth/value
  rebalanced at the same frequency, always-growth, always-value.
* Costs: every switch sells one ETF and buys the other (two one-way trades).
  We report break-even one-way cost (bps) and results net of COST_BPS.
  Corwin-Schultz high-low spread estimates give a data-based cost reference.
* Inference: Ledoit-Wolf (2008) HAC test for Sharpe ratio differences and
  Hansen (2005) SPA test across the whole family of strategies per triplet.

Returns built from Yahoo data (see download_yahoo.py):
    close-to-close total return  cc = AdjClose_t / AdjClose_{t-1} - 1
    intraday (open to close)     id = Close_t / Open_t - 1
    overnight (close to open)    on = (1 + cc) / (1 + id) - 1
so the overnight leg automatically carries dividends (ex-dates fall at the open).

Usage:  python backtest.py            (reads data/, writes results/)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "results"

START = "2001-08-01"          # after Russell Midcap launch; post-decimalization
END = None                    # None = use all data
COST_BPS = 2.0                # one-way cost used for "net" results (bps)
TWICE_DAILY_SETS = ["main"]   # sets for which the overnight/intraday rotation runs
SPA_BOOT = 1000               # bootstrap draws for the SPA test
SPA_BLOCK = 20                # mean block length (days) for the stationary bootstrap
SEED = 12345

FREQS = {  # frequency -> lookbacks, in periods of that frequency
    "D": [1, 5, 21, 63],
    "W": [1, 4, 13, 52],
    "M": [1, 3, 6, 12],
    "Q": [1, 2, 4],
}
TWICE_DAILY_LOOKBACKS = [1, 5, 21, 63]   # days
PERIODS_PER_YEAR = 252


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _file(ticker: str) -> Path:
    return DATA / (ticker.replace("^", "IDX_") + ".csv")


def load_etf(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(_file(ticker), index_col="Date", parse_dates=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df["Open"] > 0) & (df["Close"] > 0) & (df["Adj Close"] > 0)]
    out = pd.DataFrame(index=df.index)
    out["cc"] = df["Adj Close"].pct_change()
    out["id"] = df["Close"] / df["Open"] - 1.0
    out["on"] = (1.0 + out["cc"]) / (1.0 + out["id"]) - 1.0
    out["high"], out["low"] = df["High"], df["Low"]
    out["open"], out["close"] = df["Open"], df["Close"]
    return out


def load_series(ticker: str) -> pd.Series:
    df = pd.read_csv(_file(ticker), index_col="Date", parse_dates=True).sort_index()
    return df["Close"].replace(0, np.nan).ffill()


def data_quality(ticker: str, d: pd.DataFrame) -> pd.DataFrame:
    """Share of days per year with suspicious opens (matters for twice-daily)."""
    prev_close = d["close"].shift(1)
    flags = pd.DataFrame({
        "open_eq_prev_close": np.isclose(d["open"], prev_close),
        "flat_bar": np.isclose(d["high"], d["low"]),
        "open_outside_hl": (d["open"] > d["high"] * 1.0001) | (d["open"] < d["low"] * 0.9999),
    }, index=d.index)
    q = flags.groupby(flags.index.year).mean()
    q.insert(0, "ticker", ticker)
    return q


def corwin_schultz(d: pd.DataFrame) -> pd.Series:
    """Corwin-Schultz (2012) spread estimate, monthly mean, negatives set to 0.
    Returns the proportional full spread; one-way cost is half of it."""
    h, l = np.log(d["high"]), np.log(d["low"])
    beta = (h - l) ** 2 + (h.shift(1) - l.shift(1)) ** 2
    h2 = np.maximum(d["high"], d["high"].shift(1))
    l2 = np.minimum(d["low"], d["low"].shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0).resample("ME").mean()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "D":
        return idx
    per = {"W": "W-FRI", "M": "M", "Q": "Q"}[freq]
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period(per)).max().values)


def period_returns(r: pd.Series, reb: pd.DatetimeIndex) -> pd.Series:
    """Compound daily returns into returns between consecutive rebalance dates."""
    grp = np.searchsorted(reb.values, r.index.values, side="left")
    pr = (1 + r).groupby(grp).prod() - 1
    pr = pr[pr.index < len(reb)]
    return pd.Series(pr.values, index=reb[pr.index])


def switch_returns(g: pd.Series, v: pd.Series, w_reb: pd.Series, cost: float):
    """Daily returns of a long-only switch strategy.
    w_reb: weight in value (0/1) decided at each rebalance close."""
    w = w_reb.reindex(g.index).ffill().shift(1)
    w = w.fillna(0.5)   # 50/50 during the signal's warm-up period
    turnover = 2.0 * w.diff().abs().fillna(0.0)          # one-way trades
    gross = w * v + (1 - w) * g
    return gross - cost * turnover, gross, turnover


def static_mix(g: pd.Series, v: pd.Series, reb: pd.DatetimeIndex, cost: float,
               wv: float = 0.5):
    """50/50 mix with drift, rebalanced back to target at each rebalance date."""
    rebset = set(reb)
    a, b = 1 - wv, wv
    net, gross, to = [], [], []
    for t, rg, rv in zip(g.index, g.values, v.values):
        port = a * (1 + rg) + b * (1 + rv)
        r = port - 1
        a, b = a * (1 + rg) / port, b * (1 + rv) / port
        turn = 0.0
        if t in rebset:
            turn = abs(a - (1 - wv)) + abs(b - wv)
            a, b = 1 - wv, wv
        gross.append(r); net.append(r - cost * turn); to.append(turn)
    idx = g.index
    return pd.Series(net, idx), pd.Series(gross, idx), pd.Series(to, idx)


def perf(r: pd.Series, rf: pd.Series) -> dict:
    ex = r - rf
    n = len(r)
    wealth = (1 + r).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    return {
        "ann_ret": wealth.iloc[-1] ** (PERIODS_PER_YEAR / n) - 1,
        "ann_vol": r.std() * np.sqrt(PERIODS_PER_YEAR),
        "sharpe": ex.mean() / ex.std() * np.sqrt(PERIODS_PER_YEAR),
        "max_dd": dd,
    }


def _newey_west(u: np.ndarray, lags: int) -> np.ndarray:
    u = u - u.mean(0)
    n = len(u)
    S = u.T @ u / n
    for j in range(1, lags + 1):
        G = u[j:].T @ u[:-j] / n
        S += (1 - j / (lags + 1)) * (G + G.T)
    return S


def ledoit_wolf_sharpe(r1: pd.Series, r2: pd.Series, rf: pd.Series) -> tuple[float, float]:
    """Ledoit-Wolf (2008) HAC test of H0: SR1 = SR2. Returns (diff_annual, p)."""
    x = np.column_stack([r1 - rf, r2 - rf])
    x = x[~np.isnan(x).any(1)]
    n = len(x)
    m = x.mean(0); m2 = (x ** 2).mean(0)
    sr = m / np.sqrt(m2 - m ** 2)
    u = np.column_stack([x, x ** 2])
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    psi = _newey_west(u, lags)
    grad = np.array([
        m2[0] / (m2[0] - m[0] ** 2) ** 1.5,
        -m2[1] / (m2[1] - m[1] ** 2) ** 1.5,
        -0.5 * m[0] / (m2[0] - m[0] ** 2) ** 1.5,
        0.5 * m[1] / (m2[1] - m[1] ** 2) ** 1.5,
    ])
    se = np.sqrt(grad @ psi @ grad / n)
    diff = sr[0] - sr[1]
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(diff / se) / sqrt(2))))
    return diff * np.sqrt(PERIODS_PER_YEAR), p


def _stationary_bootstrap_idx(n: int, block: int, rng) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    newblock = rng.random(n) < 1.0 / block
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if newblock[t] else (idx[t - 1] + 1) % n
    return idx


def spa_test(d: pd.DataFrame, boot: int, block: int, seed: int) -> dict:
    """Hansen (2005) SPA test, consistent p-value.
    d: daily loss differentials (strategy minus comparison), one column each.
    H0: no strategy beats the comparison."""
    x = d.dropna().values
    n, k = x.shape
    rng = np.random.default_rng(seed)
    mean = x.mean(0)
    boots = np.empty((boot, k))
    for b in range(boot):
        boots[b] = x[_stationary_bootstrap_idx(n, block, rng)].mean(0)
    omega = np.sqrt(n) * boots.std(0)
    omega[omega == 0] = np.inf
    t_stat = max(0.0, np.max(np.sqrt(n) * mean / omega))
    keep = np.sqrt(n) * mean / omega >= -np.sqrt(2 * np.log(np.log(n)))
    mu_c = mean * keep
    z = np.sqrt(n) * (boots - mu_c) / omega
    t_boot = np.maximum(0.0, z.max(1))
    best = d.columns[int(np.argmax(mean / (omega / np.sqrt(n))))]
    return {"spa_stat": t_stat, "spa_p": float((t_boot > t_stat).mean()),
            "n_strategies": k, "best_strategy": best}


# ----------------------------------------------------------------------------
# Strategy runners
# ----------------------------------------------------------------------------
@dataclass
class Triplet:
    set: str
    name: str
    bench: str
    growth: str
    value: str


def run_frequency(tp: Triplet, R: dict, rf: pd.Series, tnx: pd.Series,
                  freq: str, cost: float):
    g, v, b = R[tp.growth]["cc"], R[tp.value]["cc"], R[tp.bench]["cc"]
    reb = rebalance_dates(g.index, freq)
    sg, sv = period_returns(g, reb), period_returns(v, reb)
    spread = sv - sg
    tnx_reb = tnx.reindex(reb, method="ffill")

    signals = {}
    for L in FREQS[freq]:
        m = spread.rolling(L).sum()
        signals[f"mom_{L}"] = (m > 0).astype(float).where(m.notna())
        signals[f"rev_{L}"] = (m <= 0).astype(float).where(m.notna())
        dy = tnx_reb.diff(L)
        signals[f"rates_{L}"] = (dy > 0).astype(float).where(dy.notna())

    rows, series = [], {}
    mix_net, mix_gross, mix_to = static_mix(g, v, reb, cost)
    comps = {"benchmark": b, "mix5050": mix_gross, "value": v, "growth": g}
    for sig, w in signals.items():
        net, gross, to = switch_returns(g, v, w, cost)
        row = {"set": tp.set, "triplet": tp.name, "freq": freq, "signal": sig,
               "switches_per_year": to.sum() / 2 / len(to) * PERIODS_PER_YEAR}
        row.update({f"gross_{k}": x for k, x in perf(gross, rf).items()})
        row.update({f"net_{k}": x for k, x in perf(net, rf).items()})
        avg_to = to.mean()
        for cname, c in comps.items():
            dmean = (gross - c).mean()
            row[f"breakeven_bps_vs_{cname}"] = (dmean / avg_to * 1e4) if avg_to > 0 else np.nan
        for cname in ["benchmark", "mix5050"]:
            dsr, p = ledoit_wolf_sharpe(net, comps[cname], rf)
            row[f"dSharpe_net_vs_{cname}"], row[f"p_LW_vs_{cname}"] = dsr, p
        rows.append(row)
        series[f"{freq}:{sig}"] = net
    return rows, series


def run_twice_daily(tp: Triplet, R: dict, rf: pd.Series, cost: float):
    """Separate positions for the overnight and the intraday session.
    Signals (decided with information available before each session starts):
      sess_mom_L : hold value overnight if value beat growth overnight on
                   average over the last L days; same for the intraday session.
      sess_rev_L : the opposite.
      on2id_mom  : overnight as sess_mom_1; intraday follows today's overnight winner.
      on2id_rev  : overnight as sess_mom_1; intraday holds today's overnight loser.
    """
    G, V, B = R[tp.growth], R[tp.value], R[tp.bench]
    s_on, s_id = V["on"] - G["on"], V["id"] - G["id"]
    sigs = {}
    for L in TWICE_DAILY_LOOKBACKS:
        m_on = s_on.rolling(L).mean().shift(1)
        m_id = s_id.rolling(L).mean().shift(1)
        sigs[f"sess_mom_{L}"] = ((m_on > 0).astype(float), (m_id > 0).astype(float), m_on, m_id)
        sigs[f"sess_rev_{L}"] = ((m_on <= 0).astype(float), (m_id <= 0).astype(float), m_on, m_id)
    m_on1 = s_on.shift(1)
    sigs["on2id_mom"] = ((m_on1 > 0).astype(float), (s_on > 0).astype(float), m_on1, s_on)
    sigs["on2id_rev"] = ((m_on1 > 0).astype(float), (s_on <= 0).astype(float), m_on1, s_on)

    rows, series = [], {}
    mix_net, mix_gross, _ = static_mix(G["cc"], V["cc"], G.index, cost)
    comps = {"benchmark": B["cc"], "mix5050": mix_gross, "value": V["cc"], "growth": G["cc"]}
    for name, (w_on, w_id, a, b_) in sigs.items():
        valid = a.notna() & b_.notna()
        w_on, w_id = w_on.where(valid), w_id.where(valid)
        # interleave sessions: on(t), id(t), on(t+1), ...
        seq = pd.concat([w_on.rename(0), w_id.rename(1)], axis=1).stack(future_stack=True)
        seq = seq.ffill().fillna(0.5)
        to = (2.0 * seq.diff().abs().fillna(0.0)).unstack()
        r_on_g = w_on.fillna(0.5) * V["on"] + (1 - w_on.fillna(0.5)) * G["on"]
        r_id_g = w_id.fillna(0.5) * V["id"] + (1 - w_id.fillna(0.5)) * G["id"]
        gross = (1 + r_on_g) * (1 + r_id_g) - 1
        net = (1 + r_on_g - cost * to[0]) * (1 + r_id_g - cost * to[1]) - 1
        daily_to = to.sum(1)
        row = {"set": tp.set, "triplet": tp.name, "freq": "2xD", "signal": name,
               "switches_per_year": daily_to.sum() / 2 / len(daily_to) * PERIODS_PER_YEAR}
        row.update({f"gross_{k}": x for k, x in perf(gross, rf).items()})
        row.update({f"net_{k}": x for k, x in perf(net, rf).items()})
        avg_to = daily_to.mean()
        for cname, c in comps.items():
            dmean = (gross - c).mean()
            row[f"breakeven_bps_vs_{cname}"] = (dmean / avg_to * 1e4) if avg_to > 0 else np.nan
        for cname in ["benchmark", "mix5050"]:
            dsr, p = ledoit_wolf_sharpe(net, comps[cname], rf)
            row[f"dSharpe_net_vs_{cname}"], row[f"p_LW_vs_{cname}"] = dsr, p
        rows.append(row)
        series[f"2xD:{name}"] = net
    return rows, series


def style_decomposition(tp: Triplet, R: dict) -> dict:
    """Annualized mean value-minus-growth spread, by session (descriptive)."""
    G, V = R[tp.growth], R[tp.value]
    out = {"set": tp.set, "triplet": tp.name}
    for leg in ["cc", "on", "id"]:
        s = V[leg] - G[leg]
        out[f"spread_{leg}_ann_mean"] = s.mean() * PERIODS_PER_YEAR
        out[f"spread_{leg}_tstat"] = s.mean() / s.std() * np.sqrt(len(s))
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(exist_ok=True)
    meta = pd.read_csv(HERE / "tickers.csv")
    etfs = meta[meta["set"] != "signal"]
    triplets = []
    for (s, name), grp in etfs.groupby(["set", "triplet"], sort=False):
        legs = dict(zip(grp["leg"], grp["ticker"]))
        triplets.append(Triplet(s, name, legs["benchmark"], legs["growth"], legs["value"]))

    raw = {t: load_etf(t) for t in etfs["ticker"]}
    irx = load_series("^IRX")
    tnx = load_series("^TNX")

    pd.concat([data_quality(t, d) for t, d in raw.items()]).to_csv(OUT / "data_quality.csv")
    spreads = pd.DataFrame({t: corwin_schultz(d) for t, d in raw.items()})
    spreads.groupby(spreads.index.year).median().mul(1e4 / 2).round(2) \
        .to_csv(OUT / "one_way_cost_bps_corwin_schultz.csv")

    cost = COST_BPS / 1e4
    all_rows, spa_rows, decomp = [], [], []
    for tp in triplets:
        tick = [tp.bench, tp.growth, tp.value]
        idx = raw[tick[0]].index
        for t in tick[1:]:
            idx = idx.intersection(raw[t].index)
        idx = idx[idx >= pd.Timestamp(START)]
        if END:
            idx = idx[idx <= pd.Timestamp(END)]
        R = {t: raw[t].reindex(idx) for t in tick}
        first = max(R[t]["cc"].first_valid_index() for t in tick)
        idx = idx[idx >= first]
        R = {t: R[t].loc[idx].dropna(subset=["cc", "on", "id"]) for t in tick}
        idx = R[tick[0]].index.intersection(R[tick[1]].index).intersection(R[tick[2]].index)
        R = {t: R[t].loc[idx] for t in tick}
        rf = (irx.reindex(idx, method="ffill") / 100 / PERIODS_PER_YEAR).fillna(0.0)
        tnx_t = tnx.reindex(idx, method="ffill")
        print(f"{tp.name}: {idx[0].date()} to {idx[-1].date()} ({len(idx)} days)")

        fam = {}
        for freq in FREQS:
            rows, ser = run_frequency(tp, R, rf, tnx_t, freq, cost)
            all_rows += rows; fam.update(ser)
        if tp.set in TWICE_DAILY_SETS:
            rows, ser = run_twice_daily(tp, R, rf, cost)
            all_rows += rows; fam.update(ser)
            decomp.append(style_decomposition(tp, R))

        bench = R[tp.bench]["cc"]
        d = pd.DataFrame({k: s - bench for k, s in fam.items()})
        res = spa_test(d, SPA_BOOT, SPA_BLOCK, SEED)
        res.update({"set": tp.set, "triplet": tp.name, "comparison": "benchmark",
                    "cost_bps": COST_BPS, "start": idx[0].date(), "end": idx[-1].date()})
        spa_rows.append(res)

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT / "rotation_results.csv", index=False)
    pd.DataFrame(spa_rows).to_csv(OUT / "spa_tests.csv", index=False)
    if decomp:
        pd.DataFrame(decomp).to_csv(OUT / "session_decomposition.csv", index=False)

    # Compact view: best net Sharpe per triplet x frequency
    best = res.loc[res.groupby(["triplet", "freq"])["net_sharpe"].idxmax(),
                   ["set", "triplet", "freq", "signal", "net_sharpe",
                    "dSharpe_net_vs_benchmark", "p_LW_vs_benchmark",
                    "breakeven_bps_vs_benchmark", "switches_per_year"]]
    best.to_csv(OUT / "best_by_frequency.csv", index=False)
    print(best.round(3).to_string(index=False))
    print(pd.DataFrame(spa_rows)[["triplet", "n_strategies", "spa_p", "best_strategy"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()

"""Build LaTeX tables and figures for the paper from ../results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as A

ROOT = A.ROOT
RES, TAB, FIG = ROOT / "results", ROOT / "tables", ROOT / "figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

r = pd.read_csv(RES / "results.csv")
r["variant"] = r["variant"].fillna("")
spa = pd.read_csv(RES / "spa.csv")
ac = pd.read_csv(RES / "autocorr.csv")
ses = pd.read_csv(RES / "spread_by_session.csv")
sub = pd.read_csv(RES / "subperiods.csv")
sub["variant"] = sub["variant"].fillna("")

INK, INK2 = "#1f1f1e", "#6b6a64"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

STRATS = [("2xD", "prev", "Inertia"), ("2xD", "prev", "Reversal"),
          ("2xD", "same", "Inertia"), ("2xD", "same", "Reversal"),
          ("D", "", "Inertia"), ("D", "", "Reversal"),
          ("W", "", "Inertia"), ("W", "", "Reversal"),
          ("M", "", "Inertia"), ("M", "", "Reversal"),
          ("Q", "", "Inertia"), ("Q", "", "Reversal")]
FLAB = {"2xD": "Twice-daily", "D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly"}


def slab(f, v, rule):
    if f == "2xD":
        return f"2$\\times$D-{'P' if v == 'prev' else 'S'} {rule}"
    return f"{FLAB[f]} {rule}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def get(tr, f, v, rule, c):
    q = r[(r.triplet == tr) & (r.freq == f) & (r.variant == v) & (r["rule"] == rule) & (r.cost_bps == c)]
    return q.iloc[0] if len(q) else None


def star(t):
    a = abs(t)
    return "$^{***}$" if a >= 2.576 else "$^{**}$" if a >= 1.96 else "$^{*}$" if a >= 1.645 else ""


# ------------------------------------------------------------------ Table 1
def table_data():
    rows = []
    for grp, name, b, g, v in A.TRIPLETS:
        for leg, t in [("Benchmark", b), ("Growth", g), ("Value", v)]:
            x = get(name, "B&H", "", leg, 0.0)
            d = A.load(t)
            d = d.loc[str(x.start):str(x.end)]
            cs = (0.005 / d["Close"].loc["2025-10-01":]).median() * 1e4
            rows.append((grp, name, leg, t, x, d["Volume"].median() / 1e6, cs))
    L = []
    L.append(r"\begin{table}[H]")
    L.append(r"\caption{Sample, buy-and-hold performance and liquidity of the ETF triplets. CAGR is the compound annual growth rate of total return; Vol is annualized volatility; Sharpe uses the 13-week Treasury bill as the risk-free rate; MaxDD is the maximum drawdown. Volume is the median daily share volume (millions). Half-tick is half a one-cent tick divided by the median closing price over the last twelve months of the sample, in basis points: a lower bound on the one-way cost of a market order for a fund quoted at the minimum spread.\label{tab:data}}")
    L.append(r"\begin{adjustwidth}{-\extralength}{0cm}")
    L.append(r"\newcolumntype{C}{>{\centering\arraybackslash}X}")
    L.append(r"\begin{tabularx}{\fulllength}{llllCCCCCC}")
    L.append(r"\toprule")
    L.append(r"\textbf{Triplet} & \textbf{Sample} & \textbf{Leg} & \textbf{Ticker} & \textbf{CAGR (\%)} & \textbf{Vol (\%)} & \textbf{Sharpe} & \textbf{MaxDD (\%)} & \textbf{Volume} & \textbf{Half-tick (bps)} \\")
    L.append(r"\midrule")
    last = None
    for grp, name, leg, t, x, vol, cs in rows:
        if grp != last:
            if last is not None:
                L.append(r"\midrule")
            L.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{grp} sample}}}} \\")
            last = grp
        tri = name.replace("&", r"\&") if leg == "Benchmark" else ""
        samp = f"{str(x.start)[:7]} to {str(x.end)[:7]}" if leg == "Benchmark" else ""
        L.append(f"{tri} & {samp} & {leg} & {t} & {pct(x.CAGR)} & {pct(x.Vol)} & {x.Sharpe:.2f} & {pct(x.MaxDD)} & {vol:.2f} & {cs:.1f} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabularx}")
    L.append(r"\end{adjustwidth}")
    L.append(r"\end{table}")
    (TAB / "tab_data.tex").write_text("\n".join(L))


def cs_spread(d):
    h, l = np.log(d["High"]), np.log(d["Low"])
    beta = (h - l) ** 2 + (h.shift(1) - l.shift(1)) ** 2
    h2 = np.maximum(d["High"], d["High"].shift(1)); l2 = np.minimum(d["Low"], d["Low"].shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0).resample("ME").mean()


# ------------------------------------------------------------------ Table 2
def table_spread():
    L = [r"\begin{table}[H]",
         r"\caption{The value-minus-growth return spread. Panel A reports the annualized mean of the daily spread (value ETF return minus growth ETF return, in percent) for the close-to-close, overnight and intraday legs, with $t$-statistics in parentheses. Panel B reports the first-order autocorrelation of the spread measured at each rebalancing frequency and, for twice-daily rotation, the correlation between consecutive sessions. Under the null of no autocorrelation the standard error is approximately $1/\sqrt{n}$; significance at the 10\%, 5\% and 1\% levels is marked by $^{*}$, $^{**}$ and $^{***}$.\label{tab:spread}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
         r"\begin{tabularx}{\fulllength}{lCCCCCCC}",
         r"\toprule",
         r" & \textbf{R1000} & \textbf{R Midcap} & \textbf{R2000} & \textbf{S\&P 500} & \textbf{S\&P 400} & \textbf{S\&P 600} & \textbf{EAFE} \\",
         r"\midrule",
         r"\multicolumn{8}{l}{\textit{Panel A: Annualized mean spread, \% (t-statistic)}} \\"]
    names = [t[1] for t in A.TRIPLETS]
    for leg in ["Close-to-close", "Overnight", "Intraday"]:
        cells = []
        for n in names:
            x = ses[(ses.triplet == n) & (ses.leg == leg)].iloc[0]
            cells.append(f"{100 * x.ann_mean:.2f} ({x.t:.2f})")
        L.append(f"{leg} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{8}{l}{\textit{Panel B: First-order autocorrelation of the spread}} \\")
    for f, lab in [("D", "Daily"), ("W", "Weekly"), ("M", "Monthly"), ("Q", "Quarterly"),
                   ("2xD: Overnight(t) on intraday(t-1)", "Overnight$_t$ on intraday$_{t-1}$"),
                   ("2xD: Intraday(t) on overnight(t)", "Intraday$_t$ on overnight$_t$"),
                   ("2xD: Overnight(t) on overnight(t-1)", "Overnight$_t$ on overnight$_{t-1}$"),
                   ("2xD: Intraday(t) on intraday(t-1)", "Intraday$_t$ on intraday$_{t-1}$")]:
        cells = []
        for n in names:
            x = ac[(ac.triplet == n) & (ac.freq == f)].iloc[0]
            cells.append(f"{x.ac1:.3f}{star(x.z)}")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_spread.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ Tables 3-5 (+EAFE)
def table_full(tr, label, extra="", short=False):
    L = [r"\begin{table}[H]",
         (rf"\caption{{Rotation strategies versus buy-and-hold: {tr.replace('&', chr(92) + '&')}{extra}. Definitions as in Table~\ref{{tab:r1000}}.\label{{{label}}}}}" if short else None) or rf"\caption{{Rotation strategies versus buy-and-hold: {tr.replace('&', chr(92) + '&')}{extra}. Inertia holds the ETF (growth or value) that outperformed over the previous period; Reversal holds the one that underperformed. 2$\times$D-P rotates at the open and the close and follows the immediately preceding session; 2$\times$D-S follows the same session on the previous day. Sw/yr is the number of switches per year. BE is the break-even one-way cost (bps) at which the strategy's mean return equals the benchmark's. CAGR, Sharpe and $\alpha$ (annualized intercept from a regression of strategy excess returns on benchmark excess returns, \%, with Newey--West $t$-statistic in parentheses) are reported for one-way costs of 0, 1 and 2 bps. $^{{*}}$, $^{{**}}$, $^{{***}}$: Ledoit--Wolf test of equal Sharpe ratios with the benchmark rejects at 10\%, 5\%, 1\%.\label{{{label}}}}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
         r"\footnotesize\centering\setlength{\tabcolsep}{3.2pt}",
         r"\begin{tabular}{lrrccccccccc}",
         r"\toprule",
         r" & & & \multicolumn{3}{c}{\textbf{CAGR (\%)}} & \multicolumn{3}{c}{\textbf{Sharpe}} & \multicolumn{3}{c}{\textbf{$\alpha$ (\%) and $t$-stat}} \\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
         r"\textbf{Strategy} & \textbf{Sw/yr} & \textbf{BE} & \textbf{0} & \textbf{1} & \textbf{2} & \textbf{0} & \textbf{1} & \textbf{2} & \textbf{0 bps} & \textbf{1 bps} & \textbf{2 bps} \\",
         r"\midrule"]
    for leg in ["Benchmark", "Growth", "Value"]:
        x = get(tr, "B&H", "", leg, 0.0)
        L.append(f"Buy-and-hold {leg.lower()} & -- & -- & {pct(x.CAGR)} & {pct(x.CAGR)} & {pct(x.CAGR)} & {x.Sharpe:.2f} & {x.Sharpe:.2f} & {x.Sharpe:.2f} & -- & -- & -- \\\\")
    L.append(r"\midrule")
    for f, v, rule in STRATS:
        xs = [get(tr, f, v, rule, c) for c in A.COSTS]
        if xs[0] is None:
            continue
        x0 = xs[0]
        cagr = " & ".join(pct(x.CAGR) for x in xs)
        sr = " & ".join(f"{x.Sharpe:.2f}{pstar(x.p_LW)}" for x in xs)
        al = " & ".join(f"{100 * x.alpha:.1f} ({x.alpha_t:.1f})" for x in xs)
        L.append(f"{slab(f, v, rule)} & {x0.switches_yr:.0f} & {x0.breakeven_bps:.1f} & {cagr} & {sr} & {al} \\\\")
        if rule == "Reversal" and f != "Q":
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    fn = "tab_" + label.split(":")[1] + ".tex"
    (TAB / fn).write_text("\n".join(L))


def pstar(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


# ------------------------------------------------------------------ robustness compact
def table_robust():
    trs = ["S&P 500", "S&P MidCap 400", "S&P SmallCap 600"]
    L = [r"\begin{table}[H]",
         r"\caption{Robustness: S\&P triplets (IVV/IVW/IVE, IJH/IJK/IJJ, IJR/IJT/IJS), September 2001 to September 2026. Panel A reports Sharpe ratios and Panel B the Newey--West $t$-statistics of the annualized alpha relative to the benchmark ETF, at one-way costs of 0, 1 and 2 bps. Buy-and-hold benchmark Sharpe ratios are 0.51 (S\&P 500), 0.47 (S\&P 400) and 0.45 (S\&P 600). Stars in Panel A: Ledoit--Wolf test of equal Sharpe ratios with the benchmark rejects at 10\%, 5\%, 1\%.\label{tab:robust}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
         r"\footnotesize",
         r"\begin{tabularx}{\fulllength}{lCCCCCCCCC}",
         r"\toprule",
         r" & \multicolumn{3}{c}{\textbf{S\&P 500}} & \multicolumn{3}{c}{\textbf{S\&P MidCap 400}} & \multicolumn{3}{c}{\textbf{S\&P SmallCap 600}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"\textbf{Strategy} & \textbf{0} & \textbf{1} & \textbf{2} & \textbf{0} & \textbf{1} & \textbf{2} & \textbf{0} & \textbf{1} & \textbf{2} \\",
         r"\midrule"]
    for panel, key in [("Panel A: Sharpe ratio", "Sharpe"), ("Panel B: $t$-statistic of $\\alpha$", "alpha_t")]:
        L.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{panel}}}}} \\")
        for f, v, rule in STRATS:
            cells = []
            for tr in trs:
                for c in A.COSTS:
                    x = get(tr, f, v, rule, c)
                    cells.append(f"{x[key]:.2f}{pstar(x.p_LW) if key == 'Sharpe' else ''}")
            L.append(f"{slab(f, v, rule)} & " + " & ".join(cells) + r" \\")
        if key == "Sharpe":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_robust.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ SPA
def table_spa():
    L = [r"\begin{table}[H]",
         r"\caption{Data-snooping-robust inference. Consistent $p$-values of Hansen's (2005) test for superior predictive ability (SPA) of the null hypothesis that no rotation strategy in the family earns a higher mean daily return than buy-and-hold of the benchmark ETF. The full family contains all inertia and reversal strategies at all frequencies (12 for US triplets, 8 for EAFE); the close-to-close family excludes twice-daily rotation (8 strategies). Stationary bootstrap with mean block length of 20 days and 2000 replications.\label{tab:spa}}",
         r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
         r"\begin{tabularx}{\textwidth}{lCCCCCC}",
         r"\toprule",
         r" & \multicolumn{3}{c}{\textbf{Full family}} & \multicolumn{3}{c}{\textbf{Close-to-close family}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"\textbf{Triplet} & \textbf{0 bps} & \textbf{1 bps} & \textbf{2 bps} & \textbf{0 bps} & \textbf{1 bps} & \textbf{2 bps} \\",
         r"\midrule"]
    for _, name, *_ in A.TRIPLETS:
        q = spa[spa.triplet == name].set_index("cost_bps")
        a = " & ".join(f"{q.loc[c, 'spa_p_all']:.3f}" for c in A.COSTS)
        b = " & ".join(f"{q.loc[c, 'spa_p_cc']:.3f}" for c in A.COSTS)
        L.append(f"{name.replace('&', chr(92) + '&')} & {a} & {b} \\\\")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    (TAB / "tab_spa.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ subperiods
def table_sub():
    names = [t[1] for t in A.TRIPLETS]
    rows = [("B&H", "", "Benchmark", 0.0, "Buy-and-hold benchmark"),
            ("M", "", "Inertia", 2.0, "Monthly inertia (2 bps)"),
            ("D", "", "Reversal", 0.0, "Daily reversal (0 bps)"),
            ("2xD", "prev", "Reversal", 0.0, "2$\\times$D-P reversal (0 bps)"),
            ("2xD", "prev", "Reversal", 2.0, "2$\\times$D-P reversal (2 bps)")]
    L = [r"\begin{table}[H]",
         r"\caption{Sharpe ratios by subperiod for the benchmark and the main rotation strategies. The EAFE sample starts in September 2005 and has no twice-daily strategy.\label{tab:sub}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\newcolumntype{C}{>{\centering\arraybackslash}X}",
         r"\footnotesize",
         r"\begin{tabularx}{\fulllength}{llCCCCCCC}",
         r"\toprule",
         r"\textbf{Strategy} & \textbf{Period} & \textbf{R1000} & \textbf{R Midcap} & \textbf{R2000} & \textbf{S\&P 500} & \textbf{S\&P 400} & \textbf{S\&P 600} & \textbf{EAFE} \\",
         r"\midrule"]
    for f, v, rule, c, lab in rows:
        for i, per in enumerate(["2001--2007", "2008--2019", "2020--2026"]):
            cells = []
            for n in names:
                q = sub[(sub.triplet == n) & (sub.freq == f) & (sub.variant == v) & (sub["rule"] == rule)
                        & (sub.period == per) & ((sub.cost_bps == c) | (f == "B&H"))]
                cells.append(f"{q.iloc[0].Sharpe:.2f}" if len(q) else "--")
            L.append(f"{lab if i == 0 else ''} & {per} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabularx}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_sub.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ figures
def fig_wealth():
    series = pd.read_pickle(RES / "series.pkl")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, tr in zip(axes, ["Russell 1000", "Russell Midcap", "Russell 2000"]):
        lines = [(series[(tr, "B&H", "Benchmark", "", 0.0)], "Buy-and-hold benchmark", INK, "-"),
                 (series[(tr, "M", "Inertia", "", 1.0)], "Monthly inertia", BLUE, "-"),
                 (series[(tr, "D", "Reversal", "", 1.0)], "Daily reversal", ORANGE, "--"),
                 (series[(tr, "2xD", "Reversal", "prev", 1.0)], "Twice-daily reversal (P)", AQUA, "-.")]
        for s, lab, col, ls in lines:
            w = (1 + s).cumprod()
            ax.plot(w.index, w.values, color=col, lw=1.4 if col != INK else 1.2, ls=ls, label=lab)
        ax.set_yscale("log")
        ax.set_title(tr, fontsize=9, color=INK)
        ax.grid(axis="y", color="#e4e3dd", lw=0.6)
        ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "40"])
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(6, month=1, day=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("Growth of \\$1 (log scale)")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_wealth.pdf"); fig.savefig(FIG / "fig_wealth.png", dpi=200)


def fig_sharpe():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    freqs = ["2xD", "D", "W", "M", "Q"]
    x = np.arange(len(freqs))
    for ax, tr in zip(axes, ["Russell 1000", "Russell Midcap", "Russell 2000"]):
        b = get(tr, "B&H", "", "Benchmark", 0.0).Sharpe
        for k, (rule, col, hatch) in enumerate([("Inertia", BLUE, ""), ("Reversal", ORANGE, "////")]):
            vals = [get(tr, f, "prev" if f == "2xD" else "", rule, 0.0).Sharpe for f in freqs]
            ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=col, edgecolor="white",
                   linewidth=0.8, hatch=hatch, label=rule)
        ax.axhline(b, color=INK, lw=1, ls="--", label="Benchmark")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(["2×D", "D", "W", "M", "Q"])
        ax.set_title(tr, fontsize=9, color=INK)
        ax.grid(axis="y", color="#e4e3dd", lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Sharpe ratio (0 bps)")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sharpe.pdf"); fig.savefig(FIG / "fig_sharpe.png", dpi=200)


def fig_costs():
    """Sharpe ratio as a function of the one-way cost, Russell 2000."""
    costs = np.arange(0, 4.01, 0.25)
    tr = "Russell 2000"
    _, _, b, g, v = [t for t in A.TRIPLETS if t[1] == tr][0]
    R = {t: A.returns(t) for t in (b, g, v)}
    idx = R[b].index.intersection(R[g].index).intersection(R[v].index)
    idx = idx[(idx >= A.START["Main"]) & (idx <= A.END)]
    R = {t: R[t].loc[idx].dropna() for t in R}
    idx = R[b].index.intersection(R[g].index).intersection(R[v].index)
    R = {t: R[t].loc[idx] for t in R}
    irx = A.load("^IRX")["Close"].replace(0, np.nan).ffill()
    rf = (irx.reindex(idx, method="ffill") / 100 / A.N).fillna(0)
    bench = A.perf(R[b]["cc"], rf)["Sharpe"]
    specs = [("Monthly inertia", BLUE, "-", lambda c: A.strat_freq(R[g]["cc"], R[v]["cc"], "M", "Inertia", c)),
             ("Daily reversal", ORANGE, "--", lambda c: A.strat_freq(R[g]["cc"], R[v]["cc"], "D", "Reversal", c)),
             ("Twice-daily reversal (P)", AQUA, "-.", lambda c: A.strat_2xd(R[g], R[v], "Reversal", "prev", c))]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for lab, col, ls, fn in specs:
        s = [A.perf(fn(c / 1e4)[0], rf.loc[fn(c / 1e4)[0].index])["Sharpe"] for c in costs]
        ax.plot(costs, s, color=col, ls=ls, lw=1.6, label=lab)
    ax.axhline(bench, color=INK, lw=1, ls=":", label="Buy-and-hold benchmark")
    ax.set_xlabel("One-way trading cost (bps)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Russell 2000: sensitivity to trading costs", fontsize=9, color=INK)
    ax.grid(color="#e4e3dd", lw=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_costs.pdf"); fig.savefig(FIG / "fig_costs.png", dpi=200)


if __name__ == "__main__":
    table_data()
    table_spread()
    table_full("Russell 1000", "tab:r1000")
    table_full("Russell Midcap", "tab:rmid", short=True)
    table_full("Russell 2000", "tab:r2000", short=True)
    table_full("MSCI EAFE", "tab:eafe", " (international sample, September 2005 to September 2026; twice-daily rotation is not applicable because the constituents trade outside US hours)", short=True)
    table_robust()
    table_spa()
    table_sub()
    fig_wealth()
    fig_sharpe()
    fig_costs()
    print("done")

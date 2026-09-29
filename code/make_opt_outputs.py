"""LaTeX tables and figures for the optimization sections (reads ../results)."""
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as A
import dp_policy as DP
import optimization as O

RES, TAB, FIG = A.ROOT / "results", A.ROOT / "tables", A.ROOT / "figures"
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"
FCOL = {"D": ORANGE, "W": AQUA, "M": BLUE, "Q": PURPLE, "none": "#c9c8c2"}
FLS = {"D": "--", "W": "-.", "M": "-", "Q": ":"}
FLAB = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "none": "No rotation"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
NAMES = [t[1] for t in A.TRIPLETS]
RUSSELL = NAMES[:3]
esc = lambda s: s.replace("&", r"\&")

val = pd.read_csv(RES / "model_validation.csv")
oos = pd.read_csv(RES / "oos.csv")
use = pd.read_csv(RES / "oos_usage.csv")
mv = pd.read_csv(RES / "mv_kelly.csv")
zs = pd.read_csv(RES / "oos_z_sensitivity.csv")


def pstar(p):
    if not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


# ------------------------------------------------------------------ Table: model validation
def table_model():
    L = [r"\begin{table}[H]",
         r"\caption{The AR(1) model of the value-minus-growth spread against realized rotation outcomes, full sample. $\rho$ and $\sigma$ are the first-order autocorrelation and the standard deviation (\%) of the spread at each rebalancing frequency. Gain is the annual gross return of the inertia rule relative to the equal-weighted growth--value mix (\%): the model value is $N_f\rho\sigma/\sqrt{2\pi}$ (Proposition~\ref{prop:sign}) and the realized value is the sample mean, with its $t$-statistic in parentheses. Sw/yr is the number of switches per year, $N_f\,(1/2-\arcsin(\rho)/\pi)$ in the model. BE is the break-even one-way cost in bps, gain divided by twice the switching rate. Negative gains and break-even costs mean that reversal, not inertia, is profitable.\label{tab:model}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\footnotesize\centering\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrrrrrrr}",
         r"\toprule",
         r" & & & & \multicolumn{2}{c}{\textbf{Gain (\%/yr)}} & \multicolumn{2}{c}{\textbf{Sw/yr}} & \multicolumn{2}{c}{\textbf{BE (bps)}} \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"\textbf{Triplet} & \textbf{Freq.} & \textbf{$\rho$} & \textbf{$\sigma$ (\%)} & \textbf{Model} & \textbf{Realized} & \textbf{Model} & \textbf{Realized} & \textbf{Model} & \textbf{Realized} \\",
         r"\midrule"]
    for k, n in enumerate(NAMES):
        q = val[val.triplet == n]
        for i, (_, x) in enumerate(q.iterrows()):
            L.append(f"{esc(n) if i == 0 else ''} & {FLAB[x.freq]} & {x.rho:.3f} & {100 * x.sigma:.2f} & "
                     f"{100 * x.gross_model:.2f} & {100 * x.gross_real:.2f} ({x.gross_real_t:.1f}) & "
                     f"{x.sw_model:.1f} & {x.sw_real:.1f} & {x.be_model_bps:.1f} & {x.be_real_bps:.1f} \\\\")
        if k < len(NAMES) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_model.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ Table: OOS
SLAB = {"Benchmark": "Buy-and-hold benchmark", "Monthly inertia (fixed)": "Monthly inertia (fixed rule)",
        "Plug-in optimal": "Plug-in optimizer", "Robust optimal": "Robust optimizer"}


def table_oos():
    costs = O.COSTS_OOS
    L = [r"\begin{table}[H]",
         r"\caption{Out-of-sample performance of the cost-aware optimizer. From month 61 onward (October 2006 for the US, October 2010 for EAFE), at each month end the optimizer estimates $\rho$ and $\sigma$ of the spread at the daily, weekly, monthly and quarterly frequencies on an expanding window, solves problem~(\ref{eq:bellman}) for each, and implements the frequency, direction and no-trade band with the highest expected annual net gain, or holds the equal-weighted mix if no gain is positive. The plug-in optimizer uses the point estimate $\hat\rho$; the robust optimizer uses the worst case over $|\rho| \in [|\hat\rho| - 1.645\,\mathrm{se}, |\hat\rho| + 1.645\,\mathrm{se}]$ with White standard errors. The fixed monthly inertia rule is shown for comparison; its choice uses the full-sample evidence. Columns report Sharpe ratios at one-way costs of 0 to 10 bps, and switches per year and the annualized alpha against the benchmark (\%, Newey--West $t$-statistic) at 2 bps. Stars: Ledoit--Wolf test of equal Sharpe ratios with the benchmark.\label{tab:oos}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\footnotesize\centering\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{ll" + "c" * len(costs) + "rr}",
         r"\toprule",
         r" & & \multicolumn{" + str(len(costs)) + r"}{c}{\textbf{Sharpe ratio at one-way cost (bps)}} & \multicolumn{2}{c}{\textbf{At 2 bps}} \\",
         r"\cmidrule(lr){3-" + str(2 + len(costs)) + r"}\cmidrule(lr){" + str(3 + len(costs)) + "-" + str(4 + len(costs)) + "}",
         r"\textbf{Triplet} & \textbf{Strategy} & " + " & ".join(f"\\textbf{{{c:g}}}" for c in costs) + r" & \textbf{Sw/yr} & \textbf{$\alpha$ (\%)} \\",
         r"\midrule"]
    for k, n in enumerate(NAMES):
        for i, s in enumerate(SLAB):
            q = oos[(oos.triplet == n) & (oos.strategy == s)].set_index("cost_bps")
            cells = [f"{q.loc[c, 'Sharpe']:.2f}{pstar(q.loc[c, 'p_LW']) if s != 'Benchmark' else ''}" for c in costs]
            x = q.loc[2.0]
            tail = "-- & --" if s == "Benchmark" else f"{x.switches_yr:.1f} & {100 * x.alpha:.1f} ({x.alpha_t:.1f})"
            L.append(f"{esc(n) if i == 0 else ''} & {SLAB[s]} & " + " & ".join(cells) + f" & {tail} \\\\")
        if k < len(NAMES) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_oos.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ Table: frequency usage
def table_usage():
    costs = [0.0, 2.0, 5.0, 10.0]
    L = [r"\begin{table}[H]",
         r"\caption{Decisions of the robust optimizer: share of out-of-sample months (\%) in which each rebalancing frequency is selected, or no rotation is undertaken (None), by one-way cost.\label{tab:usage}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\footnotesize\centering\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "rrrrr" * len(costs) + "}",
         r"\toprule",
         " & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{c:g} bps}}}}" for c in costs) + r" \\",
         "".join(f"\\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}" for i in range(len(costs))),
         r"\textbf{Triplet} & " + " & ".join(["D & W & M & Q & None"] * len(costs)) + r" \\",
         r"\midrule"]
    for n in NAMES:
        cells = []
        for c in costs:
            x = use[(use.triplet == n) & (use.strategy == "Robust optimal") & (use.cost_bps == c)].iloc[0]
            cells += [f"{100 * x[k]:.0f}" for k in ["D", "W", "M", "Q", "none"]]
        L.append(f"{esc(n)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_usage.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ Table: risk
def table_risk():
    L = [r"\begin{table}[H]",
         r"\caption{Risk profile over the out-of-sample period (October 2006 to September 2026), Russell triplets, net of a one-way cost of 2 bps. Sortino is the annualized mean excess return divided by the annualized downside deviation of excess returns. MaxDD is the maximum drawdown. ES$_{95}$ is the expected shortfall of monthly returns at the 95\% level (the mean of the worst 5\% of months). Calmar is the CAGR divided by the absolute maximum drawdown. $\beta$ is the slope on the benchmark's excess return.\label{tab:risk}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\footnotesize\centering\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrrrrrrr}",
         r"\toprule",
         r"\textbf{Triplet} & \textbf{Strategy} & \textbf{CAGR (\%)} & \textbf{Vol (\%)} & \textbf{Sharpe} & \textbf{Sortino} & \textbf{MaxDD (\%)} & \textbf{ES$_{95}$ (\%)} & \textbf{Calmar} & \textbf{$\beta$} \\",
         r"\midrule"]
    for k, n in enumerate(RUSSELL):
        for i, s in enumerate(SLAB):
            x = oos[(oos.triplet == n) & (oos.strategy == s) & (oos.cost_bps == 2.0)].iloc[0]
            L.append(f"{esc(n) if i == 0 else ''} & {SLAB[s]} & {100 * x.CAGR:.1f} & {100 * x.Vol:.1f} & {x.Sharpe:.2f} & "
                     f"{x.Sortino:.2f} & {100 * x.MaxDD:.1f} & {100 * x.ES95_m:.1f} & {x.Calmar:.2f} & {x.beta:.2f} \\\\")
        if k < 2:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_risk.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ Table: MV / Kelly
def table_mv():
    gams = [1, 2, 5, 10, 25, 50, 100]
    L = [r"\begin{table}[H]",
         r"\caption{Benchmark-relative mean--variance weights with a long-only constraint, monthly rebalancing, out of sample (from October 2006), net of 2 bps. The value weight is $w_{k+1} = \min\{1, \max\{0, \tfrac12 + \hat\rho_k s_k/(\gamma\hat\sigma^2_k)\}\}$, with $\hat\rho_k$ and $\hat\sigma_k$ estimated on an expanding window. Bound is the share of months (\%) in which the constraint binds; $|w-\tfrac12|$ is the mean absolute tilt; TO is the annual sum of $|\Delta w|$, which equals the number of switches per year for the binary rule. The binary inertia rule is the limit $\gamma \to 0$.\label{tab:mv}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\footnotesize\centering\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{l" + "rrrrr" * 3 + "}",
         r"\toprule",
         " & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{esc(n)}}}}}" for n in RUSSELL) + r" \\",
         "".join(f"\\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}" for i in range(3)),
         r"$\boldsymbol{\gamma}$ & " + " & ".join([r"Bound & $|w-\tfrac12|$ & TO & Sharpe & $\alpha$ ($t$)"] * 3) + r" \\",
         r"\midrule"]
    for gm in gams:
        cells = []
        for n in RUSSELL:
            x = mv[(mv.triplet == n) & (mv.gamma == gm)].iloc[0]
            cells += [f"{100 * x.share_at_bound:.0f}", f"{x.mean_abs_tilt:.2f}", f"{x.turnover_yr:.1f}",
                      f"{x.Sharpe:.2f}", f"{100 * x.alpha:.1f} ({x.alpha_t:.1f})"]
        L.append(f"{gm} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_mv.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ Table: z sensitivity
def table_z():
    zz = [0.0, 1.0, 1.645, 2.326]
    L = [r"\begin{table}[H]",
         r"\caption{Sensitivity of the out-of-sample optimizer to the size of the uncertainty set, net of 2 bps. $z$ is the number of White standard errors by which $|\hat\rho|$ is shrunk; $z=0$ is the plug-in optimizer and $z=1.645$ the robust optimizer of Table~\ref{tab:oos}. Sharpe is the Sharpe ratio; Sw/yr the switches per year. The benchmark's Sharpe ratio over the same period is shown in the first column.\label{tab:zsens}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}",
         r"\footnotesize\centering\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lr" + "rr" * len(zz) + "}",
         r"\toprule",
         r" & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{$z = {z:g}$}}}}" for z in zz) + r" \\",
         "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(zz))),
         r"\textbf{Triplet} & \textbf{B\&H} & " + " & ".join([r"Sharpe & Sw/yr"] * len(zz)) + r" \\",
         r"\midrule"]
    for n in NAMES:
        b = oos[(oos.triplet == n) & (oos.strategy == "Benchmark") & (oos.cost_bps == 2.0)].iloc[0].Sharpe
        cells = []
        for z in zz:
            x = zs[(zs.triplet == n) & (np.isclose(zs.z, z))].iloc[0]
            cells += [f"{x.Sharpe:.2f}", f"{x.switches_yr:.1f}"]
        L.append(f"{esc(n)} & {b:.2f} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    (TAB / "tab_zsens.tex").write_text("\n".join(L))


# ------------------------------------------------------------------ Figure: optimal band
def fig_band():
    kap = np.linspace(0, 0.3, 61)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for r, col, ls in [(0.05, PURPLE, ":"), (0.1, AQUA, "-."), (0.2, BLUE, "-"), (0.3, ORANGE, "--")]:
        bands, ratio, ratio0 = [], [], []
        for k in kap:
            g, b, _ = DP.solve(r, k)
            gross0 = r / sqrt(2 * np.pi)
            bands.append(b if np.isfinite(b) else np.nan)
            ratio.append(g / gross0)
            ratio0.append(DP.closed_form(r, k)[2] / gross0)
        axes[0].plot(kap, bands, color=col, ls=ls, lw=1.6, label=f"$\\rho = {r}$")
        axes[1].plot(kap, ratio, color=col, ls=ls, lw=1.6, label=f"$\\rho = {r}$")
        axes[1].plot(kap, ratio0, color=col, ls=ls, lw=0.8, alpha=0.55)
    axes[0].set_ylim(0, 4)
    axes[0].set_xlabel(r"Cost relative to spread volatility, $\kappa = c/\sigma$")
    axes[0].set_ylabel(r"Optimal band $b^*$ (in $\sigma$)")
    axes[0].set_title("Optimal no-trade band", fontsize=9, color=INK)
    axes[1].axhline(0, color=INK2, lw=0.6)
    axes[1].set_ylim(-1.0, 1.05)
    axes[1].set_xlabel(r"Cost relative to spread volatility, $\kappa = c/\sigma$")
    axes[1].set_ylabel("Net gain / zero-cost gain")
    axes[1].set_title("Net gain: optimal band (thick) vs. no band (thin)", fontsize=9, color=INK)
    for ax in axes:
        ax.grid(color=GRID, lw=0.6)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_band.pdf"); fig.savefig(FIG / "fig_band.png", dpi=200)


# ------------------------------------------------------------------ Figure: frequency frontier
def fig_frontier():
    costs = np.linspace(0, 20, 81)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, n in zip(axes, ["Russell 1000", "Russell 2000"]):
        q = val[val.triplet == n].set_index("freq")
        for f in ["D", "W", "M", "Q"]:
            rho, sig = q.loc[f, "rho"], q.loc[f, "sigma"]
            y = [O.NPER[f] * sig * DP.lookup(rho, c / 1e4 / sig)[0] * 100 for c in costs]
            ax.plot(costs, y, color=FCOL[f], ls=FLS[f], lw=1.6, label=FLAB[f])
        rho, sig = q.loc["D", "rho"], q.loc["D", "sigma"]
        y0 = [O.NPER["D"] * sig * DP.closed_form(rho, c / 1e4 / sig)[2] * 100 for c in costs]
        ax.plot(costs, y0, color=FCOL["D"], ls="--", lw=0.8, alpha=0.6, label="Daily, no band")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylim(-1, 3.5)
        ax.set_xlabel("One-way trading cost (bps)")
        ax.set_title(n, fontsize=9, color=INK)
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("Model net gain vs. 50/50 mix (%/yr)")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_frontier.pdf"); fig.savefig(FIG / "fig_frontier.png", dpi=200)


# ------------------------------------------------------------------ Figure: OOS decisions and wealth
def fig_oos():
    paths = pd.read_pickle(RES / "oos_paths.pkl")
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.62)
    ax = fig.add_subplot(gs[0])
    rows = [(n, c) for n in RUSSELL for c in [2.0, 5.0]]
    for i, (n, c) in enumerate(rows):
        _, ch = paths[(n, c, "Robust optimal")]
        ch.index = pd.to_datetime(ch.index)
        ends = list(ch.index[1:]) + [ch.index[-1] + pd.offsets.MonthEnd(1)]
        for t0, t1, f in zip(ch.index, ends, ch.values):
            ax.barh(len(rows) - 1 - i, (t1 - t0).days, left=t0, height=0.72, color=FCOL[f], edgecolor="none")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{n.replace('Russell ', 'R')}, {c:g} bps" for n, c in rows][::-1], fontsize=7.5)
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("Rebalancing frequency chosen by the robust optimizer", fontsize=9, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FCOL[f], label=FLAB[f]) for f in ["D", "W", "M", "none"]],
              ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax2 = fig.add_subplot(gs[1])
    n = "Russell 2000"
    grp, g, v, b, rf, idx = O.prep(n)
    net_r, _ = paths[(n, 2.0, "Robust optimal")]
    net_p, _ = paths[(n, 2.0, "Plug-in optimal")]
    fm, _ = O.fixed_monthly(n, 2.0, net_r.index[0])
    bb = b.loc[net_r.index]
    wb = (1 + bb).cumprod()
    for s, lab, col, ls in [(fm, "Monthly inertia (fixed rule)", BLUE, "-"),
                            (net_r, "Robust optimizer", ORANGE, "--"),
                            (net_p, "Plug-in optimizer", AQUA, "-.")]:
        w = (1 + s.loc[net_r.index]).cumprod() / wb
        ax2.plot(w.index, w.values, color=col, ls=ls, lw=1.5, label=lab)
    ax2.axhline(1, color=INK, lw=1, ls=":", label="Buy-and-hold benchmark")
    ax2.set_ylabel("Wealth relative to benchmark")
    ax2.set_title("Russell 2000, net of 2 bps: cumulative wealth relative to the benchmark", fontsize=9, color=INK)
    ax2.grid(color=GRID, lw=0.6)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2)); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig_oos.pdf", bbox_inches="tight"); fig.savefig(FIG / "fig_oos.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    table_model(); table_oos(); table_usage(); table_risk(); table_mv(); table_z()
    fig_band(); fig_frontier(); fig_oos()
    print("done")

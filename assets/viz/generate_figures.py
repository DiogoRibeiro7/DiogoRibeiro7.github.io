"""Generate the figures embedded in posts.

Every figure on the site is produced here so it can be regenerated rather than
being an opaque binary. Run from this directory:

    python generate_figures.py            # all figures
    python generate_figures.py clt lorenz # named figures only

Each generator returns (slug, alt_text). Charts follow the house style:
validated palette, legend whenever there are two or more series, selective
direct labels, no dual axes.
"""
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import housestyle as hs
from housestyle import PALETTE as P

hs.use()
RNG = np.random.default_rng(20260816)
FIGURES = {}


def figure(slug, alt):
    def deco(fn):
        FIGURES[slug] = (fn, alt)
        return fn
    return deco


# --------------------------------------------------------------------------
@figure("clt_convergence",
        "Sampling distributions of the mean for n = 2, 5, and 30 drawn from a "
        "strongly skewed exponential population. The spread narrows and the "
        "shape becomes symmetric as n grows.")
def clt_convergence():
    # One hue across all three panels: this is the same quantity at three
    # sample sizes, not three different series. Colour follows the entity.
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    for ax, n in zip(axes, (2, 5, 30)):
        means = RNG.exponential(1.0, size=(40000, n)).mean(axis=1)
        ax.hist(means, bins=60, range=(0, 3), density=True,
                color=P[0], alpha=0.85)
        grid = np.linspace(0, 3, 400)
        ax.plot(grid, np.exp(-(grid - 1) ** 2 / (2 * (1 / n))) /
                np.sqrt(2 * np.pi / n), color=hs.INK_PRIMARY, lw=1.6)
        ax.set_title(f"n = {n}", fontsize=11)
        ax.set_xlabel("sample mean")
    axes[0].set_ylabel("density")
    fig.suptitle("The mean becomes normal long before the data does",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    axes[2].annotate("normal reference", xy=(1.55, 1.2), xytext=(1.8, 1.7),
                     fontsize=9, color=hs.INK_SECONDARY,
                     arrowprops=dict(arrowstyle="-", lw=1,
                                     color=hs.INK_MUTED))
    return fig


# --------------------------------------------------------------------------
@figure("lorenz_gini",
        "Lorenz curve for a simulated income distribution, plotted against the "
        "line of perfect equality. The shaded gap between them is the area "
        "the Gini coefficient measures.")
def lorenz_gini():
    income = np.sort(RNG.lognormal(mean=10.2, sigma=0.85, size=20000))
    cum = np.cumsum(income) / income.sum()
    cum = np.insert(cum, 0, 0)
    pop = np.linspace(0, 1, cum.size)
    gini = 1 - 2 * np.trapezoid(cum, pop)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], color=hs.INK_MUTED, lw=1.6,
            label="Perfect equality")
    ax.plot(pop, cum, color=P[0], label="Observed distribution")
    ax.fill_between(pop, cum, pop, color=P[0], alpha=0.10)
    ax.annotate(f"Gini = {gini:.2f}", xy=(0.36, 0.56), fontsize=11,
                color=hs.INK_PRIMARY, fontweight="semibold")
    ax.set_xlabel("cumulative share of population")
    ax.set_ylabel("cumulative share of income")
    ax.set_title("The Gini coefficient is twice the shaded area")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(axis="both")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("savitzky_golay",
        "A noisy signal smoothed by a moving average and by a Savitzky-Golay "
        "filter. The moving average flattens the peaks; Savitzky-Golay keeps "
        "their height and position.")
def savitzky_golay():
    from scipy.signal import savgol_filter
    x = np.linspace(0, 6, 400)
    clean = np.exp(-((x - 1.6) ** 2) / 0.05) + 0.8 * np.exp(-((x - 3.6) ** 2) / 0.03)
    noisy = clean + RNG.normal(0, 0.06, x.size)
    ma = np.convolve(noisy, np.ones(31) / 31, mode="same")
    sg = savgol_filter(noisy, 31, 3)

    fig, ax = plt.subplots()
    ax.plot(x, noisy, color=hs.INK_MUTED, lw=1.0, alpha=0.7, label="Raw signal")
    ax.plot(x, ma, color=P[1], label="Moving average (31)")
    ax.plot(x, sg, color=P[0], label="Savitzky-Golay (31, order 3)")
    ax.set_title("Savitzky-Golay preserves peak height; a moving average does not")
    ax.set_xlabel("time"); ax.set_ylabel("amplitude")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("kaplan_meier",
        "Kaplan-Meier survival curves for two treatment groups, with censored "
        "observations marked. The treatment group's curve stays above the "
        "control throughout follow-up.")
def kaplan_meier():
    def km(times, events):
        order = np.argsort(times)
        t, e = times[order], events[order]
        uniq = np.unique(t[e == 1])
        surv, s, out = [], 1.0, []
        for u in uniq:
            at_risk = (t >= u).sum()
            d = ((t == u) & (e == 1)).sum()
            s *= (1 - d / at_risk)
            surv.append(s); out.append(u)
        return np.array([0] + out), np.array([1.0] + surv)

    fig, ax = plt.subplots()
    for i, (scale, name) in enumerate([(9.0, "Control"), (15.0, "Treatment")]):
        t = RNG.exponential(scale, 160)
        c = RNG.exponential(20.0, 160)
        obs, ev = np.minimum(t, c), (t <= c).astype(int)
        xs, ys = km(obs, ev)
        ax.step(xs, ys, where="post", color=P[i], label=name)
        cens = obs[ev == 0]
        cy = [ys[max(0, np.searchsorted(xs, v) - 1)] for v in cens]
        ax.plot(cens, cy, linestyle="none", marker="|", markersize=7,
                color=P[i], markeredgecolor=P[i], markeredgewidth=1.4)
    ax.set_title("Kaplan-Meier estimate, censored observations ticked")
    ax.set_xlabel("time"); ax.set_ylabel("survival probability")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 40)
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("acf_pacf",
        "Autocorrelation and partial autocorrelation of a simulated AR(2) "
        "series. The ACF decays gradually while the PACF cuts off after lag 2, "
        "the signature used to identify the order.")
def acf_pacf():
    n = 600
    e = RNG.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.35 * y[t - 2] + e[t]
    y = y[50:]

    def acf(v, k):
        v = v - v.mean()
        d = (v * v).sum()
        return np.array([1.0] + [(v[i:] * v[:-i]).sum() / d for i in range(1, k + 1)])

    def pacf(v, k):
        out = [1.0]
        for j in range(1, k + 1):
            X = np.column_stack([v[j - l - 1: -l - 1] for l in range(j)])
            beta, *_ = np.linalg.lstsq(X, v[j:], rcond=None)
            out.append(beta[-1])
        return np.array(out)

    K = 18
    ci = 1.96 / np.sqrt(y.size)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, vals, name in ((axes[0], acf(y, K), "ACF"),
                           (axes[1], pacf(y, K), "PACF")):
        lags = np.arange(K + 1)
        ax.axhspan(-ci, ci, color=hs.INK_MUTED, alpha=0.15)
        ax.vlines(lags, 0, vals, color=P[0], lw=2)
        ax.plot(lags, vals, "o", color=P[0], markersize=5,
                markeredgecolor=hs.SURFACE, markeredgewidth=2)
        ax.axhline(0, color=hs.BASELINE, lw=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("lag")
        ax.set_xticks(np.arange(0, K + 1, 3))   # lags are discrete
    axes[0].set_ylabel("correlation")
    fig.suptitle("AR(2): the PACF cuts off at lag 2, the ACF does not",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    return fig


# --------------------------------------------------------------------------
@figure("splines_fit",
        "A cubic spline, a degree-10 polynomial, and a straight line fitted to "
        "the same curved data. The polynomial oscillates near the edges while "
        "the spline follows the shape.")
def splines_fit():
    from scipy.interpolate import UnivariateSpline
    x = np.linspace(0, 10, 90)
    truth = np.sin(x) + 0.15 * x
    y = truth + RNG.normal(0, 0.28, x.size)

    spline = UnivariateSpline(x, y, s=len(x) * 0.09)
    poly = np.polyval(np.polyfit(x, y, 10), x)
    line = np.polyval(np.polyfit(x, y, 1), x)

    fig, ax = plt.subplots()
    ax.plot(x, y, "o", color=hs.INK_MUTED, markersize=4, alpha=0.6,
            markeredgecolor="none", label="Observations")
    ax.plot(x, line, color=P[3], label="Linear fit")
    ax.plot(x, poly, color=P[1], label="Degree-10 polynomial")
    ax.plot(x, spline(x), color=P[0], label="Cubic spline")
    ax.set_title("A spline bends locally; a high-degree polynomial wobbles globally")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(ncol=2)
    return fig


# --------------------------------------------------------------------------
@figure("mcmc_trace",
        "Markov chain Monte Carlo output: four chains exploring the same "
        "posterior on the left, and the pooled posterior histogram against the "
        "analytic density on the right.")
def mcmc_trace():
    def target(x):
        return np.exp(-0.5 * ((x - 2.0) / 0.8) ** 2)

    chains = []
    for _ in range(4):
        x, out = RNG.normal(2, 3), []
        for _ in range(4000):
            prop = x + RNG.normal(0, 0.9)
            if RNG.random() < min(1.0, target(prop) / target(x)):
                x = prop
            out.append(x)
        chains.append(np.array(out))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    for i, ch in enumerate(chains):
        axes[0].plot(ch[:800], color=P[i], lw=1.0, alpha=0.9,
                     label=f"Chain {i + 1}")
    axes[0].set_title("Traces (first 800 draws)", fontsize=11)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("value")
    axes[0].legend(ncol=4, fontsize=8.5)

    pooled = np.concatenate([c[1000:] for c in chains])
    axes[1].hist(pooled, bins=60, density=True, color=P[0], alpha=0.85)
    g = np.linspace(pooled.min(), pooled.max(), 300)
    axes[1].plot(g, target(g) / (0.8 * np.sqrt(2 * np.pi)),
                 color=hs.INK_PRIMARY, lw=1.6)
    axes[1].set_title("Pooled posterior vs analytic density", fontsize=11)
    axes[1].set_xlabel("value")
    fig.suptitle("Well-mixed chains converge on the same posterior",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    return fig


# --------------------------------------------------------------------------
@figure("bias_variance",
        "Training and test error against model complexity. Training error "
        "falls monotonically while test error turns upward, and the gap "
        "between them is the overfitting penalty.")
def bias_variance():
    deg = np.arange(1, 16)
    x = np.linspace(-1, 1, 40)
    truth = np.cos(2.2 * x)
    tr, te = [], []
    for d in deg:
        a, b = [], []
        for _ in range(160):
            ytr = truth + RNG.normal(0, 0.32, x.size)
            yte = truth + RNG.normal(0, 0.32, x.size)
            co = np.polyfit(x, ytr, d)
            a.append(np.mean((np.polyval(co, x) - ytr) ** 2))
            b.append(np.mean((np.polyval(co, x) - yte) ** 2))
        tr.append(np.mean(a)); te.append(np.mean(b))

    fig, ax = plt.subplots()
    ax.plot(deg, tr, color=P[0], marker="o", markersize=5,
            markeredgecolor=hs.SURFACE, markeredgewidth=2, label="Training error")
    ax.plot(deg, te, color=P[1], marker="o", markersize=5,
            markeredgecolor=hs.SURFACE, markeredgewidth=2, label="Test error")
    best = deg[int(np.argmin(te))]
    ax.axvline(best, color=hs.INK_MUTED, lw=1.0)
    ax.annotate(f"test error minimised\nat degree {best}", xy=(best, min(te)),
                xytext=(best + 1.2, min(te) + 0.10), fontsize=9,
                color=hs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", lw=1, color=hs.INK_MUTED))
    ax.set_title("Training error keeps falling; test error does not")
    ax.set_xlabel("polynomial degree"); ax.set_ylabel("mean squared error")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("type_i_ii",
        "Null and alternative sampling distributions overlapping at a critical "
        "value. The shaded tail on the left is the Type I error rate; the "
        "shaded region on the right is the Type II error rate.")
def type_i_ii():
    x = np.linspace(-4, 8, 700)
    h0 = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    h1 = np.exp(-0.5 * (x - 3) ** 2) / np.sqrt(2 * np.pi)
    crit = 1.645

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(x, h0, color=P[0], label="Null ($H_0$ true)")
    ax.plot(x, h1, color=P[1], label="Alternative ($H_1$ true)")
    ax.fill_between(x, 0, h0, where=x >= crit, color=P[0], alpha=0.28)
    ax.fill_between(x, 0, h1, where=x < crit, color=P[1], alpha=0.28)
    ax.axvline(crit, color=hs.INK_PRIMARY, lw=1.2)
    ax.annotate("critical value", xy=(crit, 0.42), xytext=(crit + 0.25, 0.42),
                fontsize=9, color=hs.INK_SECONDARY)
    ax.annotate("Type I ($\\alpha$)", xy=(2.15, 0.02), xytext=(3.1, 0.10),
                fontsize=10, color=hs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", lw=1, color=hs.INK_MUTED))
    ax.annotate("Type II ($\\beta$)", xy=(0.9, 0.03), xytext=(-2.6, 0.14),
                fontsize=10, color=hs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", lw=1, color=hs.INK_MUTED))
    ax.set_title("Lowering $\\alpha$ moves the line right and enlarges $\\beta$")
    ax.set_xlabel("test statistic"); ax.set_ylabel("density")
    ax.set_ylim(0, 0.46)
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("kde_bandwidth",
        "Kernel density estimates of the same sample under three bandwidths. "
        "Too small is spiky and overfits; too large washes out the two modes "
        "that are really there.")
def kde_bandwidth():
    sample = np.concatenate([RNG.normal(-1.6, 0.5, 220),
                             RNG.normal(1.7, 0.7, 280)])
    grid = np.linspace(-4.5, 5, 600)

    def kde(data, h):
        u = (grid[:, None] - data[None, :]) / h
        return np.exp(-0.5 * u ** 2).sum(axis=1) / (data.size * h * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots()
    ax.plot(sample, np.full(sample.size, -0.006), "|", color=hs.INK_MUTED,
            markeredgecolor=hs.INK_MUTED, markeredgewidth=0.8, markersize=6)
    for i, (h, name) in enumerate([(0.12, "h = 0.12 (undersmoothed)"),
                                   (0.45, "h = 0.45 (about right)"),
                                   (1.40, "h = 1.40 (oversmoothed)")]):
        ax.plot(grid, kde(sample, h), color=P[i], label=name)
    ax.set_title("Bandwidth decides whether you see one mode or two")
    ax.set_xlabel("value"); ax.set_ylabel("density")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("regularization_paths",
        "Lasso coefficient paths against the regularisation strength. "
        "Coefficients shrink to exactly zero one after another as the penalty "
        "grows, which is how lasso performs variable selection.")
def regularization_paths():
    from sklearn.linear_model import lasso_path
    from sklearn.preprocessing import StandardScaler
    n, p = 120, 8
    X = RNG.normal(size=(n, p))
    beta = np.array([3.0, -2.0, 1.4, 0.0, 0.0, 0.8, 0.0, -0.5])
    y = X @ beta + RNG.normal(0, 1.0, n)
    X = StandardScaler().fit_transform(X)
    alphas, coefs, _ = lasso_path(X, y, n_alphas=120)

    # All eight paths are the same kind of thing, so they share one hue;
    # identity comes from direct labels on the survivors, not from eight
    # colours (four of which would sit indistinguishably on top of zero).
    fig, ax = plt.subplots()
    ax.set_xlim(alphas[0], alphas[-1] * 0.45)      # room at the right for labels
    for j in range(p):
        survives = abs(coefs[j][-1]) > 0.25
        ax.plot(alphas, coefs[j], color=P[0],
                alpha=1.0 if survives else 0.30,
                lw=2.0 if survives else 1.2)
        if survives:
            # x-axis is inverted, so the final value sits at the LEFT edge:
            # offset leftwards into the margin, clear of the line.
            ax.annotate(f"$x_{{{j + 1}}}$", xy=(alphas[-1], coefs[j][-1]),
                        xytext=(-10, 0), textcoords="offset points",
                        fontsize=9.5, color=hs.INK_SECONDARY,
                        va="center", ha="right")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.axhline(0, color=hs.BASELINE, lw=0.8)
    ax.set_title("Lasso drives coefficients to exactly zero, one by one")
    ax.set_xlabel("regularisation strength $\\alpha$ (log scale, decreasing)")
    ax.set_ylabel("coefficient")
    return fig


# --------------------------------------------------------------------------
@figure("monte_carlo_fan",
        "Ten thousand simulated GDP paths summarised as a median line with 50% "
        "and 90% bands. The bands widen with the horizon, showing how "
        "uncertainty compounds over time.")
def monte_carlo_fan():
    years, sims = 11, 10000
    rho = 0.7
    paths = np.full((sims, years), 100.0)
    eps = np.zeros(sims)
    for t in range(1, years):
        eps = rho * eps + RNG.standard_t(df=5, size=sims) * 0.011
        paths[:, t] = paths[:, t - 1] * (1 + 0.02 + eps)

    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    xs = np.arange(years)
    fig, ax = plt.subplots()
    ax.fill_between(xs, q[0], q[4], color=P[0], alpha=0.12, label="90% interval")
    ax.fill_between(xs, q[1], q[3], color=P[0], alpha=0.24, label="50% interval")
    ax.plot(xs, q[2], color=P[0], label="Median path")
    ax.set_title("Persistent shocks make the fan widen faster than the horizon")
    ax.set_xlabel("year"); ax.set_ylabel("GDP index (start = 100)")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("drift_monitoring_architecture",
        "Architecture diagram for production drift monitoring. Production data "
        "flows through validation, feature monitoring, prediction monitoring, "
        "label monitoring, decision monitoring, and response actions.")
def drift_monitoring_architecture():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    boxes = [
        ("Production\ninputs", 0.25, 3.25, 1.55, 0.75, P[0]),
        ("Data quality\nchecks", 2.15, 3.25, 1.55, 0.75, P[2]),
        ("Feature drift\nmonitoring", 4.05, 3.25, 1.55, 0.75, P[0]),
        ("Prediction\nmonitoring", 5.95, 3.25, 1.55, 0.75, P[3]),
        ("Labels and\noutcomes", 4.05, 1.45, 1.55, 0.75, P[1]),
        ("Decision\nmonitoring", 5.95, 1.45, 1.55, 0.75, P[4]),
        ("Response:\nfix, recalibrate,\nretrain, pause", 8.0, 2.35, 1.75, 1.05, P[7]),
    ]

    for text, x, y, w, h, color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.08",
            facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10.2, color=hs.INK_PRIMARY, fontweight="semibold")

    arrows = [
        ((1.8, 3.63), (2.15, 3.63)),
        ((3.7, 3.63), (4.05, 3.63)),
        ((5.6, 3.63), (5.95, 3.63)),
        ((7.5, 3.63), (8.0, 3.0)),
        ((4.83, 3.25), (4.83, 2.2)),
        ((5.6, 1.83), (5.95, 1.83)),
        ((7.5, 1.83), (8.0, 2.75)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.4,
                                     color=hs.INK_MUTED))

    ax.text(0.25, 4.55, "Monitoring should separate signal, impact, and action",
            fontsize=13.5, fontweight="semibold", color=hs.INK_PRIMARY)
    ax.text(0.25, 4.25,
            "Feature changes are early signals; mature labels and decision metrics tell whether action is needed.",
            fontsize=10.2, color=hs.INK_SECONDARY)
    ax.text(4.15, 0.72, "Label delay means outcome monitoring lags production.",
            fontsize=9.6, color=hs.INK_SECONDARY)
    return fig


# --------------------------------------------------------------------------
@figure("feature_drift_distribution",
        "Training and production feature distributions after a sensor scaling "
        "change. The production distribution shifts right, creating feature "
        "drift before labels are available.")
def feature_drift_distribution():
    train = RNG.normal(0.0, 1.0, 6000)
    prod = RNG.normal(0.85, 1.15, 6000)
    bins = np.linspace(-4, 5, 80)

    fig, ax = plt.subplots()
    ax.hist(train, bins=bins, density=True, alpha=0.38, color=P[0],
            label="Training period")
    ax.hist(prod, bins=bins, density=True, alpha=0.42, color=P[1],
            label="Production period")
    ax.axvline(np.mean(train), color=P[0], lw=2)
    ax.axvline(np.mean(prod), color=P[1], lw=2)
    ax.annotate("mean shifted", xy=(np.mean(prod), 0.32), xytext=(1.95, 0.42),
                fontsize=9.5, color=hs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", color=hs.INK_MUTED, lw=1))
    ax.set_title("Feature drift: the input distribution moved")
    ax.set_xlabel("standardized sensor value")
    ax.set_ylabel("density")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("concept_drift_boundary",
        "The same input feature has a different relationship with the outcome "
        "after deployment. The production risk curve shifts relative to the "
        "training risk curve, illustrating concept drift.")
def concept_drift_boundary():
    x = np.linspace(-4, 4, 400)
    train = 1 / (1 + np.exp(-1.6 * (x - 0.1)))
    prod = 1 / (1 + np.exp(-1.15 * (x - 1.0)))

    fig, ax = plt.subplots()
    ax.plot(x, train, color=P[0], label="Training relationship")
    ax.plot(x, prod, color=P[1], label="Production relationship")
    ax.axhline(0.5, color=hs.BASELINE, lw=1)
    ax.axvline(0.1, color=P[0], lw=1.4, alpha=0.75)
    ax.axvline(1.0, color=P[1], lw=1.4, alpha=0.75)
    ax.annotate("same score threshold,\ndifferent real risk",
                xy=(0.55, 0.5), xytext=(-2.8, 0.64),
                fontsize=9.5, color=hs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", color=hs.INK_MUTED, lw=1))
    ax.set_title("Concept drift: the feature-target relationship changed")
    ax.set_xlabel("feature value")
    ax.set_ylabel("probability of event")
    ax.set_ylim(0, 1)
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("prediction_drift_threshold",
        "Daily alert volume from a fixed score threshold. Alert counts rise "
        "after a distribution shift even though the threshold does not change.")
def prediction_drift_threshold():
    days = np.arange(1, 91)
    baseline = 95 + 8 * np.sin(days / 5.5) + RNG.normal(0, 5, days.size)
    shifted = baseline.copy()
    shifted[54:] += np.linspace(25, 95, days.size - 54)
    shifted = np.maximum(shifted, 0)

    fig, ax = plt.subplots()
    ax.plot(days, shifted, color=P[0], label="Alerts above fixed threshold")
    ax.axvline(55, color=P[1], lw=2, label="Distribution shift")
    ax.axhline(140, color=P[7], lw=1.8, label="Review capacity")
    ax.fill_between(days, 140, shifted, where=shifted > 140,
                    color=P[7], alpha=0.14)
    ax.annotate("threshold unchanged,\nworkload changed",
                xy=(75, shifted[74]), xytext=(43, 188),
                fontsize=9.5, color=hs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", color=hs.INK_MUTED, lw=1))
    ax.set_title("Prediction drift: scores cross the action threshold more often")
    ax.set_xlabel("day")
    ax.set_ylabel("daily alerts")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("solow_steady_state",
        "Solow model capital accumulation: the saving curve s*f(k) crosses the "
        "break-even line (n+g+d)k once, at the steady state. Below it capital "
        "per worker grows; above it, it shrinks.")
def solow_steady_state():
    alpha, s, n, g, delta = 0.35, 0.25, 0.01, 0.02, 0.05
    k = np.linspace(0.01, 14, 600)
    saving = s * k ** alpha
    breakeven = (n + g + delta) * k
    kstar = (s / (n + g + delta)) ** (1 / (1 - alpha))

    fig, ax = plt.subplots()
    ax.plot(k, k ** alpha, color=hs.INK_MUTED, lw=1.6, label="Output per worker $f(k)$")
    ax.plot(k, saving, color=P[0], label="Saving $s\\,f(k)$")
    ax.plot(k, breakeven, color=P[1], label="Break-even $(n+g+\\delta)k$")
    ax.fill_between(k, saving, breakeven, where=saving > breakeven,
                    color=P[0], alpha=0.10)
    ax.axvline(kstar, color=hs.INK_PRIMARY, lw=1.0)
    ax.annotate(f"steady state $k^*$ = {kstar:.1f}",
                xy=(kstar, s * kstar ** alpha), xytext=(kstar + 1.1, 0.55),
                fontsize=9.5, color=hs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", lw=1, color=hs.INK_MUTED))
    ax.set_title("Concave returns give the Solow model one stable steady state")
    ax.set_xlabel("capital per effective worker $k$")
    ax.set_ylabel("output / investment per worker")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.6)
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("linear_vs_logistic",
        "A linear probability model and a logistic curve fitted to the same "
        "binary outcome. The straight line passes below zero and above one, "
        "predicting impossible probabilities; the logistic curve cannot.")
def linear_vs_logistic():
    x = np.concatenate([RNG.normal(-1.4, 1.1, 160), RNG.normal(2.2, 1.1, 160)])
    p = 1 / (1 + np.exp(-(1.4 * x - 0.6)))
    y = (RNG.random(x.size) < p).astype(float)

    b1, b0 = np.polyfit(x, y, 1)
    # logistic fit by Newton steps on the log-likelihood
    w = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(60):
        eta = X @ w
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        w += np.linalg.solve((X * W[:, None]).T @ X, X.T @ (y - mu))

    grid = np.linspace(x.min() - 1.5, x.max() + 1.5, 400)
    lin = b0 + b1 * grid
    log = 1 / (1 + np.exp(-(w[0] + w[1] * grid)))

    fig, ax = plt.subplots()
    ax.axhspan(1.0, 1.35, color=hs.INK_MUTED, alpha=0.10)
    ax.axhspan(-0.35, 0.0, color=hs.INK_MUTED, alpha=0.10)
    # observed 0/1 outcomes as a rug, drawn after the bands so it stays visible
    ax.plot(x, y, "|", color=hs.INK_SECONDARY,
            markeredgecolor=hs.INK_SECONDARY, markeredgewidth=1.0,
            markersize=10, alpha=0.55, zorder=4)
    ax.plot(grid, lin, color=P[1], label="Linear probability model")
    ax.plot(grid, log, color=P[0], label="Logistic regression")
    ax.axhline(0, color=hs.BASELINE, lw=0.8)
    ax.axhline(1, color=hs.BASELINE, lw=0.8)
    # point at the line where it has already dropped below zero
    xneg = -b0 / b1 - 1.2
    ax.annotate("impossible probabilities", xy=(xneg, b0 + b1 * xneg),
                xytext=(xneg + 0.6, -0.27), fontsize=9.5,
                color=hs.INK_SECONDARY, ha="left",
                arrowprops=dict(arrowstyle="-", lw=1, color=hs.INK_MUTED))
    ax.set_title("A straight line leaves the unit interval; a logistic curve cannot")
    ax.set_xlabel("predictor"); ax.set_ylabel("P(outcome = 1)")
    ax.set_ylim(-0.35, 1.35)
    ax.legend(loc="center right")
    return fig


# --------------------------------------------------------------------------
@figure("hist2d_outliers",
        "A two-dimensional histogram of bivariate data. Density is shown as a "
        "single-hue heatmap and the sparse cells at the edges hold the points "
        "an outlier detector flags.")
def hist2d_outliers():
    core = RNG.multivariate_normal([0, 0], [[1, 0.65], [0.65, 1]], 4000)
    strays = RNG.uniform(-5, 5, size=(45, 2))
    data = np.vstack([core, strays])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    h = axes[0].hist2d(data[:, 0], data[:, 1], bins=40, range=[[-5, 5], [-5, 5]],
                       cmap=hs.sequential_cmap(), cmin=1)
    axes[0].set_title("2D histogram: density per cell", fontsize=11)
    cb = fig.colorbar(h[3], ax=axes[0])
    cb.outline.set_visible(False)
    cb.set_label("count", color=hs.INK_SECONDARY)

    counts, xe, ye = np.histogram2d(data[:, 0], data[:, 1], bins=40,
                                    range=[[-5, 5], [-5, 5]])
    ix = np.clip(np.digitize(data[:, 0], xe) - 1, 0, 39)
    iy = np.clip(np.digitize(data[:, 1], ye) - 1, 0, 39)
    sparse = counts[ix, iy] <= 1
    axes[1].plot(data[~sparse, 0], data[~sparse, 1], "o", markersize=3,
                 color=hs.INK_MUTED, alpha=0.35, markeredgecolor="none",
                 label="Dense cells")
    axes[1].plot(data[sparse, 0], data[sparse, 1], "o", markersize=5,
                 color=P[1], markeredgecolor=hs.SURFACE, markeredgewidth=1.2,
                 label="Sparse cells (flagged)")
    axes[1].set_title("Points falling in near-empty cells", fontsize=11)
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-5, 5)
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    return fig


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
@figure("regression_to_the_mean_scatter",
        "Monthly failure counts for 500 machines in two consecutive months with "
        "nothing changed between them. The worst decile in month one, "
        "highlighted, falls back toward the fleet mean in month two and lands "
        "on the regression line rather than the identity line.")
def regression_to_the_mean_scatter():
    r = np.random.default_rng(42)
    n = 500
    rate = r.gamma(4.0, 1.0, n)
    m1 = r.poisson(rate)
    m2 = r.poisson(rate)
    k = n // 10
    worst = np.argsort(m1)[-k:]
    rest = np.setdiff1d(np.arange(n), worst)
    rho = np.corrcoef(m1, m2)[0, 1]
    mu = m1.mean()
    jit = lambda v: v + r.uniform(-0.3, 0.3, v.size)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(jit(m1[rest]), jit(m2[rest]), s=14, color=hs.INK_MUTED,
               alpha=0.5, label="Other machines")
    ax.scatter(jit(m1[worst]), jit(m2[worst]), s=20, color=P[1],
               alpha=0.9, label="Worst 10% in month 1")
    lim = np.array([-0.5, max(m1.max(), m2.max()) + 1.0])
    ax.plot(lim, lim, color=hs.BASELINE, lw=1.4, label="No change (y = x)")
    ax.plot(lim, mu + rho * (lim - mu), color=P[0], lw=2.2,
            label=f"Expected month 2 (rho = {rho:.2f})")
    mx, my = m1[worst].mean(), m2[worst].mean()
    ax.plot([mx], [my], marker="D", markersize=9, color=P[1],
            markeredgecolor=hs.SURFACE, zorder=6)
    ax.annotate(f"worst-decile mean\n{mx:.1f} in month 1, {my:.1f} in month 2",
                xy=(mx, my), xytext=(-150, 60), textcoords="offset points",
                fontsize=9.5, color=hs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", color=hs.INK_MUTED, lw=1))
    ax.set_xlabel("failures in month 1")
    ax.set_ylabel("failures in month 2")
    ax.set_title("Selected on month 1, the worst decile lands on the regression line")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(axis="both")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("drift_alert_power_curves",
        "Share of drifting features flagged per day against the size of the "
        "shift, for an uncorrected daily test, Benjamini-Hochberg and "
        "Bonferroni, with 200 features monitored and five drifting. The "
        "corrections cost power only for subtle shifts, and the uncorrected "
        "test pays for its power with two false alerts for every true one.")
def drift_alert_power_curves():
    from matplotlib.ticker import PercentFormatter
    from scipy import stats
    r = np.random.default_rng(7)
    m, n_ref, n_batch, alpha, days = 200, 2000, 500, 0.05, 30
    drifted = np.arange(5)

    def bh(p, q):
        order = np.argsort(p)
        below = np.where(p[order] <= q * np.arange(1, m + 1) / m)[0]
        rej = np.zeros(m, bool)
        if below.size:
            rej[order[:below.max() + 1]] = True
        return rej

    shifts = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    power = {"raw": [], "bonf": [], "bh": []}
    for delta in shifts:
        acc = {k: [] for k in power}
        for _ in range(days):
            p = np.array([stats.ks_2samp(
                r.normal(size=n_ref),
                r.normal(loc=delta if j < 5 else 0.0, size=n_batch)).pvalue
                for j in range(m)])
            flags = {"raw": p < alpha, "bonf": p < alpha / m, "bh": bh(p, 0.05)}
            for k, f in flags.items():
                acc[k].append(f[drifted].mean())
        for k in power:
            power[k].append(np.mean(acc[k]))

    fig, ax = plt.subplots()
    series = (("raw", "Uncorrected, alpha = 0.05", P[1]),
              ("bh", "Benjamini-Hochberg, q = 0.05", P[0]),
              ("bonf", "Bonferroni", P[2]))
    for k, label, col in series:
        ax.plot(shifts, power[k], marker="o", color=col, label=label)
    ax.set_xlabel("shift in the drifting features (standard deviations)")
    ax.set_ylabel("drifting features flagged per day")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Corrections cost power only where the shift is subtle")
    ax.legend(loc="lower right")
    return fig


# --------------------------------------------------------------------------
@figure("drift_alert_bursts",
        "Daily false-alert counts over 60 days with no drift anywhere, for 200 "
        "independent features and for 200 features that share ten latent "
        "factors. The average is the same; correlated features produce quiet "
        "days and bursts, so a burst on its own is not evidence of drift.")
def drift_alert_bursts():
    from scipy import stats
    r = np.random.default_rng(21)
    m, n_ref, n_batch, alpha = 200, 2000, 500, 0.05
    load = r.normal(size=(m, 10)) * np.sqrt(0.8 / 10)

    def batch(n):
        f = r.normal(size=(n, 10))
        return f @ load.T + r.normal(scale=np.sqrt(0.2), size=(n, m))

    ind, cor = [], []
    for _ in range(60):
        p_ind = np.array([stats.ks_2samp(r.normal(size=n_ref),
                                         r.normal(size=n_batch)).pvalue
                          for _ in range(m)])
        ref, new = batch(n_ref), batch(n_batch)
        p_cor = np.array([stats.ks_2samp(ref[:, j], new[:, j]).pvalue
                          for j in range(m)])
        ind.append((p_ind < alpha).sum())
        cor.append((p_cor < alpha).sum())

    fig, ax = plt.subplots()
    bins = np.arange(0, 40, 2)
    ax.hist(ind, bins=bins, color=P[0], alpha=0.85,
            label=f"Independent features (sd {np.std(ind):.1f})")
    ax.hist(cor, bins=bins, color=P[1], alpha=0.6,
            label=f"Correlated features (sd {np.std(cor):.1f})")
    ax.axvline(alpha * m, color=hs.INK_MUTED, lw=1.2)
    ax.annotate("expected: 10 per day", xy=(alpha * m, ax.get_ylim()[1] * 0.92),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9.5, color=hs.INK_SECONDARY)
    ax.set_xlabel("false alerts per day (no drift anywhere)")
    ax.set_ylabel("days")
    ax.set_title("Correlated features turn a steady trickle of false alerts into bursts")
    ax.legend()
    return fig


# --------------------------------------------------------------------------
@figure("permutation_importance_methods",
        "Four importance methods applied to the same random forest and the "
        "same four features: a driver, a near-duplicate of it with no effect "
        "of its own, a weak independent feature and noise. Each method ranks "
        "the features differently because each answers a different question.")
def permutation_importance_methods():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    r = np.random.default_rng(0)
    n = 4000
    x1 = r.normal(size=n)
    x2 = x1 + r.normal(scale=0.2, size=n)
    x3 = r.normal(size=n)
    x4 = r.normal(size=n)
    y = 2.0 * x1 + 0.5 * x3 + r.normal(size=n)
    X = np.column_stack([x1, x2, x3, x4])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)

    def fit(Xa, ya):
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                     max_features=2, random_state=0,
                                     n_jobs=-1).fit(Xa, ya)

    rf = fit(Xtr, ytr)
    base = mean_squared_error(yte, rf.predict(Xte))
    perm = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=0,
                                  scoring="neg_mean_squared_error").importances_mean

    g = np.random.default_rng(0)
    def group(cols):
        out = []
        for _ in range(20):
            Xp = Xte.copy()
            p = g.permutation(len(Xte))
            Xp[:, cols] = Xp[p][:, cols]
            out.append(mean_squared_error(yte, rf.predict(Xp)) - base)
        return np.mean(out)

    def conditional(col, cond, n_bins=20, repeats=20):
        c = np.random.default_rng(0)
        edges = np.quantile(Xte[:, cond], np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.searchsorted(edges, Xte[:, cond], side="right") - 1,
                       0, n_bins - 1)
        out = []
        for _ in range(repeats):
            Xp = Xte.copy()
            for b in range(n_bins):
                idx = np.where(bins == b)[0]
                Xp[idx, col] = Xp[c.permutation(idx), col]
            out.append(mean_squared_error(yte, rf.predict(Xp)) - base)
        return np.mean(out)

    def drop(j):
        cols = [c for c in range(4) if c != j]
        mdl = fit(Xtr[:, cols], ytr)
        return mean_squared_error(yte, mdl.predict(Xte[:, cols])) - base

    names = ["x1\ndriver", "x2\nduplicate", "x3\nweak", "x4\nnoise"]
    panels = [
        ("Permutation", names, perm),
        ("Group permutation", ["x1 + x2", "x3", "x4"],
         [group([0, 1]), perm[2], perm[3]]),
        ("Conditional permutation", names,
         [conditional(0, 1), conditional(1, 0), conditional(2, 0), conditional(3, 0)]),
        ("Drop-column refit", names, [drop(j) for j in range(4)]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.4))
    for ax, (title, labels, vals) in zip(axes, panels):
        ax.bar(labels, vals, color=P[0], width=0.62)
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color=hs.BASELINE, lw=0.8)
        ax.tick_params(axis="x", labelsize=8.5)
    axes[0].set_ylabel("increase in test MSE")
    fig.suptitle("Same model, same data, four different answers",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    return fig


# --------------------------------------------------------------------------
@figure("test_set_size_ranking",
        "How often a test set ranks two classifiers correctly, against test-set "
        "size, when model B is one accuracy point better than model A and the "
        "two disagree on about eight percent of cases. Evaluating both on the same "
        "test set beats separate test sets at every size, and a paired test "
        "reaches significance long before two independent confidence "
        "intervals stop overlapping.")
def test_set_size_ranking():
    from matplotlib.ticker import PercentFormatter
    from scipy import stats
    tau = 0.5
    qA = stats.norm.ppf(0.90) * np.sqrt(1 + tau ** 2)
    qB = stats.norm.ppf(0.91) * np.sqrt(1 + tau ** 2)

    def draw(n, r):
        d = r.normal(size=n)
        a = d + r.normal(scale=tau, size=n) < qA
        b = d + r.normal(scale=tau, size=n) < qB
        return a, b

    rng3 = np.random.default_rng(3)
    reps = 2000
    sizes = [200, 500, 1000, 2000, 5000, 10000, 20000]
    rows = []
    for n in sizes:
        same = sep = mcn = disj = 0
        for _ in range(reps):
            a, b = draw(n, rng3)
            same += b.mean() > a.mean()
            a2, _ = draw(n, rng3)
            sep += b.mean() > a2.mean()
            b_, c_ = np.sum(~a & b), np.sum(a & ~b)
            if b_ + c_ > 0:
                mcn += (stats.binomtest(int(b_), int(b_ + c_)).pvalue < 0.05
                        and b_ > c_)
            ha = 1.96 * np.sqrt(a.mean() * (1 - a.mean()) / n)
            hb = 1.96 * np.sqrt(b.mean() * (1 - b.mean()) / n)
            disj += (b.mean() - hb) > (a.mean() + ha)
        rows.append([same, sep, mcn, disj])
    rows = np.array(rows) / reps

    fig, ax = plt.subplots()
    series = ((0, "Same test set: B measures higher", P[0]),
              (1, "Separate test sets: B measures higher", P[2]),
              (2, "Paired (McNemar) test significant", P[1]),
              (3, "Independent 95% intervals disjoint", P[3]))
    for j, label, col in series:
        ax.plot(sizes, rows[:, j], marker="o", color=col, label=label)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("test-set size")
    ax.set_ylabel("share of test sets")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("A one-point difference needs thousands of test cases to show")
    ax.legend(loc="center right")
    return fig


# --------------------------------------------------------------------------
@figure("extreme_value_tail_plot",
        "Exceedance probability per hour against load level for three years "
        "of simulated hourly peak load, on a log scale. The empirical tail "
        "ends at the sample maximum; a normal fit falls off far too quickly; "
        "the generalised Pareto fit above the 98th percentile extrapolates to "
        "the ten-year and hundred-year levels close to the true tail.")
def extreme_value_tail_plot():
    from scipy import stats
    r = np.random.default_rng(0)
    df, loc, scale, H = 4, 100.0, 10.0, 8760
    n = 3 * H
    x = loc + scale * r.standard_t(df, n)
    u = np.quantile(x, 0.98)
    exc = x[x > u] - u
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    zeta = np.mean(x > u)

    grid = np.linspace(u, 620, 400)
    true_sf = stats.t.sf((grid - loc) / scale, df)
    norm_sf = stats.norm.sf(grid, x.mean(), x.std())
    gpd_sf = zeta * (1 + xi * (grid - u) / sigma) ** (-1 / xi)
    top = np.sort(x[x > u])[::-1]
    emp_p = (np.arange(len(top)) + 1) / n

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(top, emp_p, s=9, color=hs.INK_MUTED, alpha=0.6,
               label="Observed (3 years)")
    ax.plot(grid, true_sf, color=hs.INK_PRIMARY, lw=1.6, label="True tail")
    ax.plot(grid, gpd_sf, color=P[0], label="Generalised Pareto fit")
    ax.plot(grid, norm_sf, color=P[1], label="Normal fit")
    for T, name in ((10, "10-year level"), (100, "100-year level")):
        p = 1 / (T * H)
        ax.axhline(p, color=hs.BASELINE, lw=1.0)
        ax.annotate(name, xy=(132, p), xytext=(0, 4), textcoords="offset points",
                    fontsize=9, color=hs.INK_SECONDARY)
    ax.set_yscale("log")
    ax.set_ylim(3e-7, 3e-2)
    ax.set_xlim(125, 620)
    ax.set_xlabel("hourly peak load")
    ax.set_ylabel("probability an hour exceeds the level")
    ax.set_title("The data end at the sample maximum; the tail does not")
    ax.legend(loc="upper right")
    return fig


# --------------------------------------------------------------------------
@figure("censored_labels_cohorts",
        "Mean predicted twelve-month churn probability by months since "
        "signup, for held-out customers, from three models trained on the "
        "same extract. The naive model, trained on whether a customer had "
        "churned by the extract date, predicts almost no risk for recent "
        "customers and too much for old ones. A fixed-horizon label and a "
        "discrete-time hazard model both track the true probability.")
def censored_labels_cohorts():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    r = np.random.default_rng(0)
    N, H = 20000, 12
    signup = r.uniform(0, 36, N)
    x1, x2 = r.normal(size=N), r.normal(size=N)
    lam = 18 * np.exp(-(0.8 * x1 - 0.6 * x2))
    k = 1.3
    T = lam * r.weibull(k, N)
    followup = 36 - signup
    churned_by_extract = T <= followup
    p12_true = 1 - np.exp(-(H / lam) ** k)
    event12 = T <= H
    test = r.uniform(size=N) < 0.25
    tr = ~test
    full = followup >= H

    gbm = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                 random_state=0)
    Xn = np.column_stack([x1, x2, followup])
    p_naive = gbm().fit(Xn[tr], churned_by_extract[tr]).predict_proba(Xn[test])[:, 1]
    Xf = np.column_stack([x1, x2])
    p_fixed = gbm().fit(Xf[tr & full], event12[tr & full]).predict_proba(Xf[test])[:, 1]

    def person_months(idx):
        rows, ys = [], []
        for i in idx:
            last = int(np.ceil(min(T[i], followup[i], H)))
            for m in range(1, max(last, 1) + 1):
                rows.append((x1[i], x2[i], m))
                ys.append(int(T[i] <= m and T[i] > m - 1 and m <= followup[i] + 1e-9))
        return np.array(rows), np.array(ys)

    def design(rows):
        m = rows[:, 2].astype(int)
        return np.column_stack([rows[:, :2], np.eye(H)[m - 1]])

    R, Y = person_months(np.where(tr)[0])
    haz = LogisticRegression(C=10.0, max_iter=2000).fit(design(R), Y)
    idx = np.where(test)[0]
    surv = np.ones(len(idx))
    for m in range(1, H + 1):
        rows = np.column_stack([x1[idx], x2[idx], np.full(len(idx), m)])
        surv *= 1 - haz.predict_proba(design(rows))[:, 1]
    p_haz = 1 - surv

    ten = followup[test]
    bins = [(0, 3), (3, 6), (6, 12), (12, 24), (24, 36)]
    labels = [f"{lo}-{hi}" for lo, hi in bins]
    def by_cohort(p):
        return [p[(ten >= lo) & (ten < hi)].mean() for lo, hi in bins]

    fig, ax = plt.subplots()
    xs = np.arange(len(bins))
    ax.plot(xs, by_cohort(p12_true[test]), color=hs.INK_PRIMARY, lw=1.6,
            label="True 12-month probability")
    ax.plot(xs, by_cohort(p_naive), marker="o", color=P[1],
            label="Naive: churned by extract date")
    ax.plot(xs, by_cohort(p_fixed), marker="o", color=P[2],
            label="Fixed 12-month horizon")
    ax.plot(xs, by_cohort(p_haz), marker="o", color=P[0],
            label="Discrete-time hazard model")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("months since signup at the data extract")
    ax.set_ylabel("mean predicted 12-month churn")
    ax.set_ylim(0, 0.85)
    ax.set_title("Censored labels teach the model that new customers never churn")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("learning_curves_extrapolation",
        "Test error against training-set size for a logistic regression and "
        "a gradient boosting classifier on the same simulated task, with the "
        "Bayes error rate. Power-law curves fitted to the five smallest sizes "
        "predict the held-out larger sizes: the boosting curve heads for the "
        "Bayes rate, the logistic curve for a floor three times higher.")
def learning_curves_extrapolation():
    from scipy.optimize import curve_fit
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    r = np.random.default_rng(0)

    def make(n, r):
        X = r.normal(size=(n, 12))
        logit = (1.2 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2]
                 + 1.5 * X[:, 0] * X[:, 1] + 1.0 * (X[:, 3] ** 2 - 1) - 0.6 * X[:, 4])
        p = 1 / (1 + np.exp(-logit))
        return X, (r.uniform(size=n) < p).astype(int), p

    Xte, yte, pte = make(50000, r)
    bayes = np.mean(np.minimum(pte, 1 - pte))
    sizes = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    Xpool, ypool, _ = make(32000 * 3, r)
    models = {
        "logistic": lambda: LogisticRegression(max_iter=2000),
        "boosting": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0),
    }
    curves = {k: [] for k in models}
    for n in sizes:
        for name, mk in models.items():
            errs = []
            for rep in range(3):
                idx = np.random.default_rng(100 * n + rep).choice(len(Xpool), n, replace=False)
                m = mk().fit(Xpool[idx], ypool[idx])
                errs.append(1 - (m.predict(Xte) == yte).mean())
            curves[name].append(np.mean(errs))

    power = lambda n, a, b, c: a + b * n ** (-c)
    grid = np.logspace(np.log10(250), np.log10(32000), 200)
    fig, ax = plt.subplots()
    for name, col, label in (("logistic", P[1], "Logistic regression"),
                             ("boosting", P[0], "Gradient boosting")):
        es = np.array(curves[name])
        (a, b, c), _ = curve_fit(power, np.array(sizes[:5], float), es[:5],
                                 p0=[es[4] * 0.9, 1.0, 0.5],
                                 bounds=([0, 0, 0.05], [1, 100, 2]), maxfev=20000)
        ax.plot(grid, power(grid, a, b, c), color=col, lw=1.2, alpha=0.7)
        ax.plot(sizes[:5], es[:5], "o", color=col, markersize=6, label=f"{label}, used for fit")
        ax.plot(sizes[5:], es[5:], "o", markerfacecolor=hs.SURFACE, markeredgecolor=col,
                markeredgewidth=1.8, markersize=6, label=f"{label}, held out")
        hs.label_end(ax, 32000, power(32000, a, b, c), f"fit predicts {power(32000, a, b, c):.3f}", col)
    ax.axhline(bayes, color=hs.INK_MUTED, lw=1.2)
    ax.annotate(f"Bayes error {bayes:.3f}", xy=(260, bayes), xytext=(0, -12),
                textcoords="offset points", fontsize=9.5, color=hs.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("training examples")
    ax.set_ylabel("test error rate")
    ax.set_ylim(0.19, 0.34)
    ax.set_title("A power law fitted to small sizes predicts the large ones")
    ax.legend(loc="upper right", ncol=2)
    return fig


# --------------------------------------------------------------------------
@figure("quantile_regression_bands",
        "Simulated delivery times for 5 km deliveries against network load, "
        "with the 90 percent interval from ordinary least squares plus a "
        "normal residual, and from linear quantile regression at the 5th and "
        "95th percentiles. The least-squares band has constant width and "
        "misses the pattern; the quantile band widens with load and sits "
        "asymmetrically around the median.")
def quantile_regression_bands():
    import statsmodels.api as sm
    r = np.random.default_rng(0)
    n = 6000
    distance = r.uniform(1, 10, n)
    load = r.uniform(0, 1, n)
    mu = 20 + 3.0 * distance
    sigma = 2 + 12 * load
    y = mu + sigma * (r.gamma(2.0, 1.0, n) - 2.0) / np.sqrt(2.0)
    X = sm.add_constant(np.column_stack([distance, load]))
    tr = np.arange(n) < 4000
    ols = sm.OLS(y[tr], X[tr]).fit()
    sd = ols.resid.std()
    q05 = sm.QuantReg(y[tr], X[tr]).fit(q=0.05)
    q50 = sm.QuantReg(y[tr], X[tr]).fit(q=0.5)
    q95 = sm.QuantReg(y[tr], X[tr]).fit(q=0.95)

    # fresh deliveries at 5 km for the picture
    m = 2500
    lp = r.uniform(0, 1, m)
    yp = 20 + 15.0 + (2 + 12 * lp) * (r.gamma(2.0, 1.0, m) - 2.0) / np.sqrt(2.0)
    grid = np.linspace(0, 1, 100)
    Xg = sm.add_constant(np.column_stack([np.full(100, 5.0), grid]), has_constant="add")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(lp, yp, s=7, color=hs.INK_MUTED, alpha=0.35, label="Deliveries at 5 km")
    ax.plot(grid, ols.predict(Xg) + 1.645 * sd, color=P[1], label="OLS + normal, 5th to 95th")
    ax.plot(grid, ols.predict(Xg) - 1.645 * sd, color=P[1])
    ax.plot(grid, q95.predict(Xg), color=P[0], label="Quantile regression, 5th to 95th")
    ax.plot(grid, q05.predict(Xg), color=P[0])
    ax.plot(grid, q50.predict(Xg), color=P[0], lw=1.2, alpha=0.7, label="Quantile regression, median")
    ax.set_xlabel("network load")
    ax.set_ylabel("delivery time (minutes)")
    ax.set_ylim(0, 110)
    ax.set_title("One interval width cannot fit a spread that changes with load")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("measurement_error_attenuation_simex",
        "Left: the fitted regression slope against the reliability of the "
        "predictor, when the true slope is one; noise in the predictor "
        "shrinks the slope in proportion to the share of variance that is "
        "noise. Right: simulation-extrapolation for a predictor with "
        "reliability 0.6, refitting the slope with extra noise added and "
        "extrapolating back to the noise-free case; the quadratic "
        "extrapolant under-corrects, the rational one recovers the truth.")
def measurement_error_attenuation_simex():
    from scipy.optimize import curve_fit
    r = np.random.default_rng(0)
    n = 20000
    x_true = r.normal(size=n)
    y = 1.0 * x_true + r.normal(scale=1.0, size=n)
    rels = [1.0, 0.8, 0.6, 0.4, 0.2]
    slopes_rel = []
    for rel in rels:
        su = np.sqrt((1 - rel) / rel)
        slopes_rel.append(np.polyfit(x_true + r.normal(scale=su, size=n), y, 1)[0])
    # consume the two-predictor draws so the SIMEX section matches the post
    z = r.normal(size=n)
    x1 = z + r.normal(scale=0.7, size=n)
    x2 = z + r.normal(scale=0.7, size=n)
    y2 = 1.0 * x1 + r.normal(size=n)
    for su in (0.0, 0.5, 1.0, 1.5):
        r.normal(scale=su, size=n)
    rel = 0.6
    su = np.sqrt((1 - rel) / rel)
    x_obs = x_true + r.normal(scale=su, size=n)
    lams = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    slopes = np.array([np.mean([np.polyfit(x_obs + r.normal(scale=np.sqrt(l) * su, size=n),
                                           y, 1)[0] for _ in range(20)]) for l in lams])
    quad = np.polyfit(lams, slopes, 2)
    rational = lambda lam, a, b: a / (b + lam)
    (a, b), _ = curve_fit(rational, lams, slopes, p0=[1.0, 2.0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    rg = np.linspace(0.1, 1.0, 100)
    ax1.plot(rg, rg, color=hs.INK_MUTED, lw=1.4, label="Theory: slope = reliability")
    ax1.plot(rels, slopes_rel, "o", color=P[0], markersize=7, label="Simulated fit")
    ax1.axhline(1.0, color=hs.BASELINE, lw=1.0)
    ax1.set_xlabel("reliability of the predictor (signal share of its variance)")
    ax1.set_ylabel("fitted slope (true slope = 1)")
    ax1.set_xlim(0, 1.05); ax1.set_ylim(0, 1.1)
    ax1.set_title("Attenuation", fontsize=11.5)
    ax1.legend(loc="upper left")

    lg = np.linspace(-1, 2, 200)
    ax2.plot(lg, np.polyval(quad, lg), color=P[1], lw=1.6, label="Quadratic extrapolant")
    ax2.plot(lg, rational(lg, a, b), color=P[0], lw=1.6, label="Rational extrapolant")
    ax2.plot(lams, slopes, "o", color=hs.INK_PRIMARY, markersize=6, label="Refits with added noise")
    ax2.plot([-1], [1.0], marker="D", markersize=8, color=P[2], markeredgecolor=hs.SURFACE,
             label="True slope")
    ax2.axvline(0, color=hs.BASELINE, lw=1.0)
    ax2.annotate("observed data", xy=(0, 0.35), xytext=(4, 0), textcoords="offset points",
                 fontsize=9, color=hs.INK_SECONDARY)
    ax2.set_xlabel("added noise variance, as a multiple of the existing noise")
    ax2.set_ylabel("fitted slope")
    ax2.set_ylim(0.2, 1.1)
    ax2.set_title("Simulation-extrapolation", fontsize=11.5)
    ax2.legend(loc="upper right")
    fig.suptitle("Noise in a predictor pulls its slope toward zero, predictably",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    return fig


# --------------------------------------------------------------------------
@figure("equivalence_testing_power",
        "Share of paired comparisons reaching each conclusion against the "
        "number of test cases, with a one-point equivalence margin and a "
        "six-point standard deviation of the per-item difference. A "
        "challenger that is truly 1.5 points worse produces a non-significant "
        "t-test more than half the time at 50 cases, which is not evidence "
        "that it is no worse; two identical models need about 300 cases "
        "before the equivalence test can say so.")
def equivalence_testing_power():
    from matplotlib.ticker import PercentFormatter
    from scipy import stats
    r = np.random.default_rng(0)
    sigma_d, margin = 6.0, 1.0

    def outcomes(n, delta, reps=4000):
        t_sig = tost_eq = 0
        for _ in range(reps):
            d = r.normal(delta, sigma_d, n)
            m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
            t_sig += abs(m / se) > stats.t.ppf(0.975, n - 1)
            lo = m - stats.t.ppf(0.95, n - 1) * se
            hi = m + stats.t.ppf(0.95, n - 1) * se
            tost_eq += (lo > -margin) and (hi < margin)
        return t_sig / reps, tost_eq / reps

    sizes = [50, 100, 200, 400, 800, 1600, 3200]
    ident = [outcomes(n, 0.0)[1] for n in sizes]
    worse15 = [1 - outcomes(n, -1.5)[0] for n in sizes]
    worse05 = [outcomes(n, -0.5)[1] for n in sizes]

    fig, ax = plt.subplots()
    ax.plot(sizes, worse15, marker="o", color=P[1],
            label="t-test not significant, challenger 1.5 points worse")
    ax.plot(sizes, ident, marker="o", color=P[0],
            label="Equivalence shown, models identical")
    ax.plot(sizes, worse05, marker="o", color=P[2],
            label="Equivalence shown, challenger 0.5 points worse")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("paired test cases")
    ax.set_ylabel("share of comparisons")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("A non-significant difference is not equivalence")
    ax.legend(loc="center right")
    return fig


# --------------------------------------------------------------------------
@figure("block_bootstrap_coverage",
        "Coverage of a nominal 95 percent bootstrap interval for the mean of "
        "an autocorrelated series of 200 points, against the block length "
        "used in the bootstrap, for four levels of autocorrelation. With "
        "independent data any block length near one is fine; as dependence "
        "grows the naive bootstrap collapses and longer blocks recover most, "
        "though not all, of the nominal coverage.")
def block_bootstrap_coverage():
    from matplotlib.ticker import PercentFormatter

    def ar1(n, phi, r):
        x = np.empty(n)
        x[0] = r.normal(0, 1 / np.sqrt(1 - phi ** 2))
        e = r.normal(size=n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        return x

    def block_ci(x, block, B, r):
        n = len(x)
        nb = int(np.ceil(n / block))
        means = np.empty(B)
        for b in range(B):
            starts = r.integers(0, n - block + 1, nb)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
            means[b] = x[idx].mean()
        return np.percentile(means, [2.5, 97.5])

    n, reps = 200, 400
    blocks = [1, 2, 5, 10, 20, 40]
    phis = [0.0, 0.3, 0.7, 0.9]
    cov = np.zeros((len(phis), len(blocks)))
    for i, phi in enumerate(phis):
        for s in range(reps):
            r = np.random.default_rng(s)
            x = ar1(n, phi, r)
            for j, b in enumerate(blocks):
                lo, hi = block_ci(x, b, 400, r)
                cov[i, j] += lo <= 0 <= hi
    cov /= reps

    fig, ax = plt.subplots()
    for i, (phi, col) in enumerate(zip(phis, (hs.INK_MUTED, P[2], P[0], P[1]))):
        ax.plot(blocks, cov[i], marker="o", color=col,
                label=f"autocorrelation {phi:.1f}")
    ax.axhline(0.95, color=hs.BASELINE, lw=1.2)
    ax.annotate("nominal 95%", xy=(1, 0.95), xytext=(0, 5), textcoords="offset points",
                fontsize=9.5, color=hs.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xticks(blocks)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("block length (1 = ordinary bootstrap)")
    ax.set_ylabel("coverage of the 95% interval")
    ax.set_ylim(0.25, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("The ordinary bootstrap breaks as soon as observations are dependent")
    ax.legend(loc="lower right")
    return fig


# --------------------------------------------------------------------------
@figure("synthetic_control_paths",
        "Left: monthly outcome for a treated unit and its synthetic control, "
        "a weighted average of donor units chosen to match the treated unit "
        "over the 36 months before the intervention; the gap after month 36 "
        "is the estimated effect. Right: the same gap for the treated unit "
        "against the gaps obtained by treating each donor as if it had been "
        "treated, which is the placebo distribution the effect is judged "
        "against.")
def synthetic_control_paths():
    from scipy.optimize import minimize
    r = np.random.default_rng(0)
    n_donors, pre, post, effect = 20, 36, 12, -8.0
    T = pre + post
    trend = np.cumsum(r.normal(0.3, 1.0, T)) + 100
    season = 10 * np.sin(np.arange(T) * 2 * np.pi / 12)
    f = np.column_stack([trend, season])
    loads = r.uniform(0.3, 1.5, (n_donors + 1, 2))
    loads[0] = (1.2, 0.9)
    mu = r.uniform(-20, 20, n_donors + 1)
    Y = mu[:, None] + loads @ f.T + r.normal(0, 2.0, (n_donors + 1, T))
    Y[0, pre:] += effect

    def synth_weights(y_treated, Y_donors):
        k = Y_donors.shape[0]
        obj = lambda w: np.mean((y_treated - w @ Y_donors) ** 2)
        res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                       options={"maxiter": 500})
        return res.x

    gaps = []
    for j in range(n_donors + 1):
        others = np.delete(Y, j, axis=0)
        wj = synth_weights(Y[j, :pre], others[:, :pre])
        gaps.append(Y[j] - wj @ others)
    gaps = np.array(gaps)
    synth = Y[0] - gaps[0]
    months = np.arange(1, T + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax1.plot(months, Y[0], color=P[0], label="Treated unit")
    ax1.plot(months, synth, color=P[1], label="Synthetic control")
    ax1.axvline(pre + 0.5, color=hs.INK_MUTED, lw=1.2)
    ax1.annotate("intervention", xy=(pre + 0.5, ax1.get_ylim()[1]), xytext=(-6, -14),
                 textcoords="offset points", ha="right", fontsize=9.5, color=hs.INK_SECONDARY)
    ax1.set_xlabel("month")
    ax1.set_ylabel("outcome")
    ax1.set_title("Treated unit and its synthetic control", fontsize=11.5)
    ax1.legend(loc="upper left")

    for j in range(1, n_donors + 1):
        ax2.plot(months, gaps[j], color=hs.INK_MUTED, lw=1.0, alpha=0.45)
    ax2.plot(months, gaps[0], color=P[0], lw=2.2, label="Treated unit")
    ax2.plot([], [], color=hs.INK_MUTED, lw=1.0, label="Each donor treated as a placebo")
    ax2.axvline(pre + 0.5, color=hs.INK_MUTED, lw=1.2)
    ax2.axhline(0, color=hs.BASELINE, lw=1.0)
    ax2.set_xlabel("month")
    ax2.set_ylabel("gap: unit minus its synthetic control")
    ax2.set_title("Placebo gaps", fontsize=11.5)
    ax2.legend(loc="lower left")
    fig.suptitle("The effect is the gap after the intervention, judged against placebo gaps",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    return fig


# --------------------------------------------------------------------------
@figure("queue_wait_vs_utilisation",
        "Left: mean time waiting in queue, in units of the mean service time, "
        "against server utilisation for three levels of service-time "
        "variability, with simulated points for exponential service. The "
        "curves are flat below 70 percent and vertical above 90. Right: "
        "waiting times through a day in which one hour runs at 130 percent "
        "load and every other hour at 80; the backlog takes four hours to "
        "drain.")
def queue_wait_vs_utilisation():
    r = np.random.default_rng(0)

    def gg1_wait(arrivals, services):
        w = np.empty(len(services))
        w[0] = 0.0
        for i in range(1, len(services)):
            w[i] = max(0.0, w[i - 1] + services[i - 1] - arrivals[i])
        return w

    N = 200000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    rhos = np.linspace(0.02, 0.97, 300)
    for cs2, col, label in ((0.0, P[2], "Deterministic service"),
                            (1.0, P[0], "Exponential service"),
                            (4.0, P[1], "High-variability service (c² = 4)")):
        ax1.plot(rhos, rhos / (1 - rhos) * (1 + cs2) / 2, color=col, label=label)
    sim_rhos = [0.5, 0.7, 0.8, 0.9, 0.95]
    pts = []
    for rho in sim_rhos:
        a = r.exponential(1 / rho, N)
        s = r.exponential(1.0, N)
        pts.append(gg1_wait(a, s)[N // 10:].mean())
    ax1.plot(sim_rhos, pts, "o", color=P[0], markersize=7, label="Simulated, exponential")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 25)
    ax1.set_xlabel("utilisation")
    ax1.set_ylabel("mean wait (multiples of service time)")
    ax1.set_title("Waiting against utilisation", fontsize=11.5)
    ax1.legend(loc="upper left")

    rr = np.random.default_rng(3)
    hours, per_hour = 24, 60
    rates = np.where(np.arange(hours) == 8, 1.3, 0.8)
    a, s = [], []
    for h in range(hours):
        n = rr.poisson(rates[h] * per_hour)
        a.append(np.sort(rr.uniform(h * per_hour, (h + 1) * per_hour, n)))
        s.append(rr.exponential(1.0, n))
    arr = np.concatenate(a)
    srv = np.concatenate(s)
    inter = np.diff(np.concatenate([[0.0], arr]))
    w = gg1_wait(inter, srv)
    ax2.scatter(arr / 60, w, s=5, color=hs.INK_MUTED, alpha=0.35, label="Each arrival")
    hourly = [w[(arr >= h * 60) & (arr < (h + 1) * 60)].mean() for h in range(hours)]
    ax2.plot(np.arange(hours) + 0.5, hourly, color=P[0], marker="o", markersize=4,
             label="Hourly mean")
    ax2.axvspan(8, 9, color=P[1], alpha=0.18, lw=0)
    ax2.annotate("130% load", xy=(8.5, ax2.get_ylim()[1] * 0.93), ha="center",
                 fontsize=9.5, color=hs.INK_SECONDARY)
    ax2.set_xlabel("hour of day")
    ax2.set_ylabel("wait (minutes)")
    ax2.set_xlim(0, 24)
    ax2.set_title("One overloaded hour, four hours of backlog", fontsize=11.5)
    ax2.legend(loc="upper right")
    fig.suptitle("Utilisation is cheap until it is not",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    return fig


# --------------------------------------------------------------------------
@figure("changepoint_segmentation_penalty",
        "Left: a series with four shifts in mean, the segment means found by "
        "exact penalised partitioning, and the detected change points. "
        "Right: the number of change points found against the penalty, as a "
        "multiple of log n, for independent noise and for autocorrelated "
        "noise with the same shifts. With independent noise the count "
        "settles on the true four over a wide range of penalties; with "
        "autocorrelation the same penalties find dozens.")
def changepoint_segmentation_penalty():
    def make_series(r, n=600, phi=0.0):
        bounds = [0, 120, 200, 350, 420, 600]
        means = [0.0, 1.5, 0.0, -1.2, 1.0]
        mu = np.zeros(n)
        for (a, b), m in zip(zip(bounds[:-1], bounds[1:]), means):
            mu[a:b] = m
        e = r.normal(size=n)
        if phi:
            for t in range(1, n):
                e[t] = phi * e[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
        return mu + e, bounds[1:-1]

    def optimal_partition(x, penalty):
        n = len(x)
        cs = np.concatenate([[0.0], np.cumsum(x)])
        cs2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
        def cost(a, b):
            s, s2 = cs[b] - cs[a], cs2[b] - cs2[a]
            return s2 - s * s / (b - a)
        F = np.full(n + 1, np.inf)
        F[0] = -penalty
        last = np.zeros(n + 1, dtype=int)
        for b in range(1, n + 1):
            cands = [F[a] + cost(a, b) + penalty for a in range(b)]
            a = int(np.argmin(cands))
            F[b] = cands[a]
            last[b] = a
        cps, b = [], n
        while last[b] > 0:
            cps.append(last[b])
            b = last[b]
        return sorted(cps)

    def robust_sigma(x):
        d = np.diff(x)
        return np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2)

    x, true_cps = make_series(np.random.default_rng(0))
    xa, _ = make_series(np.random.default_rng(1), phi=0.6)
    n = len(x)
    cps = optimal_partition(x, 2 * np.log(n) * robust_sigma(x) ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax1.plot(x, color=hs.INK_MUTED, lw=0.9, alpha=0.8, label="Series")
    b = [0] + cps + [n]
    for a, c in zip(b[:-1], b[1:]):
        ax1.plot([a, c - 1], [x[a:c].mean()] * 2, color=P[0], lw=2.4)
    ax1.plot([], [], color=P[0], lw=2.4, label="Segment means")
    for cp in cps:
        ax1.axvline(cp, color=P[1], lw=1.2)
    ax1.plot([], [], color=P[1], lw=1.2, label="Detected change points")
    ax1.set_xlabel("time")
    ax1.set_ylabel("value")
    ax1.set_title("Independent noise, penalty 2 log n", fontsize=11.5)
    ax1.legend(loc="lower left")

    mults = [0.5, 1, 2, 4, 8, 16]
    counts_iid = [len(optimal_partition(x, m * np.log(n) * robust_sigma(x) ** 2)) for m in mults]
    counts_ar = [len(optimal_partition(xa, m * np.log(n) * robust_sigma(xa) ** 2)) for m in mults]
    ax2.plot(mults, counts_iid, marker="o", color=P[0], label="Independent noise")
    ax2.plot(mults, counts_ar, marker="o", color=P[1], label="Autocorrelated noise (0.6)")
    ax2.axhline(4, color=hs.BASELINE, lw=1.2)
    ax2.annotate("true count: 4", xy=(16, 4), xytext=(0, 5), textcoords="offset points",
                 ha="right", fontsize=9.5, color=hs.INK_SECONDARY)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(mults)
    ax2.set_xticklabels([str(m) for m in mults])
    ax2.set_yticks([1, 2, 4, 10, 30, 100])
    ax2.set_yticklabels(["1", "2", "4", "10", "30", "100"])
    ax2.set_xlabel("penalty, as a multiple of log n")
    ax2.set_ylabel("change points found")
    ax2.set_title("The penalty decides how many", fontsize=11.5)
    ax2.legend(loc="upper right")
    fig.suptitle("Offline change-point detection is a penalised partition",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    return fig


# --------------------------------------------------------------------------
@figure("distance_concentration",
        "Left: relative contrast, the gap between the farthest and nearest "
        "neighbour as a share of the nearest distance, against dimension for "
        "uniform and Gaussian data; it falls toward zero, so every point "
        "becomes about equally far from every other. Right: accuracy of a "
        "nearest-neighbour classifier as noise dimensions are added to two "
        "informative ones, against the same classifier after projecting to "
        "two components and against logistic regression.")
def distance_concentration():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    r = np.random.default_rng(0)
    dims = [1, 2, 5, 10, 20, 50, 100, 500, 2000]
    contrast = {"uniform": [], "gaussian": []}
    for d in dims:
        for name, gen in (("uniform", lambda n: r.uniform(size=(n, d))),
                          ("gaussian", lambda n: r.normal(size=(n, d)))):
            X = gen(1000)
            Q = gen(50)
            D = np.sqrt(((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1))
            contrast[name].append(np.mean((D.max(1) - D.min(1)) / D.min(1)))

    def dataset(n, noise_dims, rr):
        y = rr.integers(0, 2, n)
        X = rr.normal(size=(n, 2)) + 1.5 * y[:, None] * np.array([1.0, 1.0])
        return np.column_stack([X, rr.normal(size=(n, noise_dims))]), y

    noise = [0, 5, 20, 50, 100, 500]
    acc = []
    for nd in noise:
        rows = []
        for rep in range(5):
            rr = np.random.default_rng(10 * nd + rep)
            Xtr, ytr = dataset(1000, nd, rr)
            Xte, yte = dataset(2000, nd, rr)
            knn = KNeighborsClassifier(5).fit(Xtr, ytr).score(Xte, yte)
            pca = PCA(2, svd_solver="full").fit(Xtr)
            knn_pca = KNeighborsClassifier(5).fit(pca.transform(Xtr), ytr).score(
                pca.transform(Xte), yte)
            lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte)
            rows.append((knn, knn_pca, lr))
        acc.append(np.mean(rows, axis=0))
    acc = np.array(acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax1.plot(dims, contrast["uniform"], marker="o", color=P[0], label="Uniform")
    ax1.plot(dims, contrast["gaussian"], marker="o", color=P[1], label="Gaussian")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("dimension")
    ax1.set_ylabel("relative contrast (farthest - nearest) / nearest")
    ax1.set_title("Distances concentrate", fontsize=11.5)
    ax1.legend(loc="upper right")

    xs = np.arange(len(noise))
    ax2.plot(xs, acc[:, 0], marker="o", color=P[1], label="k-NN on all dimensions")
    ax2.plot(xs, acc[:, 1], marker="o", color=P[0], label="k-NN after PCA to 2")
    ax2.plot(xs, acc[:, 2], marker="o", color=P[2], label="Logistic regression")
    ax2.axhline(0.5, color=hs.BASELINE, lw=1.0)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(n) for n in noise])
    ax2.set_xlabel("noise dimensions added to two informative ones")
    ax2.set_ylabel("test accuracy")
    ax2.set_ylim(0.45, 1.0)
    ax2.set_title("Nearest neighbours drown in irrelevant dimensions", fontsize=11.5)
    ax2.legend(loc="lower left")
    fig.suptitle("High dimensions make every neighbour equally far",
                 x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
    return fig


# --------------------------------------------------------------------------
@figure("factorial_vs_ofat_optimum",
        "Share of simulated experiments that end at the best of sixteen "
        "factor settings, against the number of runs spent, for a process "
        "with four two-level factors and one strong interaction. One factor "
        "at a time from the baseline is trapped by the interaction and finds "
        "the optimum in fewer than one experiment in ten at any budget; a "
        "half fraction of eight runs finds it about two thirds of the time "
        "and a full factorial of sixteen runs seven times in ten.")
def factorial_vs_ofat_optimum():
    import itertools
    from matplotlib.ticker import PercentFormatter
    r = np.random.default_rng(0)
    sigma = 2.0
    beta = {"A": 2.0, "B": 1.5, "C": 0.0, "D": 0.5, "AB": 2.5}
    corners = np.array(list(itertools.product([-1, 1], repeat=4)), dtype=float)

    def response(x):
        A, B, C, D = x.T
        mean = 10 + beta["A"] * A + beta["B"] * B + beta["C"] * C + beta["D"] * D + beta["AB"] * A * B
        return mean + r.normal(0, sigma, len(x))

    def ofat_best(per_setting):
        base = -np.ones(4)
        settings = [base.copy()]
        for col in range(4):
            x = base.copy(); x[col] = 1; settings.append(x)
        y = np.array([response(np.repeat(s[None], per_setting, 0)).mean() for s in settings])
        chosen = base.copy()
        chosen[y[1:] - y[0] > 0] = 1
        return chosen

    def factorial_best(design):
        pairs = [(0, 1), (0, 2), (0, 3)] if len(design) % 16 else list(itertools.combinations(range(4), 2))
        cols = lambda d: np.column_stack([np.ones(len(d))] + [d[:, i] for i in range(4)] +
                                         [d[:, i] * d[:, j] for i, j in pairs])
        coef, *_ = np.linalg.lstsq(cols(design), response(design), rcond=None)
        return corners[(cols(corners) @ coef).argmax()]

    def share(fn, arg, reps=1500):
        hits = 0
        for _ in range(reps):
            c = fn(arg)
            hits += c[0] == 1 and c[1] == 1 and c[3] == 1
        return hits / reps

    half = corners[np.prod(corners, axis=1) == 1]
    ofat_budgets = [10, 15, 25, 40, 60, 100]
    ofat = [share(ofat_best, b // 5) for b in ofat_budgets]
    fact_budgets = [8, 16, 32, 64]
    fact = [share(factorial_best, half)] + [share(factorial_best, np.vstack([corners] * (b // 16))) for b in fact_budgets[1:]]

    fig, ax = plt.subplots()
    ax.plot(fact_budgets, fact, marker="o", color=P[0], label="Factorial design (8-run half fraction, then full 2⁴ replicated)")
    ax.plot(ofat_budgets, ofat, marker="o", color=P[1], label="One factor at a time from the baseline")
    ax.set_xscale("log")
    ax.set_xticks(fact_budgets + [100])
    ax.set_xticklabels([str(b) for b in fact_budgets + [100]])
    ax.set_xlabel("experimental runs")
    ax.set_ylabel("share of experiments ending at the best setting")
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("An interaction traps one-factor-at-a-time experiments")
    ax.legend(loc="center right")
    return fig


# --------------------------------------------------------------------------
@figure("winners_curse_optimism",
        "Average gap between the winning configuration's validation accuracy "
        "and its true accuracy, in accuracy points, against the number of "
        "configurations compared, for validation sets of 200, 1,000 and "
        "5,000 items. With one configuration the gap is zero; picking the "
        "best of a hundred on a thousand items overstates its accuracy by "
        "about two and a half points, and on two hundred items by more than "
        "six.")
def winners_curse_optimism():
    r = np.random.default_rng(0)
    ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    def optimism(k, n, reps=3000):
        tot = 0.0
        for _ in range(reps):
            true = np.clip(r.normal(0.80, 0.01, k), 0.5, 0.99)
            val = r.binomial(n, true) / n
            w = val.argmax()
            tot += val[w] - true[w]
        return 100 * tot / reps

    fig, ax = plt.subplots()
    for n, c in zip((200, 1000, 5000), (P[1], P[0], P[2])):
        ax.plot(ks, [optimism(k, n) for k in ks], marker="o", color=c, label=f"validation set of {n:,} items")
    ax.axhline(0, color=P[3], lw=1, ls="--", label="honest estimate (fresh test set)")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("configurations compared on the validation set")
    ax.set_ylabel("winner's validation accuracy minus its true accuracy (points)")
    ax.set_title("The best validation score is an overestimate")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("survivorship_left_truncation",
        "Survival curves against age in years for a fleet whose lifetimes "
        "follow a Weibull distribution with a rising hazard. The reference "
        "curve uses every unit ever installed. Following only the units in "
        "service at a snapshot, and counting them from age zero, roughly "
        "doubles the apparent median life; restricting each risk set to "
        "units already under observation, the left-truncated estimate, "
        "recovers the reference curve.")
def survivorship_left_truncation():
    r = np.random.default_rng(0)
    shape, years, followup = 1.5, 12.0, 2.0
    n = 20000
    env = r.choice([6.0, 4.0], n, p=[0.6, 0.4])
    install = r.uniform(0, years, n)
    life = r.weibull(shape, n) * env
    in_service = install + life > years
    entry = (years - install)[in_service]
    exit_ = np.minimum(life[in_service], entry + followup)
    event = life[in_service] <= entry + followup

    grid = np.linspace(0, 10, 201)

    def km_on_grid(times, events, entry_times=None):
        entry_times = np.zeros_like(times) if entry_times is None else entry_times
        order = np.argsort(times)
        t, e, en = times[order], events[order], entry_times[order]
        ev_t = np.unique(t[e])
        s = 1.0
        surv_at = {}
        for tt in ev_t:
            at_risk = np.sum((en < tt) & (t >= tt))
            d = np.sum((t == tt) & e)
            if at_risk > 0:
                s *= 1 - d / at_risk
            surv_at[tt] = s
        keys = np.array(list(surv_at.keys())); vals = np.array(list(surv_at.values()))
        idx = np.searchsorted(keys, grid, side="right") - 1
        return np.where(idx >= 0, vals[np.clip(idx, 0, None)], 1.0)

    ref = np.array([np.mean(life > g) for g in grid])
    naive = km_on_grid(exit_, event)
    trunc = km_on_grid(exit_, event, entry)

    fig, ax = plt.subplots()
    ax.plot(grid, ref, color=P[0], lw=2.2, label="Reference: every unit ever installed")
    ax.plot(grid, naive, color=P[1], lw=2, label="Survivors at the snapshot, counted from age zero")
    ax.plot(grid, trunc, color=P[2], lw=2, ls="--", label="Survivors with left truncation (risk set by age)")
    ax.axhline(0.5, color=P[3], lw=1, ls=":")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("share still in service")
    ax.set_ylim(0, 1.02)
    ax.set_title("Counting survivors from age zero doubles the apparent lifetime")
    ax.legend(loc="upper right")
    return fig


# --------------------------------------------------------------------------
@figure("ratio_metric_false_positives",
        "False positive rate of A/A tests on a per-session conversion rate, "
        "against the mean number of sessions per user, with 2,000 users per "
        "arm. Treating sessions as independent trials pushes the rate from 5 "
        "percent at one session per user to above 20 percent at ten, and the "
        "more users differ in their propensity to convert the worse it gets; "
        "the delta method with users as the unit stays at 5 percent.")
def ratio_metric_false_positives():
    from matplotlib.ticker import PercentFormatter
    r = np.random.default_rng(0)

    def experiment(n_users, kappa, mean_sessions):
        out = []
        for _ in range(2):
            p = r.beta(0.10 * kappa, 0.90 * kappa, n_users)
            s = 1 + r.poisson(mean_sessions - 1, n_users)
            out.append((s, r.binomial(s, p)))
        return out

    def naive_z(a, b):
        (sa, ca), (sb, cb) = a, b
        pa, pb = ca.sum() / sa.sum(), cb.sum() / sb.sum()
        pooled = (ca.sum() + cb.sum()) / (sa.sum() + sb.sum())
        return (pb - pa) / np.sqrt(pooled * (1 - pooled) * (1 / sa.sum() + 1 / sb.sum()))

    def delta_var(s, c):
        n, mx, my = len(s), s.mean(), c.mean()
        vx, vy, cxy = s.var(ddof=1), c.var(ddof=1), np.cov(s, c, ddof=1)[0, 1]
        return (vy - 2 * (my / mx) * cxy + (my / mx) ** 2 * vx) / (mx ** 2 * n)

    def delta_z(a, b):
        (sa, ca), (sb, cb) = a, b
        return (cb.sum() / sb.sum() - ca.sum() / sa.sum()) / np.sqrt(delta_var(sa, ca) + delta_var(sb, cb))

    sessions = [1, 2, 3, 5, 10]
    rows = {}
    for kappa in (20.0, 4.0):
        naive, delta = [], []
        for ms in sessions:
            hits = np.zeros(2)
            for _ in range(1200):
                a, b = experiment(2000, kappa, ms)
                hits += np.abs([naive_z(a, b), delta_z(a, b)]) > 1.96
            naive.append(hits[0] / 1200); delta.append(hits[1] / 1200)
        rows[kappa] = (naive, delta)

    fig, ax = plt.subplots()
    ax.plot(sessions, rows[4.0][0], marker="o", color=P[1], label="Sessions as independent trials, strongly heterogeneous users")
    ax.plot(sessions, rows[20.0][0], marker="o", color=P[3], label="Sessions as independent trials, mildly heterogeneous users")
    ax.plot(sessions, rows[4.0][1], marker="o", color=P[0], label="Delta method, users as the unit (strongly heterogeneous)")
    ax.axhline(0.05, color=P[2], lw=1, ls="--", label="Nominal 5 percent")
    ax.set_xscale("log")
    ax.set_xticks(sessions)
    ax.set_xticklabels([str(s) for s in sessions])
    ax.set_xlabel("mean sessions per user")
    ax.set_ylabel("A/A tests declared significant")
    ax.set_ylim(0, 0.32)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("The session-level test finds effects that are not there")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("staggered_did_event_study",
        "Treatment effect by periods since adoption in a staggered rollout "
        "where the effect grows with exposure. The group-time estimates track "
        "the true dynamic effect at every exposure; the single two-way "
        "fixed-effects coefficient, drawn as a horizontal line, reports less "
        "than half the true average effect on the treated because it uses "
        "already-treated units as controls for later adopters.")
def staggered_did_event_study():
    r = np.random.default_rng(0)
    n_units, n_periods = 60, 20
    g_of = np.array([5] * 15 + [10] * 15 + [15] * 15 + [-1] * 15)     # -1 = never treated

    def simulate():
        unit_fe = r.normal(0, 1, n_units)
        time_fe = np.linspace(0, 2, n_periods) + r.normal(0, 0.2, n_periods)
        t = np.arange(n_periods)[None, :]
        g = g_of[:, None]
        D = (g >= 0) & (t >= g)
        tau = np.where(D, 0.2 * (t - g + 1), 0.0)
        Y = unit_fe[:, None] + time_fe[None, :] + tau + r.normal(0, 1, (n_units, n_periods))
        return Y, D.astype(float), tau

    def twfe(Y, D):
        Yd = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
        Dd = D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()
        return (Dd * Yd).sum() / (Dd ** 2).sum()

    ks = np.arange(10)
    est = np.zeros(len(ks)); tw = []; att = []
    draws = 200
    for _ in range(draws):
        Y, D, tau = simulate()
        tw.append(twfe(Y, D)); att.append(tau[D == 1].mean())
        for k in ks:
            vals = []
            for g in (5, 10, 15):
                t = g + k
                if t >= n_periods:
                    continue
                tr = g_of == g
                co = (g_of == -1) | (g_of > t)
                vals.append((Y[tr, t] - Y[tr, g - 1]).mean() - (Y[co, t] - Y[co, g - 1]).mean())
            est[k] += np.mean(vals) / draws

    fig, ax = plt.subplots()
    ax.plot(ks, 0.2 * (ks + 1), color=P[0], lw=2.2, label="True effect by exposure")
    ax.plot(ks, est, marker="o", color=P[2], ls="--", label="Group-time estimates (not-yet-treated controls)")
    ax.axhline(np.mean(att), color=P[0], lw=1, ls=":", label=f"True average effect on the treated ({np.mean(att):.2f})")
    ax.axhline(np.mean(tw), color=P[1], lw=2, ls="--", label=f"Two-way fixed effects coefficient ({np.mean(tw):.2f})")
    ax.set_xlabel("periods since adoption")
    ax.set_ylabel("treatment effect")
    ax.set_xticks(ks)
    ax.set_title("A single coefficient cannot summarise a growing effect")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("berkson_selection_correlation",
        "Correlation between two independent ticket attributes, severity and "
        "customer value, among the tickets that were escalated, against the "
        "share of tickets escalated, when escalation depends on the sum of "
        "the two. In the whole population the correlation is zero; the more "
        "selective the escalation, the more negative the correlation among "
        "the escalated tickets, reaching minus 0.65 at one percent.")
def berkson_selection_correlation():
    from matplotlib.ticker import PercentFormatter
    r = np.random.default_rng(0)
    n = 200000
    severity, value = r.normal(0, 1, n), r.normal(0, 1, n)
    score = severity + value + r.normal(0, 0.5, n)
    shares = [0.5, 0.3, 0.15, 0.05, 0.02, 0.01]
    corr = []
    for sh in shares:
        sel = score > np.quantile(score, 1 - sh)
        corr.append(np.corrcoef(severity[sel], value[sel])[0, 1])

    fig, ax = plt.subplots()
    ax.plot(shares, corr, marker="o", color=P[1], label="Correlation among escalated tickets")
    ax.axhline(0, color=P[0], lw=1.5, ls="--", label="Correlation in all tickets (independent attributes)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(shares)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of tickets escalated (more selective to the right)")
    ax.set_ylabel("correlation of severity and customer value")
    ax.set_ylim(-0.8, 0.15)
    for sh, c in zip(shares, corr):
        ax.annotate(f"{c:+.2f}", (sh, c), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=P[1])
    ax.set_title("Selecting on a sum makes independent things look opposed")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("cuped_variance_reduction",
        "Standard error of the estimated treatment effect against the "
        "correlation between a pre-experiment covariate and the outcome, "
        "with 2,000 users per arm, for the plain difference in means, CUPED "
        "and stratification by covariate quartile. CUPED follows the square "
        "root of 1 minus the squared correlation; at a correlation of 0.85 "
        "its standard error is about half the unadjusted one, which is the "
        "same precision as four times the users.")
def cuped_variance_reduction():
    r = np.random.default_rng(0)
    n, sd = 2000, 10.0

    def draw(rho):
        t = np.repeat([0, 1], n)
        x = r.normal(50, sd, 2 * n)
        y = 50 + rho * (x - 50) + r.normal(0, sd * np.sqrt(1 - rho ** 2), 2 * n)
        return t, x, y

    def diff(t, y):
        return y[t == 1].mean() - y[t == 0].mean()

    def cuped(t, x, y):
        theta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
        return diff(t, y - theta * (x - x.mean()))

    def strat(t, x, y, k=4):
        s = np.digitize(x, np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))
        return sum(np.mean(s == j) * diff(t[s == j], y[s == j]) for j in range(k))

    rhos = np.linspace(0, 0.9, 10)
    se = {"diff": [], "cuped": [], "strat": []}
    for rho in rhos:
        e = {"diff": [], "cuped": [], "strat": []}
        for _ in range(1200):
            t, x, y = draw(rho)
            e["diff"].append(diff(t, y)); e["cuped"].append(cuped(t, x, y)); e["strat"].append(strat(t, x, y))
        for k in se:
            se[k].append(np.std(e[k]))
    base = sd * np.sqrt(2 / n)

    fig, ax = plt.subplots()
    ax.plot(rhos, se["diff"], marker="o", color=P[1], label="Difference in means")
    ax.plot(rhos, se["strat"], marker="o", color=P[3], label="Stratified by covariate quartile")
    ax.plot(rhos, se["cuped"], marker="o", color=P[0], label="CUPED (regression adjustment gives the same)")
    ax.plot(rhos, base * np.sqrt(1 - rhos ** 2), color=P[0], lw=1, ls="--", label="Theory: SE × √(1 − ρ²)")
    ax.set_xlabel("correlation between the pre-experiment covariate and the outcome")
    ax.set_ylabel("standard error of the estimated effect")
    ax.set_ylim(0, 0.36)
    ax.set_title("A correlated covariate buys the precision of more users")
    ax.legend(loc="lower left")
    return fig


# --------------------------------------------------------------------------
@figure("bandit_cumulative_regret",
        "Cumulative conversions lost against the number of users, relative "
        "to always showing the better of two variants with conversion rates "
        "of 10 and 13 percent, averaged over 100 runs. A fixed even split "
        "loses conversions at a constant rate for the whole test; Thompson "
        "sampling loses them quickly at first and then almost stops as it "
        "shifts traffic to the winner; an even split for the first quarter "
        "followed by exploiting the observed winner sits in between.")
def bandit_cumulative_regret():
    r = np.random.default_rng(0)
    p = np.array([0.10, 0.13])
    horizon, runs = 20000, 100
    grid = np.arange(0, horizon + 1, 500)

    def thompson():
        n = np.zeros(2); s = np.zeros(2); lost = np.zeros(horizon)
        for i in range(horizon):
            a = np.argmax(r.beta(1 + s, 1 + n - s))
            n[a] += 1; s[a] += r.random() < p[a]
            lost[i] = p.max() - p[a]
        return np.cumsum(lost)

    def even_then_exploit(share):
        n_test = int(horizon * share)
        arms = np.tile([0, 1], n_test // 2 + 1)[:n_test]
        conv = r.random(n_test) < p[arms]
        rates = [conv[arms == a].mean() for a in (0, 1)]
        best = int(np.argmax(rates))
        lost = np.concatenate([p.max() - p[arms], np.full(horizon - n_test, p.max() - p[best])])
        return np.cumsum(lost)

    curves = {"Thompson sampling": np.mean([thompson() for _ in range(runs)], axis=0),
              "Even split for a quarter, then exploit": np.mean([even_then_exploit(0.25) for _ in range(runs)], axis=0),
              "Even split for the whole test": np.mean([even_then_exploit(1.0) for _ in range(runs)], axis=0)}

    fig, ax = plt.subplots()
    for (name, c), col in zip(curves.items(), (P[0], P[2], P[1])):
        ax.plot(grid, np.concatenate([[0], c[grid[1:] - 1]]), color=col, lw=2, label=name)
    ax.set_xlabel("users")
    ax.set_ylabel("expected conversions lost to the worse variant")
    ax.set_title("What exploration costs, and when it stops costing")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("recurrent_events_mcf",
        "Mean cumulative failures per machine against age for a simulated "
        "fleet with recurring failures, estimated with proper risk sets, for "
        "all machines and for harsh and normal sites separately, alongside "
        "the naive average count that ignores which machines are still under "
        "observation. The naive curve flattens after two years, where "
        "observation ends for many machines; the proper estimate keeps "
        "climbing at the true rate.")
def recurrent_events_mcf():
    r = np.random.default_rng(0)
    n = 400
    env = r.choice(["normal", "harsh"], n, p=[0.6, 0.4])
    frailty = r.gamma(2.0, 0.5, n)
    rate = 0.8 * np.where(env == "harsh", 2.0, 1.0) * frailty
    observed = r.uniform(1.0, 3.0, n)
    events = []
    for i in range(n):
        t = 0.0
        while True:
            t += r.exponential(1 / rate[i])
            if t > observed[i]:
                break
            events.append((i, t))
    events = np.array(events)
    grid = np.linspace(0.05, 3.0, 60)

    def mcf(mask):
        ev = events[np.isin(events[:, 0], np.where(mask)[0])][:, 1]
        obs = observed[mask]
        ev = np.sort(ev)
        inc = 1.0 / np.array([np.sum(obs >= a) for a in ev])
        return np.array([inc[ev <= g].sum() for g in grid])

    def naive(mask):
        idx = np.where(mask)[0]
        return np.array([np.mean([np.sum((events[:, 0] == i) & (events[:, 1] <= g)) for i in idx]) for g in grid])

    fig, ax = plt.subplots()
    ax.plot(grid, mcf(env == "harsh"), color=P[1], lw=2, label="Harsh sites, mean cumulative function")
    ax.plot(grid, mcf(np.ones(n, bool)), color=P[0], lw=2.2, label="All machines, mean cumulative function")
    ax.plot(grid, mcf(env == "normal"), color=P[2], lw=2, label="Normal sites, mean cumulative function")
    ax.plot(grid, naive(np.ones(n, bool)), color=P[3], lw=2, ls="--", label="All machines, naive average count")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("cumulative failures per machine")
    ax.set_title("Counting failures per machine needs a risk set")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("rdd_bandwidth_tradeoff",
        "Bias, standard deviation and root mean squared error of the local "
        "linear regression discontinuity estimate against the bandwidth on "
        "each side of the cutoff, for a simulated outcome whose trend bends "
        "differently on the two sides. Narrow bandwidths are unbiased and "
        "noisy; wide ones are precise and biased; the error is smallest at "
        "an intermediate width, which is the choice a bandwidth selector "
        "makes.")
def rdd_bandwidth_tradeoff():
    r = np.random.default_rng(0)
    tau, n = 2.0, 5000

    def draw():
        x = r.uniform(-50, 50, n)
        d = (x >= 0).astype(float)
        curve = np.where(x >= 0, 0.002, -0.001) * x ** 2
        return x, 10 + 0.08 * x + curve + tau * d + r.normal(0, 4, n)

    def local_linear(x, y, h):
        est = []
        for side in (x >= 0, x < 0):
            m = side & (np.abs(x) <= h)
            X = np.column_stack([np.ones(m.sum()), x[m]])
            b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
            est.append(b[0])
        return est[0] - est[1]

    hs = [2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
    bias, sd, rmse = [], [], []
    for h in hs:
        e = np.array([local_linear(*draw(), h) for _ in range(600)])
        bias.append(abs(e.mean() - tau)); sd.append(e.std()); rmse.append(np.sqrt(np.mean((e - tau) ** 2)))

    fig, ax = plt.subplots()
    ax.plot(hs, rmse, marker="o", color=P[0], lw=2.2, label="Root mean squared error")
    ax.plot(hs, sd, marker="o", color=P[2], label="Standard deviation (noise)")
    ax.plot(hs, bias, marker="o", color=P[1], label="Absolute bias (curvature)")
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("bandwidth on each side of the cutoff (units of the running variable)")
    ax.set_ylabel("error of the estimated effect")
    ax.set_title("The bandwidth trades curvature bias against noise")
    ax.legend(loc="upper center")
    return fig


# --------------------------------------------------------------------------
@figure("noncompliance_estimators",
        "Estimated effect of a feature against the share of users who use "
        "it when assigned, for four estimators in a simulated experiment "
        "with 10 percent always-takers and a true effect of 2. Intention to "
        "treat falls in proportion to compliance; as-treated and per-protocol "
        "estimates sit above the truth at every compliance rate because "
        "users who take the feature differ from those who do not; the Wald "
        "estimate recovers the effect on compliers throughout.")
def noncompliance_estimators():
    r = np.random.default_rng(0)
    tau = 2.0

    def draw(n, compliers, always=0.1):
        z = r.integers(0, 2, n)
        u = r.random(n)
        kind = np.where(u < compliers, "c", np.where(u < compliers + always, "a", "n"))
        d = np.where(kind == "c", z, np.where(kind == "a", 1, 0))
        base = 10 + np.where(kind == "a", 3.0, np.where(kind == "n", -2.0, 0.0))
        return z, d, base + tau * d + r.normal(0, 5, n)

    def itt(z, d, y): return y[z == 1].mean() - y[z == 0].mean()
    def as_treated(z, d, y): return y[d == 1].mean() - y[d == 0].mean()
    def per_protocol(z, d, y): return y[(z == 1) & (d == 1)].mean() - y[(z == 0) & (d == 0)].mean()
    def wald(z, d, y): return itt(z, d, y) / (d[z == 1].mean() - d[z == 0].mean())

    cs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    series = {"Intention to treat": [], "As treated": [], "Per protocol": [], "Wald (complier effect)": []}
    for c in cs:
        vals = {k: [] for k in series}
        for _ in range(500):
            z, d, y = draw(4000, c)
            for k, fn in zip(series, (itt, as_treated, per_protocol, wald)):
                vals[k].append(fn(z, d, y))
        for k in series:
            series[k].append(np.mean(vals[k]))

    fig, ax = plt.subplots()
    ax.axhline(tau, color=P[3], lw=1, ls=":", label="True effect of using the feature (2.0)")
    for (k, v), col in zip(series.items(), (P[0], P[1], P[4], P[2])):
        ax.plot(cs, v, marker="o", color=col, label=k)
    ax.set_xlabel("share of users who use the feature when assigned to it")
    ax.set_ylabel("estimated effect")
    ax.set_title("Comparing users by what they did, not what they were assigned, invents effects")
    ax.legend(loc="upper right")
    return fig


# --------------------------------------------------------------------------
@figure("cv_selection_leakage",
        "Five-fold cross-validated accuracy on pure-noise data with 100 "
        "samples and a balanced binary label, against the number of "
        "candidate features, when the ten most label-correlated features are "
        "chosen on all the data before cross-validation and when they are "
        "chosen inside each training fold. With the selection outside the "
        "folds, accuracy on noise climbs above 80 percent as the pool of "
        "features grows; inside the folds it stays at chance.")
def cv_selection_leakage():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    r = np.random.default_rng(0)

    def top_k(X, y, k):
        yc = y - y.mean()
        corr = np.abs((X - X.mean(0)).T @ yc) / (X.std(0) * yc.std() * len(y) + 1e-12)
        return np.argsort(corr)[-k:]

    def cv_acc(X, y, k, inside):
        skf = StratifiedKFold(5, shuffle=True, random_state=int(r.integers(1e9)))
        cols = None if inside else top_k(X, y, k)
        correct = 0
        for tr, te in skf.split(X, y):
            c = top_k(X[tr], y[tr], k) if inside else cols
            clf = LogisticRegression(max_iter=1000).fit(X[tr][:, c], y[tr])
            correct += (clf.predict(X[te][:, c]) == y[te]).sum()
        return correct / len(y)

    ps = [20, 50, 100, 300, 1000, 3000, 10000]
    outside, inside = [], []
    for p in ps:
        o, i = [], []
        for _ in range(40):
            X = r.normal(0, 1, (100, p)); y = np.repeat([0, 1], 50); r.shuffle(y)
            o.append(cv_acc(X, y, 10, False)); i.append(cv_acc(X, y, 10, True))
        outside.append(np.mean(o)); inside.append(np.mean(i))

    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots()
    ax.plot(ps, outside, marker="o", color=P[1], lw=2, label="Ten features selected on all the data, then cross-validated")
    ax.plot(ps, inside, marker="o", color=P[0], lw=2, label="Ten features selected inside each training fold")
    ax.axhline(0.5, color=P[3], lw=1, ls="--", label="Chance")
    ax.set_xscale("log")
    ax.set_xticks(ps)
    ax.set_xticklabels([f"{p:,}" for p in ps])
    ax.set_xlabel("candidate features (all noise)")
    ax.set_ylabel("cross-validated accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Selecting features before cross-validation manufactures accuracy")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------
@figure("cluster_randomisation_false_positives",
        "False positive rate of A/A experiments in which 20 stores are "
        "randomised with 200 customers each, against the share of outcome "
        "variance that sits between stores, for a test that treats "
        "customers as independent and for a test on store means. The "
        "customer-level test passes 5 percent at any positive intraclass "
        "correlation and exceeds 50 percent at 0.05; the store-level test "
        "stays at its nominal level throughout.")
def cluster_randomisation_false_positives():
    from matplotlib.ticker import PercentFormatter
    from scipy import stats
    r = np.random.default_rng(0)

    def draw(n_c, per, icc, sd=10.0):
        sd_b, sd_w = sd * np.sqrt(icc), sd * np.sqrt(1 - icc)
        z = np.repeat([0, 1], n_c // 2); r.shuffle(z)
        store = r.normal(0, sd_b, n_c)
        cl = np.repeat(np.arange(n_c), per)
        return z, cl, 50 + store[cl] + r.normal(0, sd_w, len(cl))

    def customer_p(z, cl, y):
        t = z[cl]; a, b = y[t == 0], y[t == 1]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return 2 * stats.norm.sf(abs((b.mean() - a.mean()) / se))

    def store_p(z, cl, y):
        m = np.array([y[cl == c].mean() for c in range(z.size)])
        a, b = m[z == 0], m[z == 1]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return 2 * stats.t.sf(abs((b.mean() - a.mean()) / se), z.size - 2)

    iccs = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    fp_c, fp_s = [], []
    for icc in iccs:
        hc = hs = 0
        for _ in range(800):
            z, cl, y = draw(20, 200, icc)
            hc += customer_p(z, cl, y) < 0.05; hs += store_p(z, cl, y) < 0.05
        fp_c.append(hc / 800); fp_s.append(hs / 800)

    fig, ax = plt.subplots()
    ax.plot(iccs, fp_c, marker="o", color=P[1], lw=2, label="Customers treated as independent")
    ax.plot(iccs, fp_s, marker="o", color=P[0], lw=2, label="Test on store means (20 stores)")
    ax.axhline(0.05, color=P[3], lw=1, ls="--", label="Nominal 5 percent")
    ax.set_xlabel("intraclass correlation (share of variance between stores)")
    ax.set_ylabel("A/A experiments declared significant")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Randomise stores, analyse customers, and the test breaks")
    ax.legend(loc="center right")
    return fig


# --------------------------------------------------------------------------
@figure("marketplace_interference_lift",
        "Estimated lift from a treatment that raises buyers' purchase "
        "intent from 10 to 12 percent, against the daily inventory of a "
        "shared marketplace with 2,000 buyers a day, for a buyer-level "
        "split and for randomisation of separate markets, alongside the "
        "true lift from rolling the treatment out to everyone. When "
        "inventory is ample all three agree at about 20 percent; as stock "
        "binds, the true lift falls toward zero while the buyer-level "
        "estimate stays near 20 percent, because treated buyers take units "
        "the control buyers would have bought.")
def marketplace_interference_lift():
    from matplotlib.ticker import PercentFormatter
    r = np.random.default_rng(0)
    buyers, pc, pt = 2000, 0.10, 0.12

    def sales(intents, inventory):
        order = r.permutation(len(intents))
        served = np.cumsum(intents[order]) <= inventory
        sold = np.zeros(len(intents), bool); sold[order] = intents[order] & served
        return sold

    def experiment(inventory, design, days=20, n_markets=20):
        tc = tt = ec = et = 0.0
        for d in range(days):
            ic = r.random(buyers) < pc; it = r.random(buyers) < pt
            tc += min(ic.sum(), inventory); tt += min(it.sum(), inventory)
            if design == "buyer":
                z = r.integers(0, 2, buyers); s = sales(np.where(z == 1, it, ic), inventory)
                et += s[z == 1].sum() / (z == 1).mean(); ec += s[z == 0].sum() / (z == 0).mean()
            else:
                zm = np.repeat([0, 1], n_markets // 2); r.shuffle(zm)
                for m in range(n_markets):
                    s = sales(r.random(buyers) < (pt if zm[m] else pc), inventory).sum()
                    if zm[m]: et += s / zm.sum()
                    else: ec += s / (n_markets - zm.sum())
        return et / ec - 1, tt / tc - 1

    invs = [320, 280, 250, 230, 215, 205, 200, 195]
    buyer, market, truth = [], [], []
    for inv in invs:
        b = np.array([experiment(inv, "buyer") for _ in range(60)])
        m = np.array([experiment(inv, "market") for _ in range(60)])
        buyer.append(b[:, 0].mean()); market.append(m[:, 0].mean()); truth.append(b[:, 1].mean())

    fig, ax = plt.subplots()
    ax.plot(invs, buyer, marker="o", color=P[1], lw=2, label="Buyer-level split inside one shared market")
    ax.plot(invs, market, marker="o", color=P[2], lw=2, label="Separate cities randomised (20 cities, 10 treated)")
    ax.plot(invs, truth, color=P[0], lw=2.4, label="True lift from rolling out to everyone")
    ax.invert_xaxis()
    ax.set_xlabel("daily inventory (units); control demand is 200, treated demand 240")
    ax.set_ylabel("estimated lift in sales")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Shared inventory makes the buyer-level test overstate the lift")
    ax.legend(loc="center left")
    return fig


# --------------------------------------------------------------------------
@figure("small_count_interval_coverage",
        "Actual coverage of nominal 95 percent confidence intervals for a "
        "proportion against the expected number of events in the sample, "
        "for the Wald, Wilson and exact Clopper-Pearson intervals, with "
        "a true rate of half a percent. The Wald interval's coverage "
        "collapses below two expected events, where it is often the empty "
        "interval at zero; Wilson holds near the nominal level and the "
        "exact interval stays above it.")
def small_count_interval_coverage():
    from matplotlib.ticker import PercentFormatter
    from scipy import stats
    r = np.random.default_rng(0)
    z = stats.norm.ppf(0.975)

    def wald(k, n):
        p = k / n; h = z * np.sqrt(p * (1 - p) / n)
        return max(0, p - h), min(1, p + h)

    def wilson(k, n):
        p = k / n; c = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
        return max(0, c - h), min(1, c + h)

    def exact(k, n):
        lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
        return lo, hi

    p = 0.005
    ns = [100, 200, 400, 800, 1600, 3200, 6400]
    cov = {"Wald": [], "Wilson": [], "Exact": []}
    for n in ns:
        ks = r.binomial(n, p, 4000)
        for name, fn in (("Wald", wald), ("Wilson", wilson), ("Exact", exact)):
            cache = {}
            hits = 0
            for k in ks:
                if k not in cache:
                    cache[k] = fn(int(k), n)
                lo, hi = cache[k]; hits += lo <= p <= hi
            cov[name].append(hits / 4000)
    expected = [n * p for n in ns]

    fig, ax = plt.subplots()
    ax.plot(expected, cov["Wald"], marker="o", color=P[1], lw=2, label="Wald (normal approximation)")
    ax.plot(expected, cov["Wilson"], marker="o", color=P[0], lw=2, label="Wilson score")
    ax.plot(expected, cov["Exact"], marker="o", color=P[2], lw=2, label="Exact (Clopper-Pearson)")
    ax.axhline(0.95, color=P[3], lw=1, ls="--", label="Nominal 95 percent")
    ax.set_xscale("log")
    ax.set_xticks(expected)
    ax.set_xticklabels([f"{e:g}" for e in expected])
    ax.set_xlabel("expected number of events in the sample (true rate 0.5 percent)")
    ax.set_ylabel("share of intervals containing the true rate")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Below a handful of events, the textbook interval fails")
    ax.legend(loc="lower right")
    return fig


def main(names):
    hs.FIGDIR.mkdir(parents=True, exist_ok=True)
    chosen = names or sorted(FIGURES)
    for slug in chosen:
        if slug not in FIGURES:
            print(f"  ?? unknown figure: {slug}")
            continue
        fn, alt = FIGURES[slug]
        fig = fn()
        res = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:26} {res['width']}x{res['height']}")
    print(f"\n{len(chosen)} figures -> {hs.FIGDIR}")


if __name__ == "__main__":
    main(sys.argv[1:])

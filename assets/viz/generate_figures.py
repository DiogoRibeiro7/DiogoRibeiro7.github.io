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

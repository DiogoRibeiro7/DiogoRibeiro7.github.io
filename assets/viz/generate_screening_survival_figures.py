"""Reproduce the screening article's synthetic cohorts and sampling examples.

Run with --dry-run to print calculations without writing the two PNG figures.
No empirical patient records or estimates of clinical effectiveness are used.
"""

import argparse
import json
from math import isfinite


def cohort(earlier=False, additional=False, postponed_deaths=0):
    """Return paired histories for the same 1,000 people under a scenario.

    Progressive disease: IDs 0..119; 75 cancer deaths at year 9, 45 other
    deaths at year 20. Indolent lesions: IDs 120..299; other deaths at 20.
    Remaining people: 50 other deaths at 10 and 650 at 20. Earlier diagnosis
    moves year 6 to year 2. A benefit postpones selected cancer deaths to 20,
    when those people die of another cause. All histories are uncensored.
    """
    if (isinstance(postponed_deaths, bool)
            or not isinstance(postponed_deaths, int)
            or not 0 <= postponed_deaths <= 75):
        raise ValueError("postponed_deaths must be an integer from 0 to 75")
    if postponed_deaths and not earlier:
        raise ValueError("The benefit scenario requires earlier diagnosis")
    rows = []
    for person in range(1000):
        progressive = person < 120
        indolent = 120 <= person < 300
        diagnosed = progressive or (additional and indolent)
        cancer_death = postponed_deaths <= person < 75
        death = 9 if cancer_death else (10 if 300 <= person < 350 else 20)
        rows.append({"person": person,
                     "kind": "progressive" if progressive else ("indolent" if indolent else "none"),
                     "diagnosis": (2 if earlier else 6) if diagnosed else None,
                     "death": death, "cause": "cancer" if cancer_death else "other"})
    return rows


def summarise(rows, horizon=12, survival_years=5):
    """Use all people for death risks and diagnosed people for survival.

    Death at the horizon counts as a death; survival means strictly beyond
    diagnosis + survival_years. Complete follow-up is an explicit assumption.
    """
    diagnosed = [row for row in rows if row["diagnosis"] is not None]
    cancer = sum(row["death"] <= horizon and row["cause"] == "cancer" for row in rows)
    deaths = sum(row["death"] <= horizon for row in rows)
    survivors = sum(row["death"] > row["diagnosis"] + survival_years for row in diagnosed)
    return {"people": len(rows), "diagnoses": len(diagnosed),
            "five_year_survivors": survivors,
            "five_year_survival": survivors / len(diagnosed) if diagnosed else None,
            "cancer_deaths": cancer, "all_deaths": deaths,
            "cancer_death_risk": cancer / len(rows), "all_death_risk": deaths / len(rows)}


def sampling(rates=(40, 40), durations=(1, 4), interval=2):
    """Stationary prevalence and repeated-screen selection with perfect sensitivity.

    Entry phases are uniform relative to regular screens. Incident cases are
    counted once, at their first detection. No competing exit occurs during
    the detectable window. Returns expected stocks/annual flows, not patients.
    """
    if (len(rates) != len(durations) or not rates
            or not isfinite(interval) or interval <= 0
            or any(not isfinite(r) or r < 0 for r in rates)
            or any(not isfinite(d) or d <= 0 for d in durations)
            or sum(rates) <= 0):
        raise ValueError("Use matching rates/durations, positive durations/interval and positive total rate")
    stock = [r * d for r, d in zip(rates, durations)]
    detection = [min(1, d / interval) for d in durations]
    detected = [r * p for r, p in zip(rates, detection)]
    incident_weights = [r / sum(rates) for r in rates]
    snapshot_weights = [n / sum(stock) for n in stock]
    return {"incident_rates": list(rates), "snapshot_stock": stock,
            "incident_weights": incident_weights, "snapshot_weights": snapshot_weights,
            "detection_probabilities": detection, "detected_annual_rates": detected,
            "repeated_weights": [n / sum(detected) for n in detected],
            "incident_mean_duration": sum(w*d for w, d in zip(incident_weights, durations)),
            "snapshot_mean_duration": sum(w*d for w, d in zip(snapshot_weights, durations))}


SCENARIOS = {
    "Clinical diagnosis": {},
    "Earlier diagnosis only": {"earlier": True},
    "Additional diagnoses only": {"additional": True},
    "Both, no effect on death": {"earlier": True, "additional": True},
    "Both, with 18 deaths postponed": {"earlier": True, "additional": True, "postponed_deaths": 18},
}


def examples():
    return {"cohort_scenarios": {name: summarise(cohort(**args)) for name, args in SCENARIOS.items()},
            "duration_selection": sampling()}


def draw_figures():
    import matplotlib.pyplot as plt
    from housestyle import INK_SECONDARY, PALETTE, save, use

    use()
    rows = list(examples()["cohort_scenarios"].values())
    labels = ["Clinical diagnosis", "Earlier diagnosis only", "Additional diagnoses only",
              "Both, no effect on death", "Both, 18 deaths postponed"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.get_layout_engine().set(rect=(0, .09, 1, .80))
    fig.suptitle("Diagnosis statistics and mortality answer different questions", fontsize=13, y=.98)
    for ax, key, scale, color, title, xlabel, xmax in (
            (axes[0], "five_year_survival", 100, PALETTE[0], "Among diagnosed people",
             "Alive beyond five years after diagnosis (%)", 118),
            (axes[1], "cancer_deaths", 1, PALETTE[1], "Among all 1,000 people",
             "Cancer deaths by year 12 (count)", 92)):
        values = [row[key] * scale for row in rows]
        ax.barh(range(5), values, color=color, height=.55)
        for y, value in enumerate(values):
            ax.text(value + 2, y, f"{value:g}", va="center", fontsize=10)
        ax.set(xlim=(0, xmax), title=title, xlabel=xlabel, yticks=range(5))
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    axes[0].set_yticklabels(labels, fontsize=9.5)
    axes[0].invert_yaxis()
    fig.text(.02, .008, "Synthetic paired histories; complete follow-up. Diagnosed denominators: 120, 120, 300, 300, 300.\n"
             "Only the final scenario changes a death time. The horizon for mortality starts at common eligibility.",
             fontsize=8.5, color=INK_SECONDARY)
    print(json.dumps(save(fig, "science_screening_survival_and_mortality")))

    selection = sampling()
    fig, ax = plt.subplots(figsize=(9, 5.3))
    fig.get_layout_engine().set(rect=(0, .13, 1, .82))
    positions = range(3)
    slow = [100 * selection[key][1] for key in ("incident_weights", "snapshot_weights", "repeated_weights")]
    fast = [100 - value for value in slow]
    ax.bar(positions, fast, color=PALETTE[0], width=.56, label="Fast: 1-year detectable window")
    ax.bar(positions, slow, bottom=fast, color=PALETTE[1], width=.56,
           label="Slow: 4-year detectable window")
    for x, f, s in zip(positions, fast, slow):
        ax.text(x, f/2, f"Fast {f:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        ax.text(x, f+s/2, f"Slow {s:.1f}%", ha="center", va="center", fontweight="bold")
    ax.set(ylim=(0, 122), ylabel="Composition of cases (%)", yticks=[0, 20, 40, 60, 80, 100],
           xticks=list(positions), xticklabels=["New detectable cases\n40 fast + 40 slow / year",
               "First-round snapshot\n40 fast + 160 slow present",
               "Repeated 2-year screens\n20 fast + 40 slow detected / year"],
           title="Detectable duration changes which disease histories are observed")
    ax.legend(loc="upper center", ncol=2, fontsize=8.5)
    fig.text(.02, .006, "Synthetic stationary model; perfect sensitivity within the detectable window, uniform entry phases.\n"
             "The first-round stock and subsequent annual detection flow have different denominators.",
             fontsize=8.5, color=INK_SECONDARY)
    print(json.dumps(save(fig, "science_screening_duration_selection")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(examples(), indent=2))
    if not args.dry_run:
        draw_figures()


if __name__ == "__main__":
    main()

"""Numbers and figures for the essay on aspartame, fruit and dose.

    python generate_aspartame_dose_figures.py            # both figures and the numbers
    python generate_aspartame_dose_figures.py --dry-run  # the numbers only, no files written

The argument of the essay is that "its breakdown products occur in fruit" is
relevant and incomplete, and that the missing part is dose. This script does the
dose arithmetic from published figures: methanol from each source on one scale,
cans needed to reach the acceptable daily intake, and a one-compartment prediction
of peak blood methanol checked against the concentrations measured in volunteers.
The model needs only the standard library; matplotlib is imported to draw.
"""

import argparse
import json

METHANOL_SHARE = 0.10             # of aspartame by weight on hydrolysis (WHO EHC 196; EFSA 2013)
PHENYLALANINE_SHARE = 0.56        # EFSA 2013
ADI = {"EFSA and JECFA": 40, "FDA": 50}                    # mg aspartame per kg body weight per day
CAN_LITRES = 0.33
# mg aspartame per litre of soft drink: the EU maximum permitted level, and the means of two Portuguese
# surveys (Lino et al. 2008; Basilio et al. 2020, both Food Addit Contam Part A).
DRINK = {"maximum permitted": 600, "measured mean, 2008": 89, "measured mean, 2020": 161.5}
# mg methanol per litre of fruit juice, WHO EHC 196: mean and range.
JUICE = {"mean": 140, "low": 12, "high": 640}
APPLES_1KG = (400, 1400)          # mg methanol from the pectin in 1 kg of apples (Lindinger et al. 1997)
ENDOGENOUS = (300, 600)           # mg methanol made by the body per day (Lindinger et al. 1997)
LETHAL_PER_KG = (300, 1000)       # minimum lethal dose of methanol, mg/kg, untreated (WHO EHC 196)
VOLUME_OF_DISTRIBUTION = 0.77     # litres per kg, methanol (EFSA 2013, citing Graw et al. 2000)
# Stegink et al. 1981: mean peak blood methanol, mg/L, with SD, after one dose of aspartame in mg/kg;
# at 34 mg/kg it was below the detection limit of 4 mg/L in all 12 subjects.
STEGINK = {100: (12.7, 4.8), 150: (21.4, 3.5), 200: (25.8, 7.8)}
DETECTION_LIMIT = 4.0
BLOOD_THRESHOLDS = {"effects on the nervous system": 200, "effects on the eye": 500}      # mg/L, WHO EHC 196
# Debras et al. 2022: mean aspartame intake of the higher-consumer group, mg/day, and its hazard ratio.
NUTRINET = {"higher consumers, mg/day": 47.42, "hazard ratio": 1.15, "interval": (1.03, 1.28)}
# EFSA 2013, Table 8: highest 95th-percentile exposure, mg/kg/day, by age group.
HIGH_EXPOSURE = {"toddlers": 36.0, "children": 32.4, "adults": 27.5}


def methanol_from(aspartame_mg):
    return METHANOL_SHARE * aspartame_mg


def predicted_peak(aspartame_mg_per_kg):
    """Peak blood methanol, mg/L, if all the methanol from one dose were absorbed at once and none cleared."""
    return methanol_from(aspartame_mg_per_kg) / VOLUME_OF_DISTRIBUTION


def cans_to_reach(adi, weight_kg, mg_per_litre):
    return adi * weight_kg / (mg_per_litre * CAN_LITRES)


def summary():
    can = {name: round(level * CAN_LITRES, 1) for name, level in DRINK.items()}
    one_can_70kg = DRINK["maximum permitted"] * CAN_LITRES / 70
    return {
        "aspartame in one can, mg": can,
        "methanol from one can, mg": {name: round(methanol_from(mg), 1) for name, mg in can.items()},
        "methanol from a 250 mL glass of juice, mg": {k: round(0.25 * v, 1) for k, v in JUICE.items()},
        "methanol from aspartame at the ADI, 70 kg, mg": round(methanol_from(ADI["EFSA and JECFA"] * 70)),
        "minimum lethal dose of methanol, 70 kg, mg": [70 * v for v in LETHAL_PER_KG],
        "cans to reach the ADI": {
            f"{weight} kg, {name}": round(cans_to_reach(ADI["EFSA and JECFA"], weight, level), 1)
            for weight in (70, 20) for name, level in DRINK.items()},
        "predicted and observed peak blood methanol, mg/L": {
            str(dose): [round(predicted_peak(dose), 1), observed] for dose, (observed, _) in STEGINK.items()},
        "predicted peak at 34 mg/kg, mg/L": round(predicted_peak(34), 1),
        "predicted peak at the ADI, mg/L": round(predicted_peak(ADI["EFSA and JECFA"]), 1),
        "predicted rise from one can at 70 kg, mg/L": round(predicted_peak(one_can_70kg), 2),
        "nervous-system threshold over the rise from one can": round(BLOOD_THRESHOLDS["effects on the nervous system"] / predicted_peak(one_can_70kg)),
        "highest 95th-percentile exposure, % of the ADI": {k: round(100 * v / ADI["EFSA and JECFA"]) for k, v in HIGH_EXPOSURE.items()},
        "NutriNet higher consumers, % of the ADI at 70 kg": round(100 * NUTRINET["higher consumers, mg/day"] / (ADI["EFSA and JECFA"] * 70), 1),
        "phenylalanine from one can at the maximum level, mg": round(PHENYLALANINE_SHARE * DRINK["maximum permitted"] * CAN_LITRES),
    }


def sources_figure(plt, hs):
    can_low = methanol_from(DRINK["measured mean, 2008"] * CAN_LITRES)
    can_high = methanol_from(DRINK["maximum permitted"] * CAN_LITRES)
    adi = methanol_from(ADI["EFSA and JECFA"] * 70)
    rows = (
        ("one can of diet drink", can_low, can_high, hs.PALETTE[1]),
        ("one glass of fruit juice", 0.25 * JUICE["low"], 0.25 * JUICE["high"], hs.PALETTE[0]),
        ("aspartame at the daily limit, 70 kg", adi, adi, hs.PALETTE[1]),
        ("made by the body in a day", *ENDOGENOUS, hs.PALETTE[0]),
        ("one kilogram of apples", *APPLES_1KG, hs.PALETTE[0]),
        ("minimum lethal dose, 70 kg", 70 * LETHAL_PER_KG[0], 70 * LETHAL_PER_KG[1], hs.PALETTE[7]),
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for y, (label, low, high, colour) in enumerate(reversed(rows)):
        if low == high:
            ax.plot([low], [y], "o", color=colour, markersize=9, markeredgecolor=hs.SURFACE, markeredgewidth=2)
            text = f"{low:,.0f} mg"
        else:
            ax.plot([low, high], [y, y], color=colour, linewidth=9, solid_capstyle="round")
            text = f"{low:,.0f} to {high:,.0f} mg"
        ax.text(high * 1.35, y, text, va="center", fontsize=9, color=hs.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(1, 2_000_000)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9)
    ax.set_xticks([1, 10, 100, 1_000, 10_000, 100_000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.set_xlabel("methanol, mg (logarithmic scale)")
    ax.set_title("Where methanol comes from, and how much")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=hs.PALETTE[1], linewidth=6, label="from aspartame"),
                       Line2D([], [], color=hs.PALETTE[0], linewidth=6, label="from food and the body"),
                       Line2D([], [], color=hs.PALETTE[7], linewidth=6, label="toxic dose")],
              loc="lower right", fontsize=8.5, bbox_to_anchor=(1.0, 0.12))
    ax.grid(axis="y", visible=False)
    alt = ("Range chart of methanol in milligrams on a logarithmic axis. One can of diet drink yields "
           f"{can_low:.0f} to {can_high:.0f} mg, one glass of fruit juice {0.25 * JUICE['low']:.0f} to {0.25 * JUICE['high']:.0f} mg, "
           f"aspartame at the acceptable daily intake for a 70 kg adult {adi:.0f} mg, the body's own daily production "
           f"{ENDOGENOUS[0]} to {ENDOGENOUS[1]} mg, a kilogram of apples {APPLES_1KG[0]} to {APPLES_1KG[1]:,} mg, and the minimum "
           f"lethal dose for a 70 kg adult {70 * LETHAL_PER_KG[0]:,} to {70 * LETHAL_PER_KG[1]:,} mg.")
    return fig, "aspartame_methanol_sources", alt


def kinetics_figure(plt, hs):
    doses = [d / 2 for d in range(2, 501)]              # 1 to 250 mg/kg
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(doses, [predicted_peak(d) for d in doses], color=hs.PALETTE[0], label="predicted: a tenth of the dose in 0.77 L per kg")
    observed = list(STEGINK)
    ax.errorbar(observed, [STEGINK[d][0] for d in observed], yerr=[STEGINK[d][1] for d in observed], fmt="o", color=hs.PALETTE[1],
                markersize=7, markeredgecolor=hs.SURFACE, markeredgewidth=2, capsize=3, label="measured in volunteers, mean and SD")
    ax.axhline(DETECTION_LIMIT, color=hs.INK_MUTED, linewidth=1.2, linestyle=(0, (3, 3)))
    ax.text(1.1, DETECTION_LIMIT * 1.12, "detection limit in the studies, 4 mg/L", fontsize=8.5, color=hs.INK_SECONDARY)
    ax.axhline(BLOOD_THRESHOLDS["effects on the nervous system"], color=hs.PALETTE[7], linewidth=1.2, linestyle=(0, (3, 3)))
    ax.text(1.1, 200 * 1.12, "effects on the nervous system begin, 200 mg/L", fontsize=8.5, color=hs.INK_SECONDARY)
    for dose, text in ((ADI["EFSA and JECFA"], "the acceptable\ndaily intake"), (DRINK["maximum permitted"] * CAN_LITRES / 70, "one can,\n70 kg adult")):
        ax.plot([dose], [predicted_peak(dose)], "o", color=hs.PALETTE[0], markersize=7, markeredgecolor=hs.SURFACE, markeredgewidth=2)
        ax.annotate(text, xy=(dose, predicted_peak(dose)), xytext=(8, -24), textcoords="offset points", fontsize=8.5, color=hs.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 300)
    ax.set_ylim(0.1, 600)
    ax.set_xticks([1, 3, 10, 40, 100, 200])
    ax.set_xticklabels(["1", "3", "10", "40", "100", "200"])
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("single dose of aspartame, mg per kg of body weight")
    ax.set_ylabel("peak blood methanol, mg/L")
    ax.set_title("The dose decides: predicted and measured blood methanol")
    ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.86))
    pairs = ", ".join(f"{predicted_peak(d):.1f} against {STEGINK[d][0]} at {d} mg/kg" for d in observed)
    alt = ("Line chart on logarithmic axes of peak blood methanol against a single dose of aspartame. A one-compartment prediction "
           f"agrees with the concentrations measured in volunteers: {pairs}. One can for a 70 kg adult predicts "
           f"{predicted_peak(DRINK['maximum permitted'] * CAN_LITRES / 70):.2f} mg/L and the acceptable daily intake taken at once "
           f"{predicted_peak(40):.1f} mg/L, against a detection limit of 4 mg/L and effects on the nervous system above 200 mg/L.")
    return fig, "aspartame_blood_methanol", alt


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print the numbers and write nothing")
    args = parser.parse_args()
    print(json.dumps(summary(), indent=1))
    if args.dry_run:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import housestyle as hs
    hs.use()
    for build in (sources_figure, kinetics_figure):
        fig, slug, alt = build(plt, hs)
        out = hs.save(fig, slug, alt=alt)
        print(f"  wrote {slug:28} {out['width']}x{out['height']}")
        print(f"  alt: {alt}")


if __name__ == "__main__":
    main()

"""Check the arithmetic and the quoted numbers of the essay on aspartame, fruit and dose."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aspartame", ROOT / "assets/viz/generate_aspartame_dose_figures.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)
POST = (ROOT / "_posts/healthcare/2026-07-12-aspartame_fruit_true_premise_bad_argument.md").read_text(encoding="utf-8")
NUMBERS = model.summary()


def test_methanol_share_is_close_to_the_stoichiometric_value():
    # CH3OH 32.04 over aspartame 294.30: the published "about 10%" rounds the exact figure down
    assert 32.04 / 294.30 == pytest.approx(model.METHANOL_SHARE, abs=0.01)


def test_the_kinetic_prediction_matches_the_measured_peaks():
    for dose, (observed, sd) in model.STEGINK.items():
        predicted = model.predicted_peak(dose)
        assert abs(predicted - observed) < sd                       # inside one standard deviation of the volunteers
        assert abs(predicted / observed - 1) < 0.10                 # "to within 10%"
    # the dose at which nothing was detected is predicted to sit at the detection limit
    assert model.predicted_peak(34) == pytest.approx(model.DETECTION_LIMIT, rel=0.15)


def test_one_can_is_far_below_every_threshold():
    can = model.DRINK["maximum permitted"] * model.CAN_LITRES
    assert can == pytest.approx(198)
    rise = model.predicted_peak(can / 70)
    assert rise < 0.5                                                # below normal background
    assert model.BLOOD_THRESHOLDS["effects on the nervous system"] / rise > 500


def test_cans_to_reach_the_acceptable_intake():
    cans = NUMBERS["cans to reach the ADI"]
    assert cans["70 kg, maximum permitted"] == pytest.approx(40 * 70 / 198, abs=0.05)
    assert cans["20 kg, maximum permitted"] == pytest.approx(40 * 20 / 198, abs=0.05)
    assert cans["70 kg, measured mean, 2020"] > 50


def test_numbers_quoted_in_the_post_match_the_model():
    can = NUMBERS["methanol from one can, mg"]
    juice = NUMBERS["methanol from a 250 mL glass of juice, mg"]
    peaks = NUMBERS["predicted and observed peak blood methanol, mg/L"]
    cans = NUMBERS["cans to reach the ADI"]
    high = NUMBERS["highest 95th-percentile exposure, % of the ADI"]
    quoted = [
        f"at most {can['maximum permitted']} mg of methanol",
        f"yields {juice['low']:.0f} to {juice['high']:.0f} mg and {juice['mean']:.0f} mg on average",
        f"predicts {peaks['100'][0]}, {peaks['150'][0]} and {peaks['200'][0]}",
        f"the prediction is {NUMBERS['predicted peak at 34 mg/kg, mg/L']} mg/L",
        f"a rise of {NUMBERS['predicted rise from one can at 70 kg, mg/L']} mg/L",
        f"is {NUMBERS['nervous-system threshold over the rise from one can']} times higher",
        f"predicts {NUMBERS['predicted peak at the ADI, mg/L']} mg/L",
        f"with {cans['70 kg, maximum permitted']} cans a day and a 20 kg child with {cans['20 kg, maximum permitted']}",
        f"are {high['toddlers']}%, {high['children']}% and {high['adults']}% of the acceptable intake",
        f"which is {NUMBERS['NutriNet higher consumers, % of the ADI at 70 kg']}% of the acceptable intake",
        f"yields {NUMBERS['phenylalanine from one can at the maximum level, mg']} mg of it",
        f"{NUMBERS['methanol from aspartame at the ADI, 70 kg, mg']} mg |",
    ]
    for text in quoted:
        assert text in POST, text
    assert 52 <= cans["70 kg, measured mean, 2020"] < 53 and 95 <= cans["70 kg, measured mean, 2008"] < 96      # "52 to 95 cans"
    assert 15 <= cans["20 kg, measured mean, 2020"] < 16 and 27 <= cans["20 kg, measured mean, 2008"] < 28      # "15 to 27"


def test_every_cited_source_has_a_reference():
    body, references = POST.split("## References")
    for author in ("Basílio", "Debras", "EFSA", "Lindinger", "Lino", "Stegink", "WHO, 1997"):
        assert author in body, author
    for author in ("Basílio", "Debras", "EFSA Panel", "Lindinger", "Lino", "Stegink", "World Health Organization (1997)"):
        assert author in references, author

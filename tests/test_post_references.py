"""Every author named in an article has to appear in its reference list.

These checks came from the model tests that used to live beside them. The models
themselves moved to the blog-reproducibility repository; checking an article's
own references is a check on the article, not on the science, so it stays here.

Two lists per article, because they are not always the same string: an article
may cite "WHO, 1997" in the body and list "World Health Organization (1997)"
under the references. Adding a citation means adding it to both.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# article -> (names expected in the body, names expected in the references)
CITED_AUTHORS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "_posts/healthcare/2026-01-24-parasite_cleanse_social_media_myth.md": (
        (
            "Branda",
            "Butler",
            "Cartwright",
            "Garcia",
            "Hoang",
            "Hosiian",
            "Janes",
            "Lachenmeier",
            "Lashaki",
            "Moser",
            "Munyangi",
            "Temple",
            "Volinsky",
            "Wendt",
        ),
        (
            "Branda",
            "Butler",
            "Cartwright",
            "Garcia",
            "Hoang",
            "Hosiian",
            "Janes",
            "Lachenmeier",
            "Lashaki",
            "Moser",
            "Munyangi",
            "Temple",
            "Volinsky",
            "Wendt",
        ),
    ),
    "_posts/healthcare/2026-02-18-hormone_balance_social_media_myth.md": (
        (
            "Akturk",
            "Anckaert",
            "Andersen",
            "Brambilla",
            "Brito",
            "Cadegiani",
            "Casals",
            "Danese",
            "Kang",
            "McNulty",
            "Musazadeh",
            "Nagarajan",
            "Naugler",
            "Nickel",
            "Santoro",
            "Srinivasa Gopalan",
            "Stanczyk",
        ),
        (
            "Akturk",
            "Anckaert",
            "Andersen",
            "Brambilla",
            "Brito",
            "Cadegiani",
            "Casals",
            "Danese",
            "Kang",
            "McNulty",
            "Musazadeh",
            "Nagarajan",
            "Naugler",
            "Nickel",
            "Santoro",
            "Srinivasa Gopalan",
            "Stanczyk",
        ),
    ),
    "_posts/healthcare/2026-03-05-consumer_microbiome_testing_limits.md": (
        (
            "Bermingham",
            "Berry",
            "Falony",
            "Magne",
            "Nishijima",
            "Olsson",
            "Porcari",
            "Servetas",
            "Sze",
            "Vandeputte",
            "Wei",
            "Zhernakova",
        ),
        (
            "Bermingham",
            "Berry",
            "Falony",
            "Magne",
            "Nishijima",
            "Olsson",
            "Porcari",
            "Servetas",
            "Sze",
            "Vandeputte",
            "Wei",
            "Zhernakova",
        ),
    ),
    "_posts/healthcare/2026-05-11-results_are_not_evidence_influencer_science.md": (
        (
            "Barnett",
            "Bhasin",
            "Denniss",
            "Finley",
            "Guyatt",
            "Helou",
            "Hernán",
            "Hubal",
            "Krogsbøll",
            "Mathur",
            "Powell",
        ),
        (
            "Barnett",
            "Bhasin",
            "Denniss",
            "Finley",
            "Guyatt",
            "Helou",
            "Hernán",
            "Hubal",
            "Krogsbøll",
            "Mathur",
            "Powell",
        ),
    ),
    "_posts/healthcare/2026-06-03-leaky_gut_social_media_myths.md": (
        (
            "Abbasi",
            "Ajamian",
            "Camilleri",
            "Chantler",
            "Cummins",
            "Hoilat",
            "Nascimento",
            "Power",
            "Rath",
            "Scheffler",
            "Turpin",
            "Zheng",
            "Zhou",
        ),
        (
            "Abbasi",
            "Ajamian",
            "Camilleri",
            "Chantler",
            "Cummins",
            "Hoilat",
            "Nascimento",
            "Power",
            "Rath",
            "Scheffler",
            "Turpin",
            "Zheng",
            "Zhou",
        ),
    ),
    "_posts/healthcare/2026-07-12-aspartame_fruit_true_premise_bad_argument.md": (
        (
            "Basílio",
            "Debras",
            "EFSA",
            "Lindinger",
            "Lino",
            "Stegink",
            "WHO, 1997",
        ),
        (
            "Basílio",
            "Debras",
            "EFSA Panel",
            "Lindinger",
            "Lino",
            "Stegink",
            "World Health Organization (1997)",
        ),
    ),
    "_posts/healthcare/2026-09-02-inflammation_is_not_a_diagnosis_social_media_myths.md": (
        (
            "Bleakley",
            "Bower",
            "CRP CHD Genetics Collaboration",
            "Calder",
            "Costenbader",
            "Cushman",
            "DeGoma",
            "Dehzad",
            "Emerging Risk Factors Collaboration",
            "Furman",
            "Gleeson",
            "Hotamisligil",
            "IL6R MR Consortium",
            "Macy",
            "Medzhitov",
            "Nidorf",
            "Pedersen",
            "Pepys",
            "Ridker",
            "Roberts",
            "Sahebkar",
            "Schwingshackl",
            "Selvin",
            "Tardif",
            "Visser",
            "Wannamethee",
        ),
        (
            "Bleakley",
            "Bower",
            "CRP CHD Genetics Collaboration",
            "Calder",
            "Costenbader",
            "Cushman",
            "DeGoma",
            "Dehzad",
            "Emerging Risk Factors Collaboration",
            "Furman",
            "Gleeson",
            "Hotamisligil",
            "IL6R MR Consortium",
            "Macy",
            "Medzhitov",
            "Nidorf",
            "Pedersen",
            "Pepys",
            "Ridker",
            "Roberts",
            "Sahebkar",
            "Schwingshackl",
            "Selvin",
            "Tardif",
            "Visser",
            "Wannamethee",
        ),
    ),
}


@pytest.mark.parametrize("article", sorted(CITED_AUTHORS))
def test_every_cited_author_has_a_reference(article: str) -> None:
    """Each cited name appears in the body, and its entry under the references."""
    path = ROOT / article
    assert path.is_file(), f"missing article: {article}"

    text = path.read_text(encoding="utf-8")
    assert "## References" in text, article
    body, references = text.split("## References", 1)

    in_body, in_references = CITED_AUTHORS[article]
    for author in in_body:
        assert author in body, f"{article}: {author} not cited in the body"
    for author in in_references:
        assert author in references, f"{article}: {author} missing from the references"


def test_every_listed_article_still_exists() -> None:
    """A renamed or removed article should fail here rather than silently pass."""
    for article in CITED_AUTHORS:
        assert (ROOT / article).is_file(), article

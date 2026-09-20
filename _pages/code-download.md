---
layout: page
title: "Code Download"
permalink: /code/
author_profile: true
---

The code behind the articles lives in two places, and which one you want depends on what you are after.

## Calculations, models and figures

Anything an article computes — the numerical models, the simulations, the benchmarks, and the scripts that draw the figures — is in [blog-reproducibility](https://github.com/DiogoRibeiro7/blog-reproducibility). That repository exists so the evidence behind an article can be run, read and checked on its own, without the website around it.

Every article with computations behind it links to its own script, and the scripts are grouped by subject:

```text
scripts/figures/statistics/     confidence sets, p-values, polling, streaks
scripts/figures/health/         dose, testing, screening, risk
scripts/figures/physics/        quantum measurement, seasons, photon energy
scripts/figures/engineering/    database and data-lake benchmarks
scripts/figures/time_series/    change points, release vintages
```

Most of them take a `--dry-run` flag that prints the calculations without writing any images, which is usually the quickest way to see what an article's numbers are made of.

```bash
git clone https://github.com/DiogoRibeiro7/blog-reproducibility.git
cd blog-reproducibility
poetry install
poetry run python scripts/figures/statistics/pvalue_evidence.py --dry-run
```

`articles/manifest.yml` there maps each article to its model, its figures and its tests, so you can go from a published claim to the code that produced it without guessing.

## The website itself

This repository, [DiogoRibeiro7.github.io](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io), holds the site: the article Markdown, the layouts and includes, the styles, the navigation, the rendered images, and the scripts that build and check the site. Clone it if you want to run the site locally or see how a page is put together.

```bash
git clone https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io.git
```

---

Feel free to reach out at [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt) for professional enquiries or [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com) for personal matters. My ORCID is [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072) and I teach at the **Faculty of Media Arts and Design, Technical University of Porto**, in Vila do Conde, Portugal, alongside my work as a data scientist and research lead.

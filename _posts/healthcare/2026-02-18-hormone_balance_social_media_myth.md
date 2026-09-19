---
permalink: '/healthcare/hormone_balance_social_media_myth/'
title: 'Hormones Are Not a Single Balance'
date: '2026-02-18'
categories:
- Healthcare
tags:
- Hormones
- Endocrinology
- Menstrual Cycle
- Menopause
- Supplements
- Science Communication
author_profile: false
classes: wide
seo_title: 'Hormone Balance, Cycle Syncing and Testing: What the Evidence Supports'
seo_description: 'A healthy person given a twelve-hormone panel has a 46% chance of at least one flagged result, and two TSH results from the same person must differ by about 49% before the difference means anything. Hormone balancing, saliva testing, compounded hormones, cycle syncing and seed cycling, measured against published numbers.'
seo_type: article
excerpt: >-
  "Hormone imbalance" is used as if the body had one dial that could be pushed
  back into range. Hormones belong to separate feedback systems whose normal
  behaviour is to move: progesterone rises 136-fold within one menstrual cycle,
  and two cortisol results from the same healthy person can differ by half
  without anything having changed. The arithmetic of testing follows from that.
summary: >-
  This essay examines the online language of hormone balance using published
  numbers: how much hormones vary within a healthy person, what that does to
  multi-hormone panels and to repeat testing, what is known about salivary
  testing and compounded hormones, what "hormone support" supplements have been
  found to contain, the trial evidence on cycle syncing and seed cycling, and who
  promotes the tests.
keywords:
- hormone balance
- hormone imbalance
- saliva hormone testing
- cycle syncing
- seed cycling
- bioidentical hormones
- menopause
- endocrinology
why_this_exists: >-
  Hormonal disorders are real, but the phrase "hormone imbalance" is often used
  online without identifying which hormone, which axis, what timing or what
  diagnostic criterion. The result is a vague diagnosis that can absorb almost
  any symptom, and a market in tests whose error rates are never mentioned.
evidence: >-
  Within-person biological variation from the EFLM database and from studies of
  thyroid, testosterone and salivary cortisol; laboratory data on abnormal-result
  rates; the National Academies report and professional guidance on compounded
  hormones; assays of compounded products and of thyroid and adrenal supplements;
  prescribing data for levothyroxine; meta-analyses of menstrual-cycle phase and
  exercise and of flaxseed; and a content analysis of 982 social media posts.
methodology: >-
  Treat hormones as distinct regulated systems rather than one global variable.
  The chance of a flagged result in a panel and the reference change value are
  computed from published variation figures; the correlated-panel curve is a
  simulation and says so. Study numbers are quoted from abstracts or, where those
  lack them, from the full text. The script and a test of every quoted number
  are in the repository.
reviewed_at: '2026-09-19'
header:
  image: /assets/images/headers/photo-microscope.jpg
  og_image: /assets/images/headers/photo-microscope.jpg
  overlay_image: /assets/images/headers/photo-microscope.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-microscope.jpg
  twitter_image: /assets/images/headers/photo-microscope.jpg
---

The phrase "hormone imbalance" sounds precise and is often almost empty. It appears in posts about fatigue, acne, difficulty losing weight, poor sleep, low libido, premenstrual symptoms, anxiety, irregular cycles and menopause, the diagnosis is usually inferred from a checklist, and the proposed remedy is as broad as the complaint: balance your hormones. Hormonal disorders are real and measurable. What the phrase hides is that there is no single balance to restore. Thyroid hormones, insulin, cortisol, prolactin, testosterone, estradiol, progesterone and the pituitary hormones that drive them belong to separate feedback loops, with different rhythms and different clinical meanings, and they do not move together around one ideal centre.

A useful endocrine question therefore has several parts: which hormone, measured when, in what physiological state, with which assay, to answer what. This essay follows those parts through the products sold under the word balance, which are panels, saliva tests, compounded hormones, supplements and diets timed to the menstrual cycle. For each there are published numbers, and most of the argument is arithmetic on how much a healthy endocrine system moves.

## A System Built to Move

Hormones are signals, and a signal that did not change would carry no information. The menstrual cycle is the clearest case. In 85 healthy women followed through one natural cycle, median serum progesterone was 0.212 nmol/L in the follicular phase and 28.8 nmol/L in the luteal phase, a 136-fold rise, while estradiol went from 198 pmol/L to 757 pmol/L at ovulation and back to 412 pmol/L (Anckaert et al., 2021). None of that is instability. It is the system doing what it is for, and a progesterone value has no meaning until the day of the cycle is known.

| Hormone | Follicular phase | Ovulation | Luteal phase |
| :--- | ---: | ---: | ---: |
| Estradiol, pmol/L | 198 (114 to 332) | 757 (222 to 1,959) | 412 (222 to 854) |
| Progesterone, nmol/L | 0.212 (0.159 to 0.616) | 1.81 (0.175 to 13.2) | 28.8 (13.1 to 46.3) |
| LH, IU/L | 7.14 (4.78 to 13.2) | 22.6 (8.11 to 72.7) | 6.24 (2.73 to 13.1) |

The table gives medians with the 5th and 95th percentiles, and the ranges matter as much as the medians: at ovulation the healthy range of estradiol spans a factor of nine. Men are not exempt from timing. In a community sample in Boston, testosterone in men aged 30 to 40 was 20% to 25% lower at four in the afternoon than at eight in the morning, and of 17 men who had at least one afternoon value below the usual threshold of 300 ng/dL, nine had normal values at all three of their morning visits (Brambilla et al., 2009). A single afternoon blood test would have labelled them deficient.

Symptoms are an even weaker guide than an untimed test. Tiredness, poor concentration, irritability, acne, low mood, cravings and weight gain are real, and they accompany anaemia, sleep loss, depression, medication and ordinary life far more often than they accompany an endocrine disorder. A symptom is diagnostic to the extent that it is more common with the disease than without it, and a list written so that most adults recognise themselves in it fails that test by construction. It can prompt a question, and it cannot say which of a dozen axes the question is about.

## A Panel of Twelve

The commercial answer to a vague symptom list is a large panel: a dozen or more hormones measured at once, with every value near the edge of its range read as a finding. A reference interval is defined to contain 95% of results from healthy people, so each test flags one healthy person in twenty. If $n$ analytes were independent, the chance that a healthy person has at least one result out of range would be

$$
P(\text{at least one flag}) = 1 - 0.95^{\,n},
$$

which is 22.6% for five analytes, 46.0% for twelve and 64.2% for twenty.

![Line chart of the chance that a healthy person has at least one result outside a 95% reference interval, against the number of analytes measured. With independent analytes it rises from 5% for one test to 46% for twelve, 64% for twenty and 87% for forty. With every pair of analytes correlated at 0.5 the curve is lower but still reaches about 32% at twelve.](/assets/images/figures/hormone_panel_false_flags.png){: width="1152" height="704" loading="lazy"}

Hormones are not independent, and correlation lowers the curve, but not by enough to rescue the panel. In a simulation with every pair of analytes correlated at 0.5, which is strong, a twelve-analyte panel still flags 31.7% of healthy people. The effect is visible in real laboratories. Across 3.9 million tests, the share of abnormal results fell from about 25% when a doctor ordered one test to about 7% when nine or more were ordered together, which is what happens when tests are ordered with less reason (Naugler & Guo, 2016). Among 1,340 family physicians the abnormal rate over 39 common analytes was 8.6%, against the 5% expected from healthy people alone, and the authors conclude that about 58% of abnormal results are likely to be false positives (Naugler & Ma, 2018). Those are tests ordered by doctors for patients with some reason to be tested. A panel bought online by someone with common symptoms starts from a lower prior and does worse.

## How Much Must a Result Change?

Balancing protocols usually involve testing again after some weeks on the product, and a changed number is offered as proof that it worked. Whether two results differ depends on how much one person's values vary when nothing is done. Laboratory medicine has a standard answer, the reference change value: for a within-person coefficient of variation $CV_I$ and an analytical one $CV_A$, two results must differ by more than

$$
RCV = 1.96\,\sqrt{2}\,\sqrt{CV_A^2 + CV_I^2}
$$

before the difference exceeds what variation alone produces 95% of the time. The within-person figures are catalogued for healthy adults in the EFLM Biological Variation Database, and setting $CV_A$ to zero gives the floor that no assay can beat.

![Horizontal bar chart of the reference change value, the smallest difference between two results from one person that biological variation alone would not produce 95% of the time. It is 13% for free T4, 23% for SHBG, 26% for FSH, 40% for testosterone, 42% for estradiol, 45% for cortisol, 49% for TSH, 51% for progesterone, 55% for DHEAS, 70% for LH, 70% for insulin, 125% for prolactin, 152% for free testosterone. Measured values for salivary cortisol, which include assay variation, run from 96% to 245% depending on the time of day.](/assets/images/figures/hormone_reference_change_values.png){: width="1152" height="800" loading="lazy"}

With a perfect assay, two TSH results from the same healthy person must differ by 49%, two serum cortisol results by 45%, two testosterone results by 40% and two prolactin results by 125% before the second says anything the first did not. Direct studies agree. With monthly sampling of 16 healthy men for a year, one TSH result located a man's own set point only to within 50% (Andersen et al., 2002), and in 132 men sampled repeatedly, two testosterone measurements differed by more than 18% to 28% about half the time and by more than 27% to 54% a quarter of the time (Brambilla et al., 2007). A cortisol that "improved" by a third between two tests has done nothing that needs explaining.

The same thyroid study contains a subtler point. Each man's own range was about half as wide as the population's reference interval, so a person can move well outside their normal and still be reported as in range, and a result at the edge of the population range can be entirely normal for that person. Population intervals are a poor instrument for judging an individual, in both directions, and the remedy is a clinical question and properly timed repeat tests, not a wider panel.

## Saliva Tests and Made-to-Measure Hormones

The strongest commercial form of personalisation is the saliva panel used to tailor a compounded hormone prescription. Saliva is the noisiest of the available samples. For salivary cortisol in 20 healthy volunteers sampled six times a day for five days, the within-person variation ran from 29.3% to 56.5% depending on the hour, and the reference change value from 96% to 245% (Danese et al., 2024). An earlier study of late-night salivary cortisol reported an analytical variation of 15.4%, a within-person variation of 34.1% and a reference change value of 104%, and the formula above returns 103.7% from those components (Casals et al., 2011). A result has to double to count as a change. The diagnosis most often built on such panels, adrenal fatigue, was the subject of a systematic review of 58 studies whose tests gave conflicting results and whose title states the conclusion: adrenal fatigue does not exist (Cadegiani & Kater, 2016).

The professional bodies take the same view of testing to guide dosing. The American College of Obstetricians and Gynecologists does not recommend hormone testing to prescribe compounded menopausal therapy and notes that salivary measurements of estrogen and progesterone are not accurate or precise enough for the purpose (ACOG, 2023). The Endocrine Society's scientific statement finds no evidence that monitoring compounded therapy with serial salivary or blood tests is effective, with the exception of thyroid hormone (Santoro et al., 2016). A test personalises treatment only if the assay is valid, the target is meaningful, the result changes management and the change improves outcomes, and for saliva panels the chain breaks at the first link.

"Bioidentical" describes a molecule, not a product. Estradiol and micronised progesterone identical to the body's own are available as approved medicines, and the question about a custom-compounded version is what is in it. When identical prescriptions for 0.5 mg of estradiol with 100 mg of progesterone were sent to compounding pharmacies, the capsules contained from 0.365 to 0.551 mg of estradiol, 27% below the label to 10% above, and from 90.8 to 135 mg of progesterone, 9% below to 35% above (Stanczyk et al., 2019). The National Academies estimated that 26 to 33 million such prescriptions are dispensed each year in the United States at a cost above $2 billion, found insufficient evidence to support their overall clinical utility, and recommended restricting them to patients with a documented allergy to an ingredient of an approved product or a need for a different dosage form (National Academies, 2020).

## What Is in the Bottle

Supplements sold to "support" the thyroid or the adrenal glands raise a different problem, which is that some of them work for an undeclared reason. Of ten commercially available thyroid-support supplements, nine contained triiodothyronine, at 1.3 to 25.4 µg per tablet, and five contained thyroxine; at the recommended dose, five products delivered more than 10 µg of T3 a day and four delivered between 8.57 and 91.6 µg of T4 a day, amounts that overlap with prescribed doses (Kang et al., 2013). All twelve adrenal-support supplements in a later study contained T3, five contained pregnenolone, three contained the synthetic glucocorticoid budesonide and one contained cortisol (Akturk et al., 2018). A person who feels more energetic on such a product may be correct about the effect and entirely misled about its cause, and is taking a hormone at an unknown dose with nobody monitoring it.

Prescription medicine is not immune to the same drift towards treating numbers. Among 110,842 adults in the United States who started levothyroxine between 2008 and 2018, the median TSH at the start was 5.3 mIU/L, a little above the usual reference range, and of the 58,706 with thyroxine results, 8.4% had overt hypothyroidism, 61.0% had the subclinical form and 30.5% had normal thyroid function (Brito et al., 2021). Given that one TSH result locates a person's set point only to within 50%, treatment decisions made near the boundary on a single value are decisions made largely on noise.

## Cycle Syncing and Seed Cycling

Adjusting training or diet to the phase of the menstrual cycle is biologically plausible, since the hormones in the table above affect temperature regulation, fuel use and symptoms. The measured effect on performance is small. A meta-analysis of 78 studies, 51 of which could be pooled, found performance trivially lower in the early follicular phase, with a standardised effect of -0.06 and a 95% credible interval from -0.16 to 0.04, and rated the quality of the evidence as low; the largest contrast between any two phases was -0.14 (McNulty et al., 2020). The authors advise an individual approach and not prescriptions by phase. A later review of 25 studies with 16,557 adults and 3,715 adolescents found that what affects participation in exercise is symptoms, experience and social pressure, which vary between people in the same phase (Srinivasa Gopalan et al., 2024). Tracking one's own cycle and responding to how one feels is reasonable, and a universal four-phase training calendar is not supported.

Seed cycling, eating flax and pumpkin seeds in one half of the cycle and sesame and sunflower seeds in the other, has less behind it. A 2025 systematic review reports favourable associations from ten studies with 635 participants, but it counted studies of single seeds as well as of the protocol, and the studies were small and varied in method (Nagarajan et al., 2025). The better evidence is on the main ingredient. A meta-analysis of ten randomised trials of flaxseed found no significant effect on FSH, SHBG, total testosterone, free androgen index or DHEAS, with standardised differences between -0.11 and 0.35 and confidence intervals that all included zero (Musazadeh et al., 2023). Seeds are nutritious food. The claim that eating them on particular days steers estrogen and progesterone has not been shown.

## Who Is Selling the Test

The information that reaches people about these tests is mostly advertising. An analysis of 982 Instagram and TikTok posts about five popular tests, among them testosterone and anti-Müllerian hormone, from accounts with 194 million followers in total, found that 87.1% mentioned benefits, 14.7% mentioned any harm and 6.1% mentioned overdiagnosis or overuse. Only 6.4% referred to evidence, 83.8% were promotional in tone, and 68.0% of the account holders had a financial interest in the test (Nickel et al., 2025). Posts by physicians had 4.49 times the odds of mentioning harms.

The word balance does a great deal of work in that setting. It implies a central ideal towards which hormones should be pushed, when healthy systems vary by design. It borrows the reassurance of "natural", when natural compounds include thyroid hormone and glucocorticoids. And it treats a changed number as a benefit, when most changes between two tests are within the range of doing nothing. Nobody wants an unbalanced body, which is why the phrase sells.

## Which Hormone, Measured When

None of this is an argument against endocrinology. Hypothyroidism, hyperthyroidism, Cushing syndrome, adrenal insufficiency, hyperprolactinaemia and polycystic ovary syndrome are real and treatable, and each is diagnosed by finding a specific abnormality in a specific axis, with tests timed and repeated as the physiology requires. A person with thyroid disease does not have a hormone imbalance. They have a disorder of one axis with a recognisable pattern of results and symptoms, and the precision is what makes treatment possible.

A claim that a product "balances estrogen" can be held to the same standard. Which measure of estrogen, in which population, at what point in the cycle or in life, compared with what? By how much does it change, and is that more than the 42% by which estradiol differs between two tests in an untreated person? Does anything the person cares about improve? Some claims survive those questions and become useful, and many disappear once they have to be stated. The better question is always the more specific one: which hormone, which axis, measured when, and linked to which outcome.

## References

- Akturk, H. K., Chindris, A. M., Hines, J. M., Singh, R. J., & Bernet, V. J. (2018). Over-the-counter "adrenal support" supplements contain thyroid and steroid-based adrenal hormones. *Mayo Clinic Proceedings*, 93(3), 284-290. <https://doi.org/10.1016/j.mayocp.2017.10.019>
- American College of Obstetricians and Gynecologists (2023). Compounded bioidentical menopausal hormone therapy. Clinical Consensus No. 6. *Obstetrics & Gynecology*, 142(5), 1266-1273. <https://doi.org/10.1097/AOG.0000000000005395>
- Anckaert, E., Jank, A., Petzold, J., Rohsmann, F., Paris, R., Renggli, M., Schönfeld, K., Schiettecatte, J., & Kriner, M. (2021). Extensive monitoring of the natural menstrual cycle using the serum biomarkers estradiol, luteinizing hormone and progesterone. *Practical Laboratory Medicine*, 25, e00211. <https://doi.org/10.1016/j.plabm.2021.e00211>
- Andersen, S., Pedersen, K. M., Bruun, N. H., & Laurberg, P. (2002). Narrow individual variations in serum T4 and T3 in normal subjects: a clue to the understanding of subclinical thyroid disease. *Journal of Clinical Endocrinology & Metabolism*, 87(3), 1068-1072. <https://doi.org/10.1210/jcem.87.3.8165>
- Brambilla, D. J., Matsumoto, A. M., Araujo, A. B., & McKinlay, J. B. (2009). The effect of diurnal variation on clinical measurement of serum testosterone and other sex hormone levels in men. *Journal of Clinical Endocrinology & Metabolism*, 94(3), 907-913. <https://doi.org/10.1210/jc.2008-1902>
- Brambilla, D. J., O'Donnell, A. B., Matsumoto, A. M., & McKinlay, J. B. (2007). Intraindividual variation in levels of serum testosterone and other reproductive and adrenal hormones in men. *Clinical Endocrinology*, 67(6), 853-862. <https://doi.org/10.1111/j.1365-2265.2007.02976.x>
- Brito, J. P., Ross, J. S., El Kawkgi, O. M., Maraka, S., Deng, Y., Shah, N. D., & Lipska, K. J. (2021). Levothyroxine use in the United States, 2008-2018. *JAMA Internal Medicine*, 181(10), 1402-1405. <https://doi.org/10.1001/jamainternmed.2021.2686>
- Cadegiani, F. A., & Kater, C. E. (2016). Adrenal fatigue does not exist: a systematic review. *BMC Endocrine Disorders*, 16(1), 48. <https://doi.org/10.1186/s12902-016-0128-4>
- Casals, G., Foj, L., & de Osaba, M. J. (2011). Day-to-day variation of late-night salivary cortisol in healthy voluntaries. *Clinical Biochemistry*, 44(8-9), 665-668. <https://doi.org/10.1016/j.clinbiochem.2011.02.003>
- Danese, E., Padoan, A., Negrini, D., Paviati, E., De Pastena, M., Esposito, A., Lippi, G., & Montagnana, M. (2024). Diurnal and day-to-day biological variation of salivary cortisol and cortisone. *Clinical Chemistry and Laboratory Medicine*, 62(11), 2287-2293. <https://doi.org/10.1515/cclm-2024-0196>
- European Federation of Clinical Chemistry and Laboratory Medicine. EFLM Biological Variation Database. <https://biologicalvariation.eu>
- Kang, G. Y., Parks, J. R., Fileta, B., Chang, A., Abdel-Rahim, M. M., Burch, H. B., & Bernet, V. J. (2013). Thyroxine and triiodothyronine content in commercially available thyroid health supplements. *Thyroid*, 23(10), 1233-1237. <https://doi.org/10.1089/thy.2013.0101>
- McNulty, K. L., Elliott-Sale, K. J., Dolan, E., Swinton, P. A., Ansdell, P., Goodall, S., Thomas, K., & Hicks, K. M. (2020). The effects of menstrual cycle phase on exercise performance in eumenorrheic women: a systematic review and meta-analysis. *Sports Medicine*, 50(10), 1813-1827. <https://doi.org/10.1007/s40279-020-01319-3>
- Musazadeh, V., Nazari, A., Natami, M., Hajhashemy, Z., Kazemi, K. S., Torabi, F., Moridpour, A. H., Vajdi, M., & Askari, G. (2023). The effect of flaxseed supplementation on sex hormone profile in adults: a systematic review and meta-analysis. *Frontiers in Nutrition*, 10, 1222584. <https://doi.org/10.3389/fnut.2023.1222584>
- Nagarajan, D. R., Mani Jacob, D., Munir Mufti, M., Rajesh, S., & Dube, R. (2025). Efficacy of seed cycling as an integrative therapy for premenstrual syndrome and polycystic ovary syndrome in reproductive-aged women: a systematic review. *Cureus*, 17(8), e90997. <https://doi.org/10.7759/cureus.90997>
- National Academies of Sciences, Engineering, and Medicine (2020). *The Clinical Utility of Compounded Bioidentical Hormone Therapy: A Review of Safety, Effectiveness, and Use*. National Academies Press. <https://doi.org/10.17226/25791>
- Naugler, C. T., & Guo, M. (2016). Mean abnormal result rate: proof of concept of a new metric for benchmarking selectivity in laboratory test ordering. *American Journal of Clinical Pathology*, 145(4), 568-573. <https://doi.org/10.1093/ajcp/aqw041>
- Naugler, C., & Ma, I. (2018). More than half of abnormal results from laboratory tests ordered by family physicians could be false-positive. *Canadian Family Physician*, 64(3), 202-203. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5851398/>
- Nickel, B., Moynihan, R., Gram, E. G., Copp, T., Taba, M., Shih, P., Heiss, R., Gao, M., & Zadro, J. R. (2025). Social media posts about medical tests with potential for overdiagnosis. *JAMA Network Open*, 8(2), e2461940. <https://doi.org/10.1001/jamanetworkopen.2024.61940>
- Santoro, N., Braunstein, G. D., Butts, C. L., Martin, K. A., McDermott, M., & Pinkerton, J. V. (2016). Compounded bioidentical hormones in endocrinology practice: an Endocrine Society scientific statement. *Journal of Clinical Endocrinology & Metabolism*, 101(4), 1318-1343. <https://doi.org/10.1210/jc.2016-1271>
- Srinivasa Gopalan, S., Mann, C., & Rhodes, R. E. (2024). Impact of symptoms, experiences, and perceptions of the menstrual cycle on recreational physical activity of cyclically menstruating individuals: a systematic review. *Preventive Medicine*, 184, 107980. <https://doi.org/10.1016/j.ypmed.2024.107980>
- Stanczyk, F. Z., Niu, C., Azen, C., Mirkin, S., & Amadio, J. M. (2019). Determination of estradiol and progesterone content in capsules and creams from compounding pharmacies. *Menopause*, 26(9), 966-971. <https://doi.org/10.1097/GME.0000000000001356>

*This article discusses endocrine physiology and population-level evidence. Persistent symptoms or suspected endocrine disease require clinical assessment rather than a generic hormone-balancing protocol.*

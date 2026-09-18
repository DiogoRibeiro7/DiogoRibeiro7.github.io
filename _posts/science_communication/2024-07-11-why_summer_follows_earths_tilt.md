---
permalink: '/science-communication/why_summer_follows_earths_tilt/'
title: 'Why Summer Follows Earth’s Tilt'
date: '2024-07-11'
categories:
- Science Communication
tags:
- Astronomy
- Physics
- Scientific Literacy
- Seasons
author_profile: false
classes: wide
seo_title: 'Why Earth’s Tilt Causes the Seasons'
seo_description: 'An original sunlight calculation explains opposite seasons in the two hemispheres and separates the effects of solar angle, daylight hours, and orbital distance.'
seo_type: article
excerpt: >-
  The two hemispheres share an orbit around the Sun but experience opposite
  seasons. A calculation at 45 degrees latitude shows what Earth’s tilt changes.
summary: >-
  An idealised Earth with a fixed distance from the Sun still has opposite
  seasonal changes in daylight and incoming solar energy. Geometry explains
  the pattern without pretending to predict the temperature of a real city.
keywords:
- cause of seasons
- Earth axial tilt
- sunlight angle
- day length
why_this_exists: >-
  Replacing the distance misconception with the word tilt is incomplete.
  This article works through the two consequences of tilt that matter for
  incoming energy: the Sun’s angle and the duration of daylight.
evidence: >-
  Original spherical-geometry calculations at 45 degrees north and south,
  an original figure, and NASA explanations of seasons and Earth’s orbit.
methodology: >-
  Hold orbital distance constant, calculate daylight and projected solar
  energy for three solar declinations, and test the result against direct
  numerical integration and symmetry between the hemispheres.
reviewed_at: '2026-09-18'
header:
  image: /assets/images/headers/photo-stars.jpg
  og_image: /assets/images/headers/photo-stars.jpg
  overlay_image: /assets/images/headers/photo-stars.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-stars.jpg
  twitter_image: /assets/images/headers/photo-stars.jpg
---

<!--
Development contract
Question: How does Earth’s tilt produce opposite seasons even at a fixed distance from the Sun?
Claim: Tilt changes both the projection of sunlight onto a surface and the duration of daylight.
Counterclaim: Orbital distance does change incoming energy and contributes to seasonal asymmetry.
Evidence object: A spherical-Earth calculation, hemisphere comparison, and original figure.
Failure case: Incoming solar energy alone does not predict surface temperature; the atmosphere and heat storage matter.
Reader payoff: Explain the mechanism of seasons and test the distance-only explanation against opposite hemispheres.
Exclusions: A city weather forecast, full orbital mechanics, and long-term climate attribution.
-->

The Northern and Southern Hemispheres experience opposite seasons while travelling around the Sun together. That observation is a useful test of the claim that summer happens simply because Earth is closer to the Sun.

Earth's distance does vary. But a change in the whole planet's orbital distance cannot, on its own, explain why one hemisphere moves into summer while the other moves into winter.

The central mechanism is Earth's tilted rotation axis. It changes how sunlight reaches different latitudes through the year. NASA's explanation of orbital cycles distinguishes this tilt from the effects of changing distance. [NASA on tilt and orbit](https://science.nasa.gov/science-research/earth-science/milankovitch-orbital-cycles-and-their-role-in-earths-climate/).

To see what tilt actually does, we can temporarily remove changing distance from the calculation.

## Give the model a circular orbit

Imagine a spherical Earth at a constant distance from the Sun. Give it an axial tilt of 23.44 degrees, a representative value close to Earth's present tilt.

Choose a location at 45 degrees north. Ignore mountains, clouds, atmospheric refraction, and the apparent width of the Sun. Treat sunrise and sunset as the moments when the centre of a point-like Sun crosses an unobstructed horizon.

Those assumptions simplify the arithmetic. They also tell us why the calculated daylight hours will differ slightly from a real sunrise table.

Now compare the December solstice, an equinox, and the June solstice:

| Idealised date | Sun's altitude at local noon | Daylight at 45° north |
| --- | ---: | ---: |
| December solstice | 21.56° | 8.57 hours |
| Equinox | 45.00° | 12.00 hours |
| June solstice | 68.44° | 15.43 hours |

The distance to the Sun has not changed in this model. Both the height of the noon Sun and the length of the day have.

## The same light spreads over different areas

Picture a beam of sunlight reaching a horizontal surface. When the Sun is overhead, the beam's cross-section covers the smallest area. At a shallow angle, the same beam spreads across a larger patch of ground.

For a horizontal surface, the incoming power per unit area is proportional to the sine of the Sun's altitude, before accounting for the atmosphere. The relevant question is the energy arriving on each square metre, not whether the sunlight has travelled a slightly longer distance across that square metre.

At the two solstices in our example:

- A noon altitude of 21.56° gives a projection factor of about 0.367.
- A noon altitude of 68.44° gives a projection factor of about 0.930.

At the higher angle, each horizontal square metre receives about 2.5 times as much instantaneous noon solar power in this idealisation. The incoming beam perpendicular to its own direction has the same intensity in both cases.

Noon is only one moment. We also need to add the sunlight arriving during the rest of the day.

## A longer day changes the total

The winter day has about 8.6 hours of daylight; the summer day has about 15.4. The Sun also follows different paths above the horizon, so multiplying the noon value by daylight hours would not give the correct daily total.

Instead, we can integrate the changing projection through the day. To avoid introducing a particular value for solar intensity, express the result as **equivalent hours of overhead sunlight**. One such hour supplies the energy that a horizontal surface would receive in one hour with the Sun directly overhead, at the model's fixed distance.

| Idealised date | Daily incoming energy at 45° north |
| --- | ---: |
| December solstice | 2.05 equivalent overhead hours |
| Equinox | 5.40 equivalent overhead hours |
| June solstice | 8.81 equivalent overhead hours |

The summer value is about 4.3 times the winter value. This is a comparison of ideal incoming energy, not a prediction that the air temperature becomes 4.3 times higher.

Repeat the calculation at 45 degrees south and the solstice values exchange places.

![At a fixed Earth-Sun distance, daylight and daily incoming solar energy rise from December to June at 45 degrees north and fall at 45 degrees south. The hemispheres match at the equinox.](/assets/images/figures/science_seasons_tilt_geometry.png){: width="1464" height="697" loading="lazy"}

*Original geometric calculation. Distance is fixed; atmospheric effects, terrain, and heat storage are omitted. The right panel integrates sunlight over the day rather than using only the noon angle.*

This is the mechanism a distance-only account misses: the same orbital position can favour one hemisphere's sunlight geometry while disadvantaging the other's. NASA's satellite-based seasonal illustration shows the corresponding changes in how sunlight is distributed across Earth. [Equinoxes and solstices from space](https://science.nasa.gov/resource/seeing-equinoxes-and-solstices-from-space/).

## Distance still has an effect

Removing distance changes from the model does not imply that distance is physically irrelevant. Solar intensity decreases with the square of distance from the Sun.

Earth is closest to the Sun in early January and farthest away in early July. The closer position therefore occurs during Northern Hemisphere winter, which is another difficulty for the simple “closer means northern summer” explanation. NASA describes the present orbital distance variation as roughly 3.4%. [NASA orbital explanation](https://science.nasa.gov/science-research/earth-science/milankovitch-orbital-cycles-and-their-role-in-earths-climate/).

An illustrative distance ratio of 1.034 would correspond to an intensity ratio of about $1.034^2=1.069$, or roughly a 7% difference. That affects the incoming energy budget. It does not replace the tilt mechanism demonstrated by the opposite-hemisphere calculation.

The two effects should therefore be separated rather than treating one as nonexistent. Tilt explains the alternating seasonal geometry. Distance modifies the amount of solar energy available to that geometry.

## Why the hottest day does not have to be the longest

The calculation concerns incoming sunlight. Temperature reflects a continuing energy balance: energy arrives, energy leaves, and energy is stored and transported.

A simple analogy is filling a bath with the drain open. The moment when the tap runs fastest need not be the moment when the bath contains the most water. The amount in the bath depends on the accumulated difference between inflow and outflow.

Likewise, a solar-energy maximum does not by itself identify the warmest day. Clouds, atmospheric conditions, oceans, land, and geography matter for the temperature response. Our bare-sphere calculation deliberately cannot predict a particular location's seasonal lag or weather.

Its narrower achievement is enough to test the misconception. Opposite seasonal energy patterns emerge even with a fixed Earth-Sun distance, once the axis is tilted.

## Try the explanation against a counterfactual

In the same model, remove the tilt as well as the distance variation. At 45 degrees north, every day then has the equinox geometry: a noon altitude of 45 degrees and 12 hours of daylight.

The annual alternation disappears from this model. That comparison identifies what the tilted axis contributed, while leaving the other assumptions unchanged.

When discussing the seasons, this is more useful than pointing only to an exaggerated ellipse in a diagram. Ask which feature of the explanation predicts opposite seasons in the hemispheres, and which quantities change when that feature is removed.

## Follow the geometry

For latitude $\phi$ and solar declination $\delta$, the ideal sunset hour angle is

$$
H_0=\arccos(-\tan\phi\tan\delta).
$$

At the latitudes used in the table, daylight lasts $24H_0/\pi$ hours. The average incoming solar power over a complete day, as a fraction of the perpendicular beam intensity $S_0$, is

$$
\frac{\overline Q}{S_0}
=\frac{H_0\sin\phi\sin\delta+
\cos\phi\cos\delta\sin H_0}{\pi}.
$$

Multiplying this fraction by 24 gives the equivalent overhead hours. Angles inside the trigonometric functions are in radians. Polar day and night require the corresponding limiting cases, handled in the code.

The [figure generator](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io/blob/master/assets/viz/generate_science_communication_figures.py) reproduces both tables. Its geometry is checked against direct numerical integration of the changing solar angle through the day.

*Archive note: dated 11 July 2024 for this collection; written and source-checked on 18 September 2026.*

---
author_profile: false
categories:
- Data Science
classes: wide
date: '2021-02-17'
excerpt: Kernel density estimation can map concentrations of traffic crashes, but crash density is not crash risk unless exposure and road-network geometry are accounted for.
header:
  image: /assets/images/headers/photo-data-science-maps.jpg
  og_image: /assets/images/headers/photo-data-science-maps.jpg
  overlay_image: /assets/images/headers/photo-data-science-maps.jpg
  overlay_filter: 0.4
  show_overlay_excerpt: false
  teaser: /assets/images/headers/photo-data-science-maps.jpg
  twitter_image: /assets/images/headers/photo-data-science-maps.jpg
keywords:
- kernel density estimation
- traffic safety
- crash hotspots
- network KDE
- spatial exposure
seo_description: A rigorous guide to traffic-crash hotspot mapping with KDE, including bandwidth selection, coordinate projection, network KDE, exposure, edge effects, uncertainty, and why event density is not risk.
seo_title: 'Traffic Crash KDE: Density Is Not Risk'
seo_type: article
summary: Kernel density estimation is useful for exploratory crash-hotspot mapping, but correct interpretation requires projected coordinates, road-network geometry, bandwidth sensitivity, traffic exposure, and validation.
tags:
- Spatial Statistics
- Traffic Safety
- Kernel Density Estimation
title: 'Traffic Crash KDE: Density Is Not Risk'
---

Kernel density estimation is useful for visualizing where recorded traffic crashes concentrate.

That statement is narrower than saying that KDE identifies the most dangerous roads.

A location can contain many crashes because it is genuinely hazardous.

It can also contain many crashes because it carries much more traffic.

The central distinction is

$$
\boxed{
\text{crash density}
\neq
\text{crash risk}.
}
$$

KDE estimates the spatial intensity of observed events unless additional exposure modeling is introduced.

## Point-pattern density

For event locations

$$
x_1,\ldots,x_n
\in
\mathbb R^2,
$$

a two-dimensional KDE can be written

$$
\hat f_h(x)
=
\frac{1}
{n h^2}
\sum_{i=1}^{n}
K
\left(
\frac{x-x_i}{h}
\right),
$$

where $K$ is a kernel and $h$ is the bandwidth.

The bandwidth is usually much more important than the exact kernel shape.

Small $h$ produces localized peaks.

Large $h$ produces broader smooth regions.

## Coordinates must be metric

Longitude and latitude are angular coordinates.

A bandwidth such as

$$
h=0.01
$$

degrees has different physical meaning across latitude and is not directly interpretable as road distance.

Before Euclidean KDE, project crash coordinates into an appropriate local metric coordinate reference system.

Then a bandwidth such as

$$
h=250\text{ m}
$$

has a physical meaning.

The previous code in this article applied Euclidean KDE directly to longitude and latitude.

That is not a defensible default.

## KDE estimates events, not exposure-adjusted risk

Suppose one arterial carries

$$
100{,}000
$$

vehicles per day and another carries

$$
5{,}000.
$$

Even if the per-vehicle crash risk is lower on the arterial, it may produce more crashes.

A simple risk-like rate is

$$
R(s)
=
\frac{
\text{expected crashes at }s
}{
\text{traffic exposure at }s
}.
$$

Exposure may be measured through vehicle-kilometers traveled, pedestrian volume, cyclist flow, road length, intersection traffic, or time at risk.

The correct denominator depends on the safety question.

KDE alone provides the numerator structure, not the denominator.

## Road networks are not continuous planes

A planar KDE lets kernel mass spread through buildings, parks, rivers, and other locations where vehicles cannot travel.

Traffic crashes occur on a network.

Network KDE measures distance along roads rather than straight-line Euclidean distance.

If $d_N(x,x_i)$ is shortest-path network distance, a network kernel has the form

$$
\hat\lambda_h(x)
=
\sum_i
K_h
\left(
d_N(x,x_i)
\right).
$$

This often provides a more meaningful hotspot representation for road crashes.

## Bandwidth determines the question

A 50-meter bandwidth asks about very local concentration.

A 2-kilometer bandwidth asks about neighborhood-scale concentration.

There is no context-free optimal bandwidth.

Statistical cross-validation can help select $h$, but domain scale matters too.

Bandwidth sensitivity should be shown rather than hidden.

## Edge effects

Kernels near the study boundary lose mass outside the observation region.

Road-network endpoints and administrative boundaries create additional complications.

Without correction, density can be underestimated near boundaries.

Possible responses include boundary correction, extending the analysis region, network-specific normalization, or cautious interpretation near boundaries.

## Repeated crashes at one coordinate

Geocoding often snaps several incidents to one intersection or road centroid.

Those duplicated coordinates are not necessarily duplicate records.

They can represent multiple crashes at the same location.

A weighted KDE may be appropriate when one record contains an event count $w_i$:

$$
\hat\lambda_h(x)
=
\sum_i
w_i
K_h(x-x_i).
$$

The previous code loaded an accident count column but ignored it.

That mismatch is now explicit.

## Temporal aggregation matters

Combining ten years of crashes into one spatial map assumes the underlying process is stable enough that the aggregation is meaningful.

But road design, traffic volume, speed limits, and land use change.

A useful model may estimate

$$
\lambda(s,t)
$$

rather than only

$$
\lambda(s).
$$

At minimum, compare time periods and test whether identified hotspots persist.

## Severity should not be hidden

A location with many minor crashes and a location with a few fatal crashes may require different interventions.

Weighted maps can incorporate severity, but the weights must represent a defined decision objective.

Those weights are policy choices.

They should not be presented as natural statistical constants.

## Hotspot detection versus causal intervention evaluation

KDE can identify spatial concentration.

It cannot establish that a proposed road treatment will reduce crashes.

Suppose density falls after a new speed limit.

A before-after map is confounded by regression to the mean, traffic-volume change, broader safety trends, enforcement changes, and seasonal composition.

Evaluating an intervention requires a counterfactual design.

## Regression to the mean

Road sites are often selected for treatment because they recently had unusually many crashes.

Even without any intervention, extremely high counts tend to move closer to their long-run expectation later.

Therefore

$$
\text{high before}
\rightarrow
\text{lower after}
$$

is not sufficient evidence of treatment success.

## Uncertainty in hotspot maps

KDE maps are commonly drawn as smooth colored surfaces with no uncertainty.

But estimated density depends on finite event counts.

Bootstrap resampling can reveal which spatial peaks are stable.

A hotspot that disappears under small data perturbations should not receive the same policy confidence as one that persists across resamples and bandwidths.

## A safer Python workflow

The following example assumes longitude and latitude input but projects the points before KDE.

~~~python
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity

FloatArray = NDArray[np.float64]

frame = pd.read_csv(
    "traffic_accidents.csv"
)

geometry = [
    Point(lon, lat)
    for lon, lat in zip(
        frame["longitude"],
        frame["latitude"],
        strict=True,
    )
]

geo = gpd.GeoDataFrame(
    frame,
    geometry=geometry,
    crs="EPSG:4326",
)

geo = geo.to_crs(
    geo.estimate_utm_crs()
)

coordinates: FloatArray = np.column_stack(
    [
        geo.geometry.x.to_numpy(),
        geo.geometry.y.to_numpy(),
    ]
)

bandwidth_m: float = 250.0

kde = KernelDensity(
    kernel="gaussian",
    bandwidth=bandwidth_m,
)

kde.fit(coordinates)

log_density: FloatArray = kde.score_samples(
    coordinates
)

density: FloatArray = np.exp(
    log_density
)

geo["kde_density"] = density
~~~

This estimates planar event density in meters.

For a real road-safety analysis, network KDE and an exposure model may be more appropriate.

## Validation

A useful hotspot method should be evaluated prospectively.

One design is:

1. estimate hotspots from years 1 through $T$;
2. predict high-density locations for year $T+1$;
3. compare future crash concentration with held-out observations.

This avoids praising a map merely because it reproduces the same incidents used to construct it.

## Conclusion

KDE is an exploratory spatial estimator.

Its correct interpretation is

$$
\boxed{
\text{where observed events concentrate}
}
$$

unless exposure and network structure are explicitly added.

For traffic safety, the full chain is

$$
\boxed{
\text{crashes}
+
\text{road network}
+
\text{traffic exposure}
+
\text{severity}
+
\text{time}
\rightarrow
\text{safety analysis}.
}
$$

A heat map is one layer of that analysis, not the final definition of risk.

## References

- Xie, Z., & Yan, J. (2008). Kernel density estimation of traffic accidents in a network space. *Computers, Environment and Urban Systems*, 32(5), 396–406.
- Pulugurtha, S. S., Krishnakumar, V. K., & Nambisan, S. S. (2007). New methods to identify and rank high pedestrian crash zones. *Accident Analysis & Prevention*, 39(4), 800–811.
- Yu, H., Liu, P., Chen, J., & Wang, H. (2014). Comparative analysis of the spatial analysis methods for hotspot identification. *Accident Analysis & Prevention*, 66, 80–88.

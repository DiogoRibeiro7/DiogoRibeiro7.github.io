---
author_profile: false
categories:
- Data Science
classes: wide
date: '2021-04-27'
excerpt: A study using GIS-based techniques for forest fire hotspot identification
  and analysis, validated with contributory factors like population density, precipitation,
  elevation, and vegetation cover.
header:
  image: /assets/images/forest_fire_kde_1.jpg
  og_image: /assets/images/data_science_4.jpg
  overlay_image: /assets/images/forest_fire_kde_1.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/forest_fire_kde_1.jpg
  twitter_image: /assets/images/data_science_4.jpg
keywords:
- Forest fires
- Gis
- Kernel density estimation
- Getis-ord gi*
- Anselin local moran's i
- Hotspot analysis
- Forest fire management
- Spatial analysis
- Belait district
- Bash
- Python
seo_description: "Explore GIS techniques like KDE, Getis-Ord Gi*, and Anselin Local\
  \ Moran\u2019s I for identifying forest fire hotspots in Southeast Asia, validated\
  \ by contributory factors."
seo_title: GIS-Based Forest Fire Hotspot Identification Using Contributory Factors
seo_type: article
summary: This article explores the application of GIS-based techniques, such as Kernel
  Density Estimation (KDE), Getis-Ord Gi*, and Anselin Local Moran's I, in identifying
  forest fire hotspots. The study focuses on Belait District, Brunei Darussalam, and
  validates hotspot results using contributory factors like population density, precipitation,
  elevation, and vegetation cover.
tags:
- "Anselin local moran\u2019s i"
- Gis
- Forest fires
- Getis-ord gi*
- Python
- Kernel density estimation
- Bash
title: 'GIS-Based Forest Fire Hotspot Identification: A Comprehensive Approach Using
  Contributory Factors'
---

![Example Image](/assets/images/forest_fire_kde_3.png)

### Introduction

Forest fires pose a significant threat to ecosystems, infrastructure, and human lives across the globe. The frequency and severity of forest fires are rising due to climate change, which exacerbates conditions conducive to fires, such as prolonged droughts and extreme heat. In regions like Southeast Asia, where forest fires have a historical and cyclical nature, the situation is particularly severe. The damaging effects of these fires, including loss of biodiversity, air pollution from smoke haze, and economic losses, emphasize the need for improved forest fire management strategies.

To better manage forest fires, spatial analysis tools like Geographic Information Systems (GIS) have become crucial. These systems help identify high-risk areas, or hotspots, where forest fires are more likely to occur. This allows decision-makers to prioritize resources, implement early-warning systems, and deploy preventive measures.

While many GIS hotspot analysis techniques have been applied in various fields, from crime prediction to public health, their application in forest fire management has gained momentum. The focus of this article is to explore three widely used GIS-based hotspot analysis methods—Kernel Density Estimation (KDE), Getis-Ord Gi*, and Anselin Local Moran’s I—and validate their effectiveness in identifying forest fire hotspots in Belait District, Brunei Darussalam. By validating the results using forest fire contributory factors, we aim to ascertain the most accurate method for forest fire hotspot identification.

### The Threat of Forest Fires in Southeast Asia

#### Historical Context

Southeast Asia has faced numerous forest fire events, notably in 1982–1983, 1997–1998, and more recently, in 2013 and 2015. The El Niño phenomenon, which brings extended periods of drought, has been a major contributor to these fires. Peatland regions, especially in countries like Indonesia and Brunei Darussalam, are particularly susceptible to fires during dry spells. In 2016, fires in the Belait district’s peatlands took more than two months to extinguish and affected over 274 hectares of land .

Forest fires in Brunei have been escalating in recent years, with an 80% increase in fire incidents between 2007 and 2016. The country’s geographic and climatic characteristics—extensive peatland, high humidity, and tropical storms—contribute to the volatility of forest fire occurrences. Human activities, such as open burning during dry periods, have also exacerbated the issue.

### GIS in Forest Fire Hotspot Analysis

Hotspot analysis enables researchers and policymakers to detect geographic clusters of events, such as forest fires, and apply mitigation strategies to the most affected areas. The visual representation of hotspots aids in decision-making, allowing authorities to efficiently allocate resources and take preventive actions. Numerous GIS-based techniques exist for hotspot identification, each with its strengths and weaknesses.

#### Kernel Density Estimation (KDE)

KDE is a non-parametric technique that transforms discrete event data points into a continuous surface, representing the intensity of event occurrences across an area. By smoothing point data over a geographic area, KDE provides a visual heat map that shows high- and low-density regions of forest fires. This technique is advantageous for its simplicity and ability to handle small datasets .

KDE has been used widely in various studies, from crime prediction to environmental monitoring. However, its main limitation is that it does not provide statistical significance testing, which means it may overestimate or underestimate hotspot areas. In the context of forest fires, KDE is often employed to assess the intensity of fire events and guide the deployment of firefighting resources.

#### Getis-Ord Gi*

The Getis-Ord Gi statistic* is a spatial autocorrelation measure used to identify clusters of high or low values in a dataset. This method evaluates whether the spatial distribution of events is random or clustered by calculating z-scores and p-values. A high z-score and low p-value indicate statistically significant clustering, marking an area as a hotspot. Getis-Ord Gi* is particularly useful in determining the spatial extent of hotspots and their statistical significance .

However, Getis-Ord Gi* has limitations when it comes to the extent of clustering. The size of the study area and the chosen distance threshold significantly influence the results. In larger study areas, Getis-Ord Gi* may fail to identify small but significant clusters. For forest fire management, this means that crucial fire-prone areas may be missed due to the method’s sensitivity to scale.

#### Anselin Local Moran’s I

Anselin Local Moran’s I is another spatial autocorrelation measure used to identify local clusters and spatial outliers. Unlike Getis-Ord Gi*, which considers both the target location and its neighbors, Local Moran’s I only focuses on the neighboring points to determine if a cluster exists. A high positive Moran’s I value indicates spatial clusters of similar values, while a negative value identifies outliers .

Local Moran’s I is useful for detecting smaller clusters that might be overlooked by broader measures like Getis-Ord Gi*. However, it can be too localized for large-scale studies, and its results can sometimes conflict with those of other hotspot analysis techniques.

### Study Area: Belait District, Brunei Darussalam

Belait, the largest district in Brunei Darussalam, spans an area of 2,727 km². Its geography is dominated by forests, including peat swamp forests, heath forests, and secondary forests. Due to its location and extensive forest cover, Belait is highly vulnerable to forest fires. Peatlands in the district, in particular, are difficult to extinguish once a fire starts, as they can burn underground for extended periods .

The district is home to approximately 69,600 inhabitants, with most of the population concentrated in the northern coastal region. This demographic concentration, coupled with the oil and gas industry’s presence, heightens the risk of fire-related damage. The study focuses on analyzing forest fire hotspots in this region using forest fire call data from 2016.

### Methodology

#### Data Collection

The dataset used in this study consists of forest fire call records from January to August 2016. The locations of the calls were imported into ArcGIS software for analysis. Three different GIS-based hotspot analysis techniques—KDE, Getis-Ord Gi*, and Anselin Local Moran’s I—were applied to identify forest fire hotspots in Belait.

Additionally, four forest fire contributory factors were selected to validate the accuracy of the hotspot analysis methods:

1. **Population Density**: Higher population densities correlate with increased human activity, which may lead to more forest fires due to activities such as open burning.
2. **Elevation**: Low-elevation areas tend to experience less precipitation and higher temperatures, making them more susceptible to fires.
3. **Vegetation Cover**: Different forest types have varying degrees of susceptibility to fire. For example, peat swamp forests are more vulnerable during dry periods.
4. **Precipitation**: Areas with lower annual rainfall are more prone to forest fires .

#### Hotspot Identification Methods

##### Kernel Density Estimation (KDE)

KDE was applied to the forest fire call data to create a continuous surface map. The resulting density map identified several hotspot areas of varying intensity. While KDE does not perform statistical significance testing, it offers a useful visualization of fire-prone areas.

##### Getis-Ord Gi*

The Getis-Ord Gi* analysis was conducted to identify statistically significant clusters of high fire call densities. This method detected one major hotspot in Belait, which was further validated using the contributory factors.

##### Anselin Local Moran’s I

Surprisingly, Anselin Local Moran’s I did not identify any statistically significant hotspots in the study area. This outcome suggests that Local Moran’s I may not be as effective for large-scale analyses of forest fire incidents in regions like Belait .

### Results and Discussion

#### Hotspot Validation Using Contributory Factors

Given the discrepancies between the hotspot analysis methods, it was essential to validate the predicted hotspots against the four forest fire contributory factors. The results showed varying levels of agreement between the methods.

##### Vegetation Cover

The study found that the hotspots identified by KDE and Getis-Ord Gi* overlapped significantly with secondary forests and peat swamp forests, both of which are highly susceptible to fire during dry periods. Secondary forests, in particular, are known for their vulnerability due to open canopies and ground-level vegetation .

##### Precipitation

Areas with lower annual rainfall were more prone to fire occurrences. The northern region of Belait, where the identified hotspots are located, is one of the driest regions in Brunei, receiving less than 2500mm of annual rainfall. This further validates the results of the KDE and Getis-Ord Gi* analyses .

##### Elevation

The identified hotspots were situated in low-elevation areas, which tend to dry out faster and are more vulnerable to fires. This finding aligns with previous research showing that lower elevations experience more frequent forest fires .

##### Population Density

The northern part of Belait, which has the highest population density, also corresponded with the identified hotspots. This suggests a correlation between human activity and forest fire occurrences in the region .

#### Comparison of Methods

The validation process revealed that while both KDE and Getis-Ord Gi* identified forest fire hotspots with some degree of accuracy, KDE was more effective in detecting a broader range of hotspot areas. KDE identified four hotspots, while Getis-Ord Gi* detected only one. Anselin Local Moran’s I did not identify any hotspots, indicating that it may not be suitable for this type of large-scale analysis .

While Getis-Ord Gi* provides statistically significant results, its sensitivity to the chosen distance threshold may have limited its ability to detect smaller clusters. On the other hand, KDE’s continuous surface model allowed for a more nuanced representation of fire-prone areas, making it a valuable tool for forest fire management in Belait.

### Conclusion and Recommendations

This study demonstrates that Kernel Density Estimation (KDE) is a reliable method for identifying forest fire hotspots, particularly when validated against contributory factors such as population density, precipitation, elevation, and vegetation cover. KDE’s ability to provide a continuous density surface makes it a valuable tool for visualizing and managing forest fire risk in large and diverse geographic areas like Belait.

The validation process also highlighted the importance of using multiple contributory factors to assess hotspot accuracy. By incorporating additional factors such as air temperature and wind speed, future studies could further improve hotspot identification accuracy and support more effective forest fire prevention strategies.

### Appendix: Python Code for Forest Fire Hotspot Analysis

#### Requirements

Install the required libraries:

```bash
pip install geopandas scipy numpy esda
```

Install ArcGIS API for Python to use spatial statistics methods like KDE and Moran’s I:

```python
pip install arcgis
```

#### Step-by-step Code

```python
import geopandas as gpd
import numpy as np
from scipy.stats import zscore
from esda.moran import Moran_Local
from esda.getisord import G_Local
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# Load the Forest Fire Call Data as a GeoDataFrame
# Assuming 'forest_fire_calls.shp' contains the point data for forest fire calls.
fire_calls = gpd.read_file('forest_fire_calls.shp')

# Ensure the GeoDataFrame is projected in a coordinate system that allows distance calculation
fire_calls = fire_calls.to_crs(epsg=3395)  # World Mercator projection

# Load the contributing factors layers (population density, elevation, precipitation, etc.)
# Assuming the files are already available in shapefiles
population_density = gpd.read_file('population_density.shp')
elevation = gpd.read_file('elevation.shp')
precipitation = gpd.read_file('precipitation.shp')

# Spatial join to link fire calls with contributing factors
fire_calls_with_factors = gpd.sjoin(fire_calls, population_density, how="left", op='intersects')
fire_calls_with_factors = gpd.sjoin(fire_calls_with_factors, elevation, how="left", op='intersects')
fire_calls_with_factors = gpd.sjoin(fire_calls_with_factors, precipitation, how="left", op='intersects')

### 1. Kernel Density Estimation (KDE) ###
def kde_analysis(geo_df, bandwidth=500):
    """
    Perform Kernel Density Estimation on GeoDataFrame of points (forest fire calls).
    
    Parameters:
    geo_df (GeoDataFrame): The GeoDataFrame with points
    bandwidth (int): Smoothing parameter for KDE

    Returns:
    KDE values as a 2D grid and extent of the grid.
    """
    coords = np.vstack([geo_df.geometry.x, geo_df.geometry.y]).T
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(coords)

    # Create a grid for evaluation
    x_min, y_min, x_max, y_max = geo_df.total_bounds
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Get KDE estimates for the grid
    z = np.exp(kde.score_samples(grid_coords)).reshape(x_grid.shape)

    return z, (x_min, x_max, y_min, y_max)

# Perform KDE Analysis
kde_values, extent = kde_analysis(fire_calls)

# Plot KDE Results
plt.imshow(kde_values, extent=extent, origin='lower', cmap='hot', alpha=0.7)
plt.scatter(fire_calls.geometry.x, fire_calls.geometry.y, c='blue', s=10, label='Forest Fire Calls')
plt.title('Kernel Density Estimation of Forest Fire Hotspots')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Density')
plt.show()

### 2. Getis-Ord Gi* ###
def getis_ord_gi(fire_calls, threshold_dist=1000):
    """
    Perform Getis-Ord Gi* hotspot analysis on forest fire data.

    Parameters:
    fire_calls (GeoDataFrame): GeoDataFrame with forest fire points
    threshold_dist (int): Distance threshold for defining neighbors

    Returns:
    Z-scores for Getis-Ord Gi*.
    """
    coords = np.vstack([fire_calls.geometry.x, fire_calls.geometry.y]).T
    g = G_Local(coords, threshold_dist)

    return g.ZI

# Run Getis-Ord Gi* Analysis
gi_z_scores = getis_ord_gi(fire_calls)

# Add Z-scores to the GeoDataFrame
fire_calls['Getis_Ord_Z'] = gi_z_scores

# Plot Getis-Ord Gi* Z-scores
fire_calls.plot(column='Getis_Ord_Z', cmap='coolwarm', legend=True)
plt.title('Getis-Ord Gi* Hotspot Analysis of Forest Fire Calls')
plt.show()

### 3. Anselin Local Moran's I ###
def local_morans_i(fire_calls):
    """
    Perform Anselin Local Moran’s I hotspot analysis on forest fire calls.

    Parameters:
    fire_calls (GeoDataFrame): GeoDataFrame with forest fire points

    Returns:
    Moran’s I values and p-values for each point.
    """
    coords = np.vstack([fire_calls.geometry.x, fire_calls.geometry.y]).T
    weights = np.ones((len(coords), len(coords)))  # Simple weights for spatial autocorrelation

    moran = Moran_Local(coords, weights)
    return moran.Is, moran.p_sim

# Run Anselin Local Moran’s I Analysis
moran_values, moran_p_values = local_morans_i(fire_calls)

# Add Moran's I values to GeoDataFrame
fire_calls['Moran_I'] = moran_values
fire_calls['Moran_P'] = moran_p_values

# Plot Local Moran's I Results
fire_calls.plot(column='Moran_I', cmap='viridis', legend=True)
plt.title('Anselin Local Moran\'s I Analysis of Forest Fire Calls')
plt.show()

### 4. Validation Using Contributory Factors ###
def validate_hotspots(fire_calls_with_factors):
    """
    Validate the hotspots by checking interference with contributory factors.

    Parameters:
    fire_calls_with_factors (GeoDataFrame): GeoDataFrame with fire calls and contributory factors
    """
    # Example of validation: Check correlation between fire density and population density
    correlation = fire_calls_with_factors['fire_density'].corr(fire_calls_with_factors['population_density'])
    print(f'Correlation between Fire Density and Population Density: {correlation}')

# Example validation
validate_hotspots(fire_calls_with_factors)
```

#### Explanation of Code

##### Kernel Density Estimation (KDE):

- The `kde_analysis` function creates a continuous surface representing forest fire density based on the point data (locations of forest fires).
- The results are visualized on a heatmap using `matplotlib`.

##### Getis-Ord Gi*:

- The `getis_ord_gi` function applies the Getis-Ord Gi* spatial statistic to identify clusters of high fire activity (hotspots) based on distance thresholds.
- The results are plotted as z-scores.

##### Anselin Local Moran's I:

- The `local_morans_i` function computes the Local Moran’s I values to detect clusters and spatial outliers in the forest fire data.
- The resulting clusters are plotted as spatial autocorrelation values.

##### Validation Using Contributory Factors:

- The `validate_hotspots` function checks the correlation between identified hotspots and contributory factors like population density, elevation, and precipitation.

#### Requirements for Running the Code

- **Input Data**: Ensure the input data files (shapefiles for fire call points, population density, elevation, etc.) are available.
- **GIS Libraries**: The script relies on GIS libraries such as `geopandas` and `arcgis` for handling spatial data.

This code provides a starting point for performing GIS-based forest fire hotspot analysis and validation. You can further extend this code to suit specific datasets and additional analyses!

### References

- Zahran, E., Shams, S., & Mohd Said, S.N. (2020). Validation of Forest Fire Hotspot Analysis in GIS Using Forest Fire Contributory Factors. *Systematic Reviews in Pharmacy, 11*(12), 249-255.
- Getis, A., & Ord, J.K. (1992). The analysis of spatial association by use of distance statistics. *Geographical Analysis, 24*, 189-206.
- Anselin, L. (1995). Local indicators of spatial association—LISA. *Geographical Analysis, 27*(2), 93-115.
- Kuter, N., Yenilmez, F., & Kuter, S. (2011). Forest fire risk mapping by Kernel Density Estimation. *Croatian Journal of Forest Engineering, 32*, 599-610.
- Catry, F.X., Rego, F.C., Bação, F.L., & Moreira, F. (2009). Modeling and mapping wildfire ignition risk in Portugal. *International Journal of Wildland Fire, 18*(8), 921-931.
- Chuvieco, E., Englefield, P., Trishchenko, A., & Luo, Y. (2008). Generation of long time series of burn severity maps using Landsat data in Canada. *Remote Sensing of Environment, 112*(9), 3751-3763.
- Eastman, J.R. (2003). IDRISI Kilimanjaro: Guide to GIS and Image Processing. *Clark Labs, Clark University*.
- Stolle, F., Chomitz, K.M., Lambin, E.F., & Tomich, T.P. (2003). Land use and vegetation fires in Jambi Province, Sumatra, Indonesia. *Forest Ecology and Management, 179*(1-3), 277-292.
- Giglio, L., Randerson, J.T., & van der Werf, G.R. (2013). Analysis of daily, monthly, and annual burned area using the fourth-generation global fire emissions database (GFED4). *Journal of Geophysical Research: Biogeosciences, 118*(1), 317-328.
- Pereira, J.M.C., & Carreiras, J.M.B. (2001). Fire risk and fire danger estimation through the processing of satellite data. *Remote Sensing of Environment, 77*(1), 1-10.
- Sakellariou, S., & Anagnostopoulou, C. (2020). Assessing forest fire risk in Greece based on fire weather indices and lightning activity. *Natural Hazards, 104*(1), 507-526.
- Saura, S., & Martínez-Millán, J. (2001). Sensitivity of landscape pattern metrics to map spatial extent. *Photogrammetric Engineering & Remote Sensing, 67*(9), 1027-1036.
- Wulder, M.A., White, J.C., Loveland, T.R., Woodcock, C.E., Belward, A.S., Cohen, W.B., ... & Zhu, Z. (2016). The global Landsat archive: Status, consolidation, and direction. *Remote Sensing of Environment, 185*, 271-283.
- Zhu, Z., Woodcock, C.E., Rogan, J., & Kellndorfer, J. (2012). Assessment of spectral, polarimetric, temporal, and spatial dimensions for urban and peri-urban land cover classification using Landsat and SAR data. *Remote Sensing of Environment, 117*, 72-82.
- Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.  
- Scott, D.W. (2015). *Multivariate Density Estimation: Theory, Practice, and Visualization*. John Wiley & Sons.  
- Cressie, N. (1993). *Statistics for Spatial Data*. John Wiley & Sons.  
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers.  
- Diggle, P.J. (2013). *Statistical Analysis of Spatial and Spatio-Temporal Point Patterns*. CRC Press.  
- Bivand, R.S., Pebesma, E., & Gómez-Rubio, V. (2013). *Applied Spatial Data Analysis with R*. Springer Science & Business Media.
- Brunsdon, C., & Comber, L. (2015). *An Introduction to R for Spatial Analysis and Mapping*. Sage Publications.

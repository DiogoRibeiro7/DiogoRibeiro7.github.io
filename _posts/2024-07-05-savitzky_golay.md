---
author_profile: false
categories:
- Data Science
- Time Series Analysis
- Machine Learning
- Data Processing
classes: wide
date: '2024-07-05'
header:
  image: /assets/images/data_science_6.jpg
  og_image: /assets/images/data_science_6.jpg
  overlay_image: /assets/images/data_science_6.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_6.jpg
  twitter_image: /assets/images/data_science_6.jpg
keywords:
- Time series smoothing
- Moving averages
- Savitzky-golay filter
- Data smoothing techniques
- Python for time series
- Time series data analysis
- Signal processing
- Data visualization
- Python
- Unknown
seo_description: Learn about smoothing time series data using Moving Averages and
  Savitzky-Golay filters. Explore their differences, benefits, and Python implementations
  for signal and data processing.
seo_title: 'Time Series Smoothing: Moving Averages vs. Savitzky-Golay Filters'
seo_type: article
summary: 'This article compares two popular techniques for smoothing time series data:
  Moving Averages and Savitzky-Golay filters, focusing on their applications, benefits,
  and implementation in Python.'
tags:
- Time series
- Data smoothing
- Moving averages
- Savitzky-golay filter
- Python
- Data visualization
- Signal processing
- Data analysis
- Unknown
title: 'Smoothing Time Series Data: Moving Averages vs. Savitzky-Golay Filters'
---

## Introduction

### Brief Introduction to Time Series Data and the Importance of Smoothing

Time series data is a sequence of data points collected or recorded at specific time intervals. This type of data is ubiquitous across various fields such as finance, meteorology, healthcare, and many more. Examples include stock prices, weather measurements, and sensor readings. Analyzing time series data helps in identifying trends, patterns, and making predictions about future events. However, the raw data often contains a significant amount of noise—random variations that can obscure the underlying trends.

Smoothing is a crucial preprocessing step in time series analysis. It helps in reducing the noise and highlighting the important patterns in the data. By removing these random fluctuations, smoothing makes it easier to detect meaningful signals and trends that can be used for further analysis and forecasting.

### Common Challenges with Noisy Data in Time Series Analysis

Noisy data presents several challenges in time series analysis:
- **Obscured Trends**: Noise can mask the true underlying trends and patterns in the data, making it difficult to identify them.
- **Erroneous Predictions**: High noise levels can lead to inaccurate models and predictions, as the noise can mislead the algorithms.
- **Increased Complexity**: Analyzing noisy data is computationally more complex and can require more sophisticated techniques to process effectively.

These challenges necessitate the use of smoothing techniques to clean the data before applying analytical or predictive models.

### Introducing the Two Smoothing Techniques: Moving Averages and Savitzky-Golay Filters

Two popular techniques for smoothing time series data are Moving Averages and Savitzky-Golay Filters.

- **Moving Averages**: This is a simple and widely used method. It works by averaging the data points within a specified window size, effectively smoothing out short-term fluctuations and highlighting longer-term trends. Moving averages are easy to implement and understand but can lag behind true trends, especially when the data contains sharp changes.

- **Savitzky-Golay Filters**: This technique, on the other hand, applies a polynomial fit to the data points within a moving window. The key advantage of the Savitzky-Golay filter is its ability to preserve the features of the signal, such as peaks and troughs, better than a simple moving average. This makes it more effective in maintaining the shape and characteristics of the original data while reducing noise.

### Goal of the Article

The goal of this article is to provide a detailed explanation of Moving Averages and Savitzky-Golay Filters, demonstrating how each method works and comparing their effectiveness in smoothing time series data. By the end of the article, readers will have a better understanding of these techniques and be able to choose the appropriate method for their specific use case.


## Understanding Time Series Smoothing

### Define What Time Series Smoothing Is and Why It's Important

Time series smoothing refers to a set of techniques used to remove noise and irregularities from time series data, making the underlying patterns and trends more visible. Smoothing helps in transforming the raw data into a more understandable and interpretable form, which is crucial for accurate analysis and forecasting.

In a time series, each data point is influenced not only by the underlying trend and seasonality but also by random fluctuations or noise. Smoothing techniques aim to filter out this noise without significantly distorting the true signal, thus providing a clearer picture of the data's inherent structure.

### Discuss the Concept of Noise in Time Series Data

Noise in time series data represents the random variations that are not part of the true underlying pattern or trend. This noise can arise from various sources such as measurement errors, external influences, or inherent variability in the system being measured. 

- **Measurement Errors**: These can occur due to inaccuracies in the instruments or methods used to collect the data.
- **External Influences**: Unexpected events or external factors can introduce irregularities in the data.
- **Inherent Variability**: Some systems have natural fluctuations that can be considered as noise.

Noise can make it challenging to identify and analyze the true patterns in the data. It can obscure important trends, lead to misleading conclusions, and affect the performance of predictive models.

### Explain the General Benefits of Smoothing Data

Smoothing data offers several benefits that are critical for effective time series analysis:

- **Revealing Underlying Trends**: Smoothing helps in highlighting the true trends in the data by filtering out random noise. This makes it easier to understand the long-term direction and behavior of the series.
- **Enhancing Signal Clarity**: By reducing the noise, smoothing techniques enhance the clarity of the signal, making the data more interpretable and actionable.
- **Improving Forecast Accuracy**: Smoothing can lead to more accurate forecasts by providing a cleaner dataset for model training. Models trained on smoothed data are less likely to be misled by random fluctuations.
- **Facilitating Pattern Recognition**: Smoothed data allows for better recognition of patterns such as seasonality and cyclic behavior, which are essential for making informed decisions and predictions.
- **Reducing Data Volatility**: Smoothing reduces the volatility in the data, which can be particularly useful in applications like financial time series analysis where stability is crucial.

Overall, smoothing is a fundamental step in time series analysis that helps in deriving meaningful insights and building robust predictive models.

## Loading the Time-Series Data

### Describe the Type of Time Series Data Used in the Example

In this example, we will work with hourly data from the M4 competition dataset. The M4 competition is a well-known benchmark for evaluating time series forecasting methods. The dataset contains a wide variety of time series data from different domains, including finance, demographics, and industry. The specific series we will use consists of hourly observations, which is common in many real-world applications such as weather monitoring, stock prices, and sensor readings.

### Mention the Source of the Data and Why It Was Chosen

The M4 competition dataset was chosen for this example because it provides a diverse and challenging set of time series data that allows us to demonstrate the effectiveness of smoothing techniques. This dataset is widely recognized in the data science community for its comprehensive coverage of different time series patterns and its role in advancing the field of time series forecasting.

### Outline the Steps to Load and Prepare the Data for Analysis

Loading and preparing the time series data involves several steps. Here, we outline the process:

1. **Import Required Libraries**: First, we need to import the necessary libraries for data handling and visualization, such as pandas, numpy, and plotly.

2. **Load the Dataset**: The dataset can be loaded from a remote source or a local file. In this example, we will load the data from URLs provided by the M4 competition.

3. **Filter the Specific Series**: Since the M4 dataset contains multiple time series, we will filter the data to select a specific series for analysis. This series is identified by a unique identifier (UID).

4. **Merge Training and Testing Data**: The dataset is often split into training and testing sets. We will load both sets and prepare them for analysis.

5. **Data Preparation**: This involves basic preprocessing steps such as handling missing values, ensuring the data is in the correct format, and possibly resampling the data to a uniform time interval if necessary.

### Example Steps to Load and Prepare the Data

Here is an outline of the steps in pseudocode, which can be later translated into actual Python code:

1. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import plotly.express as px
    ```

2. **Load the Data**:
    ```python
    train = pd.read_csv('https://auto-arima-results.s3.amazonaws.com/M4-Hourly.csv')
    test = pd.read_csv('https://auto-arima-results.s3.amazonaws.com/M4-Hourly-test.csv').rename(columns={'y': 'y_test'})
    ```

3. **Filter for Specific Series**:
    ```python
    uid = np.array(['H386'])  # Unique identifier for the specific time series
    df_train = train.query('unique_id in @uid')
    df_test = test.query('unique_id in @uid')
    ```

4. **Merge Training and Testing Data**:
    ```python
    df = pd.merge(df_train, df_test, on='ds', how='left')  # 'ds' is the date column
    ```

5. **Data Preparation**:
    ```python
    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Ensure data types are correct
    df['ds'] = pd.to_datetime(df['ds'])
    ```

By following these steps, we ensure that the time series data is loaded and prepared correctly for further analysis and application of smoothing techniques.

## Moving Averages

### Define Moving Averages and How They Work

A Moving Average is a simple and widely used technique for smoothing time series data. It involves calculating the average of data points within a specific window of time and using this average as the smoothed value for the center of the window. As the window moves along the time series, new averages are calculated, producing a smoothed version of the original data.

Mathematically, the moving average for a given time point is calculated as:
$$MA_t = \frac{1}{n} \sum_{i=0}^{n-1} x_{t-i}$$
where $$MA_t$$ is the moving average at time $$t$$, $$n$$ is the window size, and $$x$$ represents the data points.

### Explain the Concept of Window Size and Its Impact on the Smoothing Process

The window size, often denoted as $$n$$, is a critical parameter in the moving average calculation. It determines the number of data points to consider for each average calculation. The choice of window size significantly impacts the smoothing process:

- **Small Window Size**: A smaller window size results in a less smooth curve that closely follows the original data. While it may capture short-term fluctuations, it might not effectively filter out noise.
- **Large Window Size**: A larger window size produces a smoother curve that emphasizes long-term trends. However, it can oversmooth the data, potentially obscuring important short-term variations and making the smoothed series lag behind actual changes in the data.

Choosing the appropriate window size depends on the specific characteristics of the time series and the goals of the analysis.

### Discuss the Advantages and Limitations of Using Moving Averages for Smoothing

#### Advantages

1. **Simplicity**: Moving averages are straightforward to implement and understand. They require minimal computational resources and are easy to interpret.
2. **Noise Reduction**: By averaging out fluctuations within the window, moving averages effectively reduce noise and make the underlying trends more apparent.
3. **Versatility**: This method can be applied to various types of time series data across different domains, making it a versatile tool for initial data smoothing.

#### Limitations

1. **Lag Effect**: Moving averages can introduce a lag in the data, particularly with larger window sizes. This means that sudden changes or trends in the data may not be immediately reflected in the smoothed series.
2. **Equal Weighting**: All data points within the window are given equal weight, which may not be ideal if more recent data points are more relevant for the analysis.
3. **Inability to Capture Complex Patterns**: Moving averages are limited in their ability to capture more complex patterns and features in the data, such as peaks and troughs, compared to more advanced smoothing techniques.

While moving averages are a useful tool for reducing noise and highlighting trends, they have limitations that need to be considered. The choice of window size is crucial, and understanding the trade-offs involved is essential for effective time series analysis.

## Savitzky-Golay Filter

### Define the Savitzky-Golay Filter and Its Mechanism

The Savitzky-Golay Filter is a digital smoothing filter that applies a polynomial regression to a set of data points within a moving window to preserve higher-order polynomial trends in the data. Unlike simple moving averages, which may smooth out important features of the signal, the Savitzky-Golay filter aims to reduce noise while maintaining the shape and characteristics of the original data, such as peaks and troughs.

### Explain How This Method Fits a Polynomial to Data Points Within a Window

The Savitzky-Golay filter works by fitting a polynomial of a specified degree to the data points within a moving window. This polynomial fitting is done using the method of linear least squares. For each position of the window, the polynomial coefficients are recalculated, and the value at the center of the window is replaced by the value of the polynomial at that point.

The polynomial fitting process can be described as follows:

1. **Select a Window**: Choose a window of data points around the current point.
2. **Fit a Polynomial**: Fit a polynomial of a given degree to the points within the window using linear least squares.
3. **Replace the Point**: Replace the current point with the value of the polynomial at that point.
4. **Move the Window**: Shift the window to the next point and repeat the process.

### Discuss the Parameters of the Filter and Their Effects

The main parameters of the Savitzky-Golay filter are the window size and the polynomial order. 

- **Window Size**: This parameter determines the number of data points included in each local polynomial fit. A larger window size results in a smoother output but can oversmooth the data, potentially missing important features. A smaller window size preserves more detail but may not effectively reduce noise.

- **Polynomial Order**: This parameter defines the degree of the polynomial used for fitting. A higher polynomial order can capture more complex patterns in the data but may also fit the noise if the order is too high relative to the window size. Common choices are linear (order 1), quadratic (order 2), or cubic (order 3) polynomials.

The appropriate choice of these parameters depends on the specific characteristics of the time series and the desired level of smoothing. Balancing the window size and polynomial order is crucial to effectively reduce noise while preserving important signal features.

### Compare the Advantages of Savitzky-Golay Filter to Moving Averages

#### Advantages

1. **Feature Preservation**: The Savitzky-Golay filter excels at preserving the original shape and features of the data, such as peaks and troughs, better than moving averages. This makes it more suitable for applications where maintaining the integrity of the signal's features is important.
2. **Flexibility**: By adjusting the polynomial order, the Savitzky-Golay filter can capture more complex patterns in the data, providing greater flexibility in smoothing.
3. **Less Lag**: Unlike moving averages, the Savitzky-Golay filter introduces less lag, making it more responsive to changes in the data.

#### Limitations

1. **Complexity**: The Savitzky-Golay filter is more complex to implement and understand compared to moving averages. It requires choosing two parameters (window size and polynomial order) instead of just one.
2. **Computational Cost**: Fitting polynomials is computationally more intensive than calculating simple averages, which may be a consideration for very large datasets or real-time applications.

The Savitzky-Golay filter offers significant advantages in preserving signal features and providing flexible smoothing, it comes with increased complexity and computational cost. Choosing between the Savitzky-Golay filter and moving averages depends on the specific requirements of the analysis and the characteristics of the data.

## Applying the Smoothing Techniques

### Outline the Steps to Apply Moving Averages to the Time Series Data

1. **Load the Data**: Ensure that the time series data is loaded into a pandas DataFrame and is properly indexed by the date or time column.
2. **Choose the Window Size**: Determine the window size (number of data points) over which the moving average will be calculated. This choice depends on the desired level of smoothing.
3. **Calculate the Moving Average**: Use the pandas `.rolling()` method to create a rolling window and the `.mean()` method to compute the average for each window.
4. **Store the Result**: Save the resulting smoothed values into a new column in the DataFrame for comparison and further analysis.
5. **Visualize the Data**: Plot the original and smoothed time series to visually inspect the effect of the moving average.

Example outline in pseudocode:
```python
# Load the data
df = pd.read_csv('your_data.csv', parse_dates=['date_column'], index_col='date_column')

# Choose window size
window_size = 10

# Calculate moving average
df['moving_average'] = df['value_column'].rolling(window=window_size, center=True).mean()

# Visualize
plt.plot(df['value_column'], label='Original')
plt.plot(df['moving_average'], label='Moving Average')
plt.legend()
plt.show()
```

### Outline the Steps to Apply Savitzky-Golay Filter to the Time Series Data

1. **Load the Data**: Ensure that the time series data is loaded into a pandas DataFrame and is properly indexed by the date or time column.
2. **Choose the Window Size and Polynomial Order**: Determine the window size and the order of the polynomial to fit within each window. The window size must be an odd number, and the polynomial order should be less than the window size.
3. **Apply the Savitzky-Golay Filter**: Use the `savgol_filter` function from the SciPy library to apply the filter to the data. This function requires specifying the window size and polynomial order.
4. **Store the Result**: Save the resulting smoothed values into a new column in the DataFrame for comparison and further analysis.
5. **Visualize the Data**: Plot the original and smoothed time series to visually inspect the effect of the Savitzky-Golay filter.

Example outline in pseudocode:

```python
# Load the data
df = pd.read_csv('your_data.csv', parse_dates=['date_column'], index_col='date_column')

# Choose window size and polynomial order
window_size = 11  # Must be odd
poly_order = 2

# Apply Savitzky-Golay filter
df['savgol_filter'] = savgol_filter(df['value_column'], window_length=window_size, polyorder=poly_order)

# Store the result
df['savgol_filter'] = savgol_filter(df['value_column'], window_length=window_size, polyorder=poly_order)

# Visualize the data
plt.plot(df['value_column'], label='Original')
plt.plot(df['savgol_filter'], label='Savitzky-Golay Filter')
plt.legend()
plt.show()
```

### Importance of Choosing Appropriate Parameters for Each Technique

Choosing appropriate parameters is crucial for the effectiveness of both smoothing techniques:

- **Window Size**: For both methods, the window size determines the number of data points considered for each smoothing calculation. A small window size will result in less smoothing and more sensitivity to short-term fluctuations, while a large window size will produce a smoother series but may oversmooth and lag behind real changes in the data.

- **Polynomial Order (Savitzky-Golay Filter)**: The polynomial order in the Savitzky-Golay filter defines the complexity of the polynomial used for fitting. A higher order can capture more complex patterns but may also fit the noise if not chosen carefully. Typically, a second or third-order polynomial is sufficient for most applications.

Careful experimentation and cross-validation are recommended to select the optimal parameters for a given dataset. It is important to balance between reducing noise and preserving significant features of the data. Visual inspection of the smoothed results can provide valuable insights into the appropriateness of the chosen parameters.

Applying smoothing techniques effectively requires understanding the impact of parameters and selecting them based on the specific characteristics and requirements of the time series data.

## Results and Comparison

### Discuss the Results of Applying Moving Averages to the Data

After applying the Moving Averages technique to the time series data, we observe a smoother version of the original series. The Moving Average reduces the short-term fluctuations and noise, making the long-term trends more visible. However, depending on the chosen window size, the Moving Average may lag behind actual changes in the data, especially if the window size is large. This lag can result in a delayed response to sudden changes or trends in the data.

### Discuss the Results of Applying Savitzky-Golay Filter to the Data

When applying the Savitzky-Golay Filter to the time series data, we see that it also smooths the data, but with some key differences. The Savitzky-Golay Filter preserves the overall shape and important features of the data, such as peaks and troughs, better than the Moving Average. This is due to the polynomial fitting process, which maintains the characteristics of the original signal while reducing noise. The result is a smoothed series that closely follows the actual data points without introducing significant lag.

### Compare the Performance of Both Methods in Terms of Preserving Trends and Reducing Noise

- **Preserving Trends**: The Savitzky-Golay Filter outperforms the Moving Average in preserving the underlying trends and features of the data. The polynomial fitting process ensures that important characteristics such as peaks and troughs are maintained, making it more suitable for applications where the shape of the data is important.
- **Reducing Noise**: Both methods effectively reduce noise, but the Moving Average does so by averaging out data points, which can lead to a loss of important details. The Savitzky-Golay Filter, on the other hand, reduces noise while preserving the essential features of the data, providing a clearer picture of the underlying trends.

### Highlight Specific Scenarios Where One Method Might Be Preferred Over the Other

- **Moving Averages**: This method is preferred when simplicity and computational efficiency are important. It is easy to implement and understand, making it suitable for quick smoothing tasks where preserving fine details is not crucial. Moving Averages are also useful when the primary goal is to observe long-term trends without much concern for short-term fluctuations.
  
- **Savitzky-Golay Filter**: This method is preferred when it is important to preserve the shape and features of the data. It is particularly useful in applications such as signal processing, spectroscopy, and any other field where maintaining the integrity of peaks and troughs is crucial. The Savitzky-Golay Filter is also beneficial when the data contains significant noise but important features need to be retained.

The choice between Moving Averages and the Savitzky-Golay Filter depends on the specific requirements of the analysis. Moving Averages offer simplicity and efficiency, while the Savitzky-Golay Filter provides superior preservation of data features and trends.

## Conclusion

In this article, we explored the concepts of time series smoothing and the importance of reducing noise to reveal underlying trends. We discussed two popular smoothing techniques: Moving Averages and Savitzky-Golay Filters. Moving Averages are straightforward to implement and effectively reduce noise by averaging data points within a specified window. However, they can introduce lag and may not preserve important features of the data. On the other hand, Savitzky-Golay Filters use polynomial fitting within a moving window, maintaining the shape and characteristics of the original data while reducing noise.

Choosing the right smoothing technique is crucial for different use cases. Moving Averages are ideal for applications where simplicity and computational efficiency are paramount, and where long-term trends are more important than short-term fluctuations. In contrast, Savitzky-Golay Filters are better suited for scenarios where preserving the detailed features of the data, such as peaks and troughs, is essential. This method is particularly beneficial in fields like signal processing and spectroscopy.

We encourage readers to experiment with both smoothing techniques on their own datasets. By adjusting parameters such as window size and polynomial order, users can find the optimal balance between noise reduction and feature preservation for their specific needs. Visual inspection of the smoothed results can provide valuable insights into the appropriateness of the chosen parameters.

For those interested in delving deeper into time series analysis and smoothing techniques, there are numerous resources and further readings available. Exploring these materials will provide a more comprehensive understanding of the underlying principles and advanced methods used in the field.

By understanding and applying the appropriate smoothing techniques, analysts and data scientists can enhance the clarity and interpretability of their time series data, leading to more accurate analysis and better-informed decision-making.

## References

1. Schafer, R. W. (2011). "What Is a Savitzky-Golay Filter?" IEEE Signal Processing Magazine, 28(4), 111–117. [DOI: 10.1109/MSP.2011.941097](https://ieeexplore.ieee.org/document/6017220)
2. Kawala-Sterniuk, A., Podpora, M., Pelc, M., Blaszczyszyn, M., Gorzelanczyk, E. J., Martinek, R., & Ozana, S. (2020). "Comparison of Smoothing Filters in Analysis of EEG Data for Medical Diagnostics Purposes." Sensors, 20(3), 807. [DOI: 10.3390/s20030807](https://doi.org/10.3390/s20030807)
3. M4 Competition Dataset. Retrieved from [https://www.mcompetitions.unic.ac.cy/the-datasets/](https://www.mcompetitions.unic.ac.cy/the-datasets/)
4. Pandas Documentation. "pandas.DataFrame.rolling." Retrieved from [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html)
5. SciPy Documentation. "scipy.signal.savgol_filter." Retrieved from [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)
6. Shumway, R. H., & Stoffer, D. S. (2017). "Time Series Analysis and Its Applications: With R Examples." Springer.
7. Chatfield, C. (2004). "The Analysis of Time Series: An Introduction." Chapman and Hall/CRC.

These references include research papers on the Savitzky-Golay filter and smoothing techniques, the M4 competition dataset, and documentation for the libraries used in the examples. Further reading on time series analysis can be found in the recommended textbooks.

## Appendix: Implementing the Savitzky-Golay Filter in Apache Flink

Here is how you can implement the Savitzky-Golay filter in Apache Flink using Python code translated into the appropriate Flink streaming context.

### Define the Savitzky-Golay Function in Flink

Although Flink primarily uses Java and Scala, Flink can run Python code using Apache Beam or PyFlink for Python support.

### Setup Your Flink Environment

You will need to install PyFlink and set up the necessary dependencies.

### Implement the Function and Apply It in a Flink Data Stream

Here's the implementation:

```python
import numpy as np
from pyflink.table import EnvironmentSettings, TableEnvironment, DataTypes
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.table.descriptors import Schema
from pyflink.table.udf import udf

def savitzky_golay(y: np.ndarray, window_size: int, poly_order: int) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter to the provided data.
    
    Args:
        y (np.ndarray): The input signal (1D array).
        window_size (int): The length of the filter window (must be a positive odd integer).
        poly_order (int): The order of the polynomial used to fit the samples (must be less than window_size).
        
    Returns:
        np.ndarray: The smoothed signal.
    """
    if window_size % 2 != 1 or window_size <= 0:
        raise ValueError("window_size must be a positive odd integer")
    if poly_order >= window_size:
        raise ValueError("poly_order must be less than window_size")
    
    half_window = (window_size - 1) // 2
    A = np.zeros((window_size, poly_order + 1))
    for i in range(window_size):
        A[i, :] = [(i - half_window) ** k for k in range(poly_order + 1)]
    ATA_inv = np.linalg.pinv(np.dot(A.T, A))
    coeffs = np.dot(ATA_inv, A.T)[0]
    
    y_padded = np.pad(y, (half_window, half_window), mode='reflect')
    y_smoothed = np.convolve(y_padded, coeffs[::-1], mode='valid')
    
    return y_smoothed

@udf(input_types=[DataTypes.ARRAY(DataTypes.FLOAT()), DataTypes.INT(), DataTypes.INT()], result_type=DataTypes.ARRAY(DataTypes.FLOAT()))
def savgol_filter_udf(y, window_size, poly_order):
    return savitzky_golay(np.array(y), window_size, poly_order).tolist()

# Set up the StreamExecutionEnvironment
env = StreamExecutionEnvironment.get_execution_environment()
env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
settings = EnvironmentSettings.new_instance().in_streaming_mode().build()
table_env = TableEnvironment.create(settings)

# Register the UDF
table_env.register_function("savgol_filter", savgol_filter_udf)

# Define the source schema
source_ddl = """
CREATE TABLE source (
    ts TIMESTAMP(3),
    value ARRAY<FLOAT>
) WITH (
    'connector' = 'filesystem',
    'path' = 'path/to/your/input/file',
    'format' = 'csv'
)
"""
table_env.execute_sql(source_ddl)

# Define the sink schema
sink_ddl = """
CREATE TABLE sink (
    ts TIMESTAMP(3),
    smoothed_value ARRAY<FLOAT>
) WITH (
    'connector' = 'filesystem',
    'path' = 'path/to/your/output/file',
    'format' = 'csv'
)
"""
table_env.execute_sql(sink_ddl)

# Apply the Savitzky-Golay filter
table_env.execute_sql("""
INSERT INTO sink
SELECT
    ts,
    savgol_filter(value, 11, 3) AS smoothed_value
FROM source
""")

# Execute the job
env.execute("Savitzky-Golay Smoothing")
```

In this example:

- **savitzky_golay function**: The core function that applies the Savitzky-Golay filter to a numpy array.
- **savitzky_golay_udf**: A PyFlink UDF wrapping the Savitzky-Golay function for use within the Flink SQL/Table API.
- **Flink Environment Setup**: Configures the stream execution environment and table environment.
- **Schema Definitions**: Defines the source and sink schemas using DDL.
- **Applying the Filter**: Applies the UDF to the data stream and writes the output to the sink.

Make sure to replace `'path/to/your/input/file'` and `'path/to/your/output/file'` with the actual paths to your data files. This setup assumes you are working with CSV files for simplicity. Adjust the file formats and connectors as needed for your specific data sources and sinks.

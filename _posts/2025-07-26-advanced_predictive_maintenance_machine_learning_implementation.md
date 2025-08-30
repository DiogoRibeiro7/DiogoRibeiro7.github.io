---
title: "Introduction to Predictive Maintenance: Transforming Industrial Operations Through Intelligent Asset Management"
categories:
- Industry 4.0
- Predictive Maintenance
- Data Analytics
tags:
- predictive maintenance
- condition monitoring
- industrial IoT
- asset management
- machine learning
author_profile: false
seo_title: "Predictive Maintenance: Transforming Industrial Operations"
seo_description: "Explore the future of maintenance strategies through predictive maintenance. Learn how AI, IoT, and data analytics are revolutionizing industrial asset management."
excerpt: "Predictive maintenance is redefining how industries manage assets, reducing downtime and costs through intelligent monitoring and data-driven decisions."
summary: "This comprehensive article examines the evolution of predictive maintenance, comparing traditional approaches and modern strategies powered by AI and analytics. It explores condition-based monitoring, implementation challenges, economic benefits, and future directions for predictive maintenance in industrial operations."
keywords: 
- "predictive maintenance"
- "intelligent asset management"
- "condition monitoring"
- "maintenance optimization"
- "industrial analytics"
classes: wide
date: '2025-07-27'
header:
  image: /assets/images/data_science_8.jpg
  og_image: /assets/images/data_science_8.jpg
  overlay_image: /assets/images/data_science_8.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science_8.jpg
  twitter_image: /assets/images/data_science_8.jpg
---

Building upon the foundational concepts of predictive maintenance, the integration of machine learning and artificial intelligence represents the cutting edge of asset management technology. While traditional predictive maintenance techniques provide valuable insights through statistical analysis and threshold-based monitoring, machine learning approaches unlock unprecedented capabilities to identify complex patterns, predict failure modes with greater accuracy, and optimize maintenance strategies through continuous learning from operational data.

The transformation from traditional condition-based monitoring to machine learning-powered predictive maintenance represents more than a technological upgrade—it embodies a paradigm shift toward intelligent systems that can adapt, learn, and improve their predictions over time. These advanced systems can process vast amounts of multi-dimensional data, identify subtle patterns that human analysts might miss, and provide actionable insights that drive significant improvements in equipment reliability and operational efficiency.

Modern industrial environments generate enormous volumes of data from sensors, control systems, maintenance records, and operational databases. Machine learning algorithms excel at extracting meaningful patterns from these complex, high-dimensional datasets, enabling organizations to move beyond simple threshold-based alerts toward sophisticated predictive models that can forecast equipment failures weeks or months in advance. This extended prediction horizon enables proactive maintenance planning, optimized resource allocation, and strategic decision-making that maximizes asset value while minimizing operational risks.

The implementation of machine learning in predictive maintenance requires understanding both the theoretical foundations of various algorithms and the practical considerations of deploying these systems in industrial environments. This comprehensive exploration will demonstrate how to implement advanced predictive maintenance solutions using Python, covering everything from data preprocessing and feature engineering to model development, validation, and deployment strategies.

## Machine Learning Fundamentals for Predictive Maintenance

Machine learning applications in predictive maintenance leverage algorithms that can automatically identify patterns in data without explicit programming for specific failure modes. These algorithms can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning approaches, each offering unique capabilities for different predictive maintenance applications.

Supervised learning algorithms form the backbone of most predictive maintenance systems, using historical data with known outcomes to train models that can predict future failures. These algorithms learn from labeled examples where equipment sensor data is paired with information about whether failures occurred, enabling them to identify patterns associated with different failure modes and predict the likelihood of future failures.

The power of supervised learning in predictive maintenance lies in its ability to automatically discover complex relationships between multiple sensor readings, operating conditions, and failure outcomes. Traditional approaches might monitor individual parameters against fixed thresholds, but machine learning models can simultaneously consider hundreds of variables and their interactions to make more accurate predictions.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Generate synthetic sensor data for demonstration
np.random.seed(42)
n_samples = 10000

# Create synthetic equipment sensor data
data = {
    'temperature': np.random.normal(75, 10, n_samples),
    'vibration': np.random.normal(0.5, 0.2, n_samples),
    'pressure': np.random.normal(100, 15, n_samples),
    'rotation_speed': np.random.normal(1800, 50, n_samples),
    'oil_viscosity': np.random.normal(40, 5, n_samples),
    'operating_hours': np.random.uniform(0, 8760, n_samples),
    'load_factor': np.random.uniform(0.3, 1.0, n_samples)
}

# Create synthetic failure labels with realistic patterns
# Higher probability of failure with extreme conditions
failure_probability = (
    (np.abs(data['temperature'] - 75) / 50) +
    (np.abs(data['vibration'] - 0.5) / 1.0) +
    (data['operating_hours'] / 10000) +
    np.random.normal(0, 0.1, n_samples)
)

# Convert probabilities to binary failures
data['failure'] = (failure_probability > np.percentile(failure_probability, 85)).astype(int)

# Create DataFrame
df = pd.DataFrame(data)

print("Synthetic Equipment Data Overview:")
print(df.head())
print(f"\nFailure rate: {df['failure'].mean():.2%}")
print(f"Dataset shape: {df.shape}")
```

Unsupervised learning algorithms provide valuable capabilities for anomaly detection and pattern discovery in predictive maintenance applications. These algorithms can identify unusual patterns in equipment behavior without requiring labeled failure data, making them particularly valuable for detecting novel failure modes or equipment operating in conditions not represented in historical training data.

Clustering algorithms can identify different operational states or failure modes by grouping similar operational patterns together. This capability enables maintenance teams to understand equipment behavior patterns and identify when equipment begins operating in unusual modes that might indicate developing problems.

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare data for clustering analysis
features = ['temperature', 'vibration', 'pressure', 'rotation_speed', 'oil_viscosity']
X = df[features]

# Standardize features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering to identify operational patterns
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Reduce dimensionality for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Visualize clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Equipment Operational Clusters')
plt.colorbar(scatter)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['failure'], cmap='RdYlBu', alpha=0.6)
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Equipment Failures in Operational Space')
plt.colorbar(scatter, label='Failure')

plt.tight_layout()
plt.show()

# Analyze failure rates by cluster
cluster_failure_analysis = df.groupby('cluster').agg({
    'failure': ['count', 'sum', 'mean'],
    'temperature': 'mean',
    'vibration': 'mean',
    'pressure': 'mean'
}).round(3)

print("Failure Analysis by Operational Cluster:")
print(cluster_failure_analysis)
```

Time series analysis represents a critical component of predictive maintenance machine learning, as equipment sensor data typically exhibits temporal patterns and trends that provide important insights into equipment health and failure progression. Time series algorithms can identify seasonal patterns, detect trend changes, and forecast future equipment states based on historical patterns.

```python
import warnings
warnings.filterwarnings('ignore')

# Create time series data for demonstration
dates = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
df['timestamp'] = dates
df = df.set_index('timestamp')

# Create rolling features for time series analysis
window_size = 24  # 24-hour rolling window

df['temp_rolling_mean'] = df['temperature'].rolling(window=window_size).mean()
df['temp_rolling_std'] = df['temperature'].rolling(window=window_size).std()
df['vibration_rolling_mean'] = df['vibration'].rolling(window=window_size).mean()
df['vibration_rolling_max'] = df['vibration'].rolling(window=window_size).max()

# Calculate rate of change features
df['temp_rate_change'] = df['temperature'].diff()
df['vibration_rate_change'] = df['vibration'].diff()

# Create lag features
df['temp_lag_1h'] = df['temperature'].shift(1)
df['temp_lag_24h'] = df['temperature'].shift(24)
df['vibration_lag_1h'] = df['vibration'].shift(1)

# Visualize time series patterns
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(df.index[:1000], df['temperature'][:1000], alpha=0.7, label='Temperature')
plt.plot(df.index[:1000], df['temp_rolling_mean'][:1000], color='red', label='24h Rolling Mean')
plt.ylabel('Temperature')
plt.title('Equipment Temperature Over Time')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df.index[:1000], df['vibration'][:1000], alpha=0.7, label='Vibration')
plt.plot(df.index[:1000], df['vibration_rolling_mean'][:1000], color='red', label='24h Rolling Mean')
plt.ylabel('Vibration')
plt.title('Equipment Vibration Over Time')
plt.legend()

plt.subplot(3, 1, 3)
failure_times = df[df['failure'] == 1].index[:50]  # Show first 50 failures
plt.scatter(failure_times, [1]*len(failure_times), color='red', alpha=0.7, s=50)
plt.ylabel('Failure Events')
plt.title('Equipment Failure Timeline')
plt.ylim(0.5, 1.5)

plt.tight_layout()
plt.show()

print("Time Series Feature Summary:")
print(df[['temp_rolling_mean', 'temp_rolling_std', 'vibration_rolling_mean', 
          'temp_rate_change', 'vibration_rate_change']].describe())
```

## Feature Engineering for Predictive Maintenance

Feature engineering represents one of the most critical aspects of successful machine learning implementation in predictive maintenance. Raw sensor data often requires significant preprocessing and transformation to create features that effectively capture the underlying patterns associated with equipment degradation and failure modes. The quality and relevance of engineered features often determine the success or failure of predictive maintenance models more than the choice of algorithm.

Statistical features derived from time series data provide fundamental insights into equipment behavior patterns and changes over time. These features capture various aspects of signal characteristics including central tendency, variability, distribution shape, and temporal patterns that can indicate developing equipment problems.

```python
import scipy.stats as stats
from scipy import signal

def calculate_statistical_features(data, window_size=24):
    """
    Calculate comprehensive statistical features for predictive maintenance
    """
    features = {}
    
    # Basic statistical measures
    features['mean'] = data.rolling(window=window_size).mean()
    features['std'] = data.rolling(window=window_size).std()
    features['var'] = data.rolling(window=window_size).var()
    features['min'] = data.rolling(window=window_size).min()
    features['max'] = data.rolling(window=window_size).max()
    features['range'] = features['max'] - features['min']
    
    # Percentiles and quantiles
    features['q25'] = data.rolling(window=window_size).quantile(0.25)
    features['q75'] = data.rolling(window=window_size).quantile(0.75)
    features['iqr'] = features['q75'] - features['q25']
    
    # Shape and distribution features
    features['skewness'] = data.rolling(window=window_size).skew()
    features['kurtosis'] = data.rolling(window=window_size).kurt()
    
    # Trend and change features
    features['trend'] = data.rolling(window=window_size).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window_size else np.nan
    )
    
    return pd.DataFrame(features)

# Apply statistical feature engineering to key sensors
temp_features = calculate_statistical_features(df['temperature'])
temp_features.columns = [f'temp_{col}' for col in temp_features.columns]

vibration_features = calculate_statistical_features(df['vibration'])
vibration_features.columns = [f'vib_{col}' for col in vibration_features.columns]

# Combine engineered features with original data
df_features = pd.concat([df, temp_features, vibration_features], axis=1)

# Calculate cross-correlation features between sensors
df_features['temp_vib_corr'] = df['temperature'].rolling(window=48).corr(df['vibration'])
df_features['temp_pressure_corr'] = df['temperature'].rolling(window=48).corr(df['pressure'])

print("Statistical Features Summary:")
print(f"Total features created: {len(temp_features.columns) + len(vibration_features.columns) + 2}")
print("\nSample of engineered features:")
print(df_features[['temp_mean', 'temp_std', 'temp_trend', 'vib_mean', 'vib_skewness']].head(10))
```

Frequency domain features provide insights into the spectral characteristics of sensor signals, which are particularly valuable for rotating machinery where different failure modes produce characteristic frequency patterns. Fourier transforms and spectral analysis can reveal bearing problems, imbalance, misalignment, and other mechanical issues through their unique frequency signatures.

```python
from scipy.fft import fft, fftfreq
from scipy.signal import welch, periodogram

def calculate_frequency_features(data, sampling_rate=1.0, window_size=256):
    """
    Calculate frequency domain features for vibration analysis
    """
    features = {}
    
    # Ensure we have enough data points
    if len(data) < window_size:
        return pd.Series(dtype=float)
    
    # Calculate FFT
    fft_vals = np.abs(fft(data[-window_size:]))
    freqs = fftfreq(window_size, 1/sampling_rate)
    
    # Take only positive frequencies
    positive_freq_idx = freqs > 0
    fft_vals = fft_vals[positive_freq_idx]
    freqs = freqs[positive_freq_idx]
    
    # Spectral features
    features['spectral_centroid'] = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    features['spectral_spread'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * fft_vals) / np.sum(fft_vals))
    features['spectral_rolloff'] = freqs[np.where(np.cumsum(fft_vals) >= 0.85 * np.sum(fft_vals))[0][0]]
    features['spectral_flatness'] = stats.gmean(fft_vals + 1e-10) / np.mean(fft_vals + 1e-10)
    
    # Energy in different frequency bands (for rotating machinery)
    # Low frequency (0-10 Hz): Imbalance, misalignment
    low_freq_mask = (freqs >= 0) & (freqs <= 10)
    features['low_freq_energy'] = np.sum(fft_vals[low_freq_mask])
    
    # Medium frequency (10-100 Hz): Gear mesh, blade pass
    med_freq_mask = (freqs > 10) & (freqs <= 100)
    features['med_freq_energy'] = np.sum(fft_vals[med_freq_mask])
    
    # High frequency (100+ Hz): Bearing problems
    high_freq_mask = freqs > 100
    features['high_freq_energy'] = np.sum(fft_vals[high_freq_mask])
    
    # Peak detection
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(fft_vals, height=np.mean(fft_vals))
    features['num_peaks'] = len(peaks)
    features['max_peak_freq'] = freqs[peaks[np.argmax(fft_vals[peaks])]] if len(peaks) > 0 else 0
    
    return pd.Series(features)

# Apply frequency domain analysis to vibration data
print("Calculating frequency domain features...")
frequency_features_list = []

# Process in chunks to calculate rolling frequency features
chunk_size = 256
step_size = 24

for i in range(chunk_size, len(df), step_size):
    chunk_data = df['vibration'].iloc[i-chunk_size:i]
    freq_features = calculate_frequency_features(chunk_data.values)
    freq_features.name = df.index[i]
    frequency_features_list.append(freq_features)

# Create DataFrame with frequency features
freq_features_df = pd.DataFrame(frequency_features_list)
print(f"Frequency features calculated for {len(freq_features_df)} time windows")

# Display sample frequency features
print("\nSample Frequency Domain Features:")
print(freq_features_df.head())
print("\nFrequency Features Description:")
print(freq_features_df.describe())
```

Degradation modeling features attempt to capture the progressive nature of equipment wear and failure progression. These features model how equipment condition changes over time and can provide insights into remaining useful life and optimal maintenance timing.

```python
def calculate_degradation_features(data, failure_labels, lookback_hours=168):  # 1 week lookback
    """
    Calculate features that model equipment degradation patterns
    """
    features = pd.DataFrame(index=data.index)
    
    # Time since last failure
    failure_times = data.index[failure_labels == 1]
    features['hours_since_failure'] = 0
    
    for i, timestamp in enumerate(data.index):
        previous_failures = failure_times[failure_times < timestamp]
        if len(previous_failures) > 0:
            features.loc[timestamp, 'hours_since_failure'] = (timestamp - previous_failures[-1]).total_seconds() / 3600
        else:
            features.loc[timestamp, 'hours_since_failure'] = (timestamp - data.index[0]).total_seconds() / 3600
    
    # Cumulative operating stress indicators
    features['cumulative_temp_stress'] = (data['temperature'] - data['temperature'].mean()).abs().cumsum()
    features['cumulative_vib_stress'] = (data['vibration'] - data['vibration'].mean()).abs().cumsum()
    
    # Degradation trend indicators
    window_sizes = [24, 168, 720]  # 1 day, 1 week, 1 month
    
    for window in window_sizes:
        # Temperature degradation trend
        temp_trend = data['temperature'].rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
        )
        features[f'temp_trend_{window}h'] = temp_trend
        
        # Vibration degradation trend
        vib_trend = data['vibration'].rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
        )
        features[f'vib_trend_{window}h'] = vib_trend
        
        # Standard deviation trend (increasing variability indicates degradation)
        temp_std_trend = data['temperature'].rolling(window=window).std().diff()
        features[f'temp_std_trend_{window}h'] = temp_std_trend
        
        vib_std_trend = data['vibration'].rolling(window=window).std().diff()
        features[f'vib_std_trend_{window}h'] = vib_std_trend
    
    # Maintenance cycle features
    features['maintenance_cycle_phase'] = features['hours_since_failure'] % (24 * 30)  # 30-day cycle
    features['maintenance_cycle_sin'] = np.sin(2 * np.pi * features['maintenance_cycle_phase'] / (24 * 30))
    features['maintenance_cycle_cos'] = np.cos(2 * np.pi * features['maintenance_cycle_phase'] / (24 * 30))
    
    return features

# Calculate degradation features
degradation_features = calculate_degradation_features(
    df[['temperature', 'vibration']], 
    df['failure']
)

print("Degradation Features Overview:")
print(degradation_features.head(10))
print(f"\nDegradation features shape: {degradation_features.shape}")

# Visualize degradation patterns
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(degradation_features['hours_since_failure'][:2000])
plt.title('Hours Since Last Failure')
plt.ylabel('Hours')

plt.subplot(2, 2, 2)
plt.plot(degradation_features['cumulative_temp_stress'][:2000])
plt.title('Cumulative Temperature Stress')
plt.ylabel('Stress Index')

plt.subplot(2, 2, 3)
plt.plot(degradation_features['temp_trend_168h'][:2000])
plt.title('Temperature Trend (Weekly)')
plt.ylabel('Trend Slope')

plt.subplot(2, 2, 4)
plt.plot(degradation_features['vib_trend_168h'][:2000])
plt.title('Vibration Trend (Weekly)')
plt.ylabel('Trend Slope')

plt.tight_layout()
plt.show()
```

## Advanced Machine Learning Models for Failure Prediction

The selection and implementation of appropriate machine learning models for predictive maintenance requires understanding the strengths and limitations of different algorithmic approaches. Each algorithm class offers unique capabilities for handling different types of data patterns, prediction horizons, and operational requirements common in industrial predictive maintenance applications.

Ensemble methods, particularly Random Forest and Gradient Boosting algorithms, have proven highly effective for predictive maintenance applications due to their ability to handle complex, non-linear relationships while providing feature importance insights and robust performance across diverse datasets.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import xgboost as xgb

# Prepare comprehensive feature set for modeling
# Combine original features with engineered features
feature_columns = [
    'temperature', 'vibration', 'pressure', 'rotation_speed', 'oil_viscosity',
    'operating_hours', 'load_factor', 'temp_rolling_mean', 'temp_rolling_std',
    'vibration_rolling_mean', 'vibration_rolling_max', 'temp_rate_change', 
    'vibration_rate_change'
]

# Create feature matrix and target vector
X = df_features[feature_columns].dropna()
y = df_features.loc[X.index, 'failure']

print(f"Model training data shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Split data chronologically to avoid data leakage
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Handle imbalanced data
)

# Fit the model
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

# Gradient Boosting Model
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_probabilities = gb_model.predict_proba(X_test)[:, 1]

# XGBoost Model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Handle imbalance
)

xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]

# Model Performance Comparison
print("Model Performance Comparison:")
print("=" * 50)

models = {
    'Random Forest': (rf_predictions, rf_probabilities),
    'Gradient Boosting': (gb_predictions, gb_probabilities),
    'XGBoost': (xgb_predictions, xgb_probabilities)
}

for name, (predictions, probabilities) in models.items():
    auc_score = roc_auc_score(y_test, probabilities)
    print(f"\n{name}:")
    print(f"  AUC Score: {auc_score:.3f}")
    print(f"  Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Failure']))
```

Deep learning models, particularly Long Short-Term Memory (LSTM) networks and Convolutional Neural Networks (CNNs), offer powerful capabilities for modeling complex temporal patterns and extracting features from raw sensor data. These models are especially valuable for applications with rich time series data and complex failure modes.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Prepare time series data for LSTM
def create_sequences(data, target, sequence_length=24):
    """
    Create sequences for time series modeling
    """
    sequences = []
    targets = []
    
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i])
        targets.append(target[i])
    
    return np.array(sequences), np.array(targets)

# Scale features for neural network
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for LSTM
sequence_length = 24
X_seq, y_seq = create_sequences(X_scaled, y.values, sequence_length)

print(f"LSTM input shape: {X_seq.shape}")
print(f"LSTM target shape: {y_seq.shape}")

# Split sequences chronologically
train_size = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:train_size], X_seq[train_size:]
y_train_seq, y_test_seq = y_seq[:train_size], y_seq[train_size:]

# LSTM Model Architecture
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("LSTM Model Architecture:")
print(lstm_model.summary())

# Train LSTM model
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    batch_size=32,
    epochs=20,
    validation_data=(X_test_seq, y_test_seq),
    verbose=1
)

# LSTM predictions
lstm_probabilities = lstm_model.predict(X_test_seq).flatten()
lstm_predictions = (lstm_probabilities > 0.5).astype(int)

# CNN Model for 1D time series
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, X.shape[1])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train CNN model
cnn_history = cnn_model.fit(
    X_train_seq, y_train_seq,
    batch_size=32,
    epochs=15,
    validation_data=(X_test_seq, y_test_seq),
    verbose=1
)

# CNN predictions
cnn_probabilities = cnn_model.predict(X_test_seq).flatten()
cnn_predictions = (cnn_probabilities > 0.5).astype(int)

# Compare deep learning models
print("\nDeep Learning Model Performance:")
print("=" * 40)

lstm_auc = roc_auc_score(y_test_seq, lstm_probabilities)
cnn_auc = roc_auc_score(y_test_seq, cnn_probabilities)

print(f"LSTM AUC Score: {lstm_auc:.3f}")
print(f"CNN AUC Score: {cnn_auc:.3f}")
```

Anomaly detection models provide complementary capabilities to supervised learning approaches by identifying unusual patterns that might indicate developing equipment problems, even when specific failure modes haven't been observed in training data.

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# Prepare normal operation data for anomaly detection
normal_data = X[y == 0]  # Non-failure data
test_data = X_test

print(f"Normal operation data: {normal_data.shape}")
print(f"Test data for anomaly detection: {test_data.shape}")

# Isolation Forest
isolation_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of outliers
    random_state=42,
    n_estimators=100
)

isolation_forest.fit(normal_data)
if_anomaly_scores = isolation_forest.decision_function(test_data)
if_predictions = isolation_forest.predict(test_data)
if_predictions = (if_predictions == -1).astype(int)  # Convert to binary

# One-Class SVM
oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
oc_svm.fit(normal_data)
svm_anomaly_scores = oc_svm.decision_function(test_data)
svm_predictions = oc_svm.predict(test_data)
svm_predictions = (svm_predictions == -1).astype(int)

# Elliptic Envelope (Robust Covariance)
elliptic_env = EllipticEnvelope(contamination=0.1, random_state=42)
elliptic_env.fit(normal_data)
ee_anomaly_scores = elliptic_env.decision_function(test_data)
ee_predictions = elliptic_env.predict(test_data)
ee_predictions = (ee_predictions == -1).astype(int)

# Compare anomaly detection methods
print("Anomaly Detection Model Performance:")
print("=" * 40)

anomaly_models = {
    'Isolation Forest': (if_predictions, if_anomaly_scores),
    'One-Class SVM': (svm_predictions, svm_anomaly_scores),
    'Elliptic Envelope': (ee_predictions, ee_anomaly_scores)
}

for name, (predictions, scores) in anomaly_models.items():
    # Calculate metrics
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print(f"\n{name}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Detected Anomalies: {predictions.sum()}/{len(predictions)} ({predictions.mean():.1%})")

# Visualize anomaly scores
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(if_anomaly_scores, bins=50, alpha=0.7, label='All Data')
plt.hist(if_anomaly_scores[y_test == 1], bins=20, alpha=0.7, label='Failures')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Isolation Forest Scores')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(svm_anomaly_scores, bins=50, alpha=0.7, label='All Data')
plt.hist(svm_anomaly_scores[y_test == 1], bins=20, alpha=0.7, label='Failures')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('One-Class SVM Scores')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(ee_anomaly_scores, bins=50, alpha=0.7, label='All Data')
plt.hist(ee_anomaly_scores[y_test == 1], bins=20, alpha=0.7, label='Failures')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Elliptic Envelope Scores')
plt.legend()

plt.tight_layout()
plt.show()
```

## Remaining Useful Life (RUL) Prediction

Remaining Useful Life prediction represents one of the most valuable applications of machine learning in predictive maintenance, providing quantitative estimates of how much longer equipment can operate before failure or maintenance is required. RUL predictions enable optimal maintenance scheduling, parts ordering, and resource allocation while maximizing asset utilization.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_rul(failure_times, current_times):
    """
    Calculate Remaining Useful Life for training data
    """
    rul = []
    failure_indices = np.where(failure_times == 1)[0]
    
    for i, current_time in enumerate(current_times):
        # Find next failure after current time
        next_failures = failure_indices[failure_indices > i]
        if len(next_failures) > 0:
            rul.append(next_failures[0] - i)
        else:
            # No future failures in dataset - use maximum observed RUL
            rul.append(100)  # Set reasonable upper bound
    
    return np.array(rul)

# Calculate RUL for training data
train_rul = calculate_rul(y_train.values, range(len(y_train)))
test_rul = calculate_rul(y_test.values, range(len(y_test)))

print(f"RUL Statistics (Training):")
print(f"  Mean: {train_rul.mean():.1f} hours")
print(f"  Std: {train_rul.std():.1f} hours")
print(f"  Min: {train_rul.min():.1f} hours")
print(f"  Max: {train_rul.max():.1f} hours")

# Train RUL prediction models
rul_rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

rul_rf_model.fit(X_train, train_rul)
rul_rf_predictions = rul_rf_model.predict(X_test)

# Gradient Boosting for RUL
from sklearn.ensemble import GradientBoostingRegressor

rul_gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

rul_gb_model.fit(X_train, train_rul)
rul_gb_predictions = rul_gb_model.predict(X_test)

# Neural Network for RUL
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# RUL Neural Network
rul_nn_input = Input(shape=(X.shape[1],))
rul_nn_hidden1 = Dense(64, activation='relu')(rul_nn_input)
rul_nn_dropout1 = Dropout(0.3)(rul_nn_hidden1)
rul_nn_hidden2 = Dense(32, activation='relu')(rul_nn_dropout1)
rul_nn_dropout2 = Dropout(0.3)(rul_nn_hidden2)
rul_nn_output = Dense(1, activation='linear')(rul_nn_dropout2)

rul_nn_model = Model(inputs=rul_nn_input, outputs=rul_nn_output)
rul_nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Scale target variable for neural network
from sklearn.preprocessing import StandardScaler
rul_scaler = StandardScaler()
train_rul_scaled = rul_scaler.fit_transform(train_rul.reshape(-1, 1)).flatten()

# Train RUL neural network
rul_nn_history = rul_nn_model.fit(
    X_train, train_rul_scaled,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    verbose=0
)

# Make predictions and inverse transform
rul_nn_predictions_scaled = rul_nn_model.predict(X_test).flatten()
rul_nn_predictions = rul_scaler.inverse_transform(rul_nn_predictions_scaled.reshape(-1, 1)).flatten()

# Evaluate RUL models
print("\nRUL Prediction Model Performance:")
print("=" * 40)

rul_models = {
    'Random Forest': rul_rf_predictions,
    'Gradient Boosting': rul_gb_predictions,
    'Neural Network': rul_nn_predictions
}

for name, predictions in rul_models.items():
    mae = mean_absolute_error(test_rul, predictions)
    rmse = np.sqrt(mean_squared_error(test_rul, predictions))
    r2 = r2_score(test_rul, predictions)
    
    print(f"\n{name}:")
    print(f"  MAE: {mae:.2f} hours")
    print(f"  RMSE: {rmse:.2f} hours")
    print(f"  R²: {r2:.3f}")

# Visualize RUL predictions
plt.figure(figsize=(15, 10))

# Actual vs Predicted RUL
plt.subplot(2, 2, 1)
plt.scatter(test_rul[:500], rul_rf_predictions[:500], alpha=0.6, label='RF Predictions')
plt.plot([0, max(test_rul)], [0, max(test_rul)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual RUL (hours)')
plt.ylabel('Predicted RUL (hours)')
plt.title('Random Forest RUL Predictions')
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(test_rul[:500], rul_gb_predictions[:500], alpha=0.6, label='GB Predictions')
plt.plot([0, max(test_rul)], [0, max(test_rul)], 'r--', label='Perfect Prediction')
plt.xlabel('Actual RUL (hours)')
plt.ylabel('Predicted RUL (hours)')
plt.title('Gradient Boosting RUL Predictions')
plt.legend()

# RUL over time
plt.subplot(2, 2, 3)
time_indices = range(500)
plt.plot(time_indices, test_rul[:500], label='Actual RUL', alpha=0.8)
plt.plot(time_indices, rul_rf_predictions[:500], label='RF Predictions', alpha=0.8)
plt.plot(time_indices, rul_gb_predictions[:500], label='GB Predictions', alpha=0.8)
plt.xlabel('Time Index')
plt.ylabel('RUL (hours)')
plt.title('RUL Predictions Over Time')
plt.legend()

# Error distribution
plt.subplot(2, 2, 4)
rf_errors = test_rul - rul_rf_predictions
gb_errors = test_rul - rul_gb_predictions
plt.hist(rf_errors, bins=50, alpha=0.6, label='RF Errors')
plt.hist(gb_errors, bins=50, alpha=0.6, label='GB Errors')
plt.xlabel('Prediction Error (hours)')
plt.ylabel('Frequency')
plt.title('RUL Prediction Error Distribution')
plt.legend()

plt.tight_layout()
plt.show()
```

## Model Validation and Performance Assessment

Robust model validation is critical for ensuring that predictive maintenance models will perform reliably in production environments. Traditional cross-validation approaches may not be appropriate for time series data, requiring specialized validation techniques that respect temporal ordering and avoid data leakage.

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.patches as patches

def time_series_cross_validation(X, y, model, n_splits=5):
    """
    Perform time series cross validation with proper temporal ordering
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model on fold
        model.fit(X_fold_train, y_fold_train)
        
        # Make predictions
        fold_predictions = model.predict(X_fold_val)
        fold_probabilities = model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        cv_scores['accuracy'].append(accuracy_score(y_fold_val, fold_predictions))
        cv_scores['precision'].append(precision_score(y_fold_val, fold_predictions, zero_division=0))
        cv_scores['recall'].append(recall_score(y_fold_val, fold_predictions, zero_division=0))
        cv_scores['f1'].append(f1_score(y_fold_val, fold_predictions, zero_division=0))
        cv_scores['auc'].append(roc_auc_score(y_fold_val, fold_probabilities))
        
        print(f"Fold {fold + 1}: AUC = {cv_scores['auc'][-1]:.3f}, "
              f"Precision = {cv_scores['precision'][-1]:.3f}, "
              f"Recall = {cv_scores['recall'][-1]:.3f}")
    
    return cv_scores

# Perform time series cross validation
print("Time Series Cross Validation Results:")
print("=" * 50)

cv_results = time_series_cross_validation(X_train, y_train, 
                                        RandomForestClassifier(n_estimators=50, random_state=42))

print(f"\nCross Validation Summary:")
for metric, scores in cv_results.items():
    print(f"{metric.upper()}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# Business-focused performance metrics
def calculate_business_metrics(y_true, y_pred_proba, maintenance_cost=1000, 
                             downtime_cost=10000, threshold=0.5):
    """
    Calculate business-focused metrics for predictive maintenance
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
    
    # Business metrics
    total_maintenance_costs = (tp + fp) * maintenance_cost
    total_downtime_costs = fn * downtime_cost
    total_costs = total_maintenance_costs + total_downtime_costs
    
    # Cost savings vs reactive maintenance (where all failures cause downtime)
    reactive_costs = np.sum(y_true) * downtime_cost
    cost_savings = reactive_costs - total_costs
    
    metrics = {
        'total_costs': total_costs,
        'maintenance_costs': total_maintenance_costs,
        'downtime_costs': total_downtime_costs,
        'cost_savings': cost_savings,
        'savings_percentage': (cost_savings / reactive_costs) * 100 if reactive_costs > 0 else 0,
        'maintenance_actions': tp + fp,
        'prevented_failures': tp,
        'missed_failures': fn,
        'unnecessary_maintenance': fp
    }
    
    return metrics

# Calculate business metrics for different thresholds
thresholds = np.arange(0.1, 1.0, 0.1)
business_results = []

for threshold in thresholds:
    metrics = calculate_business_metrics(y_test, rf_probabilities, threshold=threshold)
    metrics['threshold'] = threshold
    business_results.append(metrics)

business_df = pd.DataFrame(business_results)

print("\nBusiness Impact Analysis:")
print("=" * 30)
optimal_threshold = business_df.loc[business_df['cost_savings'].idxmax(), 'threshold']
print(f"Optimal threshold: {optimal_threshold:.1f}")
print(f"Maximum cost savings: ${business_df['cost_savings'].max():,.0f}")
print(f"Savings percentage: {business_df.loc[business_df['cost_savings'].idxmax(), 'savings_percentage']:.1f}%")

# Visualize business metrics
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(business_df['threshold'], business_df['total_costs'], 'b-', linewidth=2)
plt.xlabel('Prediction Threshold')
plt.ylabel('Total Costs ($)')
plt.title('Total Costs vs Threshold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(business_df['threshold'], business_df['cost_savings'], 'g-', linewidth=2)
plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_threshold:.1f}')
plt.xlabel('Prediction Threshold')
plt.ylabel('Cost Savings ($)')
plt.title('Cost Savings vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.plot(business_df['threshold'], business_df['maintenance_actions'], 'orange', linewidth=2)
plt.xlabel('Prediction Threshold')
plt.ylabel('Number of Maintenance Actions')
plt.title('Maintenance Actions vs Threshold')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.plot(business_df['threshold'], business_df['prevented_failures'], 'g-', linewidth=2, label='Prevented')
plt.plot(business_df['threshold'], business_df['missed_failures'], 'r-', linewidth=2, label='Missed')
plt.xlabel('Prediction Threshold')
plt.ylabel('Number of Failures')
plt.title('Prevented vs Missed Failures')
plt.legend()
plt.grid(True, alpha=0.3)

# ROC and Precision-Recall curves
plt.subplot(2, 3, 5)
fpr, tpr, _ = roc_curve(y_test, rf_probabilities)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_test, rf_probabilities):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
precision, recall, _ = precision_recall_curve(y_test, rf_probabilities)
ap_score = average_precision_score(y_test, rf_probabilities)
plt.plot(recall, precision, linewidth=2, label=f'PR (AP = {ap_score:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Real-Time Implementation and Deployment Strategies

Deploying machine learning models for predictive maintenance in production environments requires careful consideration of real-time data processing, model serving infrastructure, and integration with existing maintenance management systems. The deployment architecture must handle streaming sensor data, provide low-latency predictions, and scale to support multiple assets simultaneously.

```python
import pickle
import json
from datetime import datetime, timedelta
import threading
import queue

class PredictiveMaintenanceSystem:
    """
    Production-ready predictive maintenance system
    """
    def __init__(self, model, feature_columns, threshold=0.5):
        self.model = model
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.data_buffer = {}
        self.predictions = {}
        self.alerts = []
        
    def preprocess_sensor_data(self, sensor_data):
        """
        Preprocess incoming sensor data for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([sensor_data])
        
        # Calculate rolling features (simplified for real-time)
        # In production, maintain rolling windows in memory
        if len(self.data_buffer) > 0:
            # Add to buffer for rolling calculations
            for key, value in sensor_data.items():
                if key not in self.data_buffer:
                    self.data_buffer[key] = []
                self.data_buffer[key].append(value)
                
                # Keep only last 24 hours of data
                if len(self.data_buffer[key]) > 24:
                    self.data_buffer[key] = self.data_buffer[key][-24:]
        else:
            # Initialize buffer
            for key, value in sensor_data.items():
                self.data_buffer[key] = [value]
        
        # Calculate rolling features
        for key in ['temperature', 'vibration']:
            if key in self.data_buffer and len(self.data_buffer[key]) > 1:
                values = np.array(self.data_buffer[key])
                df[f'{key}_rolling_mean'] = values.mean()
                df[f'{key}_rolling_std'] = values.std() if len(values) > 1 else 0
                df[f'{key}_rate_change'] = values[-1] - values[-2] if len(values) > 1 else 0
            else:
                df[f'{key}_rolling_mean'] = sensor_data.get(key, 0)
                df[f'{key}_rolling_std'] = 0
                df[f'{key}_rate_change'] = 0
        
        # Add missing columns with default values
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select and order features
        features = df[self.feature_columns].fillna(0)
        
        return features
    
    def predict_failure_probability(self, sensor_data):
        """
        Predict failure probability from sensor data
        """
        try:
            # Preprocess data
            features = self.preprocess_sensor_data(sensor_data)
            
            # Make prediction
            probability = self.model.predict_proba(features)[0, 1]
            prediction = int(probability >= self.threshold)
            
            # Store prediction
            timestamp = datetime.now()
            self.predictions[timestamp] = {
                'probability': probability,
                'prediction': prediction,
                'sensor_data': sensor_data
            }
            
            # Generate alert if necessary
            if prediction == 1:
                alert = {
                    'timestamp': timestamp,
                    'asset_id': sensor_data.get('asset_id', 'unknown'),
                    'probability': probability,
                    'alert_level': 'HIGH' if probability > 0.8 else 'MEDIUM',
                    'message': f'Failure predicted with {probability:.1%} probability'
                }
                self.alerts.append(alert)
                return alert
            
            return {
                'timestamp': timestamp,
                'probability': probability,
                'prediction': prediction,
                'status': 'NORMAL'
            }
            
        except Exception as e:
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'status': 'ERROR'
            }
    
    def get_recent_predictions(self, hours=24):
        """
        Get predictions from the last N hours
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_predictions = {
            timestamp: pred for timestamp, pred in self.predictions.items()
            if timestamp > cutoff_time
        }
        return recent_predictions
    
    def get_active_alerts(self):
        """
        Get active alerts
        """
        return self.alerts[-10:]  # Return last 10 alerts

# Initialize production system
production_system = PredictiveMaintenanceSystem(
    model=rf_model,
    feature_columns=feature_columns,
    threshold=optimal_threshold
)

# Simulate real-time sensor data stream
def simulate_sensor_stream():
    """
    Simulate streaming sensor data
    """
    print("Starting sensor data simulation...")
    
    for i in range(50):  # Simulate 50 data points
        # Generate realistic sensor data with some variability
        base_temp = 75 + np.random.normal(0, 5)
        base_vibration = 0.5 + np.random.normal(0, 0.1)
        
        # Occasionally inject concerning values
        if np.random.random() < 0.1:  # 10% chance of concerning readings
            base_temp += np.random.uniform(10, 20)
            base_vibration += np.random.uniform(0.2, 0.5)
        
        sensor_data = {
            'asset_id': 'PUMP_001',
            'temperature': base_temp,
            'vibration': abs(base_vibration),
            'pressure': np.random.normal(100, 10),
            'rotation_speed': np.random.normal(1800, 30),
            'oil_viscosity': np.random.normal(40, 3),
            'operating_hours': i * 100,  # Simulated operating hours
            'load_factor': np.random.uniform(0.5, 0.9)
        }
        
        # Make prediction
        result = production_system.predict_failure_probability(sensor_data)
        
        # Print results
        if 'error' in result:
            print(f"Step {i+1}: ERROR - {result['error']}")
        elif result['status'] == 'NORMAL':
            print(f"Step {i+1}: NORMAL (Probability: {result['probability']:.3f})")
        else:
            print(f"Step {i+1}: ALERT - {result['message']} (Level: {result['alert_level']})")
        
        # Simulate time delay
        import time
        time.sleep(0.1)  # 100ms delay
    
    print(f"\nSimulation complete!")
    print(f"Total alerts generated: {len(production_system.alerts)}")

# Run simulation
simulate_sensor_stream()

# Display system status
print("\nSystem Status Summary:")
print("=" * 30)
recent_preds = production_system.get_recent_predictions()
if recent_preds:
    probabilities = [pred['probability'] for pred in recent_preds.values()]
    print(f"Recent predictions: {len(recent_preds)}")
    print(f"Average failure probability: {np.mean(probabilities):.3f}")
    print(f"Maximum failure probability: {np.max(probabilities):.3f}")

active_alerts = production_system.get_active_alerts()
if active_alerts:
    print(f"\nActive alerts: {len(active_alerts)}")
    for alert in active_alerts[-3:]:  # Show last 3 alerts
        print(f"  {alert['timestamp'].strftime('%H:%M:%S')} - {alert['alert_level']}: {alert['message']}")
```

## Model Monitoring and Maintenance

Production predictive maintenance models require continuous monitoring to ensure they maintain accuracy over time and adapt to changing operational conditions. Model performance can degrade due to concept drift, data quality issues, or changes in equipment behavior patterns.

```python
class ModelMonitor:
    """
    Monitor model performance and data drift in production
    """
    def __init__(self, reference_data, model, feature_columns):
        self.reference_data = reference_data
        self.model = model
        self.feature_columns = feature_columns
        self.performance_history = []
        self.drift_history = []
        
    def calculate_data_drift(self, new_data, method='ks_test'):
        """
        Calculate data drift using statistical tests
        """
        from scipy import stats
        
        drift_results = {}
        
        for feature in self.feature_columns:
            if feature in new_data.columns and feature in self.reference_data.columns:
                ref_values = self.reference_data[feature].dropna()
                new_values = new_data[feature].dropna()
                
                if len(ref_values) > 0 and len(new_values) > 0:
                    if method == 'ks_test':
                        # Kolmogorov-Smirnov test
                        statistic, p_value = stats.ks_2samp(ref_values, new_values)
                        drift_results[feature] = {
                            'statistic': statistic,
                            'p_value': p_value,
                            'drift_detected': p_value < 0.05
                        }
                    elif method == 'psi':
                        # Population Stability Index
                        psi = self.calculate_psi(ref_values, new_values)
                        drift_results[feature] = {
                            'psi': psi,
                            'drift_detected': psi > 0.25
                        }
        
        return drift_results
    
    def calculate_psi(self, expected, actual, buckets=10):
        """
        Calculate Population Stability Index
        """
        # Create buckets based on expected distribution
        breakpoints = np.arange(0, buckets + 1) / buckets * 100
        breakpoints = np.percentile(expected, breakpoints)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate expected and actual distributions
        expected_counts = np.histogram(expected, breakpoints)[0]
        actual_counts = np.histogram(actual, breakpoints)[0]
        
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_percents = np.maximum(expected_percents, 0.0001)
        actual_percents = np.maximum(actual_percents, 0.0001)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return psi
    
    def evaluate_model_performance(self, new_data, new_labels):
        """
        Evaluate current model performance on new data
        """
        if len(new_data) == 0 or len(new_labels) == 0:
            return None
        
        try:
            # Make predictions
            predictions = self.model.predict(new_data)
            probabilities = self.model.predict_proba(new_data)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            performance = {
                'timestamp': datetime.now(),
                'sample_size': len(new_data),
                'accuracy': accuracy_score(new_labels, predictions),
                'precision': precision_score(new_labels, predictions, zero_division=0),
                'recall': recall_score(new_labels, predictions, zero_division=0),
                'f1_score': f1_score(new_labels, predictions, zero_division=0),
                'auc_score': roc_auc_score(new_labels, probabilities) if len(np.unique(new_labels)) > 1 else 0
            }
            
            self.performance_history.append(performance)
            return performance
            
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def generate_monitoring_report(self, new_data, new_labels=None):
        """
        Generate comprehensive monitoring report
        """
        report = {
            'timestamp': datetime.now(),
            'data_summary': {
                'samples': len(new_data),
                'features': len(self.feature_columns),
                'missing_values': new_data.isnull().sum().sum(),
                'duplicate_rows': new_data.duplicated().sum()
            }
        }
        
        # Data drift analysis
        drift_results = self.calculate_data_drift(new_data)
        drift_features = [f for f, result in drift_results.items() if result.get('drift_detected', False)]
        
        report['data_drift'] = {
            'features_with_drift': len(drift_features),
            'total_features_analyzed': len(drift_results),
            'drift_percentage': len(drift_features) / len(drift_results) * 100 if drift_results else 0,
            'drifted_features': drift_features,
            'drift_details': drift_results
        }
        
        # Model performance (if labels available)
        if new_labels is not None:
            performance = self.evaluate_model_performance(new_data, new_labels)
            report['model_performance'] = performance
        
        # Generate recommendations
        recommendations = []
        
        if report['data_drift']['drift_percentage'] > 30:
            recommendations.append("HIGH PRIORITY: Significant data drift detected. Consider model retraining.")
        elif report['data_drift']['drift_percentage'] > 15:
            recommendations.append("MEDIUM PRIORITY: Moderate data drift detected. Monitor closely.")
        
        if report['data_summary']['missing_values'] > len(new_data) * 0.1:
            recommendations.append("Data quality issue: High missing value rate detected.")
        
        if new_labels is not None and 'model_performance' in report:
            if report['model_performance'].get('auc_score', 0) < 0.7:
                recommendations.append("CRITICAL: Model performance degradation detected.")
        
        report['recommendations'] = recommendations
        
        return report

# Initialize model monitor
monitor = ModelMonitor(
    reference_data=X_train,
    model=rf_model,
    feature_columns=feature_columns
)

# Simulate model monitoring with new data
print("Model Monitoring Simulation:")
print("=" * 40)

# Generate some new data with drift
np.random.seed(123)
n_new = 1000

# Simulate data drift by shifting distributions
new_data = {
    'temperature': np.random.normal(80, 12, n_new),  # Temperature drift
    'vibration': np.random.normal(0.7, 0.3, n_new),  # Vibration drift
    'pressure': np.random.normal(98, 16, n_new),
    'rotation_speed': np.random.normal(1750, 60, n_new),  # Speed drift
    'oil_viscosity': np.random.normal(38, 6, n_new),
    'operating_hours': np.random.uniform(1000, 9000, n_new),
    'load_factor': np.random.uniform(0.2, 1.0, n_new)
}

# Add engineered features (simplified)
new_data['temp_rolling_mean'] = new_data['temperature']
new_data['temp_rolling_std'] = np.random.normal(2, 0.5, n_new)
new_data['vibration_rolling_mean'] = new_data['vibration'] 
new_data['vibration_rolling_max'] = new_data['vibration'] * 1.2
new_data['temp_rate_change'] = np.random.normal(0, 1, n_new)
new_data['vibration_rate_change'] = np.random.normal(0, 0.1, n_new)

new_df = pd.DataFrame(new_data)

# Generate synthetic labels for the new data
new_failure_prob = (
    (np.abs(new_data['temperature'] - 75) / 50) +
    (np.abs(new_data['vibration'] - 0.5) / 1.0) +
    np.random.normal(0, 0.15, n_new)
)
new_labels = (new_failure_prob > np.percentile(new_failure_prob, 85)).astype(int)

# Generate monitoring report
monitoring_report = monitor.generate_monitoring_report(new_df, new_labels)

print("Monitoring Report Summary:")
print(f"Timestamp: {monitoring_report['timestamp']}")
print(f"Data samples analyzed: {monitoring_report['data_summary']['samples']}")
print(f"Features with drift: {monitoring_report['data_drift']['features_with_drift']}")
print(f"Drift percentage: {monitoring_report['data_drift']['drift_percentage']:.1f}%")

if 'model_performance' in monitoring_report:
    perf = monitoring_report['model_performance']
    print(f"Current model AUC: {perf['auc_score']:.3f}")
    print(f"Current model accuracy: {perf['accuracy']:.3f}")

print(f"\nRecommendations:")
for i, rec in enumerate(monitoring_report['recommendations'], 1):
    print(f"{i}. {rec}")

# Visualize drift detection results
plt.figure(figsize=(15, 10))

drift_features = ['temperature', 'vibration', 'rotation_speed']
for i, feature in enumerate(drift_features):
    plt.subplot(2, 2, i + 1)
    
    # Plot distributions
    plt.hist(X_train[feature].dropna(), bins=50, alpha=0.6, label='Reference Data', density=True)
    plt.hist(new_df[feature].dropna(), bins=50, alpha=0.6, label='New Data', density=True)
    
    # Add drift detection results
    drift_result = monitoring_report['data_drift']['drift_details'].get(feature, {})
    if 'p_value' in drift_result:
        drift_status = 'DRIFT DETECTED' if drift_result['drift_detected'] else 'NO DRIFT'
        plt.title(f'{feature.title()} Distribution\n{drift_status} (p={drift_result["p_value"]:.4f})')
    else:
        plt.title(f'{feature.title()} Distribution')
    
    plt.xlabel(feature.title())
    plt.ylabel('Density')
    plt.legend()

# Performance trend visualization
plt.subplot(2, 2, 4)
if monitor.performance_history:
    timestamps = [p['timestamp'] for p in monitor.performance_history]
    auc_scores = [p['auc_score'] for p in monitor.performance_history]
    plt.plot(timestamps, auc_scores, 'o-')
    plt.title('Model Performance Over Time')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
else:
    # Show single performance point
    if 'model_performance' in monitoring_report:
        perf = monitoring_report['model_performance']
        plt.bar(['AUC', 'Accuracy', 'Precision', 'Recall'], 
               [perf['auc_score'], perf['accuracy'], perf['precision'], perf['recall']])
        plt.title('Current Model Performance')
        plt.ylabel('Score')
        plt.ylim(0, 1)

plt.tight_layout()
plt.show()
```

## Advanced Implementation Strategies

Advanced implementation strategies for machine learning-powered predictive maintenance encompass sophisticated approaches to handle complex industrial environments, multi-asset optimization, and integration with enterprise systems.

```python
# Multi-Asset Predictive Maintenance System
class MultiAssetPredictiveSystem:
    """
    Advanced system for managing multiple assets with different characteristics
    """
    def __init__(self):
        self.asset_models = {}
        self.asset_configs = {}
        self.global_optimizer = None
        
    def register_asset(self, asset_id, asset_type, model, config):
        """
        Register a new asset with its specific model and configuration
        """
        self.asset_models[asset_id] = {
            'type': asset_type,
            'model': model,
            'config': config,
            'last_prediction': None,
            'prediction_history': [],
            'maintenance_schedule': []
        }
    
    def predict_asset_failure(self, asset_id, sensor_data):
        """
        Make failure prediction for specific asset
        """
        if asset_id not in self.asset_models:
            return {'error': f'Asset {asset_id} not registered'}
        
        asset_info = self.asset_models[asset_id]
        model = asset_info['model']
        config = asset_info['config']
        
        try:
            # Preprocess data according to asset-specific configuration
            features = self.preprocess_asset_data(sensor_data, config)
            
            # Make prediction
            probability = model.predict_proba(features)[0, 1]
            prediction = int(probability >= config.get('threshold', 0.5))
            
            # Store prediction
            prediction_result = {
                'timestamp': datetime.now(),
                'asset_id': asset_id,
                'probability': probability,
                'prediction': prediction,
                'confidence': self.calculate_prediction_confidence(features, model)
            }
            
            asset_info['last_prediction'] = prediction_result
            asset_info['prediction_history'].append(prediction_result)
            
            return prediction_result
            
        except Exception as e:
            return {'error': str(e), 'asset_id': asset_id}
    
    def preprocess_asset_data(self, sensor_data, config):
        """
        Asset-specific data preprocessing
        """
        # Convert to DataFrame
        df = pd.DataFrame([sensor_data])
        
        # Apply asset-specific feature engineering
        feature_columns = config.get('feature_columns', [])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = config.get('default_values', {}).get(col, 0)
        
        return df[feature_columns].fillna(0)
    
    def calculate_prediction_confidence(self, features, model):
        """
        Calculate prediction confidence based on model uncertainty
        """
        try:
            # For ensemble methods, use prediction variance
            if hasattr(model, 'estimators_'):
                # Get predictions from all trees/estimators
                predictions = np.array([tree.predict_proba(features)[0, 1] 
                                      for tree in model.estimators_])
                confidence = 1.0 - np.std(predictions)  # Lower variance = higher confidence
                return max(0.0, min(1.0, confidence))
            else:
                # Default confidence calculation
                proba = model.predict_proba(features)[0]
                return float(max(proba))  # Maximum class probability
        except:
            return 0.5  # Default moderate confidence
    
    def optimize_maintenance_schedule(self, time_horizon_days=30):
        """
        Global optimization of maintenance schedule across all assets
        """
        maintenance_plan = []
        
        for asset_id, asset_info in self.asset_models.items():
            if asset_info['last_prediction']:
                pred = asset_info['last_prediction']
                
                if pred['prediction'] == 1:  # Failure predicted
                    # Calculate optimal maintenance timing
                    urgency_score = pred['probability'] * pred['confidence']
                    
                    # Consider asset criticality and maintenance windows
                    config = asset_info['config']
                    criticality = config.get('criticality', 1.0)
                    
                    maintenance_priority = urgency_score * criticality
                    
                    maintenance_plan.append({
                        'asset_id': asset_id,
                        'priority': maintenance_priority,
                        'failure_probability': pred['probability'],
                        'confidence': pred['confidence'],
                        'recommended_action': self.determine_maintenance_action(asset_info),
                        'estimated_cost': config.get('maintenance_cost', 1000),
                        'downtime_risk': config.get('downtime_cost', 5000) * pred['probability']
                    })
        
        # Sort by priority
        maintenance_plan.sort(key=lambda x: x['priority'], reverse=True)
        
        return maintenance_plan
    
    def determine_maintenance_action(self, asset_info):
        """
        Determine specific maintenance action based on asset type and condition
        """
        asset_type = asset_info['type']
        prediction = asset_info['last_prediction']
        
        action_map = {
            'pump': {
                0.5: 'Inspect bearings and seals',
                0.7: 'Replace seals and check alignment',
                0.9: 'Complete overhaul recommended'
            },
            'motor': {
                0.5: 'Check electrical connections',
                0.7: 'Inspect and replace bearings',
                0.9: 'Motor replacement required'
            },
            'compressor': {
                0.5: 'Valve inspection and cleaning',
                0.7: 'Replace filters and check pressure',
                0.9: 'Complete system overhaul'
            }
        }
        
        probability = prediction['probability']
        actions = action_map.get(asset_type, {})
        
        for threshold in sorted(actions.keys(), reverse=True):
            if probability >= threshold:
                return actions[threshold]
        
        return 'Continue monitoring'

# Initialize multi-asset system
multi_asset_system = MultiAssetPredictiveSystem()

# Register different types of assets
asset_configs = {
    'PUMP_001': {
        'type': 'pump',
        'feature_columns': feature_columns,
        'threshold': 0.6,
        'criticality': 0.9,
        'maintenance_cost': 1500,
        'downtime_cost': 8000
    },
    'MOTOR_002': {
        'type': 'motor', 
        'feature_columns': feature_columns,
        'threshold': 0.7,
        'criticality': 0.8,
        'maintenance_cost': 2000,
        'downtime_cost': 12000
    },
    'COMPRESSOR_003': {
        'type': 'compressor',
        'feature_columns': feature_columns, 
        'threshold': 0.5,
        'criticality': 1.0,
        'maintenance_cost': 5000,
        'downtime_cost': 25000
    }
}

# Register assets with trained models
for asset_id, config in asset_configs.items():
    multi_asset_system.register_asset(asset_id, config['type'], rf_model, config)

# Simulate multi-asset monitoring
print("Multi-Asset Predictive Maintenance Simulation:")
print("=" * 50)

# Generate predictions for each asset
for asset_id in asset_configs.keys():
    # Generate asset-specific sensor data
    sensor_data = {
        'temperature': np.random.normal(78 if 'PUMP' in asset_id else 82, 8),
        'vibration': np.random.normal(0.6 if 'MOTOR' in asset_id else 0.4, 0.15),
        'pressure': np.random.normal(105 if 'COMPRESSOR' in asset_id else 95, 12),
        'rotation_speed': np.random.normal(1850 if 'MOTOR' in asset_id else 1780, 40),
        'oil_viscosity': np.random.normal(42, 4),
        'operating_hours': np.random.uniform(2000, 7000),
        'load_factor': np.random.uniform(0.4, 0.95),
        'temp_rolling_mean': 0, 'temp_rolling_std': 0, 'vibration_rolling_mean': 0,
        'vibration_rolling_max': 0, 'temp_rate_change': 0, 'vibration_rate_change': 0
    }
    
    # Add some variation to simulate different asset conditions
    if np.random.random() < 0.3:  # 30% chance of concerning readings
        sensor_data['temperature'] += np.random.uniform(5, 15)
        sensor_data['vibration'] += np.random.uniform(0.1, 0.4)
    
    # Make prediction
    result = multi_asset_system.predict_asset_failure(asset_id, sensor_data)
    
    if 'error' not in result:
        print(f"{asset_id}: Failure Probability = {result['probability']:.3f}, "
              f"Confidence = {result['confidence']:.3f}, "
              f"Prediction = {'FAILURE' if result['prediction'] else 'NORMAL'}")

# Generate optimized maintenance schedule
print("\nOptimized Maintenance Schedule:")
print("=" * 35)

maintenance_plan = multi_asset_system.optimize_maintenance_schedule()

for i, item in enumerate(maintenance_plan, 1):
    print(f"\n{i}. Asset: {item['asset_id']}")
    print(f"   Priority Score: {item['priority']:.3f}")
    print(f"   Failure Probability: {item['failure_probability']:.3f}")
    print(f"   Recommended Action: {item['recommended_action']}")
    print(f"   Estimated Cost: ${item['estimated_cost']:,}")
    print(f"   Downtime Risk: ${item['downtime_risk']:,.0f}")

# Calculate total maintenance costs and savings
if maintenance_plan:
    total_maintenance_cost = sum(item['estimated_cost'] for item in maintenance_plan)
    total_downtime_risk = sum(item['downtime_risk'] for item in maintenance_plan)
    total_potential_savings = sum(asset_configs[item['asset_id']]['downtime_cost'] 
                                for item in maintenance_plan) - total_downtime_risk
    
    print(f"\nMaintenance Plan Summary:")
    print(f"Total Maintenance Investment: ${total_maintenance_cost:,}")
    print(f"Total Downtime Risk Avoided: ${total_potential_savings:,.0f}")
    print(f"ROI: {(total_potential_savings / total_maintenance_cost - 1) * 100:.1f}%")
```

## Integration with Enterprise Systems

Real-world deployment of predictive maintenance systems requires seamless integration with existing enterprise systems including CMMS (Computerized Maintenance Management Systems), ERP (Enterprise Resource Planning), and SCADA (Supervisory Control and Data Acquisition) systems.

```python
import requests
import json
from datetime import datetime
import sqlite3

class EnterpriseIntegration:
    """
    Integration layer for connecting predictive maintenance with enterprise systems
    """
    def __init__(self, config):
        self.config = config
        self.db_connection = self.setup_database()
        
    def setup_database(self):
        """
        Setup local database for storing predictions and maintenance records
        """
        conn = sqlite3.connect(':memory:')  # In-memory database for demo
        
        # Create tables
        conn.execute('''
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                asset_id TEXT,
                failure_probability REAL,
                prediction INTEGER,
                confidence REAL,
                sensor_data TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE maintenance_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                asset_id TEXT,
                action_type TEXT,
                description TEXT,
                cost REAL,
                priority_score REAL,
                status TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                asset_id TEXT,
                alert_level TEXT,
                message TEXT,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        return conn
    
    def log_prediction(self, prediction_result):
        """
        Log prediction to database and enterprise systems
        """
        # Store in local database
        self.db_connection.execute('''
            INSERT INTO predictions (timestamp, asset_id, failure_probability, 
                                   prediction, confidence, sensor_data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            prediction_result['timestamp'].isoformat(),
            prediction_result['asset_id'],
            prediction_result['probability'],
            prediction_result['prediction'],
            prediction_result['confidence'],
            json.dumps(prediction_result.get('sensor_data', {}))
        ))
        self.db_connection.commit()
        
        # Send to CMMS system (simulated)
        if prediction_result['prediction'] == 1:
            self.create_work_order(prediction_result)
    
    def create_work_order(self, prediction_result):
        """
        Create work order in CMMS system
        """
        work_order = {
            'asset_id': prediction_result['asset_id'],
            'work_type': 'Predictive Maintenance',
            'priority': self.calculate_priority(prediction_result),
            'description': f"Predictive maintenance required. Failure probability: {prediction_result['probability']:.1%}",
            'estimated_hours': self.estimate_maintenance_hours(prediction_result),
            'created_by': 'Predictive Maintenance System',
            'created_date': prediction_result['timestamp'].isoformat()
        }
        
        # Simulate CMMS API call
        print(f"Creating work order for {prediction_result['asset_id']}")
        print(f"  Priority: {work_order['priority']}")
        print(f"  Description: {work_order['description']}")
        
        # Log to database
        self.db_connection.execute('''
            INSERT INTO maintenance_actions (timestamp, asset_id, action_type, 
                                           description, priority_score, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prediction_result['asset_id'],
            'Predictive Maintenance',
            work_order['description'],
            prediction_result['probability'] * prediction_result['confidence'],
            'Pending'
        ))
        self.db_connection.commit()
    
    def calculate_priority(self, prediction_result):
        """
        Calculate work order priority based on prediction
        """
        score = prediction_result['probability'] * prediction_result['confidence']
        
        if score >= 0.8:
            return 'Critical'
        elif score >= 0.6:
            return 'High'
        elif score >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def estimate_maintenance_hours(self, prediction_result):
        """
        Estimate maintenance hours based on asset type and failure probability
        """
        base_hours = {
            'pump': 4,
            'motor': 6,
            'compressor': 12
        }
        
        asset_type = prediction_result.get('asset_type', 'pump')
        base = base_hours.get(asset_type, 4)
        
        # Scale by failure probability
        multiplier = 1 + prediction_result['probability']
        
        return int(base * multiplier)
    
    def send_alert_notification(self, alert_data):
        """
        Send alert notifications through various channels
        """
        # Log alert
        self.db_connection.execute('''
            INSERT INTO alerts (timestamp, asset_id, alert_level, message)
            VALUES (?, ?, ?, ?)
        ''', (
            alert_data['timestamp'].isoformat(),
            alert_data['asset_id'],
            alert_data['alert_level'],
            alert_data['message']
        ))
        self.db_connection.commit()
        
        # Send notifications (simulated)
        if alert_data['alert_level'] in ['Critical', 'High']:
            self.send_email_alert(alert_data)
            self.send_sms_alert(alert_data)
        else:
            self.send_dashboard_notification(alert_data)
    
    def send_email_alert(self, alert_data):
        """
        Send email alert (simulated)
        """
        print(f"📧 EMAIL ALERT: {alert_data['message']}")
    
    def send_sms_alert(self, alert_data):
        """
        Send SMS alert (simulated) 
        """
        print(f"📱 SMS ALERT: {alert_data['asset_id']} - {alert_data['alert_level']}")
    
    def send_dashboard_notification(self, alert_data):
        """
        Send dashboard notification (simulated)
        """
        print(f"🖥️ DASHBOARD: {alert_data['message']}")
    
    def get_maintenance_report(self, days=30):
        """
        Generate maintenance report for management
        """
        cursor = self.db_connection.cursor()
        
        # Get recent predictions
        cursor.execute('''
            SELECT asset_id, COUNT(*) as prediction_count, 
                   AVG(failure_probability) as avg_probability,
                   MAX(failure_probability) as max_probability
            FROM predictions 
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
            GROUP BY asset_id
        '''.format(days))
        
        prediction_summary = cursor.fetchall()
        
        # Get maintenance actions
        cursor.execute('''
            SELECT status, COUNT(*) as count
            FROM maintenance_actions
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
            GROUP BY status
        '''.format(days))
        
        action_summary = cursor.fetchall()
        
        # Get alerts
        cursor.execute('''
            SELECT alert_level, COUNT(*) as count
            FROM alerts
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
            GROUP BY alert_level
        '''.format(days))
        
        alert_summary = cursor.fetchall()
        
        report = {
            'period_days': days,
            'generated_at': datetime.now().isoformat(),
            'prediction_summary': prediction_summary,
            'action_summary': action_summary,
            'alert_summary': alert_summary
        }
        
        return report

# Initialize enterprise integration
enterprise_config = {
    'cmms_endpoint': 'https://api.cmms-system.com',
    'erp_endpoint': 'https://api.erp-system.com',
    'notification_email': 'maintenance@company.com',
    'sms_service': 'twilio'
}

enterprise_integration = EnterpriseIntegration(enterprise_config)

# Simulate enterprise integration workflow
print("Enterprise Integration Simulation:")
print("=" * 40)

# Simulate multiple asset predictions
for i in range(5):
    asset_id = f"ASSET_{i+1:03d}"
    
    # Generate prediction result
    prediction_result = {
        'timestamp': datetime.now(),
        'asset_id': asset_id,
        'probability': np.random.uniform(0.2, 0.95),
        'prediction': np.random.choice([0, 1], p=[0.7, 0.3]),
        'confidence': np.random.uniform(0.6, 0.95),
        'asset_type': np.random.choice(['pump', 'motor', 'compressor']),
        'sensor_data': {
            'temperature': np.random.normal(80, 10),
            'vibration': np.random.normal(0.5, 0.2)
        }
    }
    
    # Log prediction
    enterprise_integration.log_prediction(prediction_result)
    
    # Generate alert if needed
    if prediction_result['prediction'] == 1:
        alert_data = {
            'timestamp': datetime.now(),
            'asset_id': asset_id,
            'alert_level': enterprise_integration.calculate_priority(prediction_result),
            'message': f"Failure predicted for {asset_id} with {prediction_result['probability']:.1%} probability"
        }
        enterprise_integration.send_alert_notification(alert_data)

print("\n" + "="*40)

# Generate and display management report
report = enterprise_integration.get_maintenance_report(days=7)

print("Management Report (Last 7 Days):")
print(f"Generated: {report['generated_at']}")
print(f"\nPrediction Summary by Asset:")
for asset_id, count, avg_prob, max_prob in report['prediction_summary']:
    print(f"  {asset_id}: {count} predictions, avg probability: {avg_prob:.3f}, max: {max_prob:.3f}")

print(f"\nMaintenance Actions by Status:")
for status, count in report['action_summary']:
    print(f"  {status}: {count}")

print(f"\nAlerts by Level:")
for level, count in report['alert_summary']:
    print(f"  {level}: {count}")

# Close database connection
enterprise_integration.db_connection.close()
```

## Conclusion and Future Directions

The implementation of advanced machine learning techniques in predictive maintenance represents a transformative approach to asset management that extends far beyond traditional condition monitoring. Through the comprehensive examples and code implementations presented, we have demonstrated how sophisticated algorithms can extract meaningful patterns from complex sensor data, predict equipment failures with remarkable accuracy, and optimize maintenance strategies to maximize both operational efficiency and cost effectiveness.

The journey from basic statistical monitoring to AI-powered predictive systems encompasses multiple technological layers, each contributing essential capabilities to the overall solution. Feature engineering transforms raw sensor data into meaningful indicators of equipment health, while machine learning algorithms identify complex patterns that would be impossible for human analysts to detect consistently. Real-time implementation architectures enable these insights to be delivered when and where they are needed most, while enterprise integration ensures that predictive insights translate into actionable maintenance decisions.

The code examples provided demonstrate not only the technical implementation of these concepts but also the practical considerations necessary for successful deployment in production environments. From handling data quality issues and managing model drift to optimizing business metrics and integrating with existing enterprise systems, these implementations address the real-world challenges that determine the success or failure of predictive maintenance initiatives.

Looking toward the future, several emerging trends promise to further enhance the capabilities of machine learning-powered predictive maintenance systems. Edge computing will enable more sophisticated real-time processing at the asset level, reducing latency and improving responsiveness. Federated learning approaches will allow organizations to benefit from collective insights while maintaining data privacy and security. Advanced AI techniques, including transformer architectures and graph neural networks, may unlock new capabilities for modeling complex equipment interactions and failure propagation patterns.

The integration of digital twin technologies with predictive maintenance systems will enable virtual testing of maintenance strategies and optimization of interventions before they are implemented on physical assets. Quantum computing may eventually provide computational capabilities that enable solving previously intractable optimization problems in maintenance scheduling and resource allocation.

Perhaps most importantly, the continued evolution of predictive maintenance will be driven by the growing availability of data, the increasing sophistication of sensor technologies, and the expanding capabilities of machine learning algorithms. Organizations that invest in building comprehensive predictive maintenance capabilities today will be well-positioned to leverage these emerging technologies as they mature.

The examples presented in this exploration provide a solid foundation for implementing advanced predictive maintenance systems, but they represent only the beginning of what is possible. The most successful implementations will be those that combine technical sophistication with deep understanding of industrial operations, maintenance practices, and business requirements. By bridging the gap between advanced AI capabilities and practical industrial applications, predictive maintenance systems can deliver transformational value that extends throughout organizations and across entire industries.

The future of industrial asset management lies in these intelligent, adaptive systems that can learn, predict, and optimize continuously. The tools and techniques demonstrated here provide the roadmap for organizations ready to embark on this transformational journey toward more reliable, efficient, and cost-effective industrial operations.

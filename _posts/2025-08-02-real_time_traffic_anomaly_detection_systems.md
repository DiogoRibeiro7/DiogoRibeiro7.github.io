---
title: "Real-Time Traffic Anomaly Detection Systems: Advanced Incident Detection and Response"
categories:
- Transportation
- Artificial Intelligence
- Data Science

tags:
- Traffic Anomaly Detection
- Incident Detection
- Smart Transportation
- Real-Time Analytics
- Machine Learning
- Statistical Process Control

author_profile: false
seo_title: "Advanced Traffic Anomaly Detection for Smart Transportation Systems"
seo_description: "Explore real-time anomaly detection systems in traffic management using statistical process control and machine learning techniques. Enhance incident response and reduce traffic disruptions."
excerpt: "A deep dive into real-time traffic anomaly detection for intelligent transportation systems, covering statistical and machine learning methods for early incident response."
summary: "This article explores the architecture, detection methodologies, and practical implementation of real-time traffic anomaly detection systems using both statistical process control and modern machine learning algorithms."
keywords:
- "Traffic Anomaly Detection"
- "Incident Detection Systems"
- "Real-Time Traffic Monitoring"
- "Machine Learning in Transportation"
- "Statistical Process Control in Traffic"

classes: wide
date: '2025-08-02'
header:
  image: /assets/images/data_science/data_science_14.jpg
  og_image: /assets/images/data_science/data_science_14.jpg
  overlay_image: /assets/images/data_science/data_science_14.jpg
  show_overlay_excerpt: false
  teaser: /assets/images/data_science/data_science_14.jpg
  twitter_image: /assets/images/data_science/data_science_14.jpg
---

Traffic anomaly detection represents a critical component of modern intelligent transportation systems, serving as the first line of defense against incidents that can cascade into major disruptions. Unlike traditional traffic prediction that focuses on forecasting normal patterns, anomaly detection systems must identify deviations from expected behavior in real-time, often with incomplete information and under severe time constraints.

The economic impact of undetected traffic incidents is substantial. A single major highway incident can cost thousands of dollars per minute in lost productivity, increased fuel consumption, and delayed emergency response. Studies indicate that every minute of incident detection delay increases the total incident impact by 4-8%. This makes real-time anomaly detection not just a technical challenge, but a critical infrastructure need.

Modern anomaly detection systems must handle multiple types of abnormalities: sudden incidents like accidents or vehicle breakdowns, gradual degradations like weather-related slowdowns, recurring but irregular patterns like construction zones, and even coordinated disruptions like protests or emergency evacuations. Each type requires different detection strategies and response protocols.

This comprehensive guide examines the technical foundations, implementation strategies, and operational considerations for building robust traffic anomaly detection systems that can operate reliably in production environments.

## Understanding Traffic Anomalies: Types and Characteristics

### Temporal Classification of Anomalies

**Point Anomalies**
These represent individual data points that deviate significantly from normal patterns. Examples include sudden speed drops at a specific sensor due to accidents, or unexpected volume spikes during off-peak hours. Point anomalies are often the easiest to detect but require careful filtering to avoid false positives from sensor noise or temporary disruptions.

**Contextual Anomalies**
Data points that appear normal in isolation but are anomalous within their specific context. A moderate traffic volume might be normal during rush hour but highly suspicious at 3 AM. These require sophisticated models that understand temporal and spatial context.

**Collective Anomalies**
Sequences of data points that collectively indicate abnormal behavior, even if individual points appear normal. Examples include gradual speed reductions across multiple sensors indicating an incident downstream, or unusual traffic patterns suggesting coordinated disruptions.

### Spatial Distribution Patterns

**Localized Incidents**
Anomalies affecting a single location or small area, typically caused by accidents, breakdowns, or local construction. These create characteristic signatures in upstream and downstream traffic flows.

**Corridor-Level Disruptions**
Anomalies spanning significant distances along a traffic corridor, often caused by weather events, major incidents, or planned closures. Detection requires analyzing correlated patterns across multiple monitoring points.

**Network-Wide Disturbances**
System-level anomalies affecting large portions of a transportation network, such as during major events, emergencies, or infrastructure failures. These require network-level analysis and coordination.

### Severity and Impact Classification

Understanding anomaly severity helps prioritize response efforts and resource allocation:

**Critical Incidents**
Complete blockages or major accidents requiring immediate emergency response. Detection latency must be minimized to prevent secondary incidents and enable rapid intervention.

**Moderate Disruptions**
Significant but non-critical slowdowns that impact traffic flow efficiency. These benefit from prompt detection to enable dynamic routing and signal adjustments.

**Minor Irregularities**
Small deviations that may indicate developing problems or normal variations. These require monitoring but may not trigger immediate response.

## Statistical Process Control for Traffic Monitoring

Statistical Process Control (SPC) provides the foundational framework for detecting when traffic systems deviate from normal operating conditions. Adapted from manufacturing quality control, SPC techniques offer robust, interpretable methods for real-time anomaly detection.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class TrafficSPCDetector:
    """
    Statistical Process Control system for traffic anomaly detection
    """
    
    def __init__(self, window_size=50, control_limit_factor=3):
        self.window_size = window_size
        self.control_limit_factor = control_limit_factor
        self.historical_data = deque(maxlen=window_size)
        self.anomaly_threshold = 0.05
        self.control_charts = {}
        
    def initialize_control_limits(self, baseline_data):
        """
        Initialize control limits based on baseline historical data
        """
        baseline_stats = {
            'mean': np.mean(baseline_data),
            'std': np.std(baseline_data),
            'upper_control_limit': np.mean(baseline_data) + self.control_limit_factor * np.std(baseline_data),
            'lower_control_limit': np.mean(baseline_data) - self.control_limit_factor * np.std(baseline_data),
            'upper_warning_limit': np.mean(baseline_data) + 2 * np.std(baseline_data),
            'lower_warning_limit': np.mean(baseline_data) - 2 * np.std(baseline_data)
        }
        
        return baseline_stats
    
    def xbar_chart_detection(self, current_value, control_limits):
        """
        X-bar control chart for detecting mean shifts
        """
        anomaly_type = None
        severity = 'normal'
        
        if current_value > control_limits['upper_control_limit']:
            anomaly_type = 'high_outlier'
            severity = 'critical'
        elif current_value < control_limits['lower_control_limit']:
            anomaly_type = 'low_outlier'
            severity = 'critical'
        elif current_value > control_limits['upper_warning_limit']:
            anomaly_type = 'high_warning'
            severity = 'moderate'
        elif current_value < control_limits['lower_warning_limit']:
            anomaly_type = 'low_warning'
            severity = 'moderate'
            
        return {
            'is_anomaly': anomaly_type is not None,
            'anomaly_type': anomaly_type,
            'severity': severity,
            'control_value': current_value,
            'deviation': abs(current_value - control_limits['mean']) / control_limits['std']
        }
    
    def cusum_detection(self, data_stream, target_mean, std_dev, drift_threshold=1.0):
        """
        CUSUM (Cumulative Sum) control chart for detecting persistent shifts
        """
        cusum_high = 0
        cusum_low = 0
        detection_threshold = 5 * std_dev
        
        anomalies = []
        
        for i, value in enumerate(data_stream):
            # Calculate deviations
            deviation = (value - target_mean) / std_dev
            
            # Update CUSUM statistics
            cusum_high = max(0, cusum_high + deviation - drift_threshold)
            cusum_low = max(0, cusum_low - deviation - drift_threshold)
            
            # Check for anomalies
            anomaly_detected = False
            anomaly_type = None
            
            if cusum_high > detection_threshold:
                anomaly_detected = True
                anomaly_type = 'persistent_increase'
                cusum_high = 0  # Reset after detection
                
            elif cusum_low > detection_threshold:
                anomaly_detected = True
                anomaly_type = 'persistent_decrease'
                cusum_low = 0  # Reset after detection
            
            anomalies.append({
                'index': i,
                'value': value,
                'cusum_high': cusum_high,
                'cusum_low': cusum_low,
                'is_anomaly': anomaly_detected,
                'anomaly_type': anomaly_type
            })
        
        return anomalies
    
    def ewma_detection(self, data_stream, lambda_factor=0.2, control_factor=3):
        """
        Exponentially Weighted Moving Average (EWMA) for detecting small persistent shifts
        """
        if len(data_stream) < 2:
            return []
        
        # Initialize EWMA
        ewma_values = [data_stream[0]]
        baseline_mean = np.mean(data_stream[:min(20, len(data_stream))])
        baseline_std = np.std(data_stream[:min(20, len(data_stream))])
        
        anomalies = []
        
        for i in range(1, len(data_stream)):
            # Calculate EWMA
            ewma_current = lambda_factor * data_stream[i] + (1 - lambda_factor) * ewma_values[-1]
            ewma_values.append(ewma_current)
            
            # Calculate control limits for EWMA
            ewma_std = baseline_std * np.sqrt(lambda_factor / (2 - lambda_factor) * 
                                            (1 - (1 - lambda_factor)**(2 * (i + 1))))
            
            upper_limit = baseline_mean + control_factor * ewma_std
            lower_limit = baseline_mean - control_factor * ewma_std
            
            # Detect anomalies
            is_anomaly = ewma_current > upper_limit or ewma_current < lower_limit
            anomaly_type = None
            
            if ewma_current > upper_limit:
                anomaly_type = 'ewma_high'
            elif ewma_current < lower_limit:
                anomaly_type = 'ewma_low'
            
            anomalies.append({
                'index': i,
                'value': data_stream[i],
                'ewma': ewma_current,
                'upper_limit': upper_limit,
                'lower_limit': lower_limit,
                'is_anomaly': is_anomaly,
                'anomaly_type': anomaly_type
            })
        
        return anomalies
    
    def multivariate_hotelling_t2(self, data_matrix, alpha=0.01):
        """
        Hotelling's T² statistic for multivariate anomaly detection
        """
        n_samples, n_features = data_matrix.shape
        
        if n_samples < n_features + 1:
            raise ValueError("Need more samples than features for reliable covariance estimation")
        
        # Calculate sample statistics
        sample_mean = np.mean(data_matrix, axis=0)
        sample_cov = np.cov(data_matrix.T)
        
        # Calculate T² statistics
        t2_stats = []
        for i in range(n_samples):
            diff = data_matrix[i] - sample_mean
            t2 = n_samples * np.dot(np.dot(diff.T, np.linalg.inv(sample_cov)), diff)
            t2_stats.append(t2)
        
        # Calculate control limit
        from scipy.stats import f
        f_critical = f.ppf(1 - alpha, n_features, n_samples - n_features)
        control_limit = ((n_samples - 1) * n_features / (n_samples - n_features)) * f_critical
        
        # Identify anomalies
        anomalies = []
        for i, t2_value in enumerate(t2_stats):
            is_anomaly = t2_value > control_limit
            anomalies.append({
                'index': i,
                't2_statistic': t2_value,
                'control_limit': control_limit,
                'is_anomaly': is_anomaly,
                'data_point': data_matrix[i]
            })
        
        return anomalies, control_limit
    
    def generate_traffic_data_with_anomalies(self, n_samples=1000, anomaly_rate=0.05):
        """
        Generate synthetic traffic data with embedded anomalies for testing
        """
        # Base traffic pattern with daily cycles
        time_index = np.arange(n_samples)
        base_pattern = 50 + 30 * np.sin(2 * np.pi * time_index / 96)  # 96 = 15-min intervals per day
        
        # Add noise
        noise = np.random.normal(0, 5, n_samples)
        normal_data = base_pattern + noise
        
        # Add anomalies
        n_anomalies = int(n_samples * anomaly_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        anomaly_data = normal_data.copy()
        anomaly_labels = np.zeros(n_samples, dtype=bool)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drop', 'gradual_increase'])
            
            if anomaly_type == 'spike':
                anomaly_data[idx] += np.random.uniform(40, 80)
            elif anomaly_type == 'drop':
                anomaly_data[idx] -= np.random.uniform(30, 60)
            elif anomaly_type == 'gradual_increase':
                # Gradual increase over next 10 points
                for i in range(min(10, n_samples - idx)):
                    if idx + i < n_samples:
                        anomaly_data[idx + i] += np.random.uniform(10, 25)
            
            anomaly_labels[idx] = True
        
        return {
            'data': anomaly_data,
            'labels': anomaly_labels,
            'anomaly_indices': anomaly_indices,
            'normal_baseline': normal_data
        }
    
    def evaluate_detection_performance(self, detected_anomalies, true_labels):
        """
        Evaluate the performance of anomaly detection
        """
        detected_binary = np.array([det['is_anomaly'] for det in detected_anomalies])
        
        # Calculate performance metrics
        true_positives = np.sum(detected_binary & true_labels)
        false_positives = np.sum(detected_binary & ~true_labels)
        true_negatives = np.sum(~detected_binary & ~true_labels)
        false_negatives = np.sum(~detected_binary & true_labels)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }

# Example usage and testing
def demonstrate_spc_detection():
    """
    Demonstrate SPC-based anomaly detection methods
    """
    detector = TrafficSPCDetector()
    
    # Generate test data
    print("Generating synthetic traffic data with anomalies...")
    test_data = detector.generate_traffic_data_with_anomalies(n_samples=500, anomaly_rate=0.08)
    
    traffic_data = test_data['data']
    true_labels = test_data['labels']
    
    # Initialize control limits based on first 100 samples (assuming they're mostly normal)
    baseline_data = traffic_data[:100]
    control_limits = detector.initialize_control_limits(baseline_data)
    
    print(f"Control Limits Established:")
    print(f"Mean: {control_limits['mean']:.2f}")
    print(f"Upper Control Limit: {control_limits['upper_control_limit']:.2f}")
    print(f"Lower Control Limit: {control_limits['lower_control_limit']:.2f}")
    
    # Test X-bar chart detection
    print("\n1. Testing X-bar Control Chart Detection...")
    xbar_results = []
    for value in traffic_data:
        result = detector.xbar_chart_detection(value, control_limits)
        xbar_results.append(result)
    
    xbar_performance = detector.evaluate_detection_performance(xbar_results, true_labels)
    print(f"X-bar Chart Performance:")
    print(f"Precision: {xbar_performance['precision']:.3f}")
    print(f"Recall: {xbar_performance['recall']:.3f}")
    print(f"F1-Score: {xbar_performance['f1_score']:.3f}")
    
    # Test CUSUM detection
    print("\n2. Testing CUSUM Detection...")
    cusum_results = detector.cusum_detection(
        traffic_data, 
        target_mean=control_limits['mean'], 
        std_dev=control_limits['std']
    )
    
    cusum_performance = detector.evaluate_detection_performance(cusum_results, true_labels)
    print(f"CUSUM Performance:")
    print(f"Precision: {cusum_performance['precision']:.3f}")
    print(f"Recall: {cusum_performance['recall']:.3f}")
    print(f"F1-Score: {cusum_performance['f1_score']:.3f}")
    
    # Test EWMA detection
    print("\n3. Testing EWMA Detection...")
    ewma_results = detector.ewma_detection(traffic_data)
    
    ewma_performance = detector.evaluate_detection_performance(ewma_results, true_labels)
    print(f"EWMA Performance:")
    print(f"Precision: {ewma_performance['precision']:.3f}")
    print(f"Recall: {ewma_performance['recall']:.3f}")
    print(f"F1-Score: {ewma_performance['f1_score']:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 12))
    
    # Plot original data with anomalies
    plt.subplot(3, 1, 1)
    plt.plot(traffic_data, 'b-', alpha=0.7, label='Traffic Data')
    anomaly_indices = np.where(true_labels)[0]
    plt.scatter(anomaly_indices, traffic_data[anomaly_indices], color='red', s=50, label='True Anomalies')
    plt.axhline(y=control_limits['upper_control_limit'], color='r', linestyle='--', alpha=0.5, label='Control Limits')
    plt.axhline(y=control_limits['lower_control_limit'], color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=control_limits['mean'], color='g', linestyle='-', alpha=0.5, label='Mean')
    plt.title('Traffic Data with True Anomalies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot X-bar chart results
    plt.subplot(3, 1, 2)
    plt.plot(traffic_data, 'b-', alpha=0.7)
    xbar_anomalies = [i for i, res in enumerate(xbar_results) if res['is_anomaly']]
    plt.scatter(xbar_anomalies, traffic_data[xbar_anomalies], color='orange', s=30, label='Detected Anomalies')
    plt.axhline(y=control_limits['upper_control_limit'], color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=control_limits['lower_control_limit'], color='r', linestyle='--', alpha=0.5)
    plt.title('X-bar Chart Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot EWMA results
    plt.subplot(3, 1, 3)
    ewma_values = [res['ewma'] for res in ewma_results]
    upper_limits = [res['upper_limit'] for res in ewma_results]
    lower_limits = [res['lower_limit'] for res in ewma_results]
    
    plt.plot(ewma_values, 'g-', label='EWMA Values')
    plt.plot(upper_limits, 'r--', alpha=0.5, label='EWMA Control Limits')
    plt.plot(lower_limits, 'r--', alpha=0.5)
    
    ewma_anomalies = [i for i, res in enumerate(ewma_results) if res['is_anomaly']]
    if ewma_anomalies:
        plt.scatter(ewma_anomalies, [ewma_values[i] for i in ewma_anomalies], 
                   color='purple', s=30, label='EWMA Detected')
    
    plt.title('EWMA Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'xbar_performance': xbar_performance,
        'cusum_performance': cusum_performance,
        'ewma_performance': ewma_performance
    }

# Run demonstration
performance_results = demonstrate_spc_detection()
```

## Machine Learning Approaches for Anomaly Detection

While statistical process control provides robust baseline methods, machine learning approaches can capture more complex patterns and adapt to changing traffic conditions. Modern ML-based anomaly detection systems combine multiple techniques to achieve high detection rates while minimizing false positives.

```python
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
import joblib

class MLAnomalyDetector:
    """
    Machine learning based traffic anomaly detection system
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
    def prepare_features(self, traffic_data, temporal_window=12):
        """
        Prepare feature matrix for ML models
        """
        features = []
        
        # Basic statistical features
        features.extend([
            traffic_data['volume'],
            traffic_data['speed'],
            traffic_data['density']
        ])
        
        # Temporal features
        features.extend([
            traffic_data['hour'],
            traffic_data['day_of_week'],
            traffic_data['month']
        ])
        
        # Cyclical encoding of temporal features
        features.extend([
            np.sin(2 * np.pi * traffic_data['hour'] / 24),
            np.cos(2 * np.pi * traffic_data['hour'] / 24),
            np.sin(2 * np.pi * traffic_data['day_of_week'] / 7),
            np.cos(2 * np.pi * traffic_data['day_of_week'] / 7)
        ])
        
        # Rolling statistics (lag features)
        for window in [3, 6, 12]:
            features.extend([
                traffic_data['volume'].rolling(window=window).mean(),
                traffic_data['volume'].rolling(window=window).std(),
                traffic_data['speed'].rolling(window=window).mean(),
                traffic_data['speed'].rolling(window=window).std()
            ])
        
        # Rate of change features
        features.extend([
            traffic_data['volume'].diff(),
            traffic_data['speed'].diff(),
            traffic_data['volume'].diff().rolling(window=3).mean()
        ])
        
        feature_matrix = np.column_stack(features)
        
        # Remove rows with NaN values
        valid_rows = ~np.isnan(feature_matrix).any(axis=1)
        feature_matrix = feature_matrix[valid_rows]
        
        return feature_matrix, valid_rows
    
    def isolation_forest_detection(self, train_data, test_data, contamination=0.1):
        """
        Isolation Forest for anomaly detection
        """
        # Train model
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        iso_forest.fit(train_data)
        
        # Predict anomalies
        train_scores = iso_forest.decision_function(train_data)
        test_scores = iso_forest.decision_function(test_data)
        
        train_predictions = iso_forest.predict(train_data)
        test_predictions = iso_forest.predict(test_data)
        
        # Convert predictions to binary (1 = normal, -1 = anomaly)
        train_anomalies = (train_predictions == -1)
        test_anomalies = (test_predictions == -1)
        
        self.models['isolation_forest'] = iso_forest
        
        return {
            'train_anomalies': train_anomalies,
            'test_anomalies': test_anomalies,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'model': iso_forest
        }
    
    def one_class_svm_detection(self, train_data, test_data, nu=0.1):
        """
        One-Class SVM for novelty detection
        """
        # Scale data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Train One-Class SVM
        oc_svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        oc_svm.fit(train_scaled)
        
        # Predict
        train_predictions = oc_svm.predict(train_scaled)
        test_predictions = oc_svm.predict(test_scaled)
        
        train_scores = oc_svm.decision_function(train_scaled)
        test_scores = oc_svm.decision_function(test_scaled)
        
        train_anomalies = (train_predictions == -1)
        test_anomalies = (test_predictions == -1)
        
        self.models['one_class_svm'] = oc_svm
        self.scalers['one_class_svm'] = scaler
        
        return {
            'train_anomalies': train_anomalies,
            'test_anomalies': test_anomalies,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'model': oc_svm,
            'scaler': scaler
        }
    
    def local_outlier_factor_detection(self, data, n_neighbors=20, contamination=0.1):
        """
        Local Outlier Factor for density-based anomaly detection
        """
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False
        )
        
        predictions = lof.fit_predict(data)
        outlier_scores = lof.negative_outlier_factor_
        
        anomalies = (predictions == -1)
        
        return {
            'anomalies': anomalies,
            'outlier_scores': outlier_scores,
            'model': lof
        }
    
    def dbscan_clustering_detection(self, data, eps=0.5, min_samples=5):
        """
        DBSCAN clustering for anomaly detection
        """
        # Scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data_scaled)
        
        # Points labeled as -1 are considered anomalies
        anomalies = (cluster_labels == -1)
        
        # Calculate cluster statistics
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        cluster_sizes = [np.sum(cluster_labels == cluster) for cluster in unique_clusters]
        
        return {
            'anomalies': anomalies,
            'cluster_labels': cluster_labels,
            'n_clusters': len(unique_clusters),
            'cluster_sizes': cluster_sizes,
            'noise_points': np.sum(anomalies),
            'model': dbscan,
            'scaler': scaler
        }
    
    def autoencoder_detection(self, train_data, test_data, encoding_dim=10, threshold_percentile=95):
        """
        Autoencoder-based anomaly detection
        """
        # Scale data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        input_dim = train_scaled.shape[1]
        
        # Build autoencoder
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder
        history = autoencoder.fit(
            train_scaled, train_scaled,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate reconstruction errors
        train_predictions = autoencoder.predict(train_scaled)
        test_predictions = autoencoder.predict(test_scaled)
        
        train_mse = np.mean(np.power(train_scaled - train_predictions, 2), axis=1)
        test_mse = np.mean(np.power(test_scaled - test_predictions, 2), axis=1)
        
        # Set threshold based on training data
        threshold = np.percentile(train_mse, threshold_percentile)
        
        train_anomalies = train_mse > threshold
        test_anomalies = test_mse > threshold
        
        self.models['autoencoder'] = autoencoder
        self.scalers['autoencoder'] = scaler
        
        return {
            'train_anomalies': train_anomalies,
            'test_anomalies': test_anomalies,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'threshold': threshold,
            'model': autoencoder,
            'scaler': scaler,
            'training_history': history
        }
    
    def lstm_autoencoder_detection(self, sequence_data, sequence_length=24, encoding_dim=50):
        """
        LSTM Autoencoder for temporal anomaly detection
        """
        # Prepare sequence data
        def create_sequences(data, seq_length):
            sequences = []
            for i in range(len(data) - seq_length + 1):
                sequences.append(data[i:i + seq_length])
            return np.array(sequences)
        
        sequences = create_sequences(sequence_data, sequence_length)
        
        # Split data
        train_size = int(0.8 * len(sequences))
        train_sequences = sequences[:train_size]
        test_sequences = sequences[train_size:]
        
        # Scale data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_sequences.reshape(-1, train_sequences.shape[-1]))
        train_scaled = train_scaled.reshape(train_sequences.shape)
        
        test_scaled = scaler.transform(test_sequences.reshape(-1, test_sequences.shape[-1]))
        test_scaled = test_scaled.reshape(test_sequences.shape)
        
        # Build LSTM Autoencoder
        input_layer = Input(shape=(sequence_length, train_scaled.shape[2]))
        
        # Encoder
        encoded = LSTM(encoding_dim, activation='relu')(input_layer)
        
        # Repeat vector for decoder
        repeated = RepeatVector(sequence_length)(encoded)
        
        # Decoder
        decoded = LSTM(encoding_dim, activation='relu', return_sequences=True)(repeated)
        decoded = TimeDistributed(Dense(train_scaled.shape[2]))(decoded)
        
        lstm_autoencoder = Model(input_layer, decoded)
        lstm_autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train model
        history = lstm_autoencoder.fit(
            train_scaled, train_scaled,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate reconstruction errors
        train_pred = lstm_autoencoder.predict(train_scaled)
        test_pred = lstm_autoencoder.predict(test_scaled)
        
        train_mse = np.mean(np.power(train_scaled - train_pred, 2), axis=(1, 2))
        test_mse = np.mean(np.power(test_scaled - test_pred, 2), axis=(1, 2))
        
        # Set threshold
        threshold = np.percentile(train_mse, 95)
        
        train_anomalies = train_mse > threshold
        test_anomalies = test_mse > threshold
        
        return {
            'train_anomalies': train_anomalies,
            'test_anomalies': test_anomalies,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'threshold': threshold,
            'model': lstm_autoencoder,
            'scaler': scaler
        }
    
    def ensemble_detection(self, train_data, test_data, methods=['isolation_forest', 'autoencoder', 'one_class_svm']):
        """
        Ensemble approach combining multiple detection methods
        """
        results = {}
        
        for method in methods:
            if method == 'isolation_forest':
                results[method] = self.isolation_forest_detection(train_data, test_data)
            elif method == 'autoencoder':
                results[method] = self.autoencoder_detection(train_data, test_data)
            elif method == 'one_class_svm':
                results[method] = self.one_class_svm_detection(train_data, test_data)
            elif method == 'local_outlier_factor':
                # LOF needs combined data
                combined_data = np.vstack([train_data, test_data])
                lof_result = self.local_outlier_factor_detection(combined_data)
                train_size = len(train_data)
                results[method] = {
                    'train_anomalies': lof_result['anomalies'][:train_size],
                    'test_anomalies': lof_result['anomalies'][train_size:]
                }
        
        # Combine predictions using voting
        test_predictions = []
        for method in methods:
            if method in results:
                test_predictions.append(results[method]['test_anomalies'])
        
        if test_predictions:
            # Majority voting
            test_predictions_array = np.array(test_predictions)
            ensemble_test_anomalies = np.sum(test_predictions_array, axis=0) >= len(methods) / 2
            
            # Calculate confidence based on agreement
            confidence = np.mean(test_predictions_array, axis=0)
            
            results['ensemble'] = {
                'test_anomalies': ensemble_test_anomalies,
                'confidence': confidence,
                'individual_results': results
            }
        
        return results

def demonstrate_ml_detection():
    """
    Demonstrate ML-based anomaly detection methods
    """
    detector = MLAnomalyDetector()
    
    # Generate synthetic traffic data
    print("Generating traffic data with temporal patterns...")
    
    # Create more realistic traffic data
    n_days = 30
    intervals_per_day = 96  # 15-minute intervals
    n_samples = n_days * intervals_per_day
    
    time_index = np.arange(n_samples)
    hours = (time_index % intervals_per_day) / 4  # Convert to hours
    days = time_index // intervals_per_day
    
    # Base traffic pattern
    daily_pattern = 50 + 40 * (np.sin((hours - 6) * np.pi / 12) ** 2)
    weekly_pattern = 1.0 - 0.3 * ((days % 7) >= 5)  # Lower on weekends
    
    # Traffic volume
    volume = daily_pattern * weekly_pattern + np.random.normal(0, 8, n_samples)
    volume = np.maximum(volume, 5)  # Minimum traffic
    
    # Speed (inversely related to volume with some noise)
    speed = 60 - 0.3 * volume + np.random.normal(0, 5, n_samples)
    speed = np.clip(speed, 10, 70)
    
    # Density
    density = volume / speed
    
    # Create DataFrame
    traffic_df = pd.DataFrame({
        'volume': volume,
        'speed': speed,
        'density': density,
        'hour': hours,
        'day_of_week': days % 7,
        'month': 1  # Simplified
    })
    
    # Add anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    true_anomalies = np.zeros(n_samples, dtype=bool)
    true_anomalies[anomaly_indices] = True
    
    # Inject different types of anomalies
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['accident', 'congestion', 'sensor_error'])
        
        if anomaly_type == 'accident':
            # Sharp drop in speed, increase in density
            traffic_df.loc[idx, 'speed'] *= 0.3
            traffic_df.loc[idx, 'density'] *= 2.0
        elif anomaly_type == 'congestion':
            # Moderate speed reduction, volume increase
            traffic_df.loc[idx, 'speed'] *= 0.6
            traffic_df.loc[idx, 'volume'] *= 1.5
        elif anomaly_type == 'sensor_error':
            # Random extreme values
            traffic_df.loc[idx, 'volume'] = np.random.uniform(0, 200)
            traffic_df.loc[idx, 'speed'] = np.random.uniform(0, 100)
    
    # Prepare features
    feature_matrix, valid_rows = detector.prepare_features(traffic_df)
    true_labels = true_anomalies[valid_rows]
    
    # Split data
    train_size = int(0.7 * len(feature_matrix))
    train_data = feature_matrix[:train_size]
    test_data = feature_matrix[train_size:]
    train_labels = true_labels[:train_size]
    test_labels = true_labels[train_size:]
    
    print(f"Dataset prepared: {len(feature_matrix)} samples, {feature_matrix.shape[1]} features")
    print(f"Training set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Anomaly rate: {np.mean(true_labels):.2%}")
    
    # Test different methods
    results = {}
    
    # 1. Isolation Forest
    print("\n1. Testing Isolation Forest...")
    iso_result = detector.isolation_forest_detection(train_data, test_data, contamination=0.1)
    iso_performance = calculate_performance_metrics(iso_result['test_anomalies'], test_labels)
    results['isolation_forest'] = iso_performance
    print(f"Isolation Forest - Precision: {iso_performance['precision']:.3f}, Recall: {iso_performance['recall']:.3f}, F1: {iso_performance['f1_score']:.3f}")
    
    # 2. One-Class SVM
    print("\n2. Testing One-Class SVM...")
    svm_result = detector.one_class_svm_detection(train_data, test_data, nu=0.1)
    svm_performance = calculate_performance_metrics(svm_result['test_anomalies'], test_labels)
    results['one_class_svm'] = svm_performance
    print(f"One-Class SVM - Precision: {svm_performance['precision']:.3f}, Recall: {svm_performance['recall']:.3f}, F1: {svm_performance['f1_score']:.3f}")
    
    # 3. Autoencoder
    print("\n3. Testing Autoencoder...")
    ae_result = detector.autoencoder_detection(train_data, test_data, encoding_dim=8)
    ae_performance = calculate_performance_metrics(ae_result['test_anomalies'], test_labels)
    results['autoencoder'] = ae_performance
    print(f"Autoencoder - Precision: {ae_performance['precision']:.3f}, Recall: {ae_performance['recall']:.3f}, F1: {ae_performance['f1_score']:.3f}")
    
    # 4. Local Outlier Factor
    print("\n4. Testing Local Outlier Factor...")
    combined_data = np.vstack([train_data, test_data])
    combined_labels = np.concatenate([train_labels, test_labels])
    lof_result = detector.local_outlier_factor_detection(combined_data, contamination=0.1)
    lof_performance = calculate_performance_metrics(lof_result['anomalies'][train_size:], test_labels)
    results['local_outlier_factor'] = lof_performance
    print(f"LOF - Precision: {lof_performance['precision']:.3f}, Recall: {lof_performance['recall']:.3f}, F1: {lof_performance['f1_score']:.3f}")
    
    # 5. Ensemble Method
    print("\n5. Testing Ensemble Method...")
    ensemble_result = detector.ensemble_detection(train_data, test_data, 
                                                 methods=['isolation_forest', 'autoencoder', 'one_class_svm'])
    ensemble_performance = calculate_performance_metrics(ensemble_result['ensemble']['test_anomalies'], test_labels)
    results['ensemble'] = ensemble_performance
    print(f"Ensemble - Precision: {ensemble_performance['precision']:.3f}, Recall: {ensemble_performance['recall']:.3f}, F1: {ensemble_performance['f1_score']:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 12))
    
    # Plot original traffic data
    plt.subplot(3, 2, 1)
    test_traffic = traffic_df.iloc[train_size:train_size+len(test_data)]
    plt.plot(test_traffic['volume'].values, 'b-', alpha=0.7, label='Volume')
    anomaly_test_indices = np.where(test_labels)[0]
    if len(anomaly_test_indices) > 0:
        plt.scatter(anomaly_test_indices, test_traffic['volume'].values[anomaly_test_indices], 
                   color='red', s=50, label='True Anomalies')
    plt.title('Traffic Volume - True Anomalies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot detection results for different methods
    methods_to_plot = ['isolation_forest', 'one_class_svm', 'autoencoder', 'ensemble']
    method_results = [iso_result, svm_result, ae_result, ensemble_result]
    
    for i, (method, method_result) in enumerate(zip(methods_to_plot, method_results)):
        plt.subplot(3, 2, i + 2)
        plt.plot(test_traffic['volume'].values, 'b-', alpha=0.7, label='Volume')
        
        if method == 'ensemble':
            detected_anomalies = np.where(method_result['ensemble']['test_anomalies'])[0]
        else:
            detected_anomalies = np.where(method_result['test_anomalies'])[0]
        
        if len(detected_anomalies) > 0:
            plt.scatter(detected_anomalies, test_traffic['volume'].values[detected_anomalies], 
                       color='orange', s=30, label='Detected')
        
        if len(anomaly_test_indices) > 0:
            plt.scatter(anomaly_test_indices, test_traffic['volume'].values[anomaly_test_indices], 
                       color='red', s=50, alpha=0.7, label='True')
        
        performance = results[method]
        plt.title(f'{method.replace("_", " ").title()}\nF1: {performance["f1_score"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

def calculate_performance_metrics(predictions, true_labels):
    """
    Calculate performance metrics for anomaly detection
    """
    tp = np.sum(predictions & true_labels)
    fp = np.sum(predictions & ~true_labels)
    tn = np.sum(~predictions & ~true_labels)
    fn = np.sum(~predictions & true_labels)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

# Run ML demonstration
ml_results = demonstrate_ml_detection()

## Real-Time Implementation and System Integration

Building a production-ready anomaly detection system requires careful consideration of real-time processing requirements, system integration, and operational workflows.

```python
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import redis
import websockets

class RealTimeAnomalyDetectionSystem:
    """
    Real-time traffic anomaly detection and alert system
    """
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # Detection models
        self.spc_detector = TrafficSPCDetector()
        self.ml_detector = MLAnomalyDetector()
        
        # Data buffers
        self.data_buffer = deque(maxlen=self.config.get('buffer_size', 1000))
        self.alert_history = deque(maxlen=self.config.get('alert_history_size', 500))
        
        # Real-time processing
        self.processing_queue = asyncio.Queue()
        self.alert_subscribers = set()
        
        # Performance monitoring
        self.performance_metrics = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'processing_times': deque(maxlen=100),
            'false_alarm_rate': 0.0
        }
        
        # Redis for caching and coordination
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                decode_responses=True
            )
            self.redis_available = True
        except:
            self.redis_available = False
            self.logger.warning("Redis not available - using in-memory storage only")
    
    def load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        default_config = {
            'detection_methods': ['spc', 'isolation_forest', 'autoencoder'],
            'alert_thresholds': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5
            },
            'processing_interval': 1.0,  # seconds
            'buffer_size': 1000,
            'alert_history_size': 500,
            'performance_logging_interval': 60,  # seconds
            'redis_host': 'localhost',
            'redis_port': 6379
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except FileNotFoundError:
                pass
        
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('anomaly_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def process_traffic_data(self, data_point: Dict):
        """
        Process incoming traffic data point for anomaly detection
        """
        start_time = time.time()
        
        try:
            # Add to processing queue
            await self.processing_queue.put(data_point)
            
            # Update buffer
            self.data_buffer.append(data_point)
            
            # Perform anomaly detection
            detection_results = await self.detect_anomalies(data_point)
            
            # Generate alerts if necessary
            if detection_results['is_anomaly']:
                await self.generate_alert(data_point, detection_results)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['total_processed'] += 1
            self.performance_metrics['processing_times'].append(processing_time)
            
            if detection_results['is_anomaly']:
                self.performance_metrics['anomalies_detected'] += 1
            
            # Cache results
            if self.redis_available:
                await self.cache_detection_result(data_point, detection_results)
            
            return detection_results
            
        except Exception as e:
            self.logger.error(f"Error processing traffic data: {str(e)}")
            return {'is_anomaly': False, 'error': str(e)}
    
    async def detect_anomalies(self, data_point: Dict) -> Dict:
        """
        Run anomaly detection using configured methods
        """
        detection_results = {
            'timestamp': data_point.get('timestamp', datetime.now().isoformat()),
            'location_id': data_point.get('location_id'),
            'is_anomaly': False,
            'confidence': 0.0,
            'anomaly_type': None,
            'severity': 'normal',
            'detection_methods': {}
        }
        
        try:
            # Extract numeric values
            volume = data_point.get('volume', 0)
            speed = data_point.get('speed', 0)
            density = data_point.get('density', 0)
            
            method_scores = []
            
            # SPC-based detection
            if 'spc' in self.config['detection_methods'] and len(self.data_buffer) >= 20:
                spc_result = self.run_spc_detection(volume)
                detection_results['detection_methods']['spc'] = spc_result
                if spc_result['is_anomaly']:
                    method_scores.append(spc_result.get('confidence', 0.5))
            
            # ML-based detection
            if len(self.data_buffer) >= 50:  # Need sufficient data for ML methods
                ml_result = await self.run_ml_detection(data_point)
                detection_results['detection_methods']['ml'] = ml_result
                if ml_result['is_anomaly']:
                    method_scores.append(ml_result.get('confidence', 0.5))
            
            # Combine results
            if method_scores:
                detection_results['is_anomaly'] = True
                detection_results['confidence'] = np.mean(method_scores)
                detection_results['severity'] = self.determine_severity(detection_results['confidence'])
                detection_results['anomaly_type'] = self.classify_anomaly_type(data_point)
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            detection_results['error'] = str(e)
        
        return detection_results
    
    def run_spc_detection(self, value: float) -> Dict:
        """
        Run statistical process control detection
        """
        try:
            # Use recent data to establish control limits
            recent_values = [dp.get('volume', 0) for dp in list(self.data_buffer)[-50:]]
            control_limits = self.spc_detector.initialize_control_limits(recent_values)
            
            # Detect anomaly
            result = self.spc_detector.xbar_chart_detection(value, control_limits)
            
            return {
                'is_anomaly': result['is_anomaly'],
                'confidence': min(result.get('deviation', 0) / 3.0, 1.0),
                'method': 'spc_xbar',
                'details': result
            }
            
        except Exception as e:
            self.logger.error(f"SPC detection error: {str(e)}")
            return {'is_anomaly': False, 'error': str(e)}
    
    async def run_ml_detection(self, data_point: Dict) -> Dict:
        """
        Run machine learning based detection
        """
        try:
            # Prepare recent data for ML detection
            recent_data = list(self.data_buffer)[-100:]  # Last 100 points
            
            # Extract features
            feature_matrix = []
            for dp in recent_data:
                features = [
                    dp.get('volume', 0),
                    dp.get('speed', 0),
                    dp.get('density', 0),
                    dp.get('hour', datetime.now().hour),
                    dp.get('day_of_week', datetime.now().weekday())
                ]
                feature_matrix.append(features)
            
            feature_matrix = np.array(feature_matrix)
            
            # Run isolation forest on recent data
            if len(feature_matrix) >= 20:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                iso_forest.fit(feature_matrix[:-1])  # Train on all but last point
                
                current_features = feature_matrix[-1].reshape(1, -1)
                prediction = iso_forest.predict(current_features)[0]
                score = iso_forest.decision_function(current_features)[0]
                
                is_anomaly = prediction == -1
                confidence = abs(score) if is_anomaly else 0
                
                return {
                    'is_anomaly': is_anomaly,
                    'confidence': min(confidence, 1.0),
                    'method': 'isolation_forest',
                    'score': score
                }
            else:
                return {'is_anomaly': False, 'reason': 'insufficient_data'}
                
        except Exception as e:
            self.logger.error(f"ML detection error: {str(e)}")
            return {'is_anomaly': False, 'error': str(e)}
    
    def determine_severity(self, confidence: float) -> str:
        """
        Determine anomaly severity based on confidence score
        """
        thresholds = self.config['alert_thresholds']
        
        if confidence >= thresholds['critical']:
            return 'critical'
        elif confidence >= thresholds['high']:
            return 'high'
        elif confidence >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def classify_anomaly_type(self, data_point: Dict) -> str:
        """
        Classify the type of anomaly based on traffic characteristics
        """
        volume = data_point.get('volume', 0)
        speed = data_point.get('speed', 0)
        density = data_point.get('density', 0)
        
        # Simple rule-based classification
        if speed < 20 and density > 50:
            return 'severe_congestion'
        elif volume > 100 and speed < 30:
            return 'moderate_congestion'
        elif volume < 5 and speed > 70:
            return 'unusual_free_flow'
        elif density > 80:
            return 'potential_incident'
        else:
            return 'general_anomaly'
    
    async def generate_alert(self, data_point: Dict, detection_results: Dict):
        """
        Generate and distribute alerts for detected anomalies
        """
        alert = {
            'alert_id': f"alert_{int(time.time())}_{data_point.get('location_id', 'unknown')}",
            'timestamp': datetime.now().isoformat(),
            'location_id': data_point.get('location_id'),
            'anomaly_type': detection_results.get('anomaly_type'),
            'severity': detection_results.get('severity'),
            'confidence': detection_results.get('confidence'),
            'traffic_data': data_point,
            'detection_methods': detection_results.get('detection_methods', {}),
            'recommended_actions': self.get_recommended_actions(detection_results)
        }
        
        # Add to alert history
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(f"ANOMALY ALERT: {alert['alert_id']} - {alert['anomaly_type']} "
                          f"at {alert['location_id']} (Severity: {alert['severity']}, "
                          f"Confidence: {alert['confidence']:.2f})")
        
        # Distribute to subscribers
        await self.distribute_alert(alert)
        
        # Store in Redis if available
        if self.redis_available:
            try:
                alert_key = f"alert:{alert['alert_id']}"
                self.redis_client.setex(alert_key, 3600, json.dumps(alert))  # 1 hour expiry
            except Exception as e:
                self.logger.error(f"Error storing alert in Redis: {str(e)}")
    
    def get_recommended_actions(self, detection_results: Dict) -> List[str]:
        """
        Generate recommended actions based on anomaly type and severity
        """
        severity = detection_results.get('severity')
        anomaly_type = detection_results.get('anomaly_type')
        
        actions = []
        
        if severity == 'critical':
            actions.extend([
                "Immediate field verification required",
                "Consider emergency response deployment",
                "Activate incident management protocols"
            ])
        
        if anomaly_type in ['severe_congestion', 'potential_incident']:
            actions.extend([
                "Deploy traffic management team",
                "Activate variable message signs",
                "Consider alternate route recommendations"
            ])
        
        if severity in ['high', 'critical']:
            actions.extend([
                "Notify traffic control center",
                "Update real-time traffic information systems",
                "Monitor upstream and downstream conditions"
            ])
        
        return actions
    
    async def distribute_alert(self, alert: Dict):
        """
        Distribute alert to all subscribers
        """
        if self.alert_subscribers:
            alert_message = json.dumps(alert)
            
            # Send to all WebSocket subscribers
            disconnected = set()
            for websocket in self.alert_subscribers:
                try:
                    await websocket.send(alert_message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
                except Exception as e:
                    self.logger.error(f"Error sending alert: {str(e)}")
                    disconnected.add(websocket)
            
            # Remove disconnected subscribers
            self.alert_subscribers -= disconnected
    
    async def cache_detection_result(self, data_point: Dict, detection_results: Dict):
        """
        Cache detection results for analysis and debugging
        """
        try:
            cache_key = f"detection:{data_point.get('location_id')}:{int(time.time())}"
            cache_data = {
                'data_point': data_point,
                'detection_results': detection_results
            }
            
            self.redis_client.setex(cache_key, 1800, json.dumps(cache_data))  # 30 minutes
            
        except Exception as e:
            self.logger.error(f"Error caching detection result: {str(e)}")
    
    def get_system_status(self) -> Dict:
        """
        Get current system status and performance metrics
        """
        avg_processing_time = np.mean(self.performance_metrics['processing_times']) if self.performance_metrics['processing_times'] else 0
        
        status = {
            'system_status': 'operational',
            'total_processed': self.performance_metrics['total_processed'],
            'anomalies_detected': self.performance_metrics['anomalies_detected'],
            'detection_rate': self.performance_metrics['anomalies_detected'] / max(1, self.performance_metrics['total_processed']),
            'avg_processing_time': avg_processing_time,
            'buffer_size': len(self.data_buffer),
            'alert_subscribers': len(self.alert_subscribers),
            'recent_alerts': len([a for a in self.alert_history if 
                                datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]),
            'uptime': time.time() - getattr(self, 'start_time', time.time()),
            'redis_status': 'connected' if self.redis_available else 'disconnected'
        }
        
        return status

# WebSocket server for real-time alerts
async def alert_websocket_handler(websocket, path, detection_system):
    """
    WebSocket handler for real-time alert distribution
    """
    detection_system.alert_subscribers.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        detection_system.alert_subscribers.discard(websocket)

# Example usage
async def simulate_real_time_detection():
    """
    Simulate real-time traffic anomaly detection
    """
    system = RealTimeAnomalyDetectionSystem()
    system.start_time = time.time()
    
    print("Starting real-time anomaly detection simulation...")
    
    # Simulate incoming traffic data
    for i in range(100):
        # Generate realistic traffic data
        hour = (i % 24)
        is_weekend = (i // 24) % 7 >= 5
        
        # Base traffic pattern
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_volume = np.random.normal(80, 15)
        elif 22 <= hour or hour <= 6:  # Night hours
            base_volume = np.random.normal(20, 5)
        else:
            base_volume = np.random.normal(50, 10)
        
        if is_weekend:
            base_volume *= 0.7
        
        # Occasionally inject anomalies
        if np.random.random() < 0.1:  # 10% chance of anomaly
            if np.random.random() < 0.5:
                base_volume *= 0.3  # Accident scenario
            else:
                base_volume *= 2.0  # Unusual congestion
        
        volume = max(5, base_volume + np.random.normal(0, 5))
        speed = max(10, 60 - 0.4 * volume + np.random.normal(0, 8))
        density = volume / speed
        
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'location_id': f'sensor_{i % 10:03d}',
            'volume': volume,
            'speed': speed,
            'density': density,
            'hour': hour,
            'day_of_week': (i // 24) % 7
        }
        
        # Process data point
        result = await system.process_traffic_data(data_point)
        
        if result.get('is_anomaly'):
            print(f"ANOMALY DETECTED at {data_point['location_id']}: "
                  f"Type: {result.get('anomaly_type')}, "
                  f"Severity: {result.get('severity')}, "
                  f"Confidence: {result.get('confidence', 0):.2f}")
        
        # Small delay to simulate real-time processing
        await asyncio.sleep(0.1)
    
    # Print system status
    status = system.get_system_status()
    print(f"\nSystem Status:")
    print(f"Total Processed: {status['total_processed']}")
    print(f"Anomalies Detected: {status['anomalies_detected']}")
    print(f"Detection Rate: {status['detection_rate']:.2%}")
    print(f"Avg Processing Time: {status['avg_processing_time']:.4f}s")

# Run simulation
# asyncio.run(simulate_real_time_detection())

## Emergency Response Integration and Alert Management

A critical component of any traffic anomaly detection system is its integration with emergency response protocols and traffic management centers. This section explores the operational workflows and system integrations necessary for effective incident response.

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from enum import Enum
import requests

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_ALARM = "false_alarm"

class EmergencyResponseIntegration:
    """
    Integration system for emergency response and traffic management
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.active_incidents = {}
        self.response_teams = {}
        self.escalation_rules = self.load_escalation_rules()
        
    def setup_logging(self):
        """Setup logging for emergency response"""
        self.logger = logging.getLogger('emergency_response')
        
    def load_escalation_rules(self) -> Dict:
        """Load escalation rules for different alert types"""
        return {
            AlertSeverity.CRITICAL: {
                'max_response_time': 300,  # 5 minutes
                'auto_escalate_after': 600,  # 10 minutes
                'notification_channels': ['sms', 'email', 'radio', 'dashboard'],
                'required_response_teams': ['traffic_control', 'emergency_services']
            },
            AlertSeverity.HIGH: {
                'max_response_time': 900,  # 15 minutes
                'auto_escalate_after': 1800,  # 30 minutes
                'notification_channels': ['email', 'dashboard'],
                'required_response_teams': ['traffic_control']
            },
            AlertSeverity.MEDIUM: {
                'max_response_time': 1800,  # 30 minutes
                'auto_escalate_after': 3600,  # 1 hour
                'notification_channels': ['email', 'dashboard'],
                'required_response_teams': ['traffic_control']
            },
            AlertSeverity.LOW: {
                'max_response_time': 3600,  # 1 hour
                'auto_escalate_after': None,
                'notification_channels': ['dashboard'],
                'required_response_teams': []
            }
        }
    
    async def process_alert(self, alert: Dict) -> Dict:
        """
        Process incoming alert and initiate appropriate response
        """
        alert_id = alert['alert_id']
        severity = AlertSeverity(alert['severity'])
        
        # Create incident record
        incident = {
            'incident_id': f"INC_{int(time.time())}",
            'alert_id': alert_id,
            'created_at': datetime.now(),
            'location_id': alert['location_id'],
            'severity': severity,
            'status': AlertStatus.ACTIVE,
            'assigned_teams': [],
            'response_actions': [],
            'estimated_resolution_time': None,
            'actual_resolution_time': None
        }
        
        self.active_incidents[incident['incident_id']] = incident
        
        # Apply escalation rules
        escalation_rule = self.escalation_rules[severity]
        
        # Send notifications
        await self.send_notifications(alert, incident, escalation_rule['notification_channels'])
        
        # Assign response teams
        await self.assign_response_teams(incident, escalation_rule['required_response_teams'])
        
        # Set up automatic escalation if configured
        if escalation_rule['auto_escalate_after']:
            await self.schedule_escalation(incident, escalation_rule['auto_escalate_after'])
        
        # Log incident creation
        self.logger.info(f"Incident {incident['incident_id']} created for alert {alert_id}")
        
        return incident
    
    async def send_notifications(self, alert: Dict, incident: Dict, channels: List[str]):
        """
        Send notifications through configured channels
        """
        message_content = self.format_alert_message(alert, incident)
        
        for channel in channels:
            try:
                if channel == 'email':
                    await self.send_email_notification(alert, message_content)
                elif channel == 'sms':
                    await self.send_sms_notification(alert, message_content)
                elif channel == 'radio':
                    await self.send_radio_notification(alert, message_content)
                elif channel == 'dashboard':
                    await self.update_dashboard(alert, incident)
                    
            except Exception as e:
                self.logger.error(f"Failed to send {channel} notification: {str(e)}")
    
    def format_alert_message(self, alert: Dict, incident: Dict) -> str:
        """
        Format alert message for human consumption
        """
        return f"""
TRAFFIC ANOMALY ALERT - {alert['severity'].upper()}

Incident ID: {incident['incident_id']}
Location: {alert['location_id']}
Time: {alert['timestamp']}
Type: {alert['anomaly_type']}
Confidence: {alert['confidence']:.1%}

Traffic Conditions:
- Volume: {alert['traffic_data'].get('volume', 'N/A')}
- Speed: {alert['traffic_data'].get('speed', 'N/A')} mph
- Density: {alert['traffic_data'].get('density', 'N/A')}

Recommended Actions:
{chr(10).join('- ' + action for action in alert.get('recommended_actions', []))}

Detection Methods: {', '.join(alert.get('detection_methods', {}).keys())}
        """.strip()
    
    async def send_email_notification(self, alert: Dict, message: str):
        """
        Send email notification to configured recipients
        """
        try:
            smtp_config = self.config.get('email', {})
            
            if not smtp_config:
                return
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_address')
            msg['Subject'] = f"Traffic Alert - {alert['severity'].upper()} - {alert['location_id']}"
            
            # Determine recipients based on severity
            if alert['severity'] == AlertSeverity.CRITICAL.value:
                recipients = smtp_config.get('critical_recipients', [])
            elif alert['severity'] == AlertSeverity.HIGH.value:
                recipients = smtp_config.get('high_recipients', [])
            else:
                recipients = smtp_config.get('general_recipients', [])
            
            msg['To'] = ', '.join(recipients)
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_config.get('smtp_server'), smtp_config.get('smtp_port', 587))
            server.starttls()
            server.login(smtp_config.get('username'), smtp_config.get('password'))
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert['alert_id']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
    
    async def send_sms_notification(self, alert: Dict, message: str):
        """
        Send SMS notification using configured service
        """
        try:
            sms_config = self.config.get('sms', {})
            
            if not sms_config:
                return
            
            # Truncate message for SMS
            sms_message = message[:160] + "..." if len(message) > 160 else message
            
            # Use SMS service API (example with Twilio-like service)
            api_url = sms_config.get('api_url')
            api_key = sms_config.get('api_key')
            
            if alert['severity'] in [AlertSeverity.CRITICAL.value, AlertSeverity.HIGH.value]:
                phone_numbers = sms_config.get('emergency_numbers', [])
            else:
                phone_numbers = sms_config.get('general_numbers', [])
            
            for phone_number in phone_numbers:
                payload = {
                    'to': phone_number,
                    'message': sms_message,
                    'from': sms_config.get('from_number')
                }
                
                headers = {'Authorization': f'Bearer {api_key}'}
                
                response = requests.post(api_url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    self.logger.info(f"SMS sent to {phone_number} for alert {alert['alert_id']}")
                else:
                    self.logger.error(f"Failed to send SMS to {phone_number}: {response.text}")
                    
        except Exception as e:
            self.logger.error(f"Failed to send SMS notification: {str(e)}")
    
    async def assign_response_teams(self, incident: Dict, required_teams: List[str]):
        """
        Assign appropriate response teams to incident
        """
        try:
            for team_type in required_teams:
                available_team = self.find_available_team(team_type, incident['location_id'])
                
                if available_team:
                    # Assign team to incident
                    incident['assigned_teams'].append({
                        'team_id': available_team['team_id'],
                        'team_type': team_type,
                        'assigned_at': datetime.now(),
                        'status': 'assigned'
                    })
                    
                    # Notify team
                    await self.notify_response_team(available_team, incident)
                    
                    self.logger.info(f"Assigned {team_type} team {available_team['team_id']} to incident {incident['incident_id']}")
                else:
                    self.logger.warning(f"No available {team_type} team found for incident {incident['incident_id']}")
                    
        except Exception as e:
            self.logger.error(f"Failed to assign response teams: {str(e)}")
    
    def find_available_team(self, team_type: str, location_id: str) -> Optional[Dict]:
        """
        Find available response team closest to incident location
        """
        # This would integrate with actual team management system
        # For now, return mock team data
        teams = self.config.get('response_teams', {}).get(team_type, [])
        
        for team in teams:
            if team.get('status') == 'available':
                return team
        
        return None
    
    async def notify_response_team(self, team: Dict, incident: Dict):
        """
        Notify assigned response team about incident
        """
        try:
            # Format team notification
            notification = {
                'incident_id': incident['incident_id'],
                'location': incident['location_id'],
                'severity': incident['severity'].value,
                'assignment_time': datetime.now().isoformat(),
                'expected_response_time': self.calculate_expected_response_time(team, incident)
            }
            
            # Send to team's communication channel
            team_channel = team.get('communication_channel')
            
            if team_channel == 'radio':
                await self.send_radio_dispatch(team, notification)
            elif team_channel == 'mobile_app':
                await self.send_mobile_notification(team, notification)
            elif team_channel == 'email':
                await self.send_team_email(team, notification)
                
        except Exception as e:
            self.logger.error(f"Failed to notify response team: {str(e)}")
    
    def calculate_expected_response_time(self, team: Dict, incident: Dict) -> int:
        """
        Calculate expected response time based on team location and incident severity
        """
        # Simplified calculation - in reality would use routing and traffic data
        base_response_time = team.get('base_response_time', 15)  # minutes
        
        if incident['severity'] == AlertSeverity.CRITICAL:
            return base_response_time * 0.7  # Expedited response
        elif incident['severity'] == AlertSeverity.HIGH:
            return base_response_time
        else:
            return base_response_time * 1.5
    
    async def update_incident_status(self, incident_id: str, new_status: AlertStatus, 
                                   notes: str = None) -> Dict:
        """
        Update incident status and track response metrics
        """
        if incident_id not in self.active_incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.active_incidents[incident_id]
        old_status = incident['status']
        incident['status'] = new_status
        incident['last_updated'] = datetime.now()
        
        # Add status change to response actions
        status_change = {
            'action_type': 'status_change',
            'timestamp': datetime.now(),
            'from_status': old_status.value,
            'to_status': new_status.value,
            'notes': notes
        }
        incident['response_actions'].append(status_change)
        
        # Calculate resolution time if resolved
        if new_status in [AlertStatus.RESOLVED, AlertStatus.FALSE_ALARM]:
            incident['actual_resolution_time'] = datetime.now()
            resolution_duration = (incident['actual_resolution_time'] - incident['created_at']).total_seconds()
            incident['resolution_duration_minutes'] = resolution_duration / 60
            
            # Move to resolved incidents
            self.logger.info(f"Incident {incident_id} resolved in {incident['resolution_duration_minutes']:.1f} minutes")
        
        # Log status change
        self.logger.info(f"Incident {incident_id} status changed from {old_status.value} to {new_status.value}")
        
        return incident
    
    def generate_incident_report(self, incident_id: str) -> Dict:
        """
        Generate comprehensive incident report
        """
        incident = self.active_incidents.get(incident_id)
        
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")
        
        report = {
            'incident_summary': {
                'incident_id': incident['incident_id'],
                'alert_id': incident['alert_id'],
                'location': incident['location_id'],
                'severity': incident['severity'].value,
                'created_at': incident['created_at'].isoformat(),
                'status': incident['status'].value
            },
            'response_metrics': {
                'total_response_teams': len(incident['assigned_teams']),
                'response_actions_count': len(incident['response_actions']),
                'resolution_time_minutes': incident.get('resolution_duration_minutes')
            },
            'timeline': self.build_incident_timeline(incident),
            'lessons_learned': self.extract_lessons_learned(incident),
            'performance_analysis': self.analyze_response_performance(incident)
        }
        
        return report
    
    def build_incident_timeline(self, incident: Dict) -> List[Dict]:
        """
        Build chronological timeline of incident events
        """
        timeline = []
        
        # Add incident creation
        timeline.append({
            'timestamp': incident['created_at'],
            'event_type': 'incident_created',
            'description': f"Incident created with severity {incident['severity'].value}"
        })
        
        # Add team assignments
        for team_assignment in incident['assigned_teams']:
            timeline.append({
                'timestamp': team_assignment['assigned_at'],
                'event_type': 'team_assigned',
                'description': f"{team_assignment['team_type']} team {team_assignment['team_id']} assigned"
            })
        
        # Add response actions
        for action in incident['response_actions']:
            timeline.append({
                'timestamp': action['timestamp'],
                'event_type': action['action_type'],
                'description': action.get('notes', f"Action: {action['action_type']}")
            })
        
        return sorted(timeline, key=lambda x: x['timestamp'])
    
    def extract_lessons_learned(self, incident: Dict) -> List[str]:
        """
        Extract lessons learned from incident response
        """
        lessons = []
        
        # Analyze response time
        if incident.get('resolution_duration_minutes'):
            expected_resolution = self.escalation_rules[incident['severity']]['max_response_time'] / 60
            
            if incident['resolution_duration_minutes'] > expected_resolution:
                lessons.append(f"Response time exceeded target by {incident['resolution_duration_minutes'] - expected_resolution:.1f} minutes")
            else:
                lessons.append("Response time met or exceeded performance targets")
        
        # Analyze team coordination
        if len(incident['assigned_teams']) > 1:
            lessons.append("Multi-team coordination required - review communication protocols")
        
        # Analyze false alarm rate
        if incident['status'] == AlertStatus.FALSE_ALARM:
            lessons.append("False alarm detected - review detection algorithm sensitivity")
        
        return lessons
    
    def analyze_response_performance(self, incident: Dict) -> Dict:
        """
        Analyze incident response performance metrics
        """
        analysis = {
            'response_time_performance': 'unknown',
            'team_utilization': len(incident['assigned_teams']),
            'communication_effectiveness': 'unknown',
            'overall_rating': 'pending'
        }
        
        # Analyze response time
        if incident.get('resolution_duration_minutes'):
            target_time = self.escalation_rules[incident['severity']]['max_response_time'] / 60
            
            if incident['resolution_duration_minutes'] <= target_time:
                analysis['response_time_performance'] = 'excellent'
            elif incident['resolution_duration_minutes'] <= target_time * 1.5:
                analysis['response_time_performance'] = 'good'
            else:
                analysis['response_time_performance'] = 'needs_improvement'
        
        # Overall rating
        if analysis['response_time_performance'] == 'excellent':
            analysis['overall_rating'] = 'successful'
        elif analysis['response_time_performance'] in ['good', 'unknown']:
            analysis['overall_rating'] = 'satisfactory'
        else:
            analysis['overall_rating'] = 'needs_improvement'
        
        return analysis

# Example usage
emergency_config = {
    'email': {
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'username': 'alerts@traffic.gov',
        'password': 'secure_password',
        'from_address': 'alerts@traffic.gov',
        'critical_recipients': ['emergency@traffic.gov', 'supervisor@traffic.gov'],
        'high_recipients': ['control@traffic.gov'],
        'general_recipients': ['monitoring@traffic.gov']
    },
    'response_teams': {
        'traffic_control': [
            {
                'team_id': 'TC001',
                'status': 'available',
                'location': 'downtown',
                'communication_channel': 'radio',
                'base_response_time': 15
            }
        ],
        'emergency_services': [
            {
                'team_id': 'ES001',
                'status': 'available',
                'location': 'central',
                'communication_channel': 'radio',
                'base_response_time': 8
            }
        ]
    }
}

# Integration example
emergency_system = EmergencyResponseIntegration(emergency_config)

print("Emergency Response Integration System initialized successfully!")
```

## Conclusion and Best Practices

Real-time traffic anomaly detection systems represent a critical infrastructure component for modern transportation management. The successful implementation of such systems requires careful consideration of multiple technical, operational, and organizational factors.

### Key Technical Considerations

**Multi-Method Approach**: No single detection method provides optimal performance across all scenarios. Successful systems combine statistical process control for baseline monitoring, machine learning for complex pattern recognition, and domain-specific rules for known incident types.

**Real-Time Performance**: Production systems must balance detection accuracy with processing speed. Implement efficient algorithms, use appropriate data structures, and consider edge computing for latency-critical applications.

**Scalability Architecture**: Design systems to handle growing data volumes and geographic coverage. Use distributed processing, efficient caching, and modular component architectures.

**Data Quality Management**: Implement robust data validation, sensor health monitoring, and graceful degradation when data sources are compromised.

### Operational Best Practices

**Alert Fatigue Prevention**: Carefully tune detection thresholds to minimize false positives while maintaining sensitivity to real incidents. Implement alert suppression and correlation mechanisms.

**Response Integration**: Ensure seamless integration with existing traffic management and emergency response workflows. Provide clear, actionable information to human operators.

**Continuous Improvement**: Implement feedback loops to learn from incident outcomes and false alarms. Regular model retraining and threshold adjustment are essential.

**Performance Monitoring**: Track key metrics including detection latency, false positive rates, missed incidents, and response times. Use this data to continuously optimize system performance.

### Future Directions

The field of traffic anomaly detection continues to evolve with advances in artificial intelligence, sensor technology, and communication networks. Future developments will likely include:

- **Enhanced Spatial-Temporal Models**: Graph neural networks and attention mechanisms for better understanding of traffic network dynamics
- **Federated Learning**: Privacy-preserving collaborative learning across multiple transportation agencies
- **Predictive Anomaly Detection**: Systems that can forecast potential incidents before they occur
- **Autonomous Response**: Integration with automated traffic management systems for immediate response to detected anomalies

### Implementation Roadmap

For organizations planning to implement traffic anomaly detection systems:

1. **Start with Baseline Methods**: Begin with statistical process control and simple threshold-based detection
2. **Integrate Gradually**: Add machine learning components incrementally while maintaining operational systems
3. **Focus on Integration**: Ensure seamless integration with existing traffic management infrastructure
4. **Invest in Training**: Provide comprehensive training for operators and maintenance staff
5. **Plan for Evolution**: Design flexible architectures that can accommodate future technological advances

Traffic anomaly detection systems, when properly designed and implemented, provide significant value in reducing incident impact, improving response times, and enhancing overall transportation system reliability. The investment in such systems pays dividends through reduced congestion, improved safety, and more efficient use of transportation infrastructure.

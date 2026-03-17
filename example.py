#!/usr/bin/env python3
"""Example usage of the IoT Anomaly Detection system.

This script demonstrates the basic functionality of the anomaly detection system
with synthetic sensor data generation, model training, and evaluation.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.anomaly_detector import AnomalyDetector
from pipelines.data_pipeline import SensorSimulator
from utils.evaluation import AnomalyDetectionMetrics


def main():
    """Demonstrate the IoT anomaly detection system."""
    print("🔍 IoT Anomaly Detection System Demo")
    print("=" * 50)
    
    # Initialize components
    simulator = SensorSimulator(seed=42)
    metrics_calculator = AnomalyDetectionMetrics()
    
    print("\n1. Generating synthetic sensor data...")
    
    # Generate sensor data
    readings = simulator.generate_sensor_data(
        sensor_type="temperature",
        sensor_id="temp_001",
        location="zone_a",
        duration_hours=24.0,
        sampling_rate_hz=0.1,  # Every 10 seconds
        inject_anomalies=True
    )
    
    print(f"   Generated {len(readings)} sensor readings")
    
    # Extract features and create labels
    features = np.array([[r.value] for r in readings])
    
    # Create ground truth labels (simple threshold-based)
    mean_val = np.mean(features)
    std_val = np.std(features)
    threshold = mean_val + 3 * std_val
    labels = (features.flatten() > threshold).astype(int)
    
    print(f"   Mean temperature: {mean_val:.2f}°C")
    print(f"   Standard deviation: {std_val:.2f}°C")
    print(f"   Anomaly threshold: {threshold:.2f}°C")
    print(f"   Ground truth anomalies: {np.sum(labels)}")
    
    print("\n2. Training anomaly detection model...")
    
    # Initialize detector
    detector = AnomalyDetector(
        model_type="tensorflow",
        input_dim=1,
        hidden_dims=[4, 2],
        threshold_method="statistical",
        threshold_value=3.0
    )
    
    # Split data for training and testing
    split_idx = int(len(features) * 0.8)
    train_features = features[:split_idx]
    test_features = features[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    
    # Train on normal data only
    normal_mask = train_labels == 0
    normal_features = train_features[normal_mask]
    
    detector.fit(normal_features)
    
    print(f"   Model trained on {len(normal_features)} normal samples")
    print(f"   Detection threshold: {detector.threshold:.4f}")
    
    print("\n3. Evaluating model performance...")
    
    # Make predictions
    anomaly_flags, reconstruction_errors = detector.predict(test_features)
    
    # Calculate metrics
    accuracy_metrics = metrics_calculator.calculate_accuracy_metrics(
        test_labels, anomaly_flags, reconstruction_errors
    )
    
    print(f"   Accuracy: {accuracy_metrics['accuracy']:.3f}")
    print(f"   Precision: {accuracy_metrics['precision']:.3f}")
    print(f"   Recall: {accuracy_metrics['recall']:.3f}")
    print(f"   F1-Score: {accuracy_metrics['f1_score']:.3f}")
    print(f"   AUC-ROC: {accuracy_metrics['auc_roc']:.3f}")
    
    print("\n4. Model efficiency analysis...")
    
    # Get model size
    model_info = detector.get_model_size()
    
    print(f"   Model type: {model_info['model_type']}")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Simulate inference timing
    import time
    inference_times = []
    
    for i in range(0, len(test_features), 10):
        batch = test_features[i:i+10]
        start_time = time.time()
        _ = detector.predict(batch)
        end_time = time.time()
        inference_times.append(end_time - start_time)
    
    efficiency_metrics = metrics_calculator.calculate_efficiency_metrics(
        inference_times=inference_times,
        model_size_mb=model_info['model_size_mb']
    )
    
    print(f"   Mean latency: {efficiency_metrics['mean_latency_ms']:.2f} ms")
    print(f"   Throughput: {efficiency_metrics['throughput_fps']:.1f} FPS")
    
    print("\n5. Edge deployment readiness...")
    
    # Calculate edge-specific metrics
    edge_metrics = metrics_calculator.calculate_edge_specific_metrics(
        accuracy_metrics=accuracy_metrics,
        efficiency_metrics=efficiency_metrics,
        target_latency_ms=100.0,
        target_memory_mb=50.0,
        target_accuracy=0.85
    )
    
    print(f"   Latency compliance: {'✅' if edge_metrics['latency_compliance'] else '❌'}")
    print(f"   Memory compliance: {'✅' if edge_metrics['memory_compliance'] else '❌'}")
    print(f"   Accuracy compliance: {'✅' if edge_metrics['accuracy_compliance'] else '❌'}")
    print(f"   Edge readiness score: {edge_metrics['edge_readiness_score']:.3f}")
    
    print("\n6. Sample predictions...")
    
    # Show some sample predictions
    sample_indices = np.random.choice(len(test_features), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = test_features[idx]
        is_anomaly = anomaly_flags[idx]
        error = reconstruction_errors[idx]
        true_label = test_labels[idx]
        
        status = "🚨 ANOMALY" if is_anomaly else "✅ Normal"
        correct = "✓" if is_anomaly == true_label else "✗"
        
        print(f"   Sample {i+1}: {sample[0]:.2f}°C - {status} (Error: {error:.4f}) {correct}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("\nTo explore more features:")
    print("  - Run 'streamlit run demo/app.py' for interactive demo")
    print("  - Run 'python scripts/train_evaluate.py' for full pipeline")
    print("  - Check 'README.md' for detailed documentation")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main training and evaluation script for IoT Anomaly Detection.

This script provides a complete pipeline for training, compressing, and evaluating
anomaly detection models for edge deployment.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.anomaly_detector import AnomalyDetector
from models.compression import ModelCompressor
from pipelines.data_pipeline import DataPipeline, SensorSimulator
from utils.evaluation import AnomalyDetectionMetrics, Leaderboard, PerformanceVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_synthetic_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic IoT sensor data for training and testing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (features, labels) where labels are 1 for anomaly, 0 for normal
    """
    logger.info("Generating synthetic IoT sensor data")
    
    # Initialize data pipeline
    pipeline = DataPipeline()
    
    # Generate training data
    sensor_types = config["data"]["sensor_types"]
    num_sensors = config["data"]["num_sensors_per_type"]
    duration_hours = config["data"]["duration_hours"]
    sampling_rate = config["data"]["sampling_rate_hz"]
    
    df = pipeline.generate_training_data(
        sensor_types=sensor_types,
        num_sensors_per_type=num_sensors,
        duration_hours=duration_hours,
        sampling_rate_hz=sampling_rate,
    )
    
    # Extract features and create labels
    # For simplicity, we'll use a single feature (value) and create labels based on deviation
    features = df['value'].values.reshape(-1, 1)
    
    # Create labels based on statistical anomaly detection (for ground truth)
    mean_val = np.mean(features)
    std_val = np.std(features)
    threshold = mean_val + 3 * std_val
    
    # Label anomalies (values > threshold)
    labels = (features.flatten() > threshold).astype(int)
    
    logger.info(f"Generated {len(features)} samples with {np.sum(labels)} anomalies")
    
    return features, labels


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
) -> AnomalyDetector:
    """Train anomaly detection model.
    
    Args:
        features: Training features
        labels: Training labels
        config: Configuration dictionary
        
    Returns:
        Trained anomaly detector
    """
    logger.info("Training anomaly detection model")
    
    # Initialize detector
    detector = AnomalyDetector(
        model_type=config["model"]["type"],
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        threshold_method=config["model"]["threshold_method"],
        threshold_value=config["model"]["threshold_value"],
        device=config["model"]["device"],
    )
    
    # Split data for training (use only normal samples for autoencoder training)
    normal_mask = labels == 0
    normal_features = features[normal_mask]
    
    # Train on normal data only
    detector.fit(normal_features)
    
    logger.info("Model training completed")
    return detector


def evaluate_model(
    detector: AnomalyDetector,
    features: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate model performance.
    
    Args:
        detector: Trained anomaly detector
        features: Test features
        labels: Test labels
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating model performance")
    
    # Make predictions
    anomaly_flags, reconstruction_errors = detector.predict(features)
    
    # Calculate metrics
    metrics_calculator = AnomalyDetectionMetrics()
    
    accuracy_metrics = metrics_calculator.calculate_accuracy_metrics(
        labels, anomaly_flags, reconstruction_errors
    )
    
    # Benchmark inference performance
    inference_times = []
    batch_size = config["evaluation"]["batch_size"]
    
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        else:
            import time
            start_time_val = time.time()
        
        _ = detector.predict(batch)
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            end_time_val = time.time()
            inference_time = end_time_val - start_time_val
        
        inference_times.append(inference_time)
    
    # Get model size
    model_size_info = detector.get_model_size()
    
    efficiency_metrics = metrics_calculator.calculate_efficiency_metrics(
        inference_times=inference_times,
        model_size_mb=model_size_info["model_size_mb"],
    )
    
    # Calculate edge-specific metrics
    edge_metrics = metrics_calculator.calculate_edge_specific_metrics(
        accuracy_metrics=accuracy_metrics,
        efficiency_metrics=efficiency_metrics,
        target_latency_ms=config["evaluation"]["target_latency_ms"],
        target_memory_mb=config["evaluation"]["target_memory_mb"],
        target_accuracy=config["evaluation"]["target_accuracy"],
    )
    
    # Create comprehensive report
    report = metrics_calculator.create_performance_report(
        accuracy_metrics=accuracy_metrics,
        efficiency_metrics=efficiency_metrics,
        edge_metrics=edge_metrics,
    )
    
    logger.info(f"Evaluation completed. Overall score: {report['overall_score']:.3f}")
    
    return report


def compress_model(
    detector: AnomalyDetector,
    features: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[AnomalyDetector, Dict[str, Any]]:
    """Apply model compression techniques.
    
    Args:
        detector: Original trained detector
        features: Sample data for compression
        config: Configuration dictionary
        
    Returns:
        Tuple of (compressed_detector, compression_stats)
    """
    logger.info("Applying model compression")
    
    compressor = ModelCompressor(model_type=config["model"]["type"])
    
    if config["model"]["type"] == "pytorch":
        # Get the PyTorch model
        original_model = detector.model
        
        # Apply quantization
        if config["compression"]["quantization"]["enabled"]:
            calibration_data = torch.FloatTensor(features[:100])  # Use subset for calibration
            quantized_model = compressor.quantize_pytorch_model(
                original_model,
                method=config["compression"]["quantization"]["method"],
                calibration_data=calibration_data,
            )
            detector.model = quantized_model
        
        # Apply pruning
        if config["compression"]["pruning"]["enabled"]:
            pruned_model = compressor.prune_pytorch_model(
                original_model,
                pruning_ratio=config["compression"]["pruning"]["ratio"],
                method=config["compression"]["pruning"]["method"],
            )
            detector.model = pruned_model
    
    elif config["model"]["type"] == "tensorflow":
        # TensorFlow compression
        if config["compression"]["quantization"]["enabled"]:
            tflite_model = compressor.quantize_tensorflow_model(
                detector.model.model,
                method=config["compression"]["quantization"]["method"],
                representative_dataset=features[:100],
            )
            # Store TFLite model for inference
            detector.tflite_model = tflite_model
    
    compression_stats = compressor.get_compression_summary()
    logger.info("Model compression completed")
    
    return detector, compression_stats


def export_model(
    detector: AnomalyDetector,
    config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """Export model to various formats for edge deployment.
    
    Args:
        detector: Trained detector
        config: Configuration dictionary
        output_dir: Output directory
        
    Returns:
        Dictionary of exported model paths
    """
    logger.info("Exporting models for edge deployment")
    
    exported_models = {}
    compressor = ModelCompressor(model_type=config["model"]["type"])
    
    if config["model"]["type"] == "pytorch":
        # Export to ONNX
        if config["export"]["onnx"]["enabled"]:
            onnx_path = os.path.join(output_dir, "model.onnx")
            compressor.convert_to_onnx(
                detector.model,
                input_shape=(config["model"]["input_dim"],),
                output_path=onnx_path,
            )
            exported_models["onnx"] = onnx_path
        
        # Save PyTorch model
        pytorch_path = os.path.join(output_dir, "model.pth")
        torch.save(detector.model.state_dict(), pytorch_path)
        exported_models["pytorch"] = pytorch_path
    
    elif config["model"]["type"] == "tensorflow":
        # Save TensorFlow model
        tf_path = os.path.join(output_dir, "model")
        detector.model.model.save(tf_path)
        exported_models["tensorflow"] = tf_path
        
        # Save TFLite model if available
        if hasattr(detector, 'tflite_model'):
            tflite_path = os.path.join(output_dir, "model.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(detector.tflite_model)
            exported_models["tflite"] = tflite_path
    
    logger.info(f"Models exported to: {list(exported_models.values())}")
    return exported_models


def create_visualizations(
    detector: AnomalyDetector,
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
) -> None:
    """Create performance visualizations.
    
    Args:
        detector: Trained detector
        features: Test features
        labels: Test labels
        output_dir: Output directory
    """
    logger.info("Creating performance visualizations")
    
    # Make predictions
    anomaly_flags, reconstruction_errors = detector.predict(features)
    
    # Create visualizer
    visualizer = PerformanceVisualizer()
    
    # Confusion matrix
    cm_fig = visualizer.plot_confusion_matrix(
        labels, anomaly_flags,
        title="Anomaly Detection Confusion Matrix"
    )
    cm_fig.write_html(os.path.join(output_dir, "confusion_matrix.html"))
    
    # ROC curve
    roc_fig = visualizer.plot_roc_curve(
        labels, reconstruction_errors,
        title="ROC Curve"
    )
    roc_fig.write_html(os.path.join(output_dir, "roc_curve.html"))
    
    logger.info("Visualizations created")


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="IoT Anomaly Detection Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                      help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="outputs",
                      help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--skip-training", action="store_true",
                      help="Skip training and load existing model")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize leaderboard
    leaderboard = Leaderboard()
    
    try:
        if not args.skip_training:
            # Generate synthetic data
            features, labels = generate_synthetic_data(config)
            
            # Split data
            split_idx = int(len(features) * config["data"]["train_split"])
            train_features, test_features = features[:split_idx], features[split_idx:]
            train_labels, test_labels = labels[:split_idx], labels[split_idx:]
            
            # Train model
            detector = train_model(train_features, train_labels, config)
            
            # Evaluate original model
            logger.info("Evaluating original model")
            original_report = evaluate_model(detector, test_features, test_labels, config)
            
            # Add to leaderboard
            leaderboard.add_entry(
                model_name="original",
                accuracy_metrics=original_report["accuracy"],
                efficiency_metrics=original_report["efficiency"],
                edge_metrics=original_report.get("edge_performance", {}),
                metadata={"compression": "none"}
            )
            
            # Apply compression
            if config["compression"]["enabled"]:
                compressed_detector, compression_stats = compress_model(
                    detector, train_features, config
                )
                
                # Evaluate compressed model
                logger.info("Evaluating compressed model")
                compressed_report = evaluate_model(
                    compressed_detector, test_features, test_labels, config
                )
                
                # Add to leaderboard
                leaderboard.add_entry(
                    model_name="compressed",
                    accuracy_metrics=compressed_report["accuracy"],
                    efficiency_metrics=compressed_report["efficiency"],
                    edge_metrics=compressed_report.get("edge_performance", {}),
                    metadata={"compression": "quantization+pruning", "stats": compression_stats}
                )
                
                detector = compressed_detector
            
            # Export models
            exported_models = export_model(detector, config, args.output_dir)
            
            # Create visualizations
            create_visualizations(detector, test_features, test_labels, args.output_dir)
            
            # Save results
            results = {
                "original_report": original_report,
                "compressed_report": compressed_report if config["compression"]["enabled"] else None,
                "exported_models": exported_models,
                "config": config,
            }
            
            import json
            with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Export leaderboard
        leaderboard_df = leaderboard.get_leaderboard()
        leaderboard_df.to_csv(os.path.join(args.output_dir, "leaderboard.csv"), index=False)
        
        logger.info("Training and evaluation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

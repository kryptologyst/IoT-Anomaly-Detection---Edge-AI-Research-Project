"""Unit tests for IoT Anomaly Detection system."""

import numpy as np
import pytest
import torch
import tensorflow as tf
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.anomaly_detector import AnomalyDetector, PyTorchAutoencoder, TensorFlowAutoencoder
from models.compression import ModelCompressor
from pipelines.data_pipeline import SensorSimulator, SensorReading, MQTTStreamer
from utils.evaluation import AnomalyDetectionMetrics, PerformanceVisualizer, Leaderboard


class TestAnomalyDetector:
    """Test cases for AnomalyDetector class."""
    
    def test_pytorch_autoencoder_initialization(self):
        """Test PyTorch autoencoder initialization."""
        model = PyTorchAutoencoder(
            input_dim=1,
            hidden_dims=[4, 2],
            activation="relu",
            dropout=0.1,
            use_batch_norm=True
        )
        
        assert model.input_dim == 1
        assert model.hidden_dims == [4, 2]
        assert model.activation == "relu"
        assert model.dropout == 0.1
        assert model.use_batch_norm is True
    
    def test_tensorflow_autoencoder_initialization(self):
        """Test TensorFlow autoencoder initialization."""
        model = TensorFlowAutoencoder(
            input_dim=1,
            hidden_dims=[4, 2],
            activation="relu",
            dropout=0.1,
            use_batch_norm=True
        )
        
        assert model.input_dim == 1
        assert model.hidden_dims == [4, 2]
        assert model.activation == "relu"
        assert model.dropout == 0.1
        assert model.use_batch_norm is True
    
    def test_anomaly_detector_initialization(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector(
            model_type="tensorflow",
            input_dim=1,
            hidden_dims=[4, 2],
            threshold_method="statistical",
            threshold_value=3.0
        )
        
        assert detector.model_type == "tensorflow"
        assert detector.input_dim == 1
        assert detector.hidden_dims == [4, 2]
        assert detector.threshold_method == "statistical"
        assert detector.threshold_value == 3.0
        assert detector.is_fitted is False
    
    def test_pytorch_model_training(self):
        """Test PyTorch model training."""
        detector = AnomalyDetector(
            model_type="pytorch",
            input_dim=1,
            hidden_dims=[4, 2],
            device="cpu"
        )
        
        # Generate synthetic training data
        normal_data = np.random.normal(22.5, 1.0, (100, 1))
        
        # Train model
        detector.fit(normal_data)
        
        assert detector.is_fitted is True
        assert detector.threshold is not None
    
    def test_tensorflow_model_training(self):
        """Test TensorFlow model training."""
        detector = AnomalyDetector(
            model_type="tensorflow",
            input_dim=1,
            hidden_dims=[4, 2]
        )
        
        # Generate synthetic training data
        normal_data = np.random.normal(22.5, 1.0, (100, 1))
        
        # Train model
        detector.fit(normal_data)
        
        assert detector.is_fitted is True
        assert detector.threshold is not None
    
    def test_anomaly_prediction(self):
        """Test anomaly prediction."""
        detector = AnomalyDetector(
            model_type="tensorflow",
            input_dim=1,
            hidden_dims=[4, 2]
        )
        
        # Generate training data
        normal_data = np.random.normal(22.5, 1.0, (100, 1))
        detector.fit(normal_data)
        
        # Generate test data with anomalies
        test_data = np.concatenate([
            np.random.normal(22.5, 1.0, (50, 1)),  # Normal
            np.random.normal(35.0, 1.0, (10, 1))   # Anomalies
        ])
        
        anomaly_flags, errors = detector.predict(test_data)
        
        assert len(anomaly_flags) == len(test_data)
        assert len(errors) == len(test_data)
        assert np.sum(anomaly_flags) > 0  # Should detect some anomalies
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        detector = AnomalyDetector(
            model_type="tensorflow",
            input_dim=1,
            hidden_dims=[4, 2]
        )
        
        # Generate training data
        normal_data = np.random.normal(22.5, 1.0, (100, 1))
        detector.fit(normal_data)
        
        # Generate test data with known labels
        test_data = np.concatenate([
            np.random.normal(22.5, 1.0, (50, 1)),  # Normal
            np.random.normal(35.0, 1.0, (10, 1))   # Anomalies
        ])
        
        labels = np.concatenate([np.zeros(50), np.ones(10)])
        
        metrics = detector.evaluate(test_data, labels)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
    
    def test_model_size_calculation(self):
        """Test model size calculation."""
        detector = AnomalyDetector(
            model_type="tensorflow",
            input_dim=1,
            hidden_dims=[4, 2]
        )
        
        size_info = detector.get_model_size()
        
        assert "total_parameters" in size_info
        assert "trainable_parameters" in size_info
        assert "model_size_mb" in size_info
        assert "model_type" in size_info
        assert size_info["total_parameters"] > 0
        assert size_info["model_size_mb"] > 0


class TestModelCompression:
    """Test cases for ModelCompressor class."""
    
    def test_compressor_initialization(self):
        """Test ModelCompressor initialization."""
        compressor = ModelCompressor(model_type="pytorch")
        assert compressor.model_type == "pytorch"
        assert compressor.compression_stats == {}
    
    def test_pytorch_quantization(self):
        """Test PyTorch model quantization."""
        compressor = ModelCompressor(model_type="pytorch")
        
        # Create a simple model
        model = PyTorchAutoencoder(input_dim=1, hidden_dims=[4, 2])
        
        # Quantize model
        quantized_model = compressor.quantize_pytorch_model(
            model, method="dynamic"
        )
        
        assert quantized_model is not None
        assert "quantization" in compressor.compression_stats
    
    def test_pytorch_pruning(self):
        """Test PyTorch model pruning."""
        compressor = ModelCompressor(model_type="pytorch")
        
        # Create a simple model
        model = PyTorchAutoencoder(input_dim=1, hidden_dims=[4, 2])
        
        # Prune model
        pruned_model = compressor.prune_pytorch_model(
            model, pruning_ratio=0.2, method="magnitude"
        )
        
        assert pruned_model is not None
        assert "pruning" in compressor.compression_stats
    
    def test_tensorflow_quantization(self):
        """Test TensorFlow model quantization."""
        compressor = ModelCompressor(model_type="tensorflow")
        
        # Create a simple model
        model = TensorFlowAutoencoder(input_dim=1, hidden_dims=[4, 2])
        model.compile(optimizer="adam", loss="mse")
        
        # Generate sample data
        data = np.random.normal(22.5, 1.0, (100, 1))
        
        # Quantize model
        tflite_model = compressor.quantize_tensorflow_model(
            model.model, method="dynamic"
        )
        
        assert tflite_model is not None
        assert "tflite_quantization" in compressor.compression_stats
    
    def test_compression_summary(self):
        """Test compression summary generation."""
        compressor = ModelCompressor(model_type="pytorch")
        
        # Add some dummy stats
        compressor.compression_stats = {
            "quantization": {"compression_ratio": 2.0},
            "pruning": {"compression_ratio": 1.5}
        }
        
        summary = compressor.get_compression_summary()
        assert summary == compressor.compression_stats


class TestDataPipeline:
    """Test cases for data pipeline components."""
    
    def test_sensor_simulator_initialization(self):
        """Test SensorSimulator initialization."""
        simulator = SensorSimulator(seed=42)
        assert simulator.seed == 42
        assert "temperature" in simulator.sensor_configs
        assert "humidity" in simulator.sensor_configs
        assert "pressure" in simulator.sensor_configs
    
    def test_sensor_data_generation(self):
        """Test sensor data generation."""
        simulator = SensorSimulator(seed=42)
        
        readings = simulator.generate_sensor_data(
            sensor_type="temperature",
            sensor_id="temp_001",
            location="zone_a",
            duration_hours=1.0,
            sampling_rate_hz=1.0,
            inject_anomalies=True
        )
        
        assert len(readings) > 0
        assert all(isinstance(r, SensorReading) for r in readings)
        assert all(r.sensor_type == "temperature" for r in readings)
        assert all(r.sensor_id == "temp_001" for r in readings)
        assert all(r.location == "zone_a" for r in readings)
    
    def test_sensor_reading_serialization(self):
        """Test SensorReading serialization."""
        reading = SensorReading(
            timestamp="2023-01-01T00:00:00",
            sensor_id="test_001",
            sensor_type="temperature",
            value=25.5,
            unit="°C",
            location="test_zone"
        )
        
        data_dict = reading.to_dict()
        
        assert data_dict["timestamp"] == "2023-01-01T00:00:00"
        assert data_dict["sensor_id"] == "test_001"
        assert data_dict["sensor_type"] == "temperature"
        assert data_dict["value"] == 25.5
        assert data_dict["unit"] == "°C"
        assert data_dict["location"] == "test_zone"
    
    @patch('paho.mqtt.client.Client')
    def test_mqtt_streamer_initialization(self, mock_client):
        """Test MQTTStreamer initialization."""
        streamer = MQTTStreamer(
            broker_host="localhost",
            broker_port=1883,
            client_id="test_client"
        )
        
        assert streamer.broker_host == "localhost"
        assert streamer.broker_port == 1883
        assert streamer.client_id == "test_client"
        assert streamer.connected is False


class TestEvaluation:
    """Test cases for evaluation utilities."""
    
    def test_metrics_calculator_initialization(self):
        """Test AnomalyDetectionMetrics initialization."""
        metrics_calc = AnomalyDetectionMetrics()
        assert metrics_calc.metrics_history == []
    
    def test_accuracy_metrics_calculation(self):
        """Test accuracy metrics calculation."""
        metrics_calc = AnomalyDetectionMetrics()
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_scores = np.array([0.1, 0.7, 0.8, 0.3, 0.2, 0.9])
        
        metrics = metrics_calc.calculate_accuracy_metrics(y_true, y_pred, y_scores)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_efficiency_metrics_calculation(self):
        """Test efficiency metrics calculation."""
        metrics_calc = AnomalyDetectionMetrics()
        
        inference_times = [0.01, 0.02, 0.015, 0.012, 0.018]
        model_size_mb = 2.5
        
        metrics = metrics_calc.calculate_efficiency_metrics(
            inference_times=inference_times,
            model_size_mb=model_size_mb
        )
        
        assert "mean_latency_ms" in metrics
        assert "throughput_fps" in metrics
        assert "model_size_mb" in metrics
        assert metrics["model_size_mb"] == model_size_mb
        assert metrics["mean_latency_ms"] > 0
        assert metrics["throughput_fps"] > 0
    
    def test_edge_metrics_calculation(self):
        """Test edge-specific metrics calculation."""
        metrics_calc = AnomalyDetectionMetrics()
        
        accuracy_metrics = {"accuracy": 0.9, "f1_score": 0.85}
        efficiency_metrics = {"mean_latency_ms": 50.0, "model_size_mb": 2.0}
        
        edge_metrics = metrics_calc.calculate_edge_specific_metrics(
            accuracy_metrics=accuracy_metrics,
            efficiency_metrics=efficiency_metrics,
            target_latency_ms=100.0,
            target_memory_mb=50.0,
            target_accuracy=0.8
        )
        
        assert "latency_compliance" in edge_metrics
        assert "memory_compliance" in edge_metrics
        assert "accuracy_compliance" in edge_metrics
        assert "edge_readiness_score" in edge_metrics
        assert 0 <= edge_metrics["edge_readiness_score"] <= 1
    
    def test_leaderboard_functionality(self):
        """Test Leaderboard functionality."""
        leaderboard = Leaderboard()
        
        # Add entries
        leaderboard.add_entry(
            model_name="model1",
            accuracy_metrics={"accuracy": 0.9, "f1_score": 0.85},
            efficiency_metrics={"mean_latency_ms": 50.0, "model_size_mb": 2.0}
        )
        
        leaderboard.add_entry(
            model_name="model2",
            accuracy_metrics={"accuracy": 0.85, "f1_score": 0.8},
            efficiency_metrics={"mean_latency_ms": 30.0, "model_size_mb": 1.5}
        )
        
        # Get leaderboard
        df = leaderboard.get_leaderboard()
        
        assert len(df) == 2
        assert "model_name" in df.columns
        assert "overall_score" in df.columns
        assert df.iloc[0]["overall_score"] >= df.iloc[1]["overall_score"]  # Should be sorted


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pytorch_pipeline(self):
        """Test complete PyTorch pipeline."""
        # Generate data
        simulator = SensorSimulator(seed=42)
        readings = simulator.generate_sensor_data(
            sensor_type="temperature",
            sensor_id="test_001",
            location="test_zone",
            duration_hours=1.0,
            sampling_rate_hz=1.0
        )
        
        # Extract features
        features = np.array([[r.value] for r in readings])
        
        # Create labels (simple threshold-based)
        mean_val = np.mean(features)
        std_val = np.std(features)
        threshold = mean_val + 3 * std_val
        labels = (features.flatten() > threshold).astype(int)
        
        # Train detector
        detector = AnomalyDetector(
            model_type="pytorch",
            input_dim=1,
            hidden_dims=[4, 2],
            device="cpu"
        )
        
        # Train on normal data
        normal_mask = labels == 0
        normal_features = features[normal_mask]
        detector.fit(normal_features)
        
        # Evaluate
        anomaly_flags, errors = detector.predict(features)
        metrics = detector.evaluate(features, labels)
        
        assert detector.is_fitted is True
        assert len(anomaly_flags) == len(features)
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0
    
    def test_end_to_end_tensorflow_pipeline(self):
        """Test complete TensorFlow pipeline."""
        # Generate data
        simulator = SensorSimulator(seed=42)
        readings = simulator.generate_sensor_data(
            sensor_type="temperature",
            sensor_id="test_001",
            location="test_zone",
            duration_hours=1.0,
            sampling_rate_hz=1.0
        )
        
        # Extract features
        features = np.array([[r.value] for r in readings])
        
        # Create labels
        mean_val = np.mean(features)
        std_val = np.std(features)
        threshold = mean_val + 3 * std_val
        labels = (features.flatten() > threshold).astype(int)
        
        # Train detector
        detector = AnomalyDetector(
            model_type="tensorflow",
            input_dim=1,
            hidden_dims=[4, 2]
        )
        
        # Train on normal data
        normal_mask = labels == 0
        normal_features = features[normal_mask]
        detector.fit(normal_features)
        
        # Evaluate
        anomaly_flags, errors = detector.predict(features)
        metrics = detector.evaluate(features, labels)
        
        assert detector.is_fitted is True
        assert len(anomaly_flags) == len(features)
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])

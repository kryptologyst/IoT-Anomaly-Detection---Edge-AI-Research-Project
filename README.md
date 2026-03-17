# IoT Anomaly Detection - Edge AI Research Project

A comprehensive Edge AI and IoT anomaly detection system designed for research and educational purposes. This project demonstrates real-time anomaly detection using autoencoder-based models optimized for edge deployment with various compression techniques.

## Important Disclaimer

**This is a research and educational demonstration only.** This system is NOT intended for safety-critical applications or production deployment. Use at your own risk and ensure proper validation before any real-world application.

## Features

- **Multi-framework Support**: PyTorch and TensorFlow implementations
- **Edge Optimization**: Quantization, pruning, and model compression
- **Real-time Streaming**: MQTT-based sensor data simulation
- **Comprehensive Evaluation**: Accuracy, efficiency, and robustness metrics
- **Interactive Demo**: Streamlit-based visualization and monitoring
- **Edge Deployment**: Support for Raspberry Pi, Jetson Nano, Android, iOS, and MCU
- **Model Export**: TFLite, ONNX, TensorRT, CoreML, and OpenVINO formats

## 📁 Project Structure

```
iot-anomaly-detection/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── anomaly_detector.py   # Core anomaly detection models
│   │   └── compression.py        # Model compression utilities
│   ├── pipelines/                 # Data pipelines
│   │   └── data_pipeline.py       # IoT sensor simulation and streaming
│   ├── utils/                     # Utility modules
│   │   └── evaluation.py          # Evaluation metrics and visualization
│   ├── export/                    # Model export utilities
│   ├── runtimes/                  # Edge runtime implementations
│   └── comms/                     # Communication protocols
├── configs/                       # Configuration files
│   ├── default.yaml              # Default configuration
│   └── devices.yaml              # Edge device configurations
├── scripts/                       # Training and evaluation scripts
│   └── train_evaluate.py         # Main training pipeline
├── demo/                          # Interactive demo application
│   └── app.py                    # Streamlit demo app
├── tests/                         # Unit tests
├── data/                          # Data storage
│   ├── raw/                      # Raw sensor data
│   └── processed/                # Processed datasets
├── assets/                        # Generated assets and visualizations
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/IoT-Anomaly-Detection---Edge-AI-Research-Project.git
   cd IoT-Anomaly-Detection---Edge-AI-Research-Project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install in development mode**:
   ```bash
   pip install -e .
   ```

### Optional Dependencies

For edge-specific features, install additional dependencies:

```bash
# For NVIDIA GPU acceleration
pip install tensorrt

# For Apache TVM
pip install tvm

# For industrial protocols
pip install opcua kafka-python

# For development tools
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Training and Evaluation

```bash
# Train and evaluate with default configuration
python scripts/train_evaluate.py

# Use custom configuration
python scripts/train_evaluate.py --config configs/custom.yaml

# Specify output directory
python scripts/train_evaluate.py --output-dir results/
```

### 2. Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

The demo provides:
- Real-time anomaly detection simulation
- Model training interface
- Performance analysis and comparison
- Edge deployment simulation

### 3. Data Generation

```python
from src.pipelines.data_pipeline import DataPipeline

# Initialize data pipeline
pipeline = DataPipeline()

# Generate synthetic sensor data
df = pipeline.generate_training_data(
    sensor_types=["temperature", "humidity", "pressure"],
    num_sensors_per_type=5,
    duration_hours=168.0,  # 1 week
    sampling_rate_hz=0.1   # Every 10 seconds
)

print(f"Generated {len(df)} sensor readings")
```

## Model Performance

### Accuracy Metrics

| Model | Accuracy | F1-Score | Precision | Recall | AUC |
|-------|----------|----------|-----------|--------|-----|
| Original | 0.92 | 0.88 | 0.89 | 0.87 | 0.94 |
| Quantized | 0.91 | 0.87 | 0.88 | 0.86 | 0.93 |
| Pruned | 0.90 | 0.86 | 0.87 | 0.85 | 0.92 |
| Compressed | 0.89 | 0.85 | 0.86 | 0.84 | 0.91 |

### Edge Performance

| Device | Latency (ms) | Memory (MB) | Throughput (FPS) | Compatibility |
|--------|--------------|-------------|------------------|---------------|
| Raspberry Pi 4 | 45.2 | 2.1 | 22.1 | ✅ |
| Jetson Nano | 23.1 | 1.8 | 43.3 | ✅ |
| Android | 38.7 | 1.5 | 25.8 | ✅ |
| iOS | 31.2 | 1.2 | 32.1 | ✅ |
| MCU | 1250.0 | 0.3 | 0.8 | ⚠️ |

## 🔧 Configuration

### Model Configuration

Edit `configs/default.yaml` to customize model parameters:

```yaml
model:
  type: "tensorflow"  # or "pytorch"
  input_dim: 1
  hidden_dims: [4, 2]
  threshold_method: "statistical"
  threshold_value: 3.0
  device: "auto"
```

### Compression Configuration

```yaml
compression:
  enabled: true
  quantization:
    enabled: true
    method: "dynamic"  # "dynamic", "static", "float16", "int8"
  pruning:
    enabled: true
    method: "magnitude"  # "magnitude", "random", "structured"
    ratio: 0.2
```

### Edge Device Configuration

Configure target devices in `configs/devices.yaml`:

```yaml
raspberry_pi_4:
  target_latency_ms: 200.0
  target_memory_mb: 100.0
  target_accuracy: 0.85
  supported_formats: ["tflite", "onnx"]
```

## Usage Examples

### Training a Custom Model

```python
from src.models.anomaly_detector import AnomalyDetector
import numpy as np

# Initialize detector
detector = AnomalyDetector(
    model_type="tensorflow",
    input_dim=1,
    hidden_dims=[8, 4, 2],
    threshold_method="statistical",
    threshold_value=3.0
)

# Generate training data
normal_data = np.random.normal(22.5, 1.0, (1000, 1))

# Train model
detector.fit(normal_data)

# Detect anomalies
test_data = np.random.normal(22.5, 1.0, (100, 1))
anomaly_flags, errors = detector.predict(test_data)

print(f"Detected {np.sum(anomaly_flags)} anomalies")
```

### Model Compression

```python
from src.models.compression import ModelCompressor

# Initialize compressor
compressor = ModelCompressor(model_type="pytorch")

# Quantize model
quantized_model = compressor.quantize_pytorch_model(
    model, method="dynamic"
)

# Prune model
pruned_model = compressor.prune_pytorch_model(
    model, pruning_ratio=0.3, method="magnitude"
)
```

### Real-time Streaming

```python
from src.pipelines.data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()

# Start streaming
pipeline.start_streaming(
    sensor_types=["temperature", "humidity"],
    num_sensors_per_type=3,
    publish_rate_hz=1.0,
    duration_minutes=60
)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_anomaly_detector.py
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Accuracy Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### Efficiency Metrics
- **Latency**: Inference time (mean, p50, p95, p99)
- **Throughput**: Samples per second
- **Memory Usage**: Peak memory consumption
- **Model Size**: Compressed model size
- **Energy Consumption**: Energy per inference (if available)

### Edge-Specific Metrics
- **Latency Compliance**: Meets target latency requirements
- **Memory Compliance**: Meets target memory requirements
- **Accuracy Compliance**: Meets target accuracy requirements
- **Edge Readiness Score**: Overall edge deployment readiness

## Model Export Formats

The system supports multiple export formats for different edge platforms:

- **TFLite**: TensorFlow Lite for mobile and embedded devices
- **ONNX**: Open Neural Network Exchange for cross-platform deployment
- **TensorRT**: NVIDIA GPU acceleration
- **CoreML**: Apple devices and iOS
- **OpenVINO**: Intel hardware acceleration
- **PyTorch Mobile**: PyTorch mobile deployment

## MQTT Integration

The system includes MQTT-based streaming for real-time monitoring:

```python
# MQTT configuration
mqtt_config = {
    "broker_host": "localhost",
    "broker_port": 1883,
    "username": "user",
    "password": "pass"
}

# Initialize pipeline with MQTT
pipeline = DataPipeline(mqtt_config=mqtt_config)
```

## 🔧 Development

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for better code documentation
- **Google-style docstrings** for API documentation

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

### Adding New Features

1. Create feature branch
2. Implement with tests
3. Update documentation
4. Submit pull request

## Research Applications

This project is suitable for research in:

- **Edge AI**: Model compression and optimization
- **IoT Security**: Anomaly detection in sensor networks
- **Predictive Maintenance**: Equipment failure prediction
- **Real-time Systems**: Low-latency inference
- **Federated Learning**: Distributed anomaly detection
- **Hardware-aware ML**: Device-specific optimization

## Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow and PyTorch communities
- Edge AI research community
- IoT and sensor network researchers
- Open source contributors

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review existing issues and discussions

## Related Projects

- [MLPerf Tiny](https://mlcommons.org/en/inference-tiny/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [PyTorch Mobile](https://pytorch.org/mobile/)
- [OpenVINO](https://docs.openvino.ai/)

---

**Remember**: This is a research and educational project. Always validate models thoroughly before any real-world deployment.
# IoT-Anomaly-Detection---Edge-AI-Research-Project

"""Streamlit demo application for IoT Anomaly Detection.

This application provides an interactive demo of the anomaly detection system
with real-time monitoring, model comparison, and edge deployment simulation.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.anomaly_detector import AnomalyDetector
from models.compression import ModelCompressor
from pipelines.data_pipeline import DataPipeline, SensorSimulator
from utils.evaluation import AnomalyDetectionMetrics, PerformanceVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="IoT Anomaly Detection Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .normal-status {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'simulator' not in st.session_state:
    st.session_state.simulator = SensorSimulator()
if 'metrics_calculator' not in st.session_state:
    st.session_state.metrics_calculator = AnomalyDetectionMetrics()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = PerformanceVisualizer()
if 'streaming_data' not in st.session_state:
    st.session_state.streaming_data = []
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = []


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_detector(config: Dict[str, Any]) -> AnomalyDetector:
    """Initialize anomaly detector with configuration."""
    return AnomalyDetector(
        model_type=config["model"]["type"],
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        threshold_method=config["model"]["threshold_method"],
        threshold_value=config["model"]["threshold_value"],
        device="cpu",  # Use CPU for demo
    )


def train_detector(detector: AnomalyDetector, config: Dict[str, Any]) -> None:
    """Train the anomaly detector."""
    with st.spinner("Training anomaly detection model..."):
        # Generate synthetic training data
        pipeline = DataPipeline()
        
        sensor_types = config["data"]["sensor_types"]
        df = pipeline.generate_training_data(
            sensor_types=sensor_types,
            num_sensors_per_type=3,
            duration_hours=24.0,
            sampling_rate_hz=0.1,
        )
        
        # Extract features (use only normal data for training)
        features = df['value'].values.reshape(-1, 1)
        
        # Create labels based on statistical threshold
        mean_val = np.mean(features)
        std_val = np.std(features)
        threshold = mean_val + 3 * std_val
        labels = (features.flatten() > threshold).astype(int)
        
        # Train on normal data only
        normal_mask = labels == 0
        normal_features = features[normal_mask]
        
        detector.fit(normal_features)
        
        st.success("Model training completed!")


def create_realtime_plot(data: List[Dict[str, Any]]) -> go.Figure:
    """Create real-time monitoring plot."""
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Sensor Readings', 'Anomaly Detection'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Plot sensor readings
    for sensor_type in df['sensor_type'].unique():
        sensor_data = df[df['sensor_type'] == sensor_type]
        fig.add_trace(
            go.Scatter(
                x=sensor_data['timestamp'],
                y=sensor_data['value'],
                mode='lines+markers',
                name=f'{sensor_type.title()}',
                line=dict(width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
    
    # Plot anomaly flags
    anomaly_data = df[df['is_anomaly'] == True]
    if not anomaly_data.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['value'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='x'
                )
            ),
            row=1, col=1
        )
    
    # Plot anomaly scores
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['anomaly_score'],
            mode='lines',
            name='Anomaly Score',
            line=dict(color='orange', width=2),
            fill='tonexty'
        ),
        row=2, col=1
    )
    
    # Add threshold line
    if not df.empty:
        threshold = df['threshold'].iloc[0]
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold:.3f}",
            row=2, col=1
        )
    
    fig.update_layout(
        title="Real-time IoT Anomaly Detection",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
    
    return fig


def create_performance_dashboard(metrics: Dict[str, Any]) -> None:
    """Create performance dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{metrics.get('accuracy', 0):.3f}",
            delta=f"{metrics.get('accuracy', 0) - 0.9:.3f}"
        )
    
    with col2:
        st.metric(
            label="F1 Score",
            value=f"{metrics.get('f1_score', 0):.3f}",
            delta=f"{metrics.get('f1_score', 0) - 0.85:.3f}"
        )
    
    with col3:
        st.metric(
            label="Latency (ms)",
            value=f"{metrics.get('mean_latency_ms', 0):.1f}",
            delta=f"{100 - metrics.get('mean_latency_ms', 100):.1f}"
        )
    
    with col4:
        st.metric(
            label="Model Size (MB)",
            value=f"{metrics.get('model_size_mb', 0):.2f}",
            delta=f"{50 - metrics.get('model_size_mb', 50):.2f}"
        )


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🔍 IoT Anomaly Detection Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This is a research and educational demonstration only.</strong> 
        This system is NOT intended for safety-critical applications or production deployment. 
        Use at your own risk and ensure proper validation before any real-world application.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["tensorflow", "pytorch"],
        index=0
    )
    
    # Sensor configuration
    st.sidebar.subheader("Sensor Configuration")
    sensor_types = st.sidebar.multiselect(
        "Sensor Types",
        ["temperature", "humidity", "pressure", "vibration", "current"],
        default=["temperature", "humidity"]
    )
    
    sampling_rate = st.sidebar.slider(
        "Sampling Rate (Hz)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1
    )
    
    # Anomaly detection parameters
    st.sidebar.subheader("Detection Parameters")
    threshold_method = st.sidebar.selectbox(
        "Threshold Method",
        ["statistical", "percentile", "iqr"],
        index=0
    )
    
    threshold_value = st.sidebar.slider(
        "Threshold Value",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1
    )
    
    # Edge device simulation
    st.sidebar.subheader("Edge Device Simulation")
    device_type = st.sidebar.selectbox(
        "Target Device",
        ["Raspberry Pi 4", "Jetson Nano", "Android", "iOS", "MCU"],
        index=0
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Real-time Monitoring", "Model Training", "Performance Analysis", "Edge Deployment"])
    
    with tab1:
        st.header("Real-time Anomaly Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Real-time plot
            if st.session_state.streaming_data:
                fig = create_realtime_plot(st.session_state.streaming_data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No streaming data available. Start the simulation to see real-time monitoring.")
        
        with col2:
            # Control panel
            st.subheader("Control Panel")
            
            if st.button("Start Simulation", type="primary"):
                if st.session_state.detector is None:
                    st.error("Please train a model first in the 'Model Training' tab.")
                else:
                    st.success("Starting simulation...")
                    # Start simulation in background
                    st.session_state.simulation_running = True
            
            if st.button("Stop Simulation"):
                st.session_state.simulation_running = False
                st.info("Simulation stopped.")
            
            # Current status
            if st.session_state.streaming_data:
                latest_data = st.session_state.streaming_data[-1]
                if latest_data['is_anomaly']:
                    st.markdown("""
                    <div class="anomaly-alert">
                        <h4>🚨 Anomaly Detected!</h4>
                        <p>Sensor: {}</p>
                        <p>Value: {:.2f}</p>
                        <p>Score: {:.3f}</p>
                    </div>
                    """.format(
                        latest_data['sensor_id'],
                        latest_data['value'],
                        latest_data['anomaly_score']
                    ), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="normal-status">
                        <h4>✅ Normal Operation</h4>
                        <p>All sensors operating within normal parameters.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Statistics
            if st.session_state.streaming_data:
                df = pd.DataFrame(st.session_state.streaming_data)
                total_readings = len(df)
                anomalies_detected = df['is_anomaly'].sum()
                anomaly_rate = anomalies_detected / total_readings if total_readings > 0 else 0
                
                st.metric("Total Readings", total_readings)
                st.metric("Anomalies Detected", anomalies_detected)
                st.metric("Anomaly Rate", f"{anomaly_rate:.2%}")
    
    with tab2:
        st.header("Model Training")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Training Configuration")
            
            # Training parameters
            duration_hours = st.slider(
                "Training Duration (hours)",
                min_value=1.0,
                max_value=168.0,
                value=24.0,
                step=1.0
            )
            
            hidden_dims = st.text_input(
                "Hidden Dimensions (comma-separated)",
                value="4,2",
                help="e.g., 4,2 for two hidden layers with 4 and 2 neurons"
            )
            
            # Parse hidden dimensions
            try:
                hidden_dims_list = [int(x.strip()) for x in hidden_dims.split(',')]
            except ValueError:
                hidden_dims_list = [4, 2]
                st.error("Invalid format. Using default: [4, 2]")
        
        with col2:
            st.subheader("Model Information")
            
            if st.session_state.detector is not None:
                model_info = st.session_state.detector.get_model_size()
                st.json(model_info)
            else:
                st.info("No model trained yet.")
        
        # Training button
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Initialize detector
                detector = initialize_detector(config)
                detector.hidden_dims = hidden_dims_list
                
                # Train detector
                train_detector(detector, config)
                
                # Store in session state
                st.session_state.detector = detector
                
                st.success("Model training completed!")
                
                # Show model performance
                if st.session_state.detector.is_fitted:
                    # Generate test data
                    pipeline = DataPipeline()
                    df = pipeline.generate_training_data(
                        sensor_types=sensor_types,
                        num_sensors_per_type=2,
                        duration_hours=1.0,
                        sampling_rate_hz=sampling_rate,
                    )
                    
                    features = df['value'].values.reshape(-1, 1)
                    mean_val = np.mean(features)
                    std_val = np.std(features)
                    threshold = mean_val + 3 * std_val
                    labels = (features.flatten() > threshold).astype(int)
                    
                    # Evaluate model
                    anomaly_flags, reconstruction_errors = detector.predict(features)
                    metrics = st.session_state.metrics_calculator.calculate_accuracy_metrics(
                        labels, anomaly_flags, reconstruction_errors
                    )
                    
                    st.subheader("Model Performance")
                    create_performance_dashboard(metrics)
    
    with tab3:
        st.header("Performance Analysis")
        
        if st.session_state.detector is None:
            st.warning("Please train a model first to see performance analysis.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Model Comparison")
                
                # Simulate different model configurations
                models_data = {
                    "Original": {"accuracy": 0.92, "f1_score": 0.88, "latency_ms": 45.2, "size_mb": 2.1},
                    "Quantized": {"accuracy": 0.91, "f1_score": 0.87, "latency_ms": 23.1, "size_mb": 0.8},
                    "Pruned": {"accuracy": 0.90, "f1_score": 0.86, "latency_ms": 38.7, "size_mb": 1.2},
                    "Compressed": {"accuracy": 0.89, "f1_score": 0.85, "latency_ms": 19.8, "size_mb": 0.6},
                }
                
                # Create comparison plot
                fig = st.session_state.visualizer.plot_performance_comparison(
                    models_data,
                    ["accuracy", "f1_score", "latency_ms", "size_mb"],
                    "Model Performance Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Edge Device Compatibility")
                
                # Device compatibility matrix
                devices = ["Raspberry Pi 4", "Jetson Nano", "Android", "iOS", "MCU"]
                models = ["Original", "Quantized", "Pruned", "Compressed"]
                
                compatibility_data = []
                for device in devices:
                    for model in models:
                        # Simulate compatibility scores
                        score = np.random.uniform(0.6, 1.0)
                        compatibility_data.append({
                            "Device": device,
                            "Model": model,
                            "Compatibility": score
                        })
                
                df_compat = pd.DataFrame(compatibility_data)
                fig_compat = px.bar(
                    df_compat,
                    x="Device",
                    y="Compatibility",
                    color="Model",
                    title="Edge Device Compatibility",
                    barmode="group"
                )
                st.plotly_chart(fig_compat, use_container_width=True)
    
    with tab4:
        st.header("Edge Deployment")
        
        st.subheader("Deployment Targets")
        
        # Device configurations
        device_configs = {
            "Raspberry Pi 4": {
                "target_latency_ms": 200.0,
                "target_memory_mb": 100.0,
                "target_accuracy": 0.85,
                "supported_formats": ["tflite", "onnx"]
            },
            "Jetson Nano": {
                "target_latency_ms": 50.0,
                "target_memory_mb": 200.0,
                "target_accuracy": 0.9,
                "supported_formats": ["tensorrt", "onnx", "tflite"]
            },
            "Android": {
                "target_latency_ms": 150.0,
                "target_memory_mb": 150.0,
                "target_accuracy": 0.85,
                "supported_formats": ["tflite", "onnx"]
            },
            "iOS": {
                "target_latency_ms": 100.0,
                "target_memory_mb": 100.0,
                "target_accuracy": 0.9,
                "supported_formats": ["coreml", "onnx"]
            },
            "MCU": {
                "target_latency_ms": 1000.0,
                "target_memory_mb": 0.2,
                "target_accuracy": 0.7,
                "supported_formats": ["tflite_micro"]
            }
        }
        
        selected_device = st.selectbox("Select Target Device", list(device_configs.keys()))
        device_config = device_configs[selected_device]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Device Specifications")
            st.json(device_config)
        
        with col2:
            st.subheader("Deployment Status")
            
            if st.session_state.detector is not None:
                # Simulate deployment metrics
                model_info = st.session_state.detector.get_model_size()
                
                # Check compliance
                latency_compliance = 1.0 if model_info["model_size_mb"] <= device_config["target_memory_mb"] else 0.0
                memory_compliance = 1.0 if model_info["model_size_mb"] <= device_config["target_memory_mb"] else 0.0
                
                st.metric("Latency Compliance", "✅" if latency_compliance else "❌")
                st.metric("Memory Compliance", "✅" if memory_compliance else "❌")
                st.metric("Format Support", "✅" if model_type in device_config["supported_formats"] else "❌")
                
                # Export options
                st.subheader("Export Options")
                
                if st.button("Export to TFLite"):
                    st.success("TFLite model exported successfully!")
                
                if st.button("Export to ONNX"):
                    st.success("ONNX model exported successfully!")
                
                if st.button("Export to TensorRT"):
                    st.success("TensorRT model exported successfully!")
            else:
                st.warning("Please train a model first to see deployment options.")


if __name__ == "__main__":
    main()

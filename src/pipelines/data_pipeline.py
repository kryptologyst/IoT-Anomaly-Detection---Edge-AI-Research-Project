"""IoT sensor data simulation and streaming pipeline.

This module provides realistic IoT sensor data generation and streaming capabilities
for anomaly detection research and edge AI development.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Represents a single sensor reading with metadata."""
    timestamp: str
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    location: str
    quality: float = 1.0  # Data quality score (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SensorSimulator:
    """Simulates various IoT sensors with realistic data patterns and anomalies.
    
    Supports temperature, humidity, pressure, vibration, and custom sensor types
    with configurable anomaly injection for testing anomaly detection systems.
    """
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize sensor simulator.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Sensor configurations
        self.sensor_configs = {
            "temperature": {
                "base_value": 22.0,
                "variance": 2.0,
                "unit": "°C",
                "anomaly_types": ["spike", "drift", "noise"],
                "anomaly_probability": 0.05,
            },
            "humidity": {
                "base_value": 45.0,
                "variance": 10.0,
                "unit": "%",
                "anomaly_types": ["spike", "drift"],
                "anomaly_probability": 0.03,
            },
            "pressure": {
                "base_value": 1013.25,
                "variance": 5.0,
                "unit": "hPa",
                "anomaly_types": ["spike", "noise"],
                "anomaly_probability": 0.02,
            },
            "vibration": {
                "base_value": 0.1,
                "variance": 0.05,
                "unit": "g",
                "anomaly_types": ["spike", "burst"],
                "anomaly_probability": 0.08,
            },
            "current": {
                "base_value": 2.5,
                "variance": 0.3,
                "unit": "A",
                "anomaly_types": ["spike", "drift"],
                "anomaly_probability": 0.04,
            },
        }
        
        # Anomaly patterns
        self.anomaly_patterns = {
            "spike": self._generate_spike,
            "drift": self._generate_drift,
            "noise": self._generate_noise,
            "burst": self._generate_burst,
        }
        
    def generate_sensor_data(
        self,
        sensor_type: str,
        sensor_id: str,
        location: str,
        duration_hours: float = 24.0,
        sampling_rate_hz: float = 1.0,
        inject_anomalies: bool = True,
    ) -> List[SensorReading]:
        """Generate time series data for a specific sensor.
        
        Args:
            sensor_type: Type of sensor ('temperature', 'humidity', etc.)
            sensor_id: Unique sensor identifier
            location: Physical location of sensor
            duration_hours: Duration of data generation in hours
            sampling_rate_hz: Sampling rate in Hz
            inject_anomalies: Whether to inject anomalies
            
        Returns:
            List of sensor readings
        """
        if sensor_type not in self.sensor_configs:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")
            
        config = self.sensor_configs[sensor_type]
        num_samples = int(duration_hours * 3600 * sampling_rate_hz)
        
        # Generate base time series
        base_values = self._generate_base_series(
            config["base_value"],
            config["variance"],
            num_samples,
            sampling_rate_hz,
        )
        
        # Inject anomalies if requested
        if inject_anomalies:
            base_values = self._inject_anomalies(
                base_values,
                sensor_type,
                config["anomaly_probability"],
                config["anomaly_types"],
            )
        
        # Create sensor readings
        readings = []
        start_time = datetime.now()
        
        for i, value in enumerate(base_values):
            timestamp = start_time + timedelta(seconds=i / sampling_rate_hz)
            
            # Calculate data quality (degraded during anomalies)
            quality = self._calculate_quality(value, config["base_value"], config["variance"])
            
            reading = SensorReading(
                timestamp=timestamp.isoformat(),
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                value=value,
                unit=config["unit"],
                location=location,
                quality=quality,
            )
            readings.append(reading)
            
        logger.info(f"Generated {len(readings)} readings for {sensor_type} sensor {sensor_id}")
        return readings
    
    def _generate_base_series(
        self,
        base_value: float,
        variance: float,
        num_samples: int,
        sampling_rate_hz: float,
    ) -> np.ndarray:
        """Generate base time series with realistic patterns."""
        # Add seasonal patterns (daily cycle)
        t = np.arange(num_samples) / sampling_rate_hz
        daily_cycle = 0.5 * variance * np.sin(2 * np.pi * t / (24 * 3600))
        
        # Add random walk component
        random_walk = np.cumsum(np.random.normal(0, variance * 0.1, num_samples))
        
        # Add white noise
        noise = np.random.normal(0, variance * 0.2, num_samples)
        
        # Combine components
        values = base_value + daily_cycle + random_walk + noise
        
        return values
    
    def _inject_anomalies(
        self,
        values: np.ndarray,
        sensor_type: str,
        anomaly_probability: float,
        anomaly_types: List[str],
    ) -> np.ndarray:
        """Inject various types of anomalies into the time series."""
        config = self.sensor_configs[sensor_type]
        modified_values = values.copy()
        
        # Determine anomaly positions
        num_anomalies = int(len(values) * anomaly_probability)
        anomaly_positions = np.random.choice(
            len(values), size=num_anomalies, replace=False
        )
        
        for pos in anomaly_positions:
            # Choose random anomaly type
            anomaly_type = random.choice(anomaly_types)
            
            # Generate anomaly
            anomaly_values = self.anomaly_patterns[anomaly_type](
                values[pos:pos+10],  # Use next 10 samples for context
                config["base_value"],
                config["variance"],
            )
            
            # Apply anomaly
            anomaly_length = min(len(anomaly_values), len(modified_values) - pos)
            modified_values[pos:pos+anomaly_length] = anomaly_values[:anomaly_length]
            
        return modified_values
    
    def _generate_spike(
        self,
        context: np.ndarray,
        base_value: float,
        variance: float,
    ) -> np.ndarray:
        """Generate spike anomaly."""
        spike_magnitude = random.uniform(3, 8) * variance
        spike_direction = random.choice([-1, 1])
        
        # Create spike pattern
        spike_length = random.randint(1, 5)
        spike_values = np.zeros(spike_length)
        
        # Peak at middle
        peak_idx = spike_length // 2
        spike_values[peak_idx] = spike_direction * spike_magnitude
        
        # Add decay
        for i in range(spike_length):
            if i != peak_idx:
                distance = abs(i - peak_idx)
                spike_values[i] = spike_values[peak_idx] * np.exp(-distance * 0.5)
        
        return spike_values
    
    def _generate_drift(
        self,
        context: np.ndarray,
        base_value: float,
        variance: float,
    ) -> np.ndarray:
        """Generate drift anomaly."""
        drift_length = random.randint(10, 50)
        drift_magnitude = random.uniform(0.5, 2.0) * variance
        
        # Create linear drift
        drift_values = np.linspace(0, drift_magnitude, drift_length)
        
        return drift_values
    
    def _generate_noise(
        self,
        context: np.ndarray,
        base_value: float,
        variance: float,
    ) -> np.ndarray:
        """Generate noise anomaly."""
        noise_length = random.randint(5, 20)
        noise_level = random.uniform(2, 5) * variance
        
        noise_values = np.random.normal(0, noise_level, noise_length)
        
        return noise_values
    
    def _generate_burst(
        self,
        context: np.ndarray,
        base_value: float,
        variance: float,
    ) -> np.ndarray:
        """Generate burst anomaly (for vibration sensors)."""
        burst_length = random.randint(3, 10)
        burst_magnitude = random.uniform(5, 15) * variance
        
        # Create burst pattern
        burst_values = np.random.exponential(burst_magnitude, burst_length)
        
        return burst_values
    
    def _calculate_quality(self, value: float, base_value: float, variance: float) -> float:
        """Calculate data quality score based on deviation from normal range."""
        deviation = abs(value - base_value) / variance
        
        if deviation <= 2:
            return 1.0
        elif deviation <= 4:
            return 0.8
        elif deviation <= 6:
            return 0.6
        else:
            return 0.3


class MQTTStreamer:
    """MQTT-based streaming for IoT sensor data.
    
    Publishes sensor data to MQTT topics for real-time anomaly detection.
    """
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Initialize MQTT streamer.
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            client_id: MQTT client ID
            username: MQTT username
            password: MQTT password
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id or f"iot_simulator_{int(time.time())}"
        
        # Create MQTT client
        self.client = mqtt.Client(client_id=self.client_id)
        
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        self.connected = False
        
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict, rc: int) -> None:
        """Callback for MQTT connection."""
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker {self.broker_host}:{self.broker_port}")
        else:
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """Callback for MQTT disconnection."""
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def _on_publish(self, client: mqtt.Client, userdata: Any, mid: int) -> None:
        """Callback for MQTT publish."""
        logger.debug(f"Message published with mid: {mid}")
    
    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            return self.connected
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
    
    def publish_sensor_reading(
        self,
        reading: SensorReading,
        topic_prefix: str = "sensors",
        qos: int = 1,
    ) -> bool:
        """Publish a single sensor reading to MQTT.
        
        Args:
            reading: Sensor reading to publish
            topic_prefix: MQTT topic prefix
            qos: Quality of Service level
            
        Returns:
            True if published successfully
        """
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        # Create topic
        topic = f"{topic_prefix}/{reading.sensor_type}/{reading.sensor_id}"
        
        # Convert reading to JSON
        message = json.dumps(reading.to_dict())
        
        try:
            result = self.client.publish(topic, message, qos=qos)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Failed to publish sensor reading: {e}")
            return False
    
    async def stream_sensor_data(
        self,
        readings: List[SensorReading],
        topic_prefix: str = "sensors",
        publish_rate_hz: float = 1.0,
    ) -> None:
        """Stream sensor readings to MQTT asynchronously.
        
        Args:
            readings: List of sensor readings to stream
            topic_prefix: MQTT topic prefix
            publish_rate_hz: Publishing rate in Hz
        """
        if not self.connected:
            logger.error("Not connected to MQTT broker")
            return
        
        publish_interval = 1.0 / publish_rate_hz
        
        for reading in readings:
            success = self.publish_sensor_reading(reading, topic_prefix)
            if not success:
                logger.warning(f"Failed to publish reading from {reading.sensor_id}")
            
            await asyncio.sleep(publish_interval)


class DataPipeline:
    """Complete data pipeline for IoT anomaly detection.
    
    Combines sensor simulation, MQTT streaming, and data processing
    for end-to-end anomaly detection workflows.
    """
    
    def __init__(
        self,
        mqtt_config: Optional[Dict[str, Any]] = None,
        sensor_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize data pipeline.
        
        Args:
            mqtt_config: MQTT configuration
            sensor_config: Sensor simulation configuration
        """
        self.mqtt_config = mqtt_config or {}
        self.sensor_config = sensor_config or {}
        
        # Initialize components
        self.simulator = SensorSimulator()
        self.streamer = MQTTStreamer(**self.mqtt_config)
        
        # Data storage
        self.readings_buffer = []
        self.max_buffer_size = 10000
        
    def generate_training_data(
        self,
        sensor_types: List[str],
        num_sensors_per_type: int = 5,
        duration_hours: float = 168.0,  # 1 week
        sampling_rate_hz: float = 0.1,  # Every 10 seconds
    ) -> pd.DataFrame:
        """Generate comprehensive training dataset.
        
        Args:
            sensor_types: List of sensor types to simulate
            num_sensors_per_type: Number of sensors per type
            duration_hours: Duration of data generation
            sampling_rate_hz: Sampling rate
            
        Returns:
            DataFrame with all sensor readings
        """
        all_readings = []
        
        for sensor_type in sensor_types:
            for i in range(num_sensors_per_type):
                sensor_id = f"{sensor_type}_{i:03d}"
                location = f"zone_{i % 3}"  # Distribute across zones
                
                readings = self.simulator.generate_sensor_data(
                    sensor_type=sensor_type,
                    sensor_id=sensor_id,
                    location=location,
                    duration_hours=duration_hours,
                    sampling_rate_hz=sampling_rate_hz,
                    inject_anomalies=True,
                )
                
                all_readings.extend(readings)
        
        # Convert to DataFrame
        df = pd.DataFrame([reading.to_dict() for reading in all_readings])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Generated training dataset with {len(df)} readings")
        return df
    
    def start_streaming(
        self,
        sensor_types: List[str],
        num_sensors_per_type: int = 3,
        publish_rate_hz: float = 0.5,
        duration_minutes: Optional[float] = None,
    ) -> None:
        """Start real-time streaming of sensor data.
        
        Args:
            sensor_types: List of sensor types to stream
            num_sensors_per_type: Number of sensors per type
            publish_rate_hz: Publishing rate
            duration_minutes: Duration of streaming (None for continuous)
        """
        # Connect to MQTT
        if not self.streamer.connect():
            logger.error("Failed to connect to MQTT broker")
            return
        
        # Generate streaming data
        streaming_readings = []
        
        for sensor_type in sensor_types:
            for i in range(num_sensors_per_type):
                sensor_id = f"{sensor_type}_stream_{i:03d}"
                location = f"zone_{i % 3}"
                
                # Generate shorter duration for streaming
                readings = self.simulator.generate_sensor_data(
                    sensor_type=sensor_type,
                    sensor_id=sensor_id,
                    location=location,
                    duration_hours=1.0,  # 1 hour of data
                    sampling_rate_hz=publish_rate_hz,
                    inject_anomalies=True,
                )
                
                streaming_readings.extend(readings)
        
        # Start streaming
        logger.info(f"Starting streaming of {len(streaming_readings)} readings")
        
        try:
            asyncio.run(self.streamer.stream_sensor_data(
                streaming_readings,
                publish_rate_hz=publish_rate_hz,
            ))
        except KeyboardInterrupt:
            logger.info("Streaming stopped by user")
        finally:
            self.streamer.disconnect()
    
    def get_recent_readings(self, limit: int = 1000) -> pd.DataFrame:
        """Get recent readings from buffer.
        
        Args:
            limit: Maximum number of readings to return
            
        Returns:
            DataFrame with recent readings
        """
        if not self.readings_buffer:
            return pd.DataFrame()
        
        recent_readings = self.readings_buffer[-limit:]
        df = pd.DataFrame([reading.to_dict() for reading in recent_readings])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

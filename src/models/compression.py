"""Model compression and optimization utilities for edge deployment.

This module provides quantization, pruning, and other compression techniques
to optimize models for edge devices with limited computational resources.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, quantize_static
import tensorflow as tf
from tensorflow import keras
import onnx
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class ModelCompressor:
    """Model compression utilities for edge deployment.
    
    Supports quantization, pruning, and model format conversion for various edge runtimes.
    """
    
    def __init__(self, model_type: str = "pytorch") -> None:
        self.model_type = model_type.lower()
        self.compression_stats = {}
        
    def quantize_pytorch_model(
        self,
        model: nn.Module,
        method: str = "dynamic",
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize PyTorch model for edge deployment.
        
        Args:
            model: PyTorch model to quantize
            method: Quantization method ('dynamic' or 'static')
            calibration_data: Data for static quantization calibration
            
        Returns:
            Quantized model
        """
        logger.info(f"Quantizing PyTorch model using {method} quantization")
        
        if method == "dynamic":
            # Dynamic quantization (weights only)
            quantized_model = quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            
        elif method == "static":
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
                
            # Set model to evaluation mode
            model.eval()
            
            # Fuse modules for better quantization
            model_fused = torch.quantization.fuse_modules(
                model, [["encoder.0", "encoder.1"], ["decoder.0", "decoder.1"]]
            )
            
            # Set quantization config
            model_fused.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            
            # Prepare model for quantization
            model_prepared = torch.quantization.prepare(model_fused)
            
            # Calibrate with sample data
            with torch.no_grad():
                for batch in calibration_data:
                    model_prepared(batch)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model_prepared)
            
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
            
        # Calculate compression ratio
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = original_size / quantized_size
        
        self.compression_stats["quantization"] = {
            "method": method,
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression_ratio,
        }
        
        logger.info(f"Quantization completed. Compression ratio: {compression_ratio:.2f}x")
        return quantized_model
    
    def prune_pytorch_model(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.2,
        method: str = "magnitude",
    ) -> nn.Module:
        """Prune PyTorch model to reduce size and improve inference speed.
        
        Args:
            model: PyTorch model to prune
            pruning_ratio: Fraction of parameters to prune (0.0 to 1.0)
            method: Pruning method ('magnitude', 'random', 'structured')
            
        Returns:
            Pruned model
        """
        logger.info(f"Pruning PyTorch model using {method} pruning (ratio: {pruning_ratio})")
        
        # Identify prunable modules
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_prune.append((module, "weight"))
        
        if method == "magnitude":
            # Magnitude-based pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
            
        elif method == "random":
            # Random pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=pruning_ratio,
            )
            
        elif method == "structured":
            # Structured pruning (removes entire channels/filters)
            for module, param_name in modules_to_prune:
                prune.ln_structured(
                    module, param_name, amount=pruning_ratio, n=2, dim=0
                )
                
        else:
            raise ValueError(f"Unsupported pruning method: {method}")
        
        # Remove pruning reparameterization
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        # Calculate compression ratio
        original_size = self._get_model_size(model)
        pruned_size = self._get_model_size(model)
        compression_ratio = original_size / pruned_size
        
        self.compression_stats["pruning"] = {
            "method": method,
            "pruning_ratio": pruning_ratio,
            "original_size_mb": original_size,
            "pruned_size_mb": pruned_size,
            "compression_ratio": compression_ratio,
        }
        
        logger.info(f"Pruning completed. Compression ratio: {compression_ratio:.2f}x")
        return model
    
    def quantize_tensorflow_model(
        self,
        model: keras.Model,
        method: str = "dynamic",
        representative_dataset: Optional[np.ndarray] = None,
    ) -> bytes:
        """Quantize TensorFlow model to TFLite format.
        
        Args:
            model: TensorFlow model to quantize
            method: Quantization method ('dynamic', 'float16', 'int8')
            representative_dataset: Dataset for int8 quantization calibration
            
        Returns:
            Quantized TFLite model as bytes
        """
        logger.info(f"Quantizing TensorFlow model using {method} quantization")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if method == "dynamic":
            # Dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        elif method == "float16":
            # Float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif method == "int8":
            # Integer quantization with calibration
            if representative_dataset is None:
                raise ValueError("Representative dataset required for int8 quantization")
                
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_data_gen(representative_dataset)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
        
        # Convert model
        tflite_model = converter.convert()
        
        # Calculate compression ratio
        original_size = len(model.get_weights()) * 4  # Rough estimate
        quantized_size = len(tflite_model)
        compression_ratio = original_size / quantized_size
        
        self.compression_stats["tflite_quantization"] = {
            "method": method,
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
        }
        
        logger.info(f"TFLite quantization completed. Compression ratio: {compression_ratio:.2f}x")
        return tflite_model
    
    def _representative_data_gen(self, dataset: np.ndarray) -> Any:
        """Generate representative data for TFLite quantization."""
        def representative_dataset():
            for i in range(min(100, len(dataset))):
                yield [dataset[i:i+1].astype(np.float32)]
        return representative_dataset
    
    def convert_to_onnx(
        self,
        pytorch_model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 11,
    ) -> str:
        """Convert PyTorch model to ONNX format.
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shape: Input tensor shape
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            
        Returns:
            Path to saved ONNX model
        """
        logger.info(f"Converting PyTorch model to ONNX (opset {opset_version})")
        
        # Set model to evaluation mode
        pytorch_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX conversion completed. Model saved to: {output_path}")
        return output_path
    
    def optimize_onnx_model(self, onnx_path: str, output_path: str) -> str:
        """Optimize ONNX model for better inference performance.
        
        Args:
            onnx_path: Path to input ONNX model
            output_path: Path to save optimized ONNX model
            
        Returns:
            Path to optimized ONNX model
        """
        logger.info("Optimizing ONNX model")
        
        # Load ONNX model
        model = onnx.load(onnx_path)
        
        # Optimize model
        from onnxruntime.tools import optimizer
        optimized_model = optimizer.optimize_model(onnx_path)
        
        # Save optimized model
        optimized_model.save_model_to_file(output_path)
        
        logger.info(f"ONNX optimization completed. Model saved to: {output_path}")
        return output_path
    
    def benchmark_model_performance(
        self,
        model: Union[nn.Module, bytes, str],
        input_data: np.ndarray,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Benchmark model inference performance.
        
        Args:
            model: Model to benchmark (PyTorch, TFLite bytes, or ONNX path)
            input_data: Input data for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Benchmarking model performance ({num_runs} runs)")
        
        if isinstance(model, nn.Module):
            return self._benchmark_pytorch(model, input_data, num_runs, warmup_runs)
        elif isinstance(model, bytes):
            return self._benchmark_tflite(model, input_data, num_runs, warmup_runs)
        elif isinstance(model, str) and model.endswith(".onnx"):
            return self._benchmark_onnx(model, input_data, num_runs, warmup_runs)
        else:
            raise ValueError("Unsupported model type for benchmarking")
    
    def _benchmark_pytorch(
        self,
        model: nn.Module,
        input_data: np.ndarray,
        num_runs: int,
        warmup_runs: int,
    ) -> Dict[str, float]:
        """Benchmark PyTorch model."""
        import time
        
        model.eval()
        device = next(model.parameters()).device
        input_tensor = torch.FloatTensor(input_data).to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Benchmark runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "p50_latency_ms": np.percentile(times, 50) * 1000,
            "p95_latency_ms": np.percentile(times, 95) * 1000,
            "p99_latency_ms": np.percentile(times, 99) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
        }
    
    def _benchmark_tflite(
        self,
        model_bytes: bytes,
        input_data: np.ndarray,
        num_runs: int,
        warmup_runs: int,
    ) -> Dict[str, float]:
        """Benchmark TFLite model."""
        import time
        
        # Load TFLite model
        interpreter = ort.Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warmup runs
        for _ in range(warmup_runs):
            interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
            interpreter.invoke()
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
            interpreter.invoke()
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "p50_latency_ms": np.percentile(times, 50) * 1000,
            "p95_latency_ms": np.percentile(times, 95) * 1000,
            "p99_latency_ms": np.percentile(times, 99) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
        }
    
    def _benchmark_onnx(
        self,
        onnx_path: str,
        input_data: np.ndarray,
        num_runs: int,
        warmup_runs: int,
    ) -> Dict[str, float]:
        """Benchmark ONNX model."""
        import time
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        
        # Warmup runs
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: input_data.astype(np.float32)})
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {input_name: input_data.astype(np.float32)})
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "p50_latency_ms": np.percentile(times, 50) * 1000,
            "p95_latency_ms": np.percentile(times, 95) * 1000,
            "p99_latency_ms": np.percentile(times, 99) * 1000,
            "throughput_fps": 1.0 / np.mean(times),
        }
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 4 / (1024 * 1024)  # Assuming float32
    
    def get_compression_summary(self) -> Dict[str, Any]:
        """Get summary of all compression operations."""
        return self.compression_stats.copy()

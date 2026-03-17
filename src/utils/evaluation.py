"""Comprehensive evaluation metrics for IoT anomaly detection.

This module provides accuracy, efficiency, and robustness metrics specifically
designed for edge AI anomaly detection systems.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class AnomalyDetectionMetrics:
    """Comprehensive metrics for anomaly detection evaluation.
    
    Provides accuracy, efficiency, and robustness metrics for edge AI systems.
    """
    
    def __init__(self) -> None:
        """Initialize metrics calculator."""
        self.metrics_history = []
        
    def calculate_accuracy_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate accuracy-based metrics.
        
        Args:
            y_true: True anomaly labels (1 for anomaly, 0 for normal)
            y_pred: Predicted anomaly labels
            y_scores: Anomaly scores (optional, for AUC calculation)
            
        Returns:
            Dictionary of accuracy metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics["true_positive_rate"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics["true_negative_rate"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Additional metrics
        metrics["specificity"] = metrics["true_negative_rate"]
        metrics["sensitivity"] = metrics["true_positive_rate"]
        
        # F1 variants
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        # AUC if scores provided
        if y_scores is not None:
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
            except ValueError:
                metrics["auc_roc"] = 0.5
                
            try:
                metrics["auc_pr"] = self._calculate_auc_pr(y_true, y_scores)
            except ValueError:
                metrics["auc_pr"] = 0.0
        
        return metrics
    
    def calculate_efficiency_metrics(
        self,
        inference_times: List[float],
        model_size_mb: float,
        memory_usage_mb: Optional[float] = None,
        energy_consumption_j: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate efficiency metrics for edge deployment.
        
        Args:
            inference_times: List of inference times in seconds
            model_size_mb: Model size in MB
            memory_usage_mb: Peak memory usage in MB
            energy_consumption_j: Energy consumption per inference in Joules
            
        Returns:
            Dictionary of efficiency metrics
        """
        metrics = {}
        
        # Latency metrics
        inference_times_ms = np.array(inference_times) * 1000
        
        metrics["mean_latency_ms"] = np.mean(inference_times_ms)
        metrics["std_latency_ms"] = np.std(inference_times_ms)
        metrics["p50_latency_ms"] = np.percentile(inference_times_ms, 50)
        metrics["p95_latency_ms"] = np.percentile(inference_times_ms, 95)
        metrics["p99_latency_ms"] = np.percentile(inference_times_ms, 99)
        metrics["min_latency_ms"] = np.min(inference_times_ms)
        metrics["max_latency_ms"] = np.max(inference_times_ms)
        
        # Throughput metrics
        metrics["throughput_fps"] = 1.0 / np.mean(inference_times)
        metrics["max_throughput_fps"] = 1.0 / np.min(inference_times)
        
        # Model size metrics
        metrics["model_size_mb"] = model_size_mb
        metrics["model_size_kb"] = model_size_mb * 1024
        
        # Memory metrics
        if memory_usage_mb is not None:
            metrics["peak_memory_mb"] = memory_usage_mb
            metrics["memory_efficiency"] = model_size_mb / memory_usage_mb if memory_usage_mb > 0 else 0
        
        # Energy metrics
        if energy_consumption_j is not None:
            metrics["energy_per_inference_j"] = energy_consumption_j
            metrics["energy_per_inference_mj"] = energy_consumption_j * 1000
            metrics["inferences_per_joule"] = 1.0 / energy_consumption_j if energy_consumption_j > 0 else 0
        
        # Efficiency ratios
        metrics["latency_efficiency"] = 1.0 / metrics["mean_latency_ms"]  # Higher is better
        metrics["size_efficiency"] = 1.0 / model_size_mb  # Higher is better
        
        return metrics
    
    def calculate_robustness_metrics(
        self,
        clean_metrics: Dict[str, float],
        noisy_metrics: Dict[str, float],
        noise_levels: List[float],
    ) -> Dict[str, float]:
        """Calculate robustness metrics against noise and perturbations.
        
        Args:
            clean_metrics: Metrics on clean data
            noisy_metrics: Metrics on noisy data
            noise_levels: List of noise levels tested
            
        Returns:
            Dictionary of robustness metrics
        """
        metrics = {}
        
        # Accuracy degradation
        clean_accuracy = clean_metrics.get("accuracy", 0)
        noisy_accuracy = noisy_metrics.get("accuracy", 0)
        
        metrics["accuracy_degradation"] = clean_accuracy - noisy_accuracy
        metrics["accuracy_robustness"] = noisy_accuracy / clean_accuracy if clean_accuracy > 0 else 0
        
        # F1 score degradation
        clean_f1 = clean_metrics.get("f1_score", 0)
        noisy_f1 = noisy_metrics.get("f1_score", 0)
        
        metrics["f1_degradation"] = clean_f1 - noisy_f1
        metrics["f1_robustness"] = noisy_f1 / clean_f1 if clean_f1 > 0 else 0
        
        # Latency impact
        clean_latency = clean_metrics.get("mean_latency_ms", 0)
        noisy_latency = noisy_metrics.get("mean_latency_ms", 0)
        
        metrics["latency_overhead"] = noisy_latency - clean_latency
        metrics["latency_overhead_ratio"] = noisy_latency / clean_latency if clean_latency > 0 else 1
        
        # Noise tolerance
        if noise_levels:
            metrics["max_noise_tolerance"] = max(noise_levels)
            metrics["avg_noise_tolerance"] = np.mean(noise_levels)
        
        return metrics
    
    def calculate_edge_specific_metrics(
        self,
        accuracy_metrics: Dict[str, float],
        efficiency_metrics: Dict[str, float],
        target_latency_ms: float = 100.0,
        target_memory_mb: float = 50.0,
        target_accuracy: float = 0.9,
    ) -> Dict[str, float]:
        """Calculate edge-specific performance metrics.
        
        Args:
            accuracy_metrics: Accuracy metrics
            efficiency_metrics: Efficiency metrics
            target_latency_ms: Target latency in ms
            target_memory_mb: Target memory usage in MB
            target_accuracy: Target accuracy
            
        Returns:
            Dictionary of edge-specific metrics
        """
        metrics = {}
        
        # Latency compliance
        actual_latency = efficiency_metrics.get("mean_latency_ms", float('inf'))
        metrics["latency_compliance"] = 1.0 if actual_latency <= target_latency_ms else 0.0
        metrics["latency_margin_ms"] = target_latency_ms - actual_latency
        
        # Memory compliance
        actual_memory = efficiency_metrics.get("peak_memory_mb", float('inf'))
        metrics["memory_compliance"] = 1.0 if actual_memory <= target_memory_mb else 0.0
        metrics["memory_margin_mb"] = target_memory_mb - actual_memory
        
        # Accuracy compliance
        actual_accuracy = accuracy_metrics.get("accuracy", 0.0)
        metrics["accuracy_compliance"] = 1.0 if actual_accuracy >= target_accuracy else 0.0
        metrics["accuracy_margin"] = actual_accuracy - target_accuracy
        
        # Overall edge readiness score
        compliance_scores = [
            metrics["latency_compliance"],
            metrics["memory_compliance"],
            metrics["accuracy_compliance"],
        ]
        metrics["edge_readiness_score"] = np.mean(compliance_scores)
        
        # Efficiency score (combination of speed and accuracy)
        latency_score = min(1.0, target_latency_ms / actual_latency) if actual_latency > 0 else 0
        accuracy_score = min(1.0, actual_accuracy / target_accuracy) if target_accuracy > 0 else 0
        metrics["efficiency_score"] = (latency_score + accuracy_score) / 2
        
        return metrics
    
    def _calculate_auc_pr(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate Area Under Precision-Recall Curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return np.trapz(precision, recall)
    
    def create_performance_report(
        self,
        accuracy_metrics: Dict[str, float],
        efficiency_metrics: Dict[str, float],
        edge_metrics: Optional[Dict[str, float]] = None,
        robustness_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create comprehensive performance report.
        
        Args:
            accuracy_metrics: Accuracy metrics
            efficiency_metrics: Efficiency metrics
            edge_metrics: Edge-specific metrics
            robustness_metrics: Robustness metrics
            
        Returns:
            Comprehensive performance report
        """
        report = {
            "summary": {
                "accuracy": accuracy_metrics.get("accuracy", 0),
                "f1_score": accuracy_metrics.get("f1_score", 0),
                "mean_latency_ms": efficiency_metrics.get("mean_latency_ms", 0),
                "model_size_mb": efficiency_metrics.get("model_size_mb", 0),
                "throughput_fps": efficiency_metrics.get("throughput_fps", 0),
            },
            "accuracy": accuracy_metrics,
            "efficiency": efficiency_metrics,
        }
        
        if edge_metrics:
            report["edge_performance"] = edge_metrics
            
        if robustness_metrics:
            report["robustness"] = robustness_metrics
        
        # Overall score
        scores = [
            accuracy_metrics.get("f1_score", 0),
            min(1.0, 100.0 / efficiency_metrics.get("mean_latency_ms", 100)),
            min(1.0, 50.0 / efficiency_metrics.get("model_size_mb", 50)),
        ]
        
        report["overall_score"] = np.mean(scores)
        
        return report


class PerformanceVisualizer:
    """Visualization utilities for anomaly detection performance metrics."""
    
    def __init__(self) -> None:
        """Initialize visualizer."""
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "warning": "#d62728",
            "info": "#9467bd",
        }
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Normal", "Anomaly"],
            y=["Normal", "Anomaly"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            font=dict(size=12),
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {auc:.3f})",
            line=dict(color=self.colors["primary"], width=2),
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color=self.colors["warning"], dash="dash"),
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            font=dict(size=12),
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_latency_distribution(
        self,
        inference_times: List[float],
        title: str = "Inference Latency Distribution",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create latency distribution plot."""
        times_ms = np.array(inference_times) * 1000
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=times_ms,
            nbinsx=50,
            name="Latency Distribution",
            marker_color=self.colors["primary"],
            opacity=0.7,
        ))
        
        # Percentile lines
        p50 = np.percentile(times_ms, 50)
        p95 = np.percentile(times_ms, 95)
        p99 = np.percentile(times_ms, 99)
        
        fig.add_vline(x=p50, line_dash="dash", line_color=self.colors["success"], 
                     annotation_text=f"P50: {p50:.1f}ms")
        fig.add_vline(x=p95, line_dash="dash", line_color=self.colors["warning"], 
                     annotation_text=f"P95: {p95:.1f}ms")
        fig.add_vline(x=p99, line_dash="dash", line_color=self.colors["info"], 
                     annotation_text=f"P99: {p99:.1f}ms")
        
        fig.update_layout(
            title=title,
            xaxis_title="Latency (ms)",
            yaxis_title="Frequency",
            font=dict(size=12),
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_performance_comparison(
        self,
        models_data: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Model Performance Comparison",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create performance comparison plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics[:4],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = list(self.colors.values())
        
        for i, metric in enumerate(metrics[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            model_names = list(models_data.keys())
            values = [models_data[model].get(metric, 0) for model in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric,
                    marker_color=colors[i % len(colors)],
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            font=dict(size=12),
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig


class Leaderboard:
    """Performance leaderboard for comparing different models and configurations."""
    
    def __init__(self) -> None:
        """Initialize leaderboard."""
        self.entries = []
        
    def add_entry(
        self,
        model_name: str,
        accuracy_metrics: Dict[str, float],
        efficiency_metrics: Dict[str, float],
        edge_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a model entry to the leaderboard.
        
        Args:
            model_name: Name of the model
            accuracy_metrics: Accuracy metrics
            efficiency_metrics: Efficiency metrics
            edge_metrics: Edge-specific metrics
            metadata: Additional metadata
        """
        entry = {
            "model_name": model_name,
            "accuracy": accuracy_metrics,
            "efficiency": efficiency_metrics,
            "edge_performance": edge_metrics or {},
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        
        # Calculate overall score
        scores = [
            accuracy_metrics.get("f1_score", 0),
            min(1.0, 100.0 / efficiency_metrics.get("mean_latency_ms", 100)),
            min(1.0, 50.0 / efficiency_metrics.get("model_size_mb", 50)),
        ]
        entry["overall_score"] = np.mean(scores)
        
        self.entries.append(entry)
        
    def get_leaderboard(self, sort_by: str = "overall_score") -> pd.DataFrame:
        """Get leaderboard as DataFrame.
        
        Args:
            sort_by: Metric to sort by
            
        Returns:
            Sorted leaderboard DataFrame
        """
        if not self.entries:
            return pd.DataFrame()
        
        # Flatten entries for DataFrame
        flattened_entries = []
        for entry in self.entries:
            flat_entry = {
                "model_name": entry["model_name"],
                "overall_score": entry["overall_score"],
                "timestamp": entry["timestamp"],
            }
            
            # Add accuracy metrics
            for key, value in entry["accuracy"].items():
                flat_entry[f"accuracy_{key}"] = value
            
            # Add efficiency metrics
            for key, value in entry["efficiency"].items():
                flat_entry[f"efficiency_{key}"] = value
            
            # Add edge metrics
            for key, value in entry["edge_performance"].items():
                flat_entry[f"edge_{key}"] = value
            
            flattened_entries.append(flat_entry)
        
        df = pd.DataFrame(flattened_entries)
        
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        
        return df
    
    def export_leaderboard(self, filepath: str, format: str = "csv") -> None:
        """Export leaderboard to file.
        
        Args:
            filepath: Output file path
            format: Export format ('csv', 'json', 'excel')
        """
        df = self.get_leaderboard()
        
        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        elif format == "excel":
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Leaderboard exported to {filepath}")

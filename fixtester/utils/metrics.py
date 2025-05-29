"""
Metrics collection and monitoring utilities for FixTester with Prometheus integration,
performance tracking, and health monitoring.
"""
import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from fixtester.utils.logger import setup_logger


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Statistical summary of metric values."""
    count: int
    min_value: float
    max_value: float
    mean: float
    median: float
    std_dev: float
    p95: float
    p99: float


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the metrics collector.

        Args:
            config: Metrics configuration dictionary
        """
        self.config = config or {}
        self.logger = setup_logger(__name__)
        self.enabled = self.config.get('enabled', True)
        
        # Internal storage
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration
        self.max_histogram_size = self.config.get('max_histogram_size', 10000)
        self.retention_seconds = self.config.get('retention_seconds', 3600)
        
        # Prometheus setup
        self.prometheus_registry = None
        self.prometheus_metrics = {}
        if PROMETHEUS_AVAILABLE and self.config.get('prometheus_enabled', True):
            self._setup_prometheus()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self._cleanup_thread.start()

    def _setup_prometheus(self) -> None:
        """Setup Prometheus metrics collection."""
        try:
            self.prometheus_registry = CollectorRegistry()
            
            # Define Prometheus metrics
            self.prometheus_metrics = {
                'messages_sent_total': Counter(
                    'fixtester_messages_sent_total',
                    'Total number of FIX messages sent',
                    ['message_type', 'session_id'],
                    registry=self.prometheus_registry
                ),
                'messages_received_total': Counter(
                    'fixtester_messages_received_total',
                    'Total number of FIX messages received',
                    ['message_type', 'session_id'],
                    registry=self.prometheus_registry
                ),
                'message_processing_duration': Histogram(
                    'fixtester_message_processing_duration_seconds',
                    'Time spent processing FIX messages',
                    ['message_type', 'direction'],
                    registry=self.prometheus_registry
                ),
                'session_connections': Gauge(
                    'fixtester_session_connections',
                    'Number of active FIX sessions',
                    ['status'],
                    registry=self.prometheus_registry
                ),
                'validation_errors_total': Counter(
                    'fixtester_validation_errors_total',
                    'Total number of message validation errors',
                    ['error_type', 'message_type'],
                    registry=self.prometheus_registry
                ),
                'test_duration': Histogram(
                    'fixtester_test_duration_seconds',
                    'Test execution duration',
                    ['test_name', 'scenario'],
                    registry=self.prometheus_registry
                ),
                'memory_usage_bytes': Gauge(
                    'fixtester_memory_usage_bytes',
                    'Memory usage in bytes',
                    registry=self.prometheus_registry
                ),
                'cpu_usage_percent': Gauge(
                    'fixtester_cpu_usage_percent',
                    'CPU usage percentage',
                    registry=self.prometheus_registry
                )
            }
            
            self.logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Prometheus metrics: {e}")
            self.prometheus_registry = None

    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels for the metric
        """
        if not self.enabled:
            return
            
        with self._lock:
            self._counters[name] += value
            
        # Update Prometheus counter if available
        if self.prometheus_registry and name in self.prometheus_metrics:
            try:
                if labels:
                    self.prometheus_metrics[name].labels(**labels).inc(value)
                else:
                    self.prometheus_metrics[name].inc(value)
            except Exception as e:
                self.logger.warning(f"Failed to update Prometheus counter {name}: {e}")

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric value.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels for the metric
        """
        if not self.enabled:
            return
            
        with self._lock:
            self._gauges[name] = value
            
        # Update Prometheus gauge if available
        if self.prometheus_registry and name in self.prometheus_metrics:
            try:
                if labels:
                    self.prometheus_metrics[name].labels(**labels).set(value)
                else:
                    self.prometheus_metrics[name].set(value)
            except Exception as e:
                self.logger.warning(f"Failed to update Prometheus gauge {name}: {e}")

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a value in a histogram metric.

        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels for the metric
        """
        if not self.enabled:
            return
            
        timestamp = time.time()
        point = MetricPoint(timestamp=timestamp, value=value, labels=labels or {})
        
        with self._lock:
            histogram = self._histograms[name]
            histogram.append(point)
            
            # Limit histogram size
            if len(histogram) > self.max_histogram_size:
                histogram.pop(0)
                
        # Update Prometheus histogram if available
        if self.prometheus_registry and name in self.prometheus_metrics:
            try:
                if labels:
                    self.prometheus_metrics[name].labels(**labels).observe(value)
                else:
                    self.prometheus_metrics[name].observe(value)
            except Exception as e:
                self.logger.warning(f"Failed to update Prometheus histogram {name}: {e}")

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None) -> None:
        """Record a timing measurement.

        Args:
            name: Timer name
            duration: Duration in seconds
            labels: Optional labels for the metric
        """
        if not self.enabled:
            return
            
        self.record_histogram(f"{name}_duration", duration, labels)
        
        with self._lock:
            self._timers[name].append(duration)
            
            # Limit timer history
            if len(self._timers[name]) > self.max_histogram_size:
                self._timers[name].pop(0)

    @contextmanager
    def timer(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations.

        Args:
            name: Timer name
            labels: Optional labels for the metric

        Usage:
            with metrics.timer('operation_name'):
                # Timed operation
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timer(name, duration, labels)

    def get_counter(self, name: str) -> int:
        """Get current counter value.

        Args:
            name: Counter name

        Returns:
            Current counter value
        """
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value.

        Args:
            name: Gauge name

        Returns:
            Current gauge value or None if not set
        """
        with self._lock:
            return self._gauges.get(name)

    def get_histogram_summary(self, name: str) -> Optional[MetricSummary]:
        """Get statistical summary of histogram values.

        Args:
            name: Histogram name

        Returns:
            MetricSummary or None if no data
        """
        with self._lock:
            points = self._histograms.get(name, [])
            
        if not points:
            return None
            
        values = [p.value for p in points]
        
        try:
            return MetricSummary(
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
                p95=self._percentile(values, 0.95),
                p99=self._percentile(values, 0.99)
            )
        except Exception as e:
            self.logger.error(f"Error calculating histogram summary for {name}: {e}")
            return None

    def get_timer_summary(self, name: str) -> Optional[MetricSummary]:
        """Get statistical summary of timer values.

        Args:
            name: Timer name

        Returns:
            MetricSummary or None if no data
        """
        with self._lock:
            values = self._timers.get(name, [])
            
        if not values:
            return None
            
        try:
            return MetricSummary(
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
                p95=self._percentile(values, 0.95),
                p99=self._percentile(values, 0.99)
            )
        except Exception as e:
            self.logger.error(f"Error calculating timer summary for {name}: {e}")
            return None

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values.

        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {name: [
                    {'timestamp': p.timestamp, 'value': p.value, 'labels': p.labels}
                    for p in points
                ] for name, points in self._histograms.items()},
                'timers': {name: list(values) for name, values in self._timers.items()}
            }

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            
        self.logger.info("All metrics reset")

    def export_prometheus_metrics(self) -> Optional[str]:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus formatted metrics string or None
        """
        if not self.prometheus_registry:
            return None
            
        try:
            return generate_latest(self.prometheus_registry).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to export Prometheus metrics: {e}")
            return None

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values.

        Args:
            values: List of values
            percentile: Percentile to calculate (0.0 to 1.0)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0
            
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]

    def _cleanup_old_metrics(self) -> None:
        """Clean up old metric data points."""
        while True:
            try:
                time.sleep(60)  # Cleanup every minute
                
                if not self.enabled:
                    continue
                    
                current_time = time.time()
                cutoff_time = current_time - self.retention_seconds
                
                with self._lock:
                    # Clean up histogram data
                    for name, points in self._histograms.items():
                        # Remove old points
                        self._histograms[name] = [
                            p for p in points if p.timestamp > cutoff_time
                        ]
                        
            except Exception as e:
                self.logger.error(f"Error in metrics cleanup: {e}")
                time.sleep(5)


class PerformanceMonitor:
    """Monitors system performance metrics."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance monitor.

        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics = metrics_collector
        self.logger = setup_logger(__name__)
        self._monitoring = False
        self._monitor_thread = None

    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start performance monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            self.logger.warning("Performance monitoring already started")
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Performance monitoring started with {interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")

    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        while self._monitoring:
            try:
                # Memory usage
                memory_info = process.memory_info()
                self.metrics.set_gauge('memory_usage_bytes', memory_info.rss)
                self.metrics.set_gauge('memory_usage_mb', memory_info.rss / 1024 / 1024)
                
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.metrics.set_gauge('cpu_usage_percent', cpu_percent)
                
                # Thread count
                thread_count = process.num_threads()
                self.metrics.set_gauge('thread_count', thread_count)
                
                # File descriptors (Unix only)
                try:
                    fd_count = process.num_fds()
                    self.metrics.set_gauge('file_descriptors', fd_count)
                except AttributeError:
                    pass  # Windows doesn't support num_fds
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                time.sleep(1)


class HealthChecker:
    """Provides health check functionality."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize health checker.

        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics = metrics_collector
        self.logger = setup_logger(__name__)
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.health_status: Dict[str, bool] = {}

    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function.

        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy, False otherwise
        """
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")

    def run_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks.

        Returns:
            Dictionary mapping check names to their results
        """
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                self.health_status[name] = result
                
                # Update metrics
                self.metrics.set_gauge(f'health_check_{name}', 1.0 if result else 0.0)
                
            except Exception as e:
                self.logger.error(f"Health check {name} failed with error: {e}")
                results[name] = False
                self.health_status[name] = False
                self.metrics.set_gauge(f'health_check_{name}', 0.0)
        
        return results

    def is_healthy(self) -> bool:
        """Check if all health checks pass.

        Returns:
            True if all checks pass, False otherwise
        """
        if not self.health_checks:
            return True  # No checks means healthy
            
        results = self.run_health_checks()
        return all(results.values())

    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status.

        Returns:
            Health status dictionary
        """
        results = self.run_health_checks()
        
        return {
            'healthy': all(results.values()),
            'checks': results,
            'timestamp': time.time(),
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }


# Global metrics collector instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics_collector(config: Dict[str, Any] = None) -> MetricsCollector:
    """Get the global metrics collector instance.

    Args:
        config: Optional configuration for first-time initialization

    Returns:
        MetricsCollector instance
    """
    global _global_metrics
    
    if _global_metrics is None:
        _global_metrics = MetricsCollector(config)
        
    return _global_metrics


def reset_global_metrics() -> None:
    """Reset the global metrics collector."""
    global _global_metrics
    _global_metrics = None
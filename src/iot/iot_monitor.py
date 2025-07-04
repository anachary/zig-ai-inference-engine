"""
IoT Monitoring Module for Zig AI Platform
Handles health monitoring, alerting, and performance tracking for edge devices
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import psutil

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    current_value: float
    threshold: float

@dataclass
class HealthCheck:
    name: str
    check_function: Callable[[], bool]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    enabled: bool = True

class IoTMonitor:
    """
    Monitoring system for IoT devices
    Tracks health, performance, and generates alerts
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('monitoring', {})
        self.enabled = self.config.get('enabled', True)
        self.running = False
        self.alerts: List[Alert] = []
        self.health_checks: List[HealthCheck] = []
        self.metrics_history: Dict[str, List[float]] = {}
        self.last_check_time = time.time()
        
        # Default thresholds
        self.thresholds = {
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 85.0,
            'disk_usage_percent': 90.0,
            'temperature_celsius': 70.0,
            'inference_latency_ms': 1000.0,
        }
        
        # Update thresholds from config
        if 'thresholds' in self.config:
            self.thresholds.update(self.config['thresholds'])
        
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks for IoT devices"""
        
        # CPU health check
        self.health_checks.append(HealthCheck(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            interval_seconds=30
        ))
        
        # Memory health check
        self.health_checks.append(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            interval_seconds=30
        ))
        
        # Disk health check
        self.health_checks.append(HealthCheck(
            name="disk_usage",
            check_function=self._check_disk_usage,
            interval_seconds=60
        ))
        
        # Temperature health check
        self.health_checks.append(HealthCheck(
            name="temperature",
            check_function=self._check_temperature,
            interval_seconds=60
        ))
        
        # Process health check
        self.health_checks.append(HealthCheck(
            name="process_health",
            check_function=self._check_process_health,
            interval_seconds=30
        ))
    
    async def start(self):
        """Start monitoring"""
        if not self.enabled:
            logger.info("🔍 Monitoring disabled")
            return
        
        self.running = True
        
        # Start monitoring tasks
        for health_check in self.health_checks:
            if health_check.enabled:
                asyncio.create_task(self._health_check_loop(health_check))
        
        # Start metrics collection
        asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("🔍 IoT monitoring started")
    
    async def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("🔍 IoT monitoring stopped")
    
    async def _health_check_loop(self, health_check: HealthCheck):
        """Run a specific health check in a loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Run health check with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(health_check.check_function),
                    timeout=health_check.timeout_seconds
                )
                
                check_duration = time.time() - start_time
                
                if not result:
                    logger.warning(f"⚠️ Health check failed: {health_check.name}")
                else:
                    logger.debug(f"✅ Health check passed: {health_check.name} ({check_duration:.2f}s)")
                
            except asyncio.TimeoutError:
                logger.error(f"❌ Health check timeout: {health_check.name}")
            except Exception as e:
                logger.error(f"❌ Health check error {health_check.name}: {e}")
            
            await asyncio.sleep(health_check.interval_seconds)
    
    async def _metrics_collection_loop(self):
        """Collect and store metrics for trend analysis"""
        while self.running:
            try:
                # Collect current metrics
                metrics = {
                    'cpu_usage_percent': psutil.cpu_percent(interval=1),
                    'memory_usage_percent': psutil.virtual_memory().percent,
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'temperature_celsius': self._get_temperature(),
                }
                
                # Store metrics history
                for metric_name, value in metrics.items():
                    if value is not None:
                        if metric_name not in self.metrics_history:
                            self.metrics_history[metric_name] = []
                        
                        self.metrics_history[metric_name].append(value)
                        
                        # Keep only last 100 values
                        if len(self.metrics_history[metric_name]) > 100:
                            self.metrics_history[metric_name].pop(0)
                        
                        # Check thresholds
                        self._check_threshold(metric_name, value)
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage health"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            return cpu_usage < self.thresholds['cpu_usage_percent']
        except Exception as e:
            logger.error(f"❌ CPU check error: {e}")
            return False
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage health"""
        try:
            memory_usage = psutil.virtual_memory().percent
            return memory_usage < self.thresholds['memory_usage_percent']
        except Exception as e:
            logger.error(f"❌ Memory check error: {e}")
            return False
    
    def _check_disk_usage(self) -> bool:
        """Check disk usage health"""
        try:
            disk_usage = psutil.disk_usage('/').percent
            return disk_usage < self.thresholds['disk_usage_percent']
        except Exception as e:
            logger.error(f"❌ Disk check error: {e}")
            return False
    
    def _check_temperature(self) -> bool:
        """Check device temperature health"""
        try:
            temp = self._get_temperature()
            if temp is None:
                return True  # Can't check, assume OK
            return temp < self.thresholds['temperature_celsius']
        except Exception as e:
            logger.error(f"❌ Temperature check error: {e}")
            return False
    
    def _check_process_health(self) -> bool:
        """Check if critical processes are running"""
        try:
            # Check if current process is healthy
            current_process = psutil.Process()
            return current_process.is_running() and current_process.status() != psutil.STATUS_ZOMBIE
        except Exception as e:
            logger.error(f"❌ Process check error: {e}")
            return False
    
    def _get_temperature(self) -> Optional[float]:
        """Get device temperature"""
        try:
            # Try Raspberry Pi thermal zone
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read()) / 1000.0
                return temp
        except:
            try:
                # Try vcgencmd for Raspberry Pi
                import subprocess
                result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    temp_str = result.stdout.strip()
                    temp = float(temp_str.split('=')[1].replace("'C", ""))
                    return temp
            except:
                pass
        return None
    
    def _check_threshold(self, metric_name: str, value: float):
        """Check if metric exceeds threshold and generate alert"""
        if metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_name]
        
        if value > threshold:
            # Determine alert level
            if value > threshold * 1.2:  # 20% over threshold
                level = AlertLevel.CRITICAL
            else:
                level = AlertLevel.WARNING
            
            # Create alert
            alert = Alert(
                level=level,
                message=f"{metric_name} is {value:.1f}, exceeding threshold of {threshold:.1f}",
                timestamp=time.time(),
                metric_name=metric_name,
                current_value=value,
                threshold=threshold
            )
            
            self.alerts.append(alert)
            
            # Keep only last 50 alerts
            if len(self.alerts) > 50:
                self.alerts.pop(0)
            
            logger.warning(f"⚠️ {level.value.upper()}: {alert.message}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        try:
            current_metrics = {
                'cpu_usage_percent': psutil.cpu_percent(),
                'memory_usage_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'temperature_celsius': self._get_temperature(),
            }
            
            # Determine overall health
            health_status = "healthy"
            for metric_name, value in current_metrics.items():
                if value is not None and metric_name in self.thresholds:
                    if value > self.thresholds[metric_name]:
                        health_status = "unhealthy"
                        break
            
            return {
                'status': health_status,
                'metrics': current_metrics,
                'thresholds': self.thresholds,
                'recent_alerts': [
                    {
                        'level': alert.level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp,
                        'metric': alert.metric_name
                    }
                    for alert in self.alerts[-5:]  # Last 5 alerts
                ],
                'uptime_seconds': time.time() - self.last_check_time
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get health status: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def add_custom_health_check(self, name: str, check_function: Callable[[], bool], 
                               interval_seconds: int = 60):
        """Add a custom health check"""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds
        )
        
        self.health_checks.append(health_check)
        
        # Start the health check if monitoring is already running
        if self.running:
            asyncio.create_task(self._health_check_loop(health_check))
        
        logger.info(f"➕ Added custom health check: {name}")
    
    def set_threshold(self, metric_name: str, threshold: float):
        """Set threshold for a specific metric"""
        self.thresholds[metric_name] = threshold
        logger.info(f"🎯 Set threshold for {metric_name}: {threshold}")
    
    def get_metrics_trend(self, metric_name: str, window_size: int = 10) -> Optional[str]:
        """Get trend for a specific metric (increasing, decreasing, stable)"""
        if metric_name not in self.metrics_history:
            return None
        
        history = self.metrics_history[metric_name]
        if len(history) < window_size:
            return "insufficient_data"
        
        recent_values = history[-window_size:]
        first_half = recent_values[:window_size//2]
        second_half = recent_values[window_size//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        diff_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if diff_percent > 5:
            return "increasing"
        elif diff_percent < -5:
            return "decreasing"
        else:
            return "stable"

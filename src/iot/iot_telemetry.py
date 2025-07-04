"""
IoT Telemetry Module for Zig AI Platform
Handles metrics collection and cloud reporting for edge devices
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import aiohttp
import psutil

logger = logging.getLogger(__name__)

class IoTTelemetry:
    """
    Telemetry system for IoT devices
    Collects and reports metrics to cloud endpoints
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('telemetry', {})
        self.enabled = self.config.get('enabled', False)
        self.endpoint = self.config.get('endpoint', '')
        self.interval_seconds = self.config.get('interval_seconds', 60)
        self.device_id = self._get_device_id()
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.metrics_buffer = []
        
    def _get_device_id(self) -> str:
        """Get unique device identifier"""
        import socket
        import hashlib
        
        hostname = socket.gethostname()
        try:
            import uuid
            mac = uuid.getnode()
            mac_str = ':'.join(('%012X' % mac)[i:i+2] for i in range(0, 12, 2))
        except:
            mac_str = "unknown"
            
        device_string = f"{hostname}-{mac_str}"
        return hashlib.md5(device_string.encode()).hexdigest()[:12]
    
    async def start(self):
        """Start telemetry collection"""
        if not self.enabled:
            logger.info("📊 Telemetry disabled")
            return
            
        if not self.endpoint:
            logger.warning("⚠️ Telemetry endpoint not configured")
            return
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        self.running = True
        
        # Start telemetry loop
        asyncio.create_task(self._telemetry_loop())
        
        logger.info(f"📊 Telemetry started - reporting to {self.endpoint} every {self.interval_seconds}s")
    
    async def stop(self):
        """Stop telemetry collection"""
        self.running = False
        
        if self.session:
            await self.session.close()
            
        logger.info("📊 Telemetry stopped")
    
    async def _telemetry_loop(self):
        """Main telemetry collection loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Send to cloud
                await self._send_metrics(metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.interval_seconds)
                
            except Exception as e:
                logger.error(f"❌ Telemetry error: {e}")
                await asyncio.sleep(self.interval_seconds)
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect device and application metrics"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'device_id': self.device_id,
            'device_type': 'iot-edge',
            
            # System metrics
            'system': {
                'cpu_usage_percent': psutil.cpu_percent(interval=1),
                'memory_usage_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'temperature_celsius': await self._get_temperature(),
                'uptime_seconds': time.time() - psutil.boot_time(),
            },
            
            # Network metrics
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv,
                'packets_sent': psutil.net_io_counters().packets_sent,
                'packets_recv': psutil.net_io_counters().packets_recv,
            },
            
            # Application metrics (placeholder)
            'application': {
                'inference_count': 0,  # Would be updated by inference engine
                'average_latency_ms': 0.0,
                'error_count': 0,
                'model_version': '1.0.0',
            }
        }
        
        return metrics
    
    async def _get_temperature(self) -> Optional[float]:
        """Get device temperature (Raspberry Pi specific)"""
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
    
    async def _send_metrics(self, metrics: Dict[str, Any]):
        """Send metrics to cloud endpoint"""
        if not self.session or not self.endpoint:
            return
            
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': f'ZigAI-IoT/{self.device_id}'
            }
            
            async with self.session.post(
                self.endpoint,
                json=metrics,
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.debug(f"📊 Metrics sent successfully")
                else:
                    logger.warning(f"⚠️ Metrics upload failed: {response.status}")
                    
        except asyncio.TimeoutError:
            logger.warning("⚠️ Metrics upload timeout")
        except Exception as e:
            logger.error(f"❌ Failed to send metrics: {e}")
    
    def update_application_metrics(self, metrics: Dict[str, Any]):
        """Update application-specific metrics"""
        # This would be called by the inference engine to update metrics
        pass
    
    def record_inference(self, latency_ms: float, success: bool = True):
        """Record an inference event"""
        # This would be called by the inference engine
        pass
    
    def get_local_metrics(self) -> Dict[str, Any]:
        """Get current metrics without sending to cloud"""
        try:
            return {
                'device_id': self.device_id,
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'temperature': asyncio.run(self._get_temperature()) if asyncio.get_event_loop().is_running() else None,
            }
        except Exception as e:
            logger.error(f"❌ Failed to get local metrics: {e}")
            return {}

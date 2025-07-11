#!/usr/bin/env python3
"""
Test script for online IoT demo
"""

import asyncio
import httpx
import json
import time
from datetime import datetime

async def test_device(port: int, device_name: str):
    """Test a single device"""
    base_url = f"http://localhost:{port}"
    
    print(f"\n🧪 Testing {device_name} on port {port}")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            # Health check
            response = await client.get(f"{base_url}/health", timeout=5.0)
            if response.status_code == 200:
                print(f"✅ Health check: OK")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return
            
            # Get status
            response = await client.get(f"{base_url}/api/status", timeout=5.0)
            if response.status_code == 200:
                status = response.json()
                print(f"📊 Status: {status['status']}")
                print(f"⏰ Uptime: {status['uptime_seconds']:.1f}s")
                print(f"💾 Memory: {status['memory_usage_percent']:.1f}%")
                print(f"⚡ CPU: {status['cpu_usage_percent']:.1f}%")
            
            # Test inference
            test_queries = [
                "Turn on the lights",
                "What's the temperature?",
                "Check system status",
                "Hello from online simulator!"
            ]
            
            print(f"\n🤖 Testing AI Inference:")
            for query in test_queries:
                print(f"\n👤 Query: '{query}'")
                
                start_time = time.time()
                response = await client.post(
                    f"{base_url}/api/inference",
                    json={"query": query, "device_id": device_name},
                    timeout=15.0
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"🤖 Response: {result['result']}")
                    print(f"⏱️ Time: {result['processing_time_ms']:.1f}ms")
                    print(f"🧠 Model: {result['model_used']}")
                else:
                    print(f"❌ Inference failed: {response.status_code}")
                
                await asyncio.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"❌ Error testing {device_name}: {e}")

async def main():
    """Main test function"""
    print("🚀 Starting Online IoT Demo Test")
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test devices
    devices = [
        (8081, "Smart Home Pi"),
        (8082, "Industrial Pi"),
        (8083, "Retail Pi")
    ]
    
    for port, name in devices:
        await test_device(port, name)
    
    print("\n🎉 Test completed!")
    print("\n💡 Next steps:")
    print("   • Open http://localhost:8081 in your browser")
    print("   • Try the API docs at http://localhost:8081/docs")
    print("   • Send custom queries to test AI responses")

if __name__ == "__main__":
    asyncio.run(main())

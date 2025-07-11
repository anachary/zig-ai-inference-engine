#!/bin/bash
# Stop Online IoT Demo

echo "🛑 Stopping Online IoT Demo..."

# Kill processes
for pidfile in *.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "Stopped process $pid"
        fi
        rm "$pidfile"
    fi
done

echo "✅ Demo stopped"

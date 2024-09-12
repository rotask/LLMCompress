#!/bin/bash

# List all PIDs of GPU processes and kill them
pids=$(nvidia-smi | grep -oP '\d+(?= +C)' | sort -u)

for pid in $pids; do
    if [ -n "$pid" ]; then
        # Check if the current user owns the process
        owner=$(ps -o user= -p $pid 2>/dev/null)
        if [ "$owner" = "$USER" ]; then
            echo "Attempting to kill process $pid owned by $USER"
            if kill -9 $pid 2>/dev/null; then
                echo "Successfully killed process $pid"
            else
                echo "Failed to kill process $pid."
            fi
        else
            echo "Skipping process $pid (owned by $owner, not $USER)"
        fi
    fi
done

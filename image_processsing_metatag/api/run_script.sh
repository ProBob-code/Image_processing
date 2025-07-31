#!/bin/bash
# List of scripts to check and run
SCRIPTS=(
    "receiver.video_initiate.py"
    "receiver.video_stream.py"
    # "receiver.video_m3u8.py"
    "receiver.video_upload.py"
    "receiver.video_callback.py"
    # "receiver.gojd_content.py"
)

# List of virtual hosts
VHOSTS=(
    "content_processing"
)

# Base directory for scripts (update this path)
BASE_DIR="subscriber"
# Virtual environment Python interpreter (update this path)
PYTHON_INTERPRETER="/opt/venv/bin/python3"

# Log directory (update this path)
LOG_DIR="/var/log"

# Loop through each virtual host
for VHOST in "${VHOSTS[@]}"; do
    echo "Processing scripts for vhost : $VHOST"
    
    # Loop through each script
    for SCRIPT in "${SCRIPTS[@]}"; do
        # Use a unique identifier for pgrep to avoid matching the grep process itself
        COUNT=$(pgrep -f "$PYTHON_INTERPRETER.*$SCRIPT.*$VHOST" | wc -l)
        
        # Check if the script is already running for this vhost
        if [ "$COUNT" -eq 0 ]; then
            echo "No instances of $SCRIPT for vhost $VHOST are running. Starting the script..."
            $PYTHON_INTERPRETER "$BASE_DIR/$SCRIPT" "$VHOST" >> "$LOG_DIR/${SCRIPT}_${VHOST}.log" 2>&1 &
            sleep 1  # Small delay to prevent overwhelming the system
        else
            echo "$SCRIPT for vhost $VHOST is already running ($COUNT instance(s)). No action."
        fi
    done
done

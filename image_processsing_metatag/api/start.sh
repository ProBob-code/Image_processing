#!/bin/bash
# Start cron in the background
/usr/sbin/crond -n &
echo "crond started in background"

# Start SSH in the background
/usr/sbin/sshd -D &
# Set core dump limit (runs in the current shell)
ulimit -c 0 && echo "Core dump limit set to 0"

# Start Gunicorn in the foreground
exec gunicorn --workers 5 app:app -b 0.0.0.0:6051 --timeout 3000 || { echo "Gunicorn failed to start"; exit 1; }
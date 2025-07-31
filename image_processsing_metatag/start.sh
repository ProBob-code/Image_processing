#!/bin/bash

# Upgrade pip and install requirements
# pip3.9 install --upgrade pip
# pip3.9 install -v -r requirements.txt

# Start SSH service and configure ulimit
service ssh start
ulimit -c 0

# Run Gunicorn with exec to replace the shell process
exec gunicorn --workers 5 app:app -b 0.0.0.0:8081 --timeout 3000

#!/bin/bash

echo "starting crond"
crond

echo "Initializing the Airflow DB"
airflow db init

echo "Creating admin User"
airflow users create --username admin --password 'AirFlw!@#' --firstname Jayendra  --lastname Patel --role Admin  --email jayendra.patel@justdial.com

echo "Starting Scheduler"
airflow scheduler -D &

echo "Starting Airflow Webserver"
airflow webserver --port 8080 -D
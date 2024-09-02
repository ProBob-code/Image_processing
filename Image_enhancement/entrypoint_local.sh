#!/bin/bash

echo "Initializing the Airflow DB"
airflow db init

echo "Creating admin User"
airflow users create --username admin --password 'admin' --firstname Bob  --lastname Jacob --role Admin  --email bobin.jacob@justdial.com

echo "Starting Scheduler"
airflow scheduler -D &

echo "Starting Airflow Webserver"
airflow webserver --port 8080 -D

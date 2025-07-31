#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "No arguments passed or missing some arguments!"
    echo "Usage: /bin/sh $0 <env> <subscriber_file> <con_cnt> <queue_name> <log_threshold_days>"
    exit 1
fi

env="$1"
subscriber_file="$2"
con_cnt="$3"
queue_name="$4"
log_threshold_days="$5"

if [ $env == "prod" ];then dir_path="/opt/airflow/dags/subscriber/"
elif [ $env == "uat" ];then dir_path="/opt/airflow/dags/subscriber/"
else echo "Plz give appropriate environment!!";fi

dir_path="/opt/airflow/dags/subscriber/"
log_directory="/tmp/"

filename=`echo $subscriber_file | cut -d '.' -f1`
datetime=$(date +"%Y-%m-%d-%T")
logfile="log_${filename}_${datetime}"

####################################################################
################ DELETE EXISTING LOG FILES START ###################
####################################################################

# CALCULATE THE log_THRESHOLD DATE
log_threshold_date=$(date -d "$log_threshold_days days ago" "+%Y-%m-%d")

# DELETE LOG FILES ORDER THAN log_THRESHOLD
find "$log_directory" -type f -name "log_${filename}*.txt" -exec bash -c '
    for file; do
        file_date=$(echo "$file" | grep -oP "\d{4}-\d{2}-\d{2}")
        log_threshold_date_seconds=$(date -d "$log_threshold_date" +%s)
        file_date_seconds=$(date -d "$file_date" +%s)
        if [[ $file_date_seconds < $log_threshold_date_seconds ]]; then
            rm "$file"
            echo "Deleted file: $file"
        fi
    done
' bash {} +

find "${log_directory}" -type f -name "*go-build*" -exec rm {} +

echo "----------------------------------------------------------------"
echo "Log files older than $log_threshold_days days have been deleted."
echo "----------------------------------------------------------------"

####################################################################
################# DELETE EXISTING LOG FILES END ####################
####################################################################

CURL='/usr/bin/curl'
QUEUE_URL="http://192.168.29.137:15672/api/queues/content_processing/${queue_name}"
response=$($CURL $QUEUE_URL -u "guest:guest")
echo "----------------------------------------------------------------"
count=$(echo "$response" | grep -o '"consumers":[0-9]*' | cut -d':' -f2)

echo ">>>> count is ${count}"
echo "${count} -eq ${con_cnt}"

if [ "$count" -eq "$con_cnt" ]
then
echo "Subscriber daemon is running..."
exit 0
else
ps aux | grep "python3.9 ${subscriber_file}" | egrep -v 'grep' | awk '{print $2}' | xargs kill -9 2> /dev/null
for i in `seq 1 $con_cnt`
        do
        log_counter="_$i"
        echo "cd ${dir_path} && PYTHONENCODING=utf-8 python3.9 ${subscriber_file}"
        cd ${dir_path} && PYTHONENCODING=utf-8 python3.9 ${subscriber_file} >> ${log_directory}${logfile}${log_counter}.log 2>&1 &        
        done
fi
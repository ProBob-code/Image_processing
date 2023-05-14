#!/bin/bash
[[ -z "{{params.env }}" ]] && echo "No arguments passed!" && exit 1

env="{{params.env }}"
subscriber_file="{{params.subscriber_file }}"
con_cnt="{{params.con_cnt}}"
req_con_cnt="{{params.req_con_cnt}}"
queue_name="{{params.queue_name}}"

if [ $env == "production" ];then dir_path="/opt/airflow/dags/subscriber/"
elif [ $env == "uat" ];then dir_path="/opt/airflow/dags/subscriber/"
else echo "Plz give appropriate environment!!";fi

dir_path="/opt/airflow/dags/subscriber/"
logfile=`echo $subscriber_file | cut -d '.' -f2`

echo $logfile
#ps aux | grep -v grep | grep  -Eo "python3 /opt/airflow/dags/subscribers/subscriber.campaign_leadgen.py | wc -l

CURL='/usr/bin/curl'
host="192.168.29.137"
QUEUE_URL="http://${host}:15672/api/queues/content_processing/${queue_name}"
response=$($CURL $QUEUE_URL -u "guest:guest")
count=`echo "$response" | jq -r ".consumers"`
#count=$(ps aux | grep -v grep | grep  -Eo "python3 ${subscriber_file}" | wc -l)

echo ">>>> count is ${count}"
echo "${count} -ge ${req_con_cnt}"

if [ "$count" -ge "$req_con_cnt" ]
then
echo "Subscriber daemon is running..."
exit 0
else
ps aux | grep "python3 ${subscriber_file}" | egrep -v 'grep | set_subscriber' | awk '{print $2}' | xargs kill -9 2> /dev/null
for i in `seq 1 $con_cnt`
        do
        echo "cd ${dir_path} && PYTHONENCODING=utf-8 python3.9 ${subscriber_file}"
        cd ${dir_path} && PYTHONENCODING=utf-8 python3 ${subscriber_file} >> /tmp/${logfile}.log 2>&1 &        
        done
fi

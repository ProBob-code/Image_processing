# # Author: Abdul Basith
# # Created on: 10/03/2023
# # Purpose: To build python flask
# # Last Modified on : 
# # Reason for Modification :  
# # ------------------------------------------------ #

# base centos 7
FROM centos:centos7.9.2009

# installing required packages
RUN yum groupinstall "Development Tools" -y && yum install epel-release net-tools sudo vim wget python3 python3-devel tar gzip gcc make expect cronie jq -y && \
    unlink /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && \
    yum install python-devel mysql-devel libffi-devel bzip2-devel mesa-libGL cmake ImageMagick mediainfo -y && wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz && \
    tar xzf Python-3.9.6.tgz && cd Python-3.9.6 && ./configure --enable-optimizations && make altinstall && \
    ln -s /usr/local/bin/python3.9 /bin/ && ln -s /usr/local/bin/pip3.9 /bin/ && \
    yum clean all && rm -rf /var/cache/yum

# set env variables 
ENV DOCROOT="/opt/airflow/api"

# work directory
WORKDIR ${DOCROOT}

# copy cron entry
# COPY crontab /var/spool/cron/root

# preparing the python dependencies
COPY requirements.txt .
RUN pip3.9 install --user --upgrade pip && \
    pip3.9 install -r requirements.txt

COPY api/ ${DOCROOT}/
COPY models/ /opt/airflow/models/
COPY dags/ /opt/airflow/dags/

# Install watchdog for hot reloading
# RUN pip3.9 install watchdog

# # call init process
CMD gunicorn --workers 2 app:app -b 0.0.0.0:7071 --timeout 300

# Start the server with hot reloading
# CMD crond && watchmedo auto-restart --recursive --patterns="*.py" --command="gunicorn --workers 2 app:app -b 0.0.0.0:8086 --timeout 300"

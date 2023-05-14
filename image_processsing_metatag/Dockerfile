# # Author: Abdul Basith
# # Created on: 10/03/2023
# # Purpose: To build airflow with mysql
# # Last Modified on : 
# # Reason for Modification :  
# # ------------------------------------------------ #

# base centos 7
FROM centos:centos7.9.2009

# installing required packages
RUN yum groupinstall "Development Tools" -y && yum install epel-release net-tools sudo vim wget python3 python3-devel tar gzip gcc make expect -y && \
    yum install python-devel mysql-devel libffi-devel -y && wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz && \
    tar xzf Python-3.9.6.tgz && cd Python-3.9.6 && ./configure --enable-optimizations && make altinstall && \
    ln -s /usr/local/bin/python3.9 /bin/ && ln -s /usr/local/bin/pip3.9 /bin/ && \
    cd ../ && rm -rf Python-3.9* && yum clean all && rm -rf /var/cache/yum

# copy airflow config
COPY airflow.cfg /opt/airflow/
COPY plugins/ /opt/airflow/plugins/ 
COPY dags/ /opt/airflow/dags/

# Preparing for airflow
RUN pip3.9 install --user --upgrade pip

ENV AIRFLOW_HOME="/opt/airflow"
ENV AIRFLOW_VERSION 2.2.3
ENV PYTHON_VERSION 3.9
ENV CONSTRAINT_URL "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

RUN pip3.9 install setuptools-rust && pip3.9 install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" && \
    pip3.9 install mysqlclient

COPY entrypoint.sh requirements.txt /
RUN pip3.9 install --no-cache-dir --user -r /requirements.txt


# Making script executable
RUN chmod a+x /*.sh

# call entrypoint.sh
CMD ["/entrypoint.sh"]

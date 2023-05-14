from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime
from random import randint
import time


with DAG(
        "setSubscriber",
        start_date=datetime(2021, 1, 1),
        schedule_interval="*/15 * * * *",
        #schedule_interval="@daily",
        catchup=False,
        tags=["Set Subscriber"],
) as dag:

        task_data = [
                {
                    "task"  : "image_processing",
                    "params": [
                        {
                            "env"               : "uat",
                            "subscriber_file"   : "subscriber.py",
                            "con_cnt"           : "1",
                            "req_con_cnt"       : "1",
                            "queue_name"        : "content_service1"
                        }
                    ]
                }
            ]
            
            
        start = DummyOperator(
            task_id='start',
            dag=dag
        )

        # set_subscriber= BashOperator(
        #     task_id="reset_subscriber",
        #     #provide_context=True,
        #     bash_command= "scripts/set_subscriber.sh",
        #     params = {
        #         'env'               : "uat",
        #         'subscriber_file'   : "subscriber.campaign_leadgen.py",
        #         'con_cnt'           : "1",
        #         'req_con_cnt'       : "1"
        #         },
        #     dag=dag,
        # )           

        with TaskGroup('reset_subscribers',
                    prefix_group_id=False,
                    ) as reset_subscribers:

            for task_data_idx in task_data:
                set_subscriber= BashOperator(
                    task_id=f"{task_data_idx['task']}",
                    #provide_context=True,
                    bash_command= "scripts/set_subscriber.sh",
                    params = {
                        'env'               : task_data_idx['params'][0]['env'],
                        'subscriber_file'   : task_data_idx['params'][0]['subscriber_file'],
                        'con_cnt'           : task_data_idx['params'][0]['con_cnt'],
                        'req_con_cnt'       : task_data_idx['params'][0]['req_con_cnt'],
                        'queue_name'        : task_data_idx['params'][0]['queue_name'],
                        },
                    dag=dag,
                )

        end = DummyOperator(
            task_id='end',
            dag=dag
        ) 

        start >> reset_subscribers >> end


"""
12:20pm
	2023-04-20 06:50:47.636655

13:07 pm	
	2023-04-20 07:37:11.593484
	

10k data	
46 minutes, and 24 seconds

overall - 47 mins

"""

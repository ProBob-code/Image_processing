
from datetime import datetime
import sys
import os
import json
from pathlib import Path
import pytz
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)

from rabbitmq import RabbitMQ
from helper import getConfigInfo
from classes.upload import UploadVideo

def main():
    try:
        # if len(sys.argv) > 1:
        #     vhost = sys.argv[1]
        # else:
        #     vhost = 'content_processing'
        
        subscribe_queue = 'upload_m3u8_video'
        # print(f"Vhost Passed {vhost}")
        rabbit_mq       = RabbitMQ()
        connection      = rabbit_mq.createConnection(getConfigInfo('rabbitmq_server1'))
        channel         = connection.channel()
        queue           = channel.queue_declare(queue=subscribe_queue, passive=False, durable=True, exclusive=False, auto_delete=False)

        message_count   = queue.method.message_count
        print(message_count)
        
        # if(message_count > 0):
        if True:
            def callback(ch, method, properties, body):
                is_queue_data_processed = True
                queue_data              = json.loads(body)
                if "message" in queue_data:
                    print("Message Found")
                    msg = dict()
                    msg = queue_data["message"]
                    print(msg)
                    # exit(1)
                    upload_video = UploadVideo
                    output = upload_video.ParallelUploadFilesToS3(msg)
                    
                    # UploadVideo.ParallelUploadFilesToS3(msg)
                    msg["thumbnail_key"] = ""
                    if msg['thumbnail'] != None:
                        print("Uploading Thumbnail")
                        thumbResponse = upload_video.UploadThumbnailToS3(msg)
                        print("Thumb Response ",thumbResponse)
                    
                else:
                    print("### NOT VALID FORMAT DATA ###")
                    errorData = dict()
                    errorData = queue_data
                    errorData["queue_name"] = "error."+subscribe_queue
                    response = rabbit_mq.postQueue(errorData)

                # current_date    = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                # print(current_date)

                # if method.delivery_tag == queue.method.message_count:
                #     channel.stop_consuming(consumer_tag=method.consumer_tag)

                if(is_queue_data_processed == True):
                    channel.basic_ack(delivery_tag = method.delivery_tag)
                else:
                    # NEED TO ADD SOME FUNCTIONALITY WHEN TASK FAILED TO SUCCESSFULLY PROCESS DATA
                    # NOTIFY OWNER
                    # PUSH TO GENERAL QUEUE
                    print('###########################')
                    print('Something went wrong! Failed to publish data in next queue')
                    print('###########################')
                    channel.basic_ack(delivery_tag = method.delivery_tag)

            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=subscribe_queue, on_message_callback=callback)
            
            print('[SUBSCRIBERS] [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()
        else:
            print("SUBSCRIBERS] No data available in queue : %s",subscribe_queue)      

    except Exception as e:

        print("[SUBSCRIBERS] failed")
        print(str(e))
        exit(1)

if __name__ == "__main__":
    main()
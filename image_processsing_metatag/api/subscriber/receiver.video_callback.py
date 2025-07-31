
from datetime import datetime
import sys
import os
import json

import pytz, traceback
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)

from rabbitmq import RabbitMQ
from helper import getConfigInfo
from classes.stream_conversion import StreamConversionProcess

def main():
    try:
        if len(sys.argv) > 1:
            vhost = sys.argv[1]
        else:
            # vhost = 'content_processing'
            print("Vhost not passed")
            exit(1)
        
        subscribe_queue = 'video_callback'
        # print(f"Vhost Passed {vhost}")
        print(f"Vhost Passed {vhost}")
        rabbit_mq       = RabbitMQ()
        
        connectServer = getConfigInfo('rabbitmq_server1')
        connectServer["host"] = vhost
        
        connection      = rabbit_mq.createConnection(connectServer)
        channel         = connection.channel()
        queue           = channel.queue_declare(queue=subscribe_queue, passive=False, durable=True, exclusive=False, auto_delete=False)

        message_count   = queue.method.message_count
        print(message_count)
        
        
        def callback(ch, method, properties, body):
            try:
                # is_queue_data_processed = True
                queue_data = json.loads(body)
                if "message" in queue_data:
                    # print("Message Found")
                    # msg = dict()
                    msg = queue_data["message"]
                    print(msg)
                    
                    stream_conversion = StreamConversionProcess()
                    output = stream_conversion.UpdateCallbackData(msg)
                    print(output)
                else:
                    print("### NOT VALID FORMAT DATA ###")
                    errorData = dict()
                    errorData = queue_data
                    errorData["queue_name"] = "error."+subscribe_queue
                    response = rabbit_mq.postQueue(errorData)

                current_date    = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                print(current_date)
                
                ch.basic_ack(delivery_tag = method.delivery_tag)
                
            except json.JSONDecodeError:
                print("Failed to decode JSON message")
                traceback.print_exc()
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            
            except Exception as e:
                print("Error processing message: ", str(e))
                traceback.print_exc()
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            finally:
                print("Message processing complete")
                # channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=subscribe_queue, on_message_callback=callback, consumer_tag='Content_Processing')
        
        print('[SUBSCRIBERS] [*] Waiting for messages. To exit press CTRL+C')
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            channel.stop_consuming()    

    except Exception as e:

        print("[SUBSCRIBERS] failed")
        print(str(e))
        exit(1)

if __name__ == "__main__":
    main()    

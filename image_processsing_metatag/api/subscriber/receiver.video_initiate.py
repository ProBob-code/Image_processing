
from datetime import datetime
import sys
import os
import json, pytz
import traceback
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)

from rabbitmq import RabbitMQ
from helper import getConfigInfo
from classes.stream_video import StreamVideo

def main():
    try:
        if len(sys.argv) > 1:
            vhost = sys.argv[1]
        else:
            print("No vhost found")
            exit(1)
        
        subscribe_queue = 'video_initiate'
        print(f"Vhost Passed {vhost}")
        rabbit_mq       = RabbitMQ()
        
        connectServer = getConfigInfo('rabbitmq_server1')
        connectServer["host"] = vhost
        
        connection = rabbit_mq.createConnection(connectServer)
        channel = connection.channel()
        queue = channel.queue_declare(
            queue=subscribe_queue,
            passive=False,
            durable=True,
            exclusive=False,
            auto_delete=False
        )

        message_count   = queue.method.message_count
        print(message_count)
        
        def callback(ch, method, properties, body):
            try:
                queue_data = json.loads(body)
                
                rabbitmq_con = RabbitMQ()
                if "message" in queue_data:
                    print("Message Found")
                    # msg = dict()
                    msg = queue_data["message"]
                    # print(msg)
                    if "video_url" in msg or "files" in msg:
                        # print(f"Valid Data Format {msg}")
                        # Start processing Data
                        output = StreamVideo.VideoInitiate(queue_data)
                        print(output)
                    else:
                        print(f"Invalid Data Format {msg}")
                        invalidData = {
                            "queue_name" : f"error.{subscribe_queue}",
                            **queue_data
                        }
                        response = rabbitmq_con.postQueue(invalidData)
                    
                else:
                    print("### NOT VALID FORMAT DATA ###")
                    errorData = {
                        "queue_name" : f"error.{subscribe_queue}",
                        **queue_data
                    }
                    response = rabbitmq_con.postQueue(errorData)

                current_date    = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                print(current_date)
                
                ch.basic_ack(delivery_tag = method.delivery_tag)
                
            except json.JSONDecodeError:
                print("Failed to decode JSON message")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            except Exception as e:
                print("Error processing message: ", str(e))
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            finally:
                print("Message processing complete")
                # channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=1)
        
        channel.basic_consume(
            queue=subscribe_queue, 
            on_message_callback=callback, 
            consumer_tag='Content_Processing'
        )
        
        print('[SUBSCRIBERS] [*] Waiting for messages. To exit press CTRL+C')
        try:
            channel.start_consuming()
        except KeyboardInterrupt:
            channel.stop_consuming()     

    except Exception as e:
        print("[SUBSCRIBERS] failed")
        print(f"ERROR : {str(e)}")
        traceback.print_exc()
        print("Exiting...")
        exit(1)

if __name__ == "__main__":
    main()
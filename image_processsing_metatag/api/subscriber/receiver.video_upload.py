
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
from rabbitmq import RabbitMQ

def main():
    try:
        if len(sys.argv) > 1:
            vhost = sys.argv[1]
        else:
            print("No vhost found")
            exit(1)
        
        subscribe_queue = "video_upload"
        # print(f"Vhost Passed {vhost}")
        connectServer = getConfigInfo('rabbitmq_server1')
        connectServer["host"] = vhost
        
        rabbit_mq       = RabbitMQ()
        connection      = rabbit_mq.createConnection(connectServer)
        channel         = connection.channel()
        queue           = channel.queue_declare(queue=subscribe_queue, passive=False, durable=True, exclusive=False, auto_delete=False)

        message_count   = queue.method.message_count
        print(message_count)
        
        def callback(ch, method, properties, body):
            try:
                queue_data = json.loads(body)
                if "message" in queue_data:
                    print("Message Found")
                    msg = queue_data["message"]
                    # print(msg)
                    upload_video = UploadVideo
                    output, m3u8VideoUrl = upload_video.UploadParallelFilesToS3(msg)
                    
                    callBackResponse = dict()
                    
                    upload_source = msg["source"]
                    instant_mp4_upload_allowed = ["catalogue", "mcatalogue", "editlisting", "image_convert", "category"]
                    if upload_source not in instant_mp4_upload_allowed:
                        mp4Output, mp4VideoUrl = upload_video.UploadMp4FileToS3(msg)
                        if mp4Output:
                            callBackResponse["mp4_url"] = mp4VideoUrl
                            
                        msg["thumbnail_key"] = ""
                        if msg['thumbnail'] != None:
                            print("Uploading Thumbnail")
                            thumbResponse, thumbRespUrl = upload_video.UploadThumbnailFileToS3(msg)
                            print("Thumb Response ",thumbResponse)
                            callBackResponse["thumb_url"] = thumbRespUrl
                    
                    if output:
                        #pass to callback
                        print("####### HIT CALLBACK #######")
                        queueData = dict()
                        rabbitMq = RabbitMQ()
                        callBackResponse["m3u8_url"] = m3u8VideoUrl
                        callBackResponse["ref_id"] = msg["random_key"]
                        
                        queueData["credentials"] = queue_data["credentials"]
                        queueData["message"] = callBackResponse
                        queueData["queue_name"] = "video_callback"
                        response2 = rabbitMq.postQueue(queueData)
                    else:
                        print("Failed in Uploading M3U8 Files")
                    
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
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e:
                print("Error processing message: ", str(e))
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
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
        print(f"ERROR : {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()

import sys
import os
import json
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)

from rabbitmq import RabbitMQ
from helper import getConfigInfo
from datetime import datetime
from classes.stream_video import StreamVideo

try:
    subscribe_queue = 'stream_video'
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
                # print("Message Found")
                msg = dict()
                msg = queue_data["message"]
                print(msg)
                stream_video = StreamVideo
                # Action on found message
                if "working_url" in msg and msg["working_url"] != "":
                    print("########### VIDEO URL #############")
                    print(msg["working_url"])
                    # result = stream_video.ExtractVideoFromUrl(msg)
                    # print(result)
                    # # exit(1)
                    # if result["status"] == False:
                    #     # pass in exception queue
                    #     print("### INVALID FORMAT DATA CONTENT - Push to exception ###")
                    #     errorData = dict()
                    #     errorData = queue_data
                    #     errorData["queue_name"] = "exception."+subscribe_queue
                    #     response = rabbit_mq.postQueue(errorData)
                    # else:
                    working_url, docid, random_key = msg["working_url"], msg["docid"], msg["random_key"]
                    video_path = stream_video.DownloadVideo(working_url, docid, random_key)
                    print("############ VIDEO PATH #################")
                    print(video_path)
                    if video_path:
                        print("checking video stream")
                        has_video, has_audio = stream_video.CheckVideoStreams(video_path)
                        if video_path:
                            if not has_audio:
                                audio_path = stream_video.DownloadAudio(working_url, docid, random_key)
                                video_path = stream_video.MergeAudio(docid, random_key, video_path, audio_path)
                        else:
                            # continue  # Skip if download failed
                            print("ERROR FOUND")

                        metadata = stream_video.GetMetadata(video_path)

                        if not metadata:
                            # continue  # Skip if metadata extraction failed
                            print("ERROR FOUND in meta data")

                        stream_video.InsertVideoMetadata(random_key, docid, video_path, metadata)
                    else:
                        print("#############Error in video path############")
                        
                elif "file_path" in msg and msg["file_path"]!="" and os.path.isfile(msg["file_path"]):
                    print("## PROCESSING OF VIDEO WITH LOCAL FILE")
                    result = stream_video.GenerateStream(msg)
                else:
                    print("File not found at the provided path")
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
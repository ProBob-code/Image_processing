from datetime import datetime
import sys
import os
import json, pytz, traceback
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

sys.path.append(parent)

from rabbitmq import RabbitMQ
from helper import getConfigInfo
from classes.stream_video import StreamVideo
from common import CheckVideoFileMimeType

def main():
    try:
        if len(sys.argv) > 1:
            vhost = sys.argv[1]
        else:
            print("No vhost found")
            exit(1)
        
        subscribe_queue = 'video_stream'
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
                queue_data = json.loads(body)
                
                rabbitmq_con = RabbitMQ()
                if "message" in queue_data:
                    print("Message Found")
                    # msg = dict()
                    msg = queue_data["message"]
                    
                    if "files" in msg:
                        print(f"Valid Data Format {msg}")
                        # Start processing Data
                        stream_video = StreamVideo
                        randomKey = msg["random_key"]
                        
                        for i in range(len(msg["files"])):
                            videoFilePath = msg["files"][i]
                            videfileStatus, convertedVideoPath, errMsg = CheckVideoFileMimeType(videoFilePath)
                            if videfileStatus:
                                # video scaling and conversion
                                convertedVideoPath = stream_video.VideoScaled(convertedVideoPath, msg)
                                msg['random_key'] = randomKey
                                reference_id = stream_video.InsertVideoLog(msg, convertedVideoPath)
                                if reference_id is not None:
                                    has_video, has_audio = stream_video.CheckVideoStreams(convertedVideoPath)
                                    if has_video:
                                        if not has_audio:
                                            # Merge silent audio
                                            video_path = stream_video.MergeSilentAudio(convertedVideoPath)
                                        else:
                                            video_path = convertedVideoPath
                                            print(f"Continue Stream Generate")
                                        
                                        print(f"Final Video Path after check {video_path}")
                                    else:
                                        # continue  # Skip if download failed
                                        print("ERROR FOUND! Not Valid file")
                                        exit(1)

                                    metadata = stream_video.GetMetadata(video_path)
                                    if not metadata:
                                        print("ERROR FOUND in meta data")
                                    updtResMeta, thumb_path = stream_video.UpdateVideoMetadata(reference_id, video_path, metadata, has_video, has_audio)
                                    # duration handling
                                    videoDuration = int(float(metadata["b_duration"]))
                                    if updtResMeta and videoDuration > 3:
                                        print("## PASS TO GENERATE M3U8")
                                        m3u8ReqData = dict()
                                        m3u8ReqData = msg
                                        m3u8ReqData["docid"] = msg["docid"]
                                        m3u8ReqData["src_video"] = video_path
                                        m3u8ReqData["bitrate"] = metadata["b_bit_rate"]
                                        m3u8ReqData["width"] = metadata["v_width"]
                                        m3u8ReqData["height"] = metadata["v_height"]
                                        m3u8ReqData["video_codec"] = metadata["v_codec_name"]
                                        m3u8ReqData["video_profile"] = metadata["v_profile"]
                                        m3u8ReqData["video_level"] = metadata["v_level"]
                                        m3u8ReqData["avg_frame_rate"] = metadata["v_avg_frame_rate"]
                                        m3u8ReqData["color_range"] = metadata["v_color_range"]
                                        m3u8ReqData["color_space"] = metadata["v_color_space"]
                                        m3u8ReqData["color_primaries"] = metadata["v_color_primaries"]
                                        m3u8ReqData["color_transfer"] = metadata["v_color_transfer"]
                                        m3u8ReqData["matrix_coefficients"] = metadata["v_matrix_coefficients"]
                                        m3u8ReqData["audio_profile"] = metadata["a_profile"]
                                        m3u8ReqData["reference_id"] = reference_id
                                        m3u8ReqData["video_thumb"] = thumb_path
                                        
                                        queueData = dict()
                                        rabbitMq = RabbitMQ()
                                        queueData["credentials"] = queue_data["credentials"]
                                        queueData["message"] = m3u8ReqData
                                        queueData["queue_name"] = "video_m3u8_generate"
                                        response2 = rabbitMq.postQueue(queueData)
                                        # return response2
                                        # genResponse = stream_conver.GenerateM3u8AndProcessData(m3u8ReqData, queue_data)
                                        print(response2)
                                        # upload mp4 and thumb if upload Source in [mcatalogue, catalogue]
                                        instant_mp4_upload_allowed = ["catalogue", "mcatalogue", "editlisting", "image_convert", "category"]
                                        if msg["source"] in instant_mp4_upload_allowed:
                                            stream_video.InstantMp4Upload(msg, reference_id, video_path, thumb_path, metadata)
                                            # print(respo)
                                    else:
                                        print(f"Short video found, duration {videoDuration}")
                            else:
                                print(f"Invalid Data Format {msg}")
                                invalidData = dict()
                                invalidData = queue_data
                                invalidData["error_msg"] = errMsg
                                invalidData["queue_name"] = "not_found."+subscribe_queue
                                response = rabbitmq_con.postQueue(invalidData)
                                print(response)
                                # break   
                    else:
                        print(f"Invalid Data Format {msg}")
                        invalidData = dict()
                        invalidData = queue_data
                        invalidData["error_msg"] = "invalid_format"
                        invalidData["queue_name"] = "invalid."+subscribe_queue
                        response = rabbitmq_con.postQueue(invalidData)
                    
                else:
                    print("### NOT VALID FORMAT DATA ###")
                    errorData = dict()
                    errorData = queue_data
                    errorData["queue_name"] = "error."+subscribe_queue
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
        print(str(e))
        exit(1)

if __name__ == "__main__":
    main()
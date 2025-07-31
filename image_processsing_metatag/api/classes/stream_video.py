import mimetypes
import os
import requests, json
import magic, subprocess
from helper import getConfigInfo
from common import downloadFile, generateRandom, RemoveFile, calculateAspectRatio
from classes.dbconnection import DbConnection
from classes.upload import UploadVideo
from classes.stream_conversion import StreamConversionProcess
from rabbitmq import RabbitMQ
import sys
import re
import shutil
import traceback

class StreamVideo():
    def __init__(self):
        pass
    
    def ExtractVideoFromUrl(post_data):
        video_url = post_data["working_url"]
        postData = dict()
        postData = post_data
        postData["video_url"] = video_url
        downloadRes = downloadFile(postData)
        mime = magic.Magic(mime=True)
        mime_type_val = mime.from_file(downloadRes["downloaded_path"])
        # Shell execute
        # minfo = subprocess.call(
        #     args="mediainfo "+ downloadRes["downloaded_path"],
        #     shell=True
        # )
        # print(minfo)
        post_data["download_res"] = downloadRes
        post_data["mime_type"] = mime_type_val
        # post_data["mediainfo"] = minfo
        post_data["status"] = downloadRes["status"]
        
        return post_data
    
    def PushVideoInQueue(limit):
        db_connection = DbConnection()
        rabbitMq = RabbitMQ()
        
        query = "SELECT video_id, docid, working_url, random_key FROM lg_videos_29L WHERE process_flag=0 LIMIT "+limit
        
        dbconn = db_connection.db_connect_dev()
        dbcursor = dbconn.cursor()
        dbcursor.execute(query)
        print(query)
        columns = dbcursor.description
        data  = []

        for row in dbcursor.fetchall():
            row_data = {}
            for (column_name, column_value) in enumerate(row):
                row_data[columns[column_name][0]] = column_value
                # data.append(row_data)
            # pass data to queue
            queueData = dict()
            queueHost = getConfigInfo('rabbitmq_server1')
            
            queueData["credentials"] = queueHost
            queueData["message"] = row_data
            queueData["queue_name"] = "stream_video"
            # print(queueData)
            respo = rabbitMq.postQueue(queueData)
            
            updt_query = "UPDATE lg_videos_29L SET process_flag=3 WHERE video_id = "+str(row_data['video_id'])
            upd_dbcursor = dbconn.cursor()
            upd_dbcursor.execute(updt_query)
            dbconn.commit()
            upd_dbcursor.close()
            # print(respo)

        json_object = json.dumps(data)
        # print(json_object)
        return True
    
    def GenerateStream(data):
        post_data = dict()
        post_data = data
        return True
    
    #Function to download video
    def DownloadVideo(working_url, docid, random_key):
        try:
            print("downloading video")
            # input_dir=f"video_input/{docid}/{random_key}"
            NFS_path = getConfigInfo('NFS_path.video_input')
            if docid!="" and random_key!="":
                input_dir = f"{NFS_path}/{docid}/{random_key}"
            else:
                random_key = generateRandom(15)
                input_dir = f"{NFS_path}/{random_key}"
            
            if not os.path.exists(input_dir):
                os.makedirs(input_dir)  # create folder if it does not exist
            
            rand = generateRandom(5)
            input_file = f"{docid}_{random_key}_{rand}.mp4"
            # input_file = f"{docid}_{random_key}.mp4"
            # os.makedirs(input_dir, exist_ok=True)
            local_path = os.path.join(input_dir, input_file)

            if not os.path.exists(local_path):
                response = requests.get(working_url, stream=True)
                if response.status_code == 200:
                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    print(f"Downloaded at : {local_path}")
                    return local_path
                else:
                    # connection = mysql.connector.connect(**DB_CONFIG)
                    # cursor = connection.cursor(dictionary=True)
                    # cursor.execute("UPDATE lg_videos_29L SET process_flag = 2 WHERE docid = %s AND random_key = %s " ,(docid,random_key)) 
                    # cursor.close()
                    # connection.close()
                    print(f"Failed to download: {local_path}")
                    return None
            else:
                return local_path
        except Exception as e:
            print(f"Error downloading {local_path}: {e}")
            return None
        

    def DownloadAudio(working_url, docid, random_key):
        try:
            contain_audio = re.sub(r"=m\d+$", "=m18", working_url)
            rand = generateRandom(6)
            audio_file = f"audio_{docid}_{random_key}_{rand}.wav"  # Ensure a proper file extension
            # input_dir = f"video_input/{docid}/{random_key}"
            NFS_path = getConfigInfo('NFS_path.video_input')
            if docid!="" and random_key!="":
                input_dir = f"{NFS_path}/{docid}/{random_key}"
            else:
                random_key = generateRandom(15)
                input_dir = f"{NFS_path}/{random_key}"
            audio_path = os.path.join(input_dir, audio_file)

            if not os.path.exists(audio_path):
                
                yt_dlp_cmd = [
                    "yt-dlp", "-x", "--audio-format", "wav", contain_audio,
                    "-o", audio_file
                ]

                print(f"🎵 Downloading & Extracting Audio: {contain_audio} → {audio_path}")
                process = subprocess.run(yt_dlp_cmd, capture_output=True, text=True, check=False)

                if process.returncode != 0:
                    print(f" yt-dlp Error:\n{process.stderr}")
                    sys.exit(1)
                else:
                    print(" Audio downloaded successfully!")
                    # Move the file
                    shutil.move(audio_file, input_dir)
                    print(" Audio moved successfully!")
                    return audio_path

                # # Convert to WAV if needed
                # wav_path = audio_path.replace(".m4a", ".wav")
                # ffmpeg_cmd = ["ffmpeg", "-i", audio_path, "-q:a", "0", "-map", "a", wav_path]
                # print("Generated yt-dlp Command:", " ".join(ffmpeg_cmd))
                # process_ffmpeg = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)

                # if process_ffmpeg.returncode != 0:
                #     print(f" FFmpeg Error:\n{process_ffmpeg.stderr}")
                #     sys.exit(1)

                # print("Audio downloaded and converted successfully!")
                # return wav_path

            else:
                return audio_path

        except Exception as e:
            print(f" Error downloading {audio_file}: {e}")
            return None


        
    def MergeAudio(docid, random_key, input_video,input_audio):

        NFS_path = getConfigInfo('NFS_path.video_input')
        rand = generateRandom(4)
        merged_video_path =f"{NFS_path}/{docid}/{random_key}/merged_{docid}_{random_key}_{rand}.mp4"
        print(input_video)
        print(input_audio)
        print(merged_video_path)
        
        ffmpeg_cmd = [
            # "/usr/local/bin/ffmpeg", "-y",  
            "ffmpeg", "-y",  
            "-i", input_video,  # Input video
            "-i", input_audio,  # Input audio
            "-c:v", "copy",     # Copy video codec
            "-c:a", "aac",      # Use AAC codec for audio
            "-strict", "experimental",
            merged_video_path   # Output merged file
        ]

        print("Generated FFmpeg Command:", " ".join(ffmpeg_cmd))
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)

        #  Debugging output
        if process.returncode != 0:
            print(f" Error occurred:\n{process.stderr}")
            sys.exit(1)
        else:
            print("Audio downloaded successfully!")
        print(" Process completed successfully!")
        return merged_video_path

  
    # Extract metadata using ffprobe
    def GetMetadata(video_path):
        try:
            command = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,bit_rate,size,format_name,format_long_name",
                "-show_entries","stream=index,codec_name,codec_type,codec_long_name,width,height,avg_frame_rate,bit_rate,profile,level,color_range,color_space,color_primaries,color_transfer,matrix_coefficients,profile,channels,codec_type,codec_name,bit_rate,sample_rate",
                "-of","json", video_path
            ]
            #"default=noprint_wrappers=1:nokey=1",

            result = subprocess.run(command, capture_output=True, text=True)
            metadata = json.loads(result.stdout)

            # Extract video & audio properties
            video_format = metadata.get("format", {})  

            # video_streams = [stream for stream in metadata["streams"] if stream["codec_type"] == "video"][0]
            # audio_streams = [stream for stream in metadata["streams"] if stream["codec_type"] == "audio"][0]

            video_streams = next((s for s in metadata["streams"] if s["codec_type"] == "video"), None)
            audio_streams = next((s for s in metadata["streams"] if s["codec_type"] == "audio"), None)

            video_streams_data = {}
            audio_streams_data = {
                    "a_audio_codec":  "unknown",
                    "a_codec_long_name": "unknown",
                    "a_profile": "unknown",
                    "a_codec_type": "unknown",
                    "a_sample_rate":  "unknown",
                    "a_channels":"unknown",
                    "a_avg_frame_rate":"unknown",
                    "a_bit_rate":"unknown"
                }

            if  video_streams:
                
                video_streams_data =  {
                    "b_format_name": video_format.get("format_name", "Unknown"),
                    "b_format_long_name": video_format.get("format_long_name", "Unknown"),
                    "b_duration": video_format.get("duration", "Unknown"),
                    "b_bit_rate": video_format.get("bit_rate", "Unknown"),
                    "b_size": video_format.get("size", "Unknown"),
                    "v_codec_name": video_streams.get("codec_name", 0),
                    "v_codec_long_name": video_streams.get("codec_long_name", 0),
                    "v_profile": video_streams.get("profile", 0),
                    "v_codec_type": video_streams.get("codec_type", 0),
                    "v_width": video_streams.get("width", 0),
                    "v_height": video_streams.get("height", 0),
                    "v_level": video_streams.get("level", 0),
                    "v_color_range": video_streams.get("color_range", "tv"),
                    "v_color_space": video_streams.get("color_space", "bt709"),
                    "v_color_transfer": video_streams.get("color_transfer", "bt709"),
                    "v_color_primaries": video_streams.get("color_primaries", "bt709"),
                    "v_matrix_coefficients": video_streams.get("matrix_coefficients", "bt709"),
                    "v_avg_frame_rate": video_streams.get("avg_frame_rate", 0),
                    "v_bit_rate": video_streams.get("bit_rate", 0)
                }
                
                
            if audio_streams:
        
                audio_streams_data = {
                    "a_audio_codec": audio_streams.get("codec_name", "unknown"),
                    "a_codec_long_name": audio_streams.get("codec_long_name", "unknown"),
                    "a_profile": audio_streams.get("profile", "unknown"),
                    "a_codec_type": audio_streams.get("codec_type", "unknown"),
                    "a_sample_rate": audio_streams.get("sample_rate", "unknown"),
                    "a_channels": audio_streams.get("channels", "unknown"),
                    "a_avg_frame_rate": audio_streams.get("avg_frame_rate", 0),
                    "a_bit_rate": audio_streams.get("bit_rate", 0)
                }

            merged_data = {**video_streams_data, **audio_streams_data}
            
            return merged_data

        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            return None

    #Function to insert video metadata into database
    def InsertVideoMetadata(random_key, docid, video_path, metadata):
        print("INSERT AND UPDATE META")
        db_connection = DbConnection()
        dbconn = db_connection.db_connect_dev()
        try:
            # connection = mysql.connector.connect(**DB_CONFIG)
            # cursor = connection.cursor(prepared=True)
            json_string = json.dumps(metadata, indent=4)
            print(json_string)
            cursor = dbconn.cursor()
            
            

            sql = """
            UPDATE lg_videos_29L SET 
                src_video = %s, 
                format = %s, duration = %s, bitrate = %s, 
                width = %s, height = %s, size = %s, 
                video_codec = %s, 
                video_profile = %s, video_level = %s, 
                video_bitrate = %s, avg_frame_rate = %s,
                color_range = %s, color_space = %s, color_primaries = %s, color_transfer = %s, matrix_coefficients = %s,
                audio_codec = %s, audio_bitrate = %s, 
                audio_sample_rate = %s, audio_channels = %s, status = 'downloded', process_flag=1
            WHERE docid = %s AND random_key = %s
            """
            
            values = (
                video_path, 
                metadata["b_format_name"], 
                metadata["b_duration"], 
                metadata["b_bit_rate"],
                metadata["v_width"], 
                metadata["v_height"], 
                metadata["b_size"], 
                metadata["v_codec_name"], 
                metadata["v_profile"],
                metadata["v_level"], 
                metadata["v_bit_rate"], 
                metadata["v_avg_frame_rate"],
                metadata["v_color_range"],
                metadata["v_color_space"],
                metadata["v_color_primaries"],
                metadata["v_color_transfer"],
                metadata["v_matrix_coefficients"],
                metadata["a_audio_codec"], 
                metadata["a_bit_rate"],
                metadata["a_sample_rate"], 
                metadata["a_channels"], docid, random_key
            )

            cursor.execute(sql, values)
            formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
            print("Generated SQL Query:", formatted_sql)
            dbconn.commit()

            try:
                # connection = mysql.connector.connect(**DB_CONFIG)
                cursor_sel = dbconn.cursor(dictionary=True)
                cursor_sel.execute("SELECT video_id,docid,uniq_id,video_url,working_url,src_video,src_audio,dest_video,status,tag,format,duration,bitrate,width,height,size,has_video,has_audio,video_codec,video_profile,video_level,video_bitrate,avg_frame_rate,avg_motion_score,frame_count,color_range,color_space,color_primaries,color_transfer,matrix_coefficients,audio_codec,audio_profile,audio_channels,audio_bitrate,audio_sample_rate,random_key FROM lg_videos_29L WHERE docid = %s AND random_key = %s ",(docid, random_key))
                videos = cursor_sel.fetchall()
                cursor_sel.close()
                # dbconn.close()

                # Initialize index for while loop
                index = 0
                total_videos = len(videos)

                thumb_path = StreamVideo.GenerateVideoThumb(video_path, int(metadata["v_width"]), int(metadata["v_height"]), metadata["b_duration"])
                print("### THUMB PATH ###", thumb_path)
                # Process videos in a while loop
                while index < total_videos:
                    video = videos[index]
                    print(f"Processing Video ID: {video['video_id']} - {video['docid']}")
                    
                    video_data = {
                        "video_id":video["video_id"],
                        "docid":video["docid"],
                        "uniq_id":video["uniq_id"],
                        "video_url":video["video_url"],
                        "working_url":video["working_url"],
                        "src_video":video["src_video"],
                        "src_audio":video["src_audio"],
                        "dest_video":video["dest_video"],
                        "status":video["status"],
                        "tag":video["tag"],
                        "format":video["format"],
                        "duration":video["duration"],
                        "bitrate":video["bitrate"],
                        "width":video["width"],
                        "height":video["height"],
                        "size":video["size"],
                        "has_video":video["has_video"],
                        "has_audio":video["has_audio"],
                        "video_codec":video["video_codec"],
                        "video_profile":video["video_profile"],
                        "video_level":video["video_level"],
                        "video_bitrate":video["video_bitrate"],
                        "avg_frame_rate":video["avg_frame_rate"],
                        "avg_motion_score":video["avg_motion_score"],
                        "frame_count":video["frame_count"],
                        "color_range":video["color_range"],
                        "color_space":video["color_space"],
                        "color_primaries":video["color_primaries"],
                        "color_transfer":video["color_transfer"],
                        "matrix_coefficients":video["matrix_coefficients"],
                        "audio_codec":video["audio_codec"],
                        "audio_profile":video["audio_profile"],
                        "audio_channels":video["audio_channels"],
                        "audio_bitrate":video["audio_bitrate"],
                        "audio_sample_rate":video["audio_sample_rate"],
                        "random_key":video["random_key"],
                        "video_thumb":thumb_path
                    }

                    # push_to_rabbitmq(video_data)
                    rabbitMq = RabbitMQ()
                    queueData = dict()
                    queueHost = getConfigInfo('rabbitmq_server1')
                    
                    queueData["credentials"] = queueHost
                    queueData["message"] = video_data
                    queueData["queue_name"] = "generate_m3u8"
                    # print(queueData)
                    response = rabbitMq.postQueue(queueData)
                    print(response)
                    
                    index += 1

            except dbconn.Error as err:
                print(f" Database Error: {err}")
                return []


            cursor.close()
            dbconn.close()
            print(f"Metadata Updated for {docid}")

        except dbconn.Error as err:
            print(f"Database Insert Error: {err}")

    def CheckVideoStreams(video_path):
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "stream=codec_type", "-of", "json", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            streams = json.loads(result.stdout)["streams"]

            has_video = any(s["codec_type"] == "video" for s in streams)
            has_audio = any(s["codec_type"] == "audio" for s in streams)
            return has_video, has_audio
        except Exception as e:
            print(f"Error checking streams: {e}")
            return False, False            

    def GenerateVideoThumb(video_path, width,height,duration):
        print("#### Generating Thumb ####")
        try:
            thumb_path = video_path.replace(".mp4", "-thumb.jpg")
            print("thumb_path Path:", thumb_path)
            
            # if width < height:
            #     thumb_resolution = '720x1280'
            # else:
            #     thumb_resolution = '1280x720'
                  
            thumb_resolution = f"{width}x{height}"
            
            if int(float(duration)) < 10:
                offset = round(float(duration)/2)
            else:
                offset = 10
            
            cmd = [
                "ffmpeg", "-y", "-itsoffset", str(offset), "-i", video_path, "-vcodec", "mjpeg", "-vframes", "1", "-an", "-f", "rawvideo", "-s", thumb_resolution, thumb_path
            ]
            print("Generated FFmpeg Thumb Generate CMD:", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return thumb_path
        except Exception as e:
            print(f"Error generating thumb: {e}")
            return None

    def VideoInitiate(postdata):
        # print(postdata)
        queueData = dict()
        rabbitMq = RabbitMQ()
        
        queueData["credentials"] = postdata["credentials"]
        message = postdata["message"]
        print(f"Passed {message}")
        
        if "files" in message and len(message["files"]) > 0:
            print(f"Files :{message['files']}")
            # pass to next queue to video_stream
            message["random_key"] = message["random_key"] if "random_key" in message else generateRandom(15)
            
            queueData["message"] = message
            queueData["queue_name"] = "video_stream"
            response1 = rabbitMq.postQueue(queueData)
            return response1
        elif "video_url" in message and message["video_url"] != "":
            download_res = downloadFile(message)
            print(download_res)
            if download_res["status"]:
                print(f"Files downloaded as :{download_res['downloaded_path']}")
                # pass to next queue to video_stream
                downloadedData = []
                downloadedData.append(download_res['downloaded_path'])
                
                message["random_key"] = message["random_key"] if "random_key" in message else generateRandom(15)
                message["files"] = downloadedData
                
                queueData["message"] = message
                queueData["queue_name"] = "video_stream"
                response2 = rabbitMq.postQueue(queueData)
                return response2
            else:
                print(f"Failed in Download or Invalid url")
                # Update action in callback
                queueData["download_msg"] = download_res["error_msg"]
                queueData["message"] = message
                queueData["queue_name"] = "failed_video_download"
                response1 = rabbitMq.postQueue(queueData)
                return response1
        else:
            print(f"Invalid data format {postdata}")
            return None
    
    # def RemoveFile(self, file_path):
    #     try:
    #         if os.path.exists(file_path):
    #             os.remove(file_path)
    #             print(f"Removed file: {file_path}")
    #     except Exception as e:
    #         print(f"Error removing file: {e}")
    #         # return None
    
    def MergeSilentAudio(video_file_path):
        try:
            # has_video, has_audio = StreamVideo.CheckVideoStreams(video_file_path)
            # if has_audio:
            #     print("Video already has audio. No silent audio merging required.")
            #     return video_file_path
            
            # Generate output file path with silent audio merged
            silent_merged_path = video_file_path.replace(".mp4", "_silent.mp4")
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_file_path,
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                silent_merged_path
            ]
            print("Generated FFmpeg Silent Audio CMD:", " ".join(ffmpeg_cmd))
            process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
            if process.returncode != 0:
                print("Error merging silent audio:", process.stderr)
                return None

            print("Silent audio merged successfully!")
            # StreamVideo.RemoveDataQueue(video_file_path)
            return silent_merged_path
        
        except Exception as e:
            print("Exception during merging silent audio:", str(e))
            return None
    
    # def InsertVideoLog(video_path):
    def InsertVideoLog(postData, video_path):
        print("######## INSERT VIDEO LOG ########")
        ref_id = str.lower(generateRandom(15))
        random_key = postData["random_key"]
        if "source" in postData and postData["source"] != "":
            module = postData["source"]
        else:
            module = "justdial"
        try:
            db_connection = DbConnection()
            dbconn = db_connection.db_connect_live()
            json_string = json.dumps(postData)
            print(json_string)
            cursor = dbconn.cursor()
            
            sql = """
            INSERT INTO tbl_video_process_log SET 
                ref_id = %s, 
                video_src = %s,
                post_data = %s,
                module = %s,
                random_key = %s,
                status = 'IN_QUEUE'
            """
            
            values = (ref_id, video_path,json_string,module,random_key)
            cursor.execute(sql, values)
            
            formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
            print("Generated INSERT SQL Query:", formatted_sql)
            dbconn.commit()
            cursor.close()
            dbconn.close()
            print(f"LOG INSERTED for {ref_id}")
            return ref_id
        except Exception as err:
            print(f"Database Insert Error: {err}")   
            return None
    
    def UpdateVideoMetadata(ref_id, video_path, metadata, has_video, has_audio):
        print("UPDATE VIDEO META")
        db_connection = DbConnection()
        dbconn = db_connection.db_connect_live()
        try:
            json_string = json.dumps(metadata, indent=4)
            print(json_string)
            # Generate Thumb
            thumb_path = StreamVideo.GenerateVideoThumb(video_path, int(metadata["v_width"]), int(metadata["v_height"]), metadata["b_duration"])
            print("### THUMB PATH ###", thumb_path)
            
            cursor = dbconn.cursor()
            
            if has_audio:
                audio_available = 1
            else:
                audio_available = 0
            
            if has_video:
                video_available = 1
            else:
                video_available = 0
            
            sql = """
            UPDATE tbl_video_process_log SET 
                video_src = %s, 
                format = %s, duration = %s, bitrate = %s, 
                width = %s, height = %s, size = %s, 
                video_codec = %s, 
                video_profile = %s, video_level = %s, 
                video_bitrate = %s, avg_frame_rate = %s,
                color_range = %s, color_space = %s, color_primaries = %s, color_transfer = %s, matrix_coefficients = %s,
                audio_codec = %s, audio_bitrate = %s, 
                audio_sample_rate = %s, audio_channels = %s, status = 'DOWNLOADED', process_flag=1, has_audio = %s, has_video = %s
            WHERE ref_id = %s
            """
            
            values = (
                video_path, 
                metadata["b_format_name"], 
                metadata["b_duration"], 
                metadata["b_bit_rate"],
                metadata["v_width"], 
                metadata["v_height"], 
                metadata["b_size"], 
                metadata["v_codec_name"], 
                metadata["v_profile"],
                metadata["v_level"], 
                metadata["v_bit_rate"], 
                metadata["v_avg_frame_rate"],
                metadata["v_color_range"],
                metadata["v_color_space"],
                metadata["v_color_primaries"],
                metadata["v_color_transfer"],
                metadata["v_matrix_coefficients"],
                metadata["a_audio_codec"], 
                metadata["a_bit_rate"],
                metadata["a_sample_rate"], 
                metadata["a_channels"], audio_available, video_available, ref_id
            )

            cursor.execute(sql, values)
            formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
            print("Generated SQL Query:", formatted_sql)
            dbconn.commit()

            cursor.close()
            dbconn.close()
            print(f"Metadata Updated for {ref_id}")
            return True, thumb_path
        except Exception as err:
            print(f"Database Insert Error: {err}")
            return False, None

    def VideoScaled(video_path, msg_data):
        # video_info = data.get('files', {}).get('video', {})
        # video_path = video_info.get('path')
        allowed_scaling_check = ["catalogue", "mcatalogue", "editlisting", "imageconvert"]
        module = msg_data["source"] if "source" in msg_data else ""
        if module in allowed_scaling_check and os.path.exists(video_path):
            # Get duration from ffprobe
            ffprobe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,bit_rate,size,format_name",
                "-of", "json",
                video_path
            ]
            try:
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                video_info = json.loads(result.stdout)
            except Exception as e:
                print("Error running ffprobe:", e)
                video_info = {}

            video_format = video_info.get("format", {})
            
            print("Video Info:", video_info)
            print("Video Format Info:", video_format)
            traceback.print_exc()
            if video_format:
                duration = int(float(video_format.get("duration", 0)))
            else:
                duration = 0

            print("duration Info and type:", duration, type(duration))
            # Calculate bitrate in bps and Mbps
            scaled_video_size = video_format.get("size", 0)
            if duration > 0:
                scaled_video_bitrate_bps = (int(scaled_video_size) * 8) / duration
                scaled_video_bitrate_mbps = scaled_video_bitrate_bps / 1000000
            else:
                scaled_video_bitrate_bps = 0
                scaled_video_bitrate_mbps = 0

            # Build scaled video path by replacing extension with "-s"
            reversed_splited_path = video_path.rsplit('.', 1)
            scaled_video_path = reversed_splited_path[0] + "-s." + reversed_splited_path[1]

            bitrate_mbps_param = f" -b:v {scaled_video_bitrate_mbps}M" if scaled_video_bitrate_mbps > 0 else ""
            # Construct and run ffmpeg command
            scaled_video_cmd = f"ffmpeg -i {video_path} {bitrate_mbps_param} -vf scale=iw:ih -c:a copy -y {scaled_video_path}"
            subprocess.run(scaled_video_cmd, shell=True)

            # Set response (if the scaled file exists and has a size)
            if os.path.exists(scaled_video_path) and os.path.getsize(scaled_video_path) > 0:
                RemoveFile(video_path)
                base = os.path.basename(scaled_video_path)
                filename, extension = os.path.splitext(base)
                mime, _ = mimetypes.guess_type(scaled_video_path)
                print(f"{filename} scaled with MIME type {mime} and extension {extension}")
        else:
            print(f"Video scaling not required for module: {module} or file does not exist: {video_path}")
            scaled_video_path = video_path    
        return scaled_video_path
    
    def InstantMp4Upload(data, reference_id, video_path, thumb_path, meta_data):
        # upload mp4 and thumb
        upload_video = UploadVideo
        stream_conv = StreamConversionProcess()
        
        v_data = {
            "docid" : data["docid"],
            "src_video" : video_path,
            "source" : data["source"],
            "random_key" : reference_id
        }
        mp4_status, mp4_res_url = upload_video.UploadMp4FileToS3(v_data)
        if mp4_status:
            print(f"Mp4 uploaded and url {mp4_res_url}")
        
        t_data = {
            "docid" : data["docid"],
            "thumbnail" : thumb_path,
            "source" : data["source"],
            "random_key" : reference_id
        }
        tmb_status, tmb_res_url = upload_video.UploadThumbnailFileToS3(t_data)
        if tmb_status:
            print(f"Thumb uploaded and url {tmb_res_url}")
        
        width = meta_data["v_width"]
        height = meta_data["v_height"]
        duration = meta_data["b_duration"]
        file_size = meta_data["b_size"]
        aspect_ratio = calculateAspectRatio(width,height)
        modified_by = "backend_process"
        
        module_type = data["module_type"] if "module_type" in data else ""
        approved    = data["approved"] if "approved" in data else 0
        reprocess   = int(data["reprocess"]) if "reprocess" in data else 0
        platform    = data["platform"] if "platform" in data else ""
        
        bitflag_cnt = 0
        
        if int(module_type) == 3:
            approved = 1
        
        if approved == 0:
            bitflag_cnt  = bitflag_cnt + 1
        elif approved == 1:
            bitflag_cnt  = bitflag_cnt + 2
        elif approved == 2:
            bitflag_cnt  = bitflag_cnt + 4

        
        if platform != '':
            if int(platform) == 1:
                bitflag_cnt  = bitflag_cnt + 32
            elif int(platform) == 2:
                bitflag_cnt  = bitflag_cnt + 64
            elif int(platform) == 3:
                bitflag_cnt  = bitflag_cnt + 128
            elif int(platform) == 4:
                bitflag_cnt  = bitflag_cnt + 16384
        
        if data["source"] == "category":
            # insert in tbl_category_video_details
            print(f"Inserting into tbl_category_video_details and data {data}")
            catid        = data["docid"] if "docid" in data else ""
            random_key   = data["random_key"] if "random_key" in data else ""
            # source       = data["source"] if "source" in data else "jd_backend"
            catname      = data["catname"] if "catname" in data else ""
            feed_url     = data["feed_url"] if "feed_url" in data else ""
            product_position = int(data["product_position"]) if "product_position" in data else 0
            image_label  = data["image_label"] if "image_label" in data else ""
            image_description = data["image_description"] if "image_description" in data else ""
            
            upload_by    = data["upload_by"] if "upload_by" in data else "backend_video_process"
            img_scope    = int(data["img_scope"]) if "img_scope" in data else 1
            priority_flag= int(data["priority_flag"]) if "priority_flag" in data else 0
            process_flag = int(data["process_flag"]) if "process_flag" in data else 0
            video_tag    = int(data["video_tag"]) if "video_tag" in data else 1
            reprocess    = int(data["reprocess"]) if "reprocess" in data else 0
            video_id     = int(data["video_id"]) if "video_id" in data else 0
            modified_by  = data["modified_by"] if "modified_by" in data else "backend_video_reprocess"
            
            # fetch max id for category in new insertion
            max_id = stream_conv.FetchMaxImgId(catid)
            img_id = max_id + 1 if max_id is not None else 1
            
            db_connection = DbConnection()
            dbconn = db_connection.db_connect_live()
            try:
                cursor = dbconn.cursor(prepared=True)
                if video_id != 0 and reprocess == 1:
                    # Update existing record
                    update_qry = """
                        UPDATE tbl_category_video_details SET
                            ref_id = %s, product_url = %s, product_image_url = %s, aspect_ratio = %s,
                            file_size=%s, duration=%s, updated_date = NOW(), updated_by = %s, process_flag = %s
                        WHERE national_catid = %s AND id = %s
                    """
                    
                    values = (
                        reference_id, mp4_res_url, tmb_res_url, aspect_ratio,
                        file_size, duration, modified_by, process_flag,
                        catid, video_id
                    )
                    
                    cursor.execute(update_qry, values)
                    formatted_sql = update_qry % tuple(map(repr, values)) # Converts to a string
                    print("Generated Update SQL Query:", formatted_sql)
                    dbconn.commit()
                    cursor.close()
                    # return True
                else:
                    insert_qry = """
                        INSERT INTO tbl_category_video_details SET 
                            national_catid = %s, category_name = %s, upload_by = %s, image_label = %s,
                            image_description = %s, feed_url = %s, img_id = %s, product_position = %s,
                            img_scope = %s, priority_flag = %s, display_flag = 1, process_flag = %s,
                            create_date = NOW(), video_tag = %s, ref_id = %s, random_key = %s,
                            product_url = %s, product_image_url = %s, aspect_ratio = %s,
                            duration = %s, file_size = %s
                        """
                
                    values = (
                        catid, catname, upload_by, image_label, 
                        image_description, feed_url, img_id, product_position, 
                        img_scope, priority_flag, process_flag,
                        video_tag, reference_id, random_key,
                        mp4_res_url, tmb_res_url, aspect_ratio,
                        duration, file_size
                    )
                    cursor.execute(insert_qry, values)
                    formatted_sql = insert_qry % tuple(map(repr, values))  # Converts to a string
                    print("Generated Insert SQL Query:", formatted_sql)
                    dbconn.commit()
                    cursor.close()
                    # return True
            except dbconn.Error as e:
                print(f"Error found in insert {e}")
                traceback.print_exc()
                # return False
        else:
            print(f"Inserting into tbl_video_details and data {data}")
            vidStatus, foundVideoId = stream_conv.checkRandomKeyExist(data)
            print(f"Check random key exist in table : {vidStatus} => {foundVideoId}")
            if vidStatus:
                video_id = foundVideoId
            else:
                video_id = 0
            db_connection = DbConnection()
            dbconn = db_connection.db_connect_live()
            try:
                cursor = dbconn.cursor(prepared=True)
                # Move to pending and no reprocess because of OCR check
                if reprocess == 0:
                    approved1 = 0
                else:
                    approved1 = approved
                if video_id != 0:
                    # Update existing record
                    update_qry = """
                        UPDATE tbl_video_details SET
                            ref_id = %s, approved = %s, video_url = %s, video_url_image = %s, bit_flag = %s, aspect_ratio = %s,
                            file_size=%s, width=%s, height=%s, duration=%s, modified_date = NOW(), modified_by = %s
                        WHERE docid = %s AND video_id = %s
                    """
                    
                    values = (
                        reference_id, approved1, mp4_res_url, tmb_res_url, bitflag_cnt, aspect_ratio,
                        file_size, width, height, duration, modified_by,
                        data["docid"], video_id
                    )
                    
                    cursor.execute(update_qry, values)
                    formatted_sql = update_qry % tuple(map(repr, values)) # Converts to a string
                    print("Generated Update SQL Query:", formatted_sql)
                    dbconn.commit()
                    cursor.close()                    
                    # return True
                else:
                    insert_qry = """
                        INSERT INTO tbl_video_details SET 
                            docid = %s, create_date = NOW(), approved = %s, upload_by = %s, module_type = %s, 
                            bit_flag = %s, process_flag = '1', random_key = %s, ref_id = %s,
                            video_tag = 0, video_url = %s, video_url_image = %s,
                            aspect_ratio = %s, file_size=%s, width=%s, height=%s, catalogue_id=0, duration=%s
                        """
                
                    values = (
                        data["docid"], approved1, data["upload_by"], data["module_type"],
                        bitflag_cnt, data["random_key"], reference_id,
                        mp4_res_url, tmb_res_url, 
                        aspect_ratio, file_size, width, height, duration
                    )
                    cursor.execute(insert_qry, values)
                    formatted_sql = insert_qry % tuple(map(repr, values))  # Converts to a string
                    print("Generated Insert SQL Query:", formatted_sql)
                    dbconn.commit()
                    cursor.close()
                    # return True
            except dbconn.Error as e:
                print(f"Error found in insert {e}")
                traceback.print_exc()
                # return False
import json
import subprocess
import hashlib
import cv2
import numpy as np
import time

class VideoMeta():
    def __init__(self):
        pass


    def shake_128(input_string, output_length=16):
        """
        Generates a variable-length output using the Shake-128 algorithm.

        Args:
            input_string: The input string to be hashed.
            output_length: The desired length of the output in bytes (default: 16).

        Returns:
            A byte string of the specified length representing the Shake-128 output.
        """

        hasher = hashlib.shake_128()
        hasher.update(input_string.encode())
        hashed_bytes = hasher.digest(output_length)
        hashed_string = hashed_bytes.hex()  # Convert bytes to string using hex encoding
        return hashed_string  

    def convert_dict_values(data):
        converted_data = {}
        for key, value in data.items():
            try:
                # print("value", value)
                converted_data[key] = int(value)
            except:
                try:
                    converted_data[key] = float(value)
                except:
                    # Handle non-numeric values (optional)
                    converted_data[key] = str(value)  # Keep the original value
        return converted_data




    def convert_dict_values(data):
        converted_data = {}
        for key, value in data.items():
            try:
                # print("value", value)
                converted_data[key] = int(value)
            except:
                try:
                    converted_data[key] = float(value)
                except:
                    # Handle non-numeric values (optional)
                    converted_data[key] = str(value)  # Keep the original value
        return converted_data
    
    def find_matching_dicts(list_of, value, key="@type"):
        for track_dict in list_of:
            if key in track_dict and track_dict[key] == value:
                return track_dict
        return "0"


    def key_value_validation_INT(video_info, key1, key2):
        if key2 in video_info[key1] and video_info[key1].get(key2, None):
            if not isinstance(video_info[key1][key2], int):
                video_info[key1][key2] = 0
        elif(video_info.get(key1, {}).get(key2, None) == None):
            video_info[key1][key2] = 0


    def key_value_validation_FLOAT(video_info, key1, key2):
        if key2 in video_info[key1] and video_info[key1].get(key2, None):
            if not isinstance(video_info[key1][key2], float):
                video_info[key1][key2] = 0
        elif(video_info.get(key1, {}).get(key2, None) == None):
            video_info[key1][key2] = 0


    def get_video_mediainfo_json(url):
        command = "mediainfo --Output=JSON "+ url
        details = subprocess.check_output(command, shell=True, encoding="utf-8")
        details = json.loads(details) 
        # print(details)
        video_info = {} 

        general_meta = VideoMeta.find_matching_dicts(details["media"]["track"], value = "General")
        video_meta = VideoMeta.find_matching_dicts(details["media"]["track"], value = "Video")
        audio_meta = VideoMeta.find_matching_dicts(details["media"]["track"], value = "Audio")

        video_info["General"] = general_meta
        video_info["Video"] = video_meta
        video_info["Audio"] = audio_meta
        
        # datatype conversion
        if(video_info["General"] != "0"):
            video_info["General"] = VideoMeta.convert_dict_values(video_info["General"])
        if(video_info["Video"] != "0"):
            video_info["Video"] = VideoMeta.convert_dict_values(video_info["Video"])
        if(video_info["Audio"] != "0"):
            video_info["Audio"] = VideoMeta.convert_dict_values(video_info["Audio"])

        if(video_info["Audio"] != "0"):
            video_info["Audio"]["AlternateGroup"] = str(video_info["Audio"].get("AlternateGroup", "0"))    
            video_info["Audio"]["ChannelLayout"] = str(video_info["Audio"].get("ChannelLayout", "0"))   
            video_info["Audio"]["Channels"] = str(video_info["Audio"].get("Channels", "0"))    
            video_info["Audio"]["CodecID"] = str(video_info["Audio"].get("CodecID", "0"))  
            video_info["Audio"]["Default"] = str(video_info["Audio"].get("Default", "0"))    
            video_info["Audio"]["ID"] = str(video_info["Audio"].get("ID", "0"))   
            VideoMeta.key_value_validation_INT(video_info, "Audio", "BitRate") 
            VideoMeta.key_value_validation_INT(video_info, "Audio", "FrameCount") 
            VideoMeta.key_value_validation_FLOAT(video_info, "Audio", "Duration") 
        
        if(video_info["General"] != "0"):
            video_info["General"]["CodecID"] = str(video_info["General"].get("CodecID", "0"))    
            VideoMeta.key_value_validation_INT(video_info, "General", "FileSize") 
            VideoMeta.key_value_validation_INT(video_info, "General", "FrameCount") 
            VideoMeta.key_value_validation_FLOAT(video_info, "General", "Duration") 
        
        if(video_info["Video"] != "0"):
            video_info["Video"]["CodecID"] = str(video_info["Video"].get("CodecID", "0"))    
            video_info["Video"]["Format_Level"] = str(video_info["Video"].get("Format_Level", "0"))    
            video_info["Video"]["ID"] = str(video_info["Video"].get("ID", "0"))    
            VideoMeta.key_value_validation_INT(video_info, "Video", "Height") 
            VideoMeta.key_value_validation_INT(video_info, "Video", "Width") 
            VideoMeta.key_value_validation_INT(video_info, "Video", "BitRate") 
            VideoMeta.key_value_validation_INT(video_info, "Video", "FrameCount") 
            VideoMeta.key_value_validation_FLOAT(video_info, "Video", "Duration") 
        error_message_sha256 = ""
        try:
            Height = video_info["Video"]["Height"]
            Width = video_info["Video"]["Width"]
            Duration = video_info["Video"]["Duration"]
            FileSize = video_info["General"]["FileSize"]
            command_sha256 = "echo "+str(Height)+str(Width)+str(Duration)+str(FileSize)+" | sha256sum"

            details_sha256 = subprocess.check_output(command_sha256, shell=True, encoding="utf-8")
            video_info["sha256"] = VideoMeta.shake_128(details_sha256[:-4])
        except:
            error_message_sha256 = "sha256 can't be generated, either Height, Width, Duration or FileSize not available"
            video_info["sha256"] = "0"
        return video_info, error_message_sha256
    
    def is_video_rotated(video_path):
        start = time.time()
        cap = cv2.VideoCapture(video_path)

        vertical_line_count = 0
        horizontal_line_count = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50) 

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0] 
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

                    if abs(angle) < 30:  # Roughly horizontal 
                        horizontal_line_count += 1
                    elif abs(angle - 90) < 30: # Roughly vertical
                        vertical_line_count += 1

            frame_count += 1

        cap.release()

        rotate_flag = vertical_line_count > horizontal_line_count 
        end = time.time()
        print("time taken:", end - start)
        return rotate_flag, frame_count  

import mimetypes
import os, re, pytz
import json
import subprocess, traceback
from helper import getConfigInfo
from rabbitmq import RabbitMQ
from classes.dbconnection import DbConnection
from classes.mediainfo_video import VideoMeta
from classes.utility import Utility
from pathlib import Path
from datetime import datetime
from classes.upload import UploadVideo
from classes.extract_color import ExtractColor
from common import generateRandom, RemoveFile, CurlPostJson, CurlGetJson, sendMediaLogs, calculateAspectRatio, DownloadImageFromUrl

CHECK_VIDEO_DUPLICACY = 'http://192.168.131.170/cs/v1/category/api/video-duplicacy-check'
SERVICE_CATALOGUE_CENTRALIZED = 'http://192.168.8.27:4000/api/sf/catalog/'

class StreamConversionProcess():
    def __init__(self):
        pass
    
    def GetResolution(width, height):
        resolution_map = {
            (1920, 1080): "1080",
            (1280, 720): "720",
            (960, 540): "540",
            (854, 480): "480",
            (640, 360): "360",
            (426, 240): "240",
            (320, 180): "180"
        }

        # If exact match found, return it
        if (width, height) in resolution_map:
            return resolution_map[(width, height)]

        # Otherwise, find the closest height match
        closest_res = min(resolution_map.keys(), key=lambda res: abs(res[1] - height))
        
        return resolution_map[closest_res]

    def GetAvgMotion(file_path):
        
        command = [
            "ffmpeg", "-i", file_path, "-vf", "select='gt(scene,0.2)',metadata=print",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
            motion_scores = [float(line.split("=")[1]) for line in result.stderr.split("\n") if "lavfi.scene_score=" in line]
            avg_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0
            return avg_motion
        except Exception as e:
            print(f" Error calculating motion: {e}")
            return 0
        
    def GetVideoSetting(resolution,v_profile,a_profile, v_level):
        # Define CRF values
        crf_values = {
            "1080": "20",
            "720": "22",
            "540": "22",
            "480": "24",
            "360": "26",
            "240": "28",
            "180": "30"
        }

        process_dict = {
            "1080": "copy_codec",
            "720": "copy_codec",
            "540": "copy_codec",
            "480": "copy_codec",
            "360": "copy_codec",
            "240": "copy_codec",
            "180": "copy_codec"
        }
        process_value = process_dict.get(resolution,"resize_n_reencode")

        # Define Preset values
        if resolution in ["1080", "720"]:
            preset = "medium"
        elif resolution in ["540", "480", "360"]:
            preset = "fast"
        elif resolution == "240":
            preset = "fast"
        elif resolution == "180":
            preset = "fast"
        else:
            preset = "fast"  # Default preset
        
        # Define Audio Settings (bitrate, sample rate)
        audio_settings = {
            "1080": ("128k", "48000"),
            "720": ("128k", "48000"),
            "540": ("128k", "48000"),
            "480": ("96k", "44100"),
            "360": ("64k", "32000"),
            "240": ("48k", "22050"),
            "180": ("48k", "22050"),
        }

        #need profile_v
        video_profile_map = {
            "High": ("64", "640028"),  # H.264 High Profile, Level 4.0
            "Main": ("4d", "4d401f"),  # H.264 Main Profile, Level 3.1
            "Baseline": ("42", "42e01e")  # H.264 Baseline Profile, Level 3.0
        }

        # **Map Audio Profiles to Profile ID**
        audio_profile_map = {
            "LC": "2",  # AAC-LC (Low Complexity)
            "HE-AAC v1": "5", "HE-AAC": "5",  # HE-AAC v1
            "HE-AAC v2": "29",  # HE-AAC v2
            "AAC-LD": "23",  # AAC Low Delay
            "AAC-ELD": "39",  # AAC Enhanced Low Delay
            "Main-AAC": "1",  # AAC Main Profile
            "SSR": "3"  # AAC Scalable Sample Rate
        }

        # Get CRF, defaulting to "22" if resolution is unknown
        crf = crf_values.get(resolution, "22")

        # Get audio settings, defaulting to ("64k", "44100") if resolution is unknown
        audio_bitrate, audio_sample_rate = audio_settings.get(resolution, ("64k", "44100"))
        profile_hex, profile_level_id = video_profile_map.get(v_profile, ("42", "42e01e"))
        audio_profile = audio_profile_map.get(a_profile, "2")  # Default to "LC" (2)
        audio_codec = f",mp4a.40.{audio_profile}"
        # Convert level to hex
        #level_hex = f"{int(v_level):02x}"
        video_codec = f"avc1.{profile_level_id}"  # Constructing codec

        # **Final Codec String**
        codecs = f"{video_codec}{audio_codec}"

        return crf, preset, audio_bitrate, audio_sample_rate, profile_level_id, codecs, process_value

    def GetProcessData(resolution, closest_res, v_codec_name, input_dir, input_file, output_dir):

        resolution = str(resolution)
        closest_res = str(closest_res)
        v_codec_name = str(v_codec_name)
        input_dir = str(input_dir)
        input_file = str(input_file)
        output_dir = str(output_dir)

        print("###################")
        print("input_dir",input_dir)
        print("input_file",input_file)
        print("output_dir",output_dir)
        print("closest_res",closest_res)
        print("v_codec_name",v_codec_name)
        print("resolution",resolution)
        print("###################")

        # Convert to integers for comparison (skip if closest_res > resolution)
        try:
            if int(resolution) > int(closest_res):
                print(f"⚠️ Skipping: resolution ({resolution}) is greater than closest_res ({closest_res})")
                return None, None, None, None, None, None, None
        except ValueError:
            print(f"⚠️ Invalid numeric values: resolution={resolution}, closest_res={closest_res}")
            return None, None, None, None, None, None, None

        process_data = {
            "1080": {    
                "1080": {
                    "output_ts"   : f"{output_dir}/1080p.ts",
                    "output_m3u8" : f"{output_dir}/1080p.m3u8",
                    "input_file"  : f"{input_file}",
                    "video_codec": "avc1.640028",
                    "video_profile": "high",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "copy_codec", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "720": {
                    "output_ts"   : f"{output_dir}/720p.ts",
                    "output_m3u8" : f"{output_dir}/720p.m3u8",
                    "input_file"  : f"{input_file}",
                    "video_codec": "avc1.640028",
                    "video_profile": "high",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "540": {
                    "output_ts"   : f"{output_dir}/540p.ts",
                    "output_m3u8" : f"{output_dir}/540p.m3u8",
                    "input_file"  : f"{output_dir}/720p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "main",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "480": {
                    "output_ts"   : f"{output_dir}/480p.ts",
                    "output_m3u8" : f"{output_dir}/480p.m3u8",
                    "input_file"  : f"{output_dir}/720p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "main",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "360": {
                    "output_ts"   : f"{output_dir}/360p.ts",
                    "output_m3u8" : f"{output_dir}/360p.m3u8",
                    "input_file"  : f"{output_dir}/720p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "240": {
                    "output_ts"   : f"{output_dir}/240p.ts",
                    "output_m3u8" : f"{output_dir}/240p.m3u8",
                    "input_file"  : f"{output_dir}/480p.ts",
                    "video_codec": "avc1.4d401f",
                    "audio_codec": "mp4a.40.5",
                    "video_profile": "baseline",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "180": {
                    "output_ts"   : f"{output_dir}/180p.ts",
                    "output_m3u8" : f"{output_dir}/180p.m3u8",
                    "input_file"  : f"{output_dir}/360p.ts",
                    "video_codec": "avc1.42001e",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                }
            },
            "720": {    
                "720": {
                    "output_ts"   : f"{output_dir}/720p.ts",
                    "output_m3u8" : f"{output_dir}/720p.m3u8",
                    "input_file"  : f"{input_file}",
                    "video_codec": "avc1.640028",
                    "video_profile": "high",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "copy_codec", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "540": {
                    "output_ts"   : f"{output_dir}/540p.ts",
                    "output_m3u8" : f"{output_dir}/540p.m3u8",
                    "input_file"  : f"{output_dir}/720p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "main",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "480": {
                    "output_ts"   : f"{output_dir}/480p.ts",
                    "output_m3u8" : f"{output_dir}/480p.m3u8",
                    "input_file"  : f"{output_dir}/720p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "main",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "360": {
                    "output_ts"   : f"{output_dir}/360p.ts",
                    "output_m3u8" : f"{output_dir}/360p.m3u8",
                    "input_file"  : f"{output_dir}/720p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "240": {
                    "output_ts"   : f"{output_dir}/240p.ts",
                    "output_m3u8" : f"{output_dir}/240p.m3u8",
                    "input_file"  : f"{output_dir}/480p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "180": {
                    "output_ts"   : f"{output_dir}/180p.ts",
                    "output_m3u8" : f"{output_dir}/180p.m3u8",
                    "input_file"  : f"{output_dir}/360p.ts",
                    "video_codec": "avc1.42001e",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                }
            }, 
            "540": {     
                "540": {
                    "output_ts"   : f"{output_dir}/540p.ts",
                    "output_m3u8" : f"{output_dir}/540p.m3u8",
                    "input_file"  : f"{input_file}",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "main",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "copy_codec", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "480": {
                    "output_ts"   : f"{output_dir}/480p.ts",
                    "output_m3u8" : f"{output_dir}/480p.m3u8",
                    "input_file"  : f"{output_dir}/520p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "main",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "360": {
                    "output_ts"   : f"{output_dir}/360p.ts",
                    "output_m3u8" : f"{output_dir}/360p.m3u8",
                    "input_file"  : f"{output_dir}/520p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "240": {
                    "output_ts"   : f"{output_dir}/240p.ts",
                    "output_m3u8" : f"{output_dir}/240p.m3u8",
                    "input_file"  : f"{output_dir}/480p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "180": {
                    "output_ts"   : f"{output_dir}/180p.ts",
                    "output_m3u8" : f"{output_dir}/180p.m3u8",
                    "input_file"  : f"{output_dir}/360p.ts",
                    "video_codec": "avc1.42001e",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                }
            }, 
            "480": {     
                "480": {
                    "output_ts"   : f"{output_dir}/480p.ts",
                    "output_m3u8" : f"{output_dir}/480p.m3u8",
                    "input_file"  : f"{input_file}",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "main",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "copy_codec", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "360": {
                    "output_ts"   : f"{output_dir}/360p.ts",
                    "output_m3u8" : f"{output_dir}/360p.m3u8",
                    "input_file"  : f"{output_dir}/480p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.2",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "240": {
                    "output_ts"   : f"{output_dir}/240p.ts",
                    "output_m3u8" : f"{output_dir}/240p.m3u8",
                    "input_file"  : f"{output_dir}/480p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "180": {
                    "output_ts"   : f"{output_dir}/180p.ts",
                    "output_m3u8" : f"{output_dir}/180p.m3u8",
                    "input_file"  : f"{output_dir}/360p.ts",
                    "video_codec": "avc1.42001e",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                }
            },
            "360": {     
            
                "360": {
                    "output_ts"   : f"{output_dir}/360p.ts",
                    "output_m3u8" : f"{output_dir}/360p.m3u8",
                    "input_file"  : f"{input_file}",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "copy_codec", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "240": {
                    "output_ts"   : f"{output_dir}/240p.ts",
                    "output_m3u8" : f"{output_dir}/240p.m3u8",
                    "input_file"  : f"{output_dir}/360p.ts",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "180": {
                    "output_ts"   : f"{output_dir}/180p.ts",
                    "output_m3u8" : f"{output_dir}/180p.m3u8",
                    "input_file"  : f"{output_dir}/360p.ts",
                    "video_codec": "avc1.42001e",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                }
            },
            "240": {     
                "240": {
                    "output_ts"   : f"{output_dir}/240p.ts",
                    "output_m3u8" : f"{output_dir}/240p.m3u8",
                    "input_file"  : f"{input_file}",
                    "video_codec": "avc1.4d401f",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "copy_codec", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                },
                "180": {
                    "output_ts"   : f"{output_dir}/180p.ts",
                    "output_m3u8" : f"{output_dir}/180p.m3u8",
                    "input_file"  : f"{output_dir}/240p.ts",
                    "video_codec": "avc1.42001e",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "resize_encode", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                }
            },
            "180": {     
                "180": {
                    "output_ts"   : f"{output_dir}/180p.ts",
                    "output_m3u8" : f"{output_dir}/180p.m3u8",
                    "input_file"  : f"{input_file}",
                    "video_codec": "avc1.42001e",
                    "video_profile": "baseline",
                    "audio_codec": "mp4a.40.5",
                    "h264" : "copy_codec", 
                    "hevc" : "p2", 
                    "vp9"  : "p3", 
                    "av1"  : "p4"
                }
            }
        }

        # Debugging: Print keys before access
        print("processing resolution ",resolution)
        print("processing closest_res ",closest_res)

        print("Available resolutions:", list(process_data.keys()))
        if resolution in process_data:
            print("Available closest resolutions:", list(process_data[resolution].keys()))
            if closest_res in process_data[resolution]:
                print("Available video codecs:", list(process_data[resolution][closest_res].keys()))
        

        input_file = os.path.abspath(input_file)

        process     = process_data[closest_res][resolution][v_codec_name]
        output_ts   = process_data[closest_res][resolution]["output_ts"].replace(".ts", "_%03d.ts")
        output_m3u8 = process_data[closest_res][resolution]["output_m3u8"]
        input_file  = process_data[closest_res][resolution]["input_file"]
        video_codec  = process_data[closest_res][resolution]["video_codec"]
        audio_codec  = process_data[closest_res][resolution]["audio_codec"]
        video_profile  = process_data[closest_res][resolution]["video_profile"]

        return process, output_ts, output_m3u8, input_file, video_codec, audio_codec, video_profile

    def IsValid(input_dir, input_file, output_dir, closest_res):

        return (
            closest_res in {"1080", "720", "540", "480", "360", "240", "180"} and  # Check if codec is valid
            all([input_dir, input_file, output_dir])  # Check all other variables are set
        )

    def CalculateGop(fps, avg_motion_score):

        # Define motion factor based on average motion score
        if avg_motion_score < 0.2:  # Low motion (talk shows, static scenes)
            motion_factor = 3.0
        elif avg_motion_score < 0.5:  # Medium motion (regular movies, YouTube)
            motion_factor = 2.0
        else:  # High motion (sports, gaming)
            motion_factor = 1.0

        # Calculate GOP size dynamically
        gop = round((fps * motion_factor) / 2)
        gop_size = fps * 2 if avg_motion_score > 0.2 else fps * 5

        return max(gop, gop_size)

        #return max(gop, fps)  # Ensure GOP is at least equal to FPS
    
    def GetMetaData(file_path):
        try:
            command = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,bit_rate,size,format_name,format_long_name",
                "-show_entries","stream=index,codec_name,codec_type,codec_long_name,width,height,avg_frame_rate,bit_rate,profile,level,color_range,color_space,color_primaries,color_transfer,profile,channels,codec_type,codec_name,bit_rate,sample_rate",
                "-of","json", file_path
            ]
            #"default=noprint_wrappers=1:nokey=1",

            result = subprocess.run(command, capture_output=True, text=True)
            metadata = json.loads(result.stdout)
            return metadata
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            return None
    
    def GenerateAndProcessData(data):
        src_video_id=data["video_id"]
        src_docid=data["docid"]
        src_uniq_id=data["uniq_id"]
        src_video_url=data["video_url"]
        src_working_url=data["working_url"]
        src_src_video=data["src_video"]
        src_src_audio=data["src_audio"]
        src_dest_video=data["dest_video"]
        src_status=data["status"]
        src_tag=data["tag"]
        src_format=data["format"]
        src_duration=data["duration"]
        src_bitrate=data["bitrate"]
        src_width=data["width"]
        src_height=data["height"]
        src_size=data["size"]
        src_has_video=data["has_video"]
        src_has_audio=data["has_audio"]
        src_video_codec=data["video_codec"]
        src_video_profile=data["video_profile"]
        src_video_level=data["video_level"]
        src_video_bitrate=data["video_bitrate"]
        src_avg_frame_rate=data["avg_frame_rate"]
        src_avg_motion_score=data["avg_motion_score"]
        src_frame_count=data["frame_count"]
        src_color_range=data["color_range"]
        src_color_space=data["color_space"]
        src_color_primaries=data["color_primaries"]
        src_color_transfer=data["color_transfer"]
        src_matrix_coefficients=data["matrix_coefficients"]
        src_audio_codec=data["audio_codec"]
        src_audio_profile=data["audio_profile"]
        src_audio_channels=data["audio_channels"]
        src_audio_bitrate=data["audio_bitrate"]
        src_audio_sample_rate=data["audio_sample_rate"]
        src_random_key=data["random_key"]
        src_video_thumb=data["video_thumb"]

        # Prepare HLS Transcoding
        resolution_arr = ["1080", "720", "540", "480", "360", "240", "180"]
        closest_res = StreamConversionProcess.GetResolution(src_width,src_height)
        avg_motion = StreamConversionProcess.GetAvgMotion(f"{src_src_video}")
        
        NFS_in_path = getConfigInfo('NFS_path.video_input')
        # input_dir = f"video_input/{src_docid}/{src_random_key}"
        input_dir = f"{NFS_in_path}/{src_docid}/{src_random_key}"
        #input_file = f"{src_docid}_{src_random_key}.mp4"
        input_file = src_src_video

        NFS_out_path = getConfigInfo('NFS_path.video_output')
        # output_dir = f"video_output/{src_docid}/{src_random_key}"
        output_dir = f"{NFS_out_path}/{src_docid}/{src_random_key}"
        #master_playlist = f"video_output/{src_docid}/{src_random_key}/master.m3u8"
        master_playlist = os.path.join(output_dir, "master.m3u8")

        os.makedirs(os.path.dirname(master_playlist), exist_ok=True)


        with open(master_playlist, "w") as f:
            f.write("#EXTM3U\n")

        for resolution in resolution_arr:

            print(f" ############# >>>>>> Process for {resolution}")

            # target_width = int(resolution)
            # target_height = int(resolution) if src_width > src_height else int(resolution) * src_width // src_height

            # target_width = target_width + 1 if target_width % 2 else target_width
            # target_height = target_height + 1 if target_height % 2 else target_height

            if src_width > src_height:
                target_width = (int(resolution) * src_width) // src_height
                target_height = int(resolution)
            else:
                target_width = int(resolution)
                target_height = (int(resolution) * src_height) // src_width

            # Ensure target dimensions are even
            target_width = (target_width + 1) // 2 * 2
            target_height = (target_height + 1) // 2 * 2

            target_bitrate = int(int(src_bitrate) * target_width * target_height / (src_width * src_height))
            maxrate = int(target_bitrate * 1.5)
            bufsize = int(target_bitrate * 2)
            
            fps = round(float(src_avg_frame_rate.split('/')[0]) / 
                float(src_avg_frame_rate.split('/')[1]) if '/' in str(src_avg_frame_rate) else float(src_avg_frame_rate or 30))

            #gop_size = fps * 2 if avg_motion > 0.2 else fps * 5
            gop_size = StreamConversionProcess.CalculateGop(fps, avg_motion)
            crf, preset, audio_bitrate, audio_sample_rate, profile_level_id, codecs, process_value = StreamConversionProcess.GetVideoSetting(resolution, src_video_profile, src_audio_profile, src_video_level)

            print(target_width,target_height)
            print(f"⚠️ ✅ processing ({resolution}) --- target width ({target_width})  --- target height ({target_height})")
            

            # Convert to integers for comparison (skip if closest_res > resolution)
            try:
                if int(resolution) > int(closest_res):
                    print(f"⚠️ Skipping: resolution ({resolution}) is greater than closest_res ({closest_res})")
                    #return None, None, None, None, None, None, None
                    continue
            except ValueError:
                print(f"⚠️ Invalid numeric values: resolution={resolution}, closest_res={closest_res}")
                return None, None, None, None, None, None, None

            if(src_video_codec and src_video_codec in {"h264", "hevc", "vp9", "av1"}):

                print("video is fragmented")
                process, output_ts, output_m3u8, input_file, video_codec, audio_codec, video_profile = StreamConversionProcess.GetProcessData(resolution, closest_res, src_video_codec, input_dir, input_file, output_dir)

                if StreamConversionProcess.IsValid(input_dir, input_file, output_dir, closest_res):

                    if None in [process, output_ts, output_m3u8, input_file]:
                        print("Skipping due to missing values.")
                        continue
                    else:
                        ffmpeg_params = {
                            "resolution": resolution,
                            "closest_resolution":closest_res,
                            "video_codec":src_video_codec,
                            "target_width":target_width,
                            "target_height":target_height,
                            "target_bitrate":target_bitrate,
                            "maxrate":maxrate,
                            "bufsize":bufsize,
                            "fps":fps,
                            "gop":gop_size,
                            "crf":crf,
                            "preset":preset,
                            "audio_bitrate":audio_bitrate,
                            "audio_sample_rate":audio_sample_rate,
                            "video_profile":src_video_profile,
                            "process":process,
                            "output_ts":output_ts,
                            "output_m3u8":output_m3u8,
                            "input_file":input_file,
                            "master_playlist":master_playlist,
                            "codecs":codecs,
                            "video_codec":video_codec,
                            "audio_codec":audio_codec
                            }
                        
                        print(ffmpeg_params)
                        
                    # Check for missing values
                    if not output_ts or not output_m3u8 or not input_file:
                        raise ValueError("Error: output_ts or output_m3u8 is missing!")
                    
                    # Define FFmpeg command based on process type
                    if process == "copy_codec":
                        print("Processing: Copy Codec (No Re-encoding)")
                        ffmpeg_cmd = [
                            # "/usr/local/bin/ffmpeg", "-i", input_file,
                            "ffmpeg", "-i", input_file,
                            "-c:v", "copy",
                            "-metadata:s:v", f"color_primaries={src_color_primaries}",
                            "-metadata:s:v", f"transfer_characteristics={src_color_transfer}",
                            "-metadata:s:v", f"matrix_coefficients={src_matrix_coefficients}",
                            "-metadata:s:v", f"color_range={src_color_range}",
                            "-c:a", "copy",
                            "-hls_time", "4",
                            "-hls_playlist_type", "vod",
                            "-hls_flags", "append_list+independent_segments+single_file",
                            "-hls_segment_type", "mpegts",
                            "-hls_segment_filename", output_ts,
                            "-hls_list_size", "0",
                            "-f", "hls",
                            output_m3u8
                        ]
            
                    elif process == "resize_encode":
                        print("Processing: Resize & Re-encode")

                        vf_filter = (
                            f"scale=trunc(iw*min({target_width}/iw\\,{target_height}/ih)/4)*4:"
                            f"trunc(ih*min({target_width}/iw\\,{target_height}/ih)/4)*4,"
                            f"colorspace=primaries={src_color_primaries}:trc={src_color_transfer}:space={src_color_space}:format=yuv420p,setsar=1"
                            
                        ).format(target_width, target_height, target_width, target_height)


                        ffmpeg_cmd = [
                            # "/usr/local/bin/ffmpeg", "-i", input_file,
                            "ffmpeg", "-i", input_file,
                            "-vf", vf_filter,
                            "-c:v", "libx264",
                            "-b:v", str(target_bitrate),
                            "-maxrate", str(maxrate),
                            "-bufsize", str(bufsize),
                            "-profile:v", video_profile,
                            "-crf", str(crf),
                            "-preset", preset,
                            "-g", str(gop_size),
                            "-keyint_min", str(gop_size),
                            "-force_key_frames", f"expr:gte(t,n_forced*{gop_size})",
                            "-r", str(fps)
                        ]

                        # Handle audio settings dynamically
                        if audio_bitrate:
                            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate, "-ar", audio_sample_rate, "-ac", "2"])
                        else:
                            ffmpeg_cmd.append("-an")  # No audio

                        # HLS options
                        ffmpeg_cmd.extend([
                            "-hls_time", "4",
                            "-hls_playlist_type", "vod",
                            "-hls_flags", "append_list+independent_segments+single_file",
                            "-hls_segment_type", "mpegts",
                            "-hls_segment_filename", output_ts,
                            "-hls_list_size", "0",
                            "-f", "hls",
                            output_m3u8
                        ])

                    elif process == "re_encode":
                        print("Processing: Re-encode")
                        ffmpeg_cmd = [
                            # "/usr/local/bin/ffmpeg", "-i", input_file,
                            "ffmpeg", "-i", input_file,
                            "-c:v", "libx264",
                            "-b:v", str(target_bitrate),
                            "-maxrate", str(maxrate),
                            "-bufsize", str(bufsize),
                            "-profile:v", video_profile,
                            "-crf", str(crf),
                            "-preset", preset,
                            "-g", str(gop_size),
                            "-keyint_min", str(gop_size),
                            "-force_key_frames", "expr:gte(t,n_forced*2)",
                            "-r", str(fps),
                            "-c:a", "aac", "-b:a", audio_bitrate, "-ar", audio_sample_rate, "-ac", "2",
                            output_m3u8
                        ]

                else:
                    print("process : ",process)
                    print("Unknown process type! Exiting.")
                    return
            
                # Execute FFmpeg command
                try:
                    
                    print("Generated FFmpeg Command:", " ".join(ffmpeg_cmd))
                    
                    result = subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    # if resolution=="720":
                    metadata_lower = StreamConversionProcess.GetMetaData(output_ts)
                    print(json.dumps(metadata_lower,indent=4))
                    #sys.exit(1)

                    with open(master_playlist, "a") as f:
                        f.write(
                            f"#EXT-X-STREAM-INF:BANDWIDTH={target_bitrate},"
                            f"CODECS=\"{video_codec},{audio_codec}\","
                            f"RESOLUTION={target_width}x{target_height}\n"
                            f"{resolution}p.m3u8\n"
                        )
                    
                    db_connection = DbConnection()
                    dbconn = db_connection.db_connect_dev()
                    try:
                        # connection = mysql.connector.connect(**DB_CONFIG)
                        cursor = dbconn.cursor(prepared=True)
                        
                        sql = """
                        INSERT INTO lg_videos_29L_details SET 
                            docid = %s, 
                            random_key = %s,
                            resolution = %s, 
                            width = %s, 
                            height = %s, 
                            bitrate = %s, 
                            maxrate = %s, 
                            buffsize = %s, 
                            fps = %s, 
                            gop = %s, 
                            audio_bitrate = %s, 
                            audio_sample_rate = %s, 
                            preset = %s,
                            crf = %s, 
                            process = %s 
                        ON DUPLICATE KEY UPDATE
                            width = VALUES(width), 
                            height = VALUES(height),
                            bitrate = VALUES(bitrate),
                            maxrate = VALUES(maxrate),
                            buffsize = VALUES(buffsize),
                            fps = VALUES(fps),
                            gop = VALUES(gop),
                            audio_bitrate = VALUES(audio_bitrate),
                            audio_sample_rate = VALUES(audio_sample_rate),
                            preset = VALUES(preset),
                            crf = VALUES(crf),
                            process = VALUES(process);
                        """

                        values = (
                            src_docid,
                            src_random_key,
                            resolution, 
                            target_width, 
                            target_height,
                            target_bitrate,
                            maxrate, 
                            bufsize, 
                            fps, 
                            gop_size, 
                            audio_bitrate,
                            audio_sample_rate, 
                            preset, 
                            crf,
                            process
                        )

                        cursor.execute(sql, values)
                        formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
                        print("Generated SQL Query:", formatted_sql)
                        dbconn.commit()
                    except dbconn.Error as err:
                        print(f"Database Insert Error: {err}")

                    print(f"✅ Master playlist generated: {master_playlist}")
                    #return result.stdout
                
                except subprocess.CalledProcessError as e:
                    print("FFmpeg Execution Failed!")
                    print(e.stderr)
                    return None

                print(f"✅ Master playlist generated: {master_playlist}")
                
            else:
                print("Conditions not met. Check your inputs.")
        
        # Example data to push
        upload_data = {
            "docid": src_docid,
            "random_key": src_random_key,
            "src_dir": input_dir,
            "dest_dir" : output_dir,
            "thumbnail": src_video_thumb,
            "resolution":resolution
        }
        
        # push_to_rabbitmq(data)
        rabbitMq = RabbitMQ()
        queueData = dict()
        queueHost = getConfigInfo('rabbitmq_server1')
        queueData["credentials"] = queueHost
        queueData["message"] = upload_data
        queueData["queue_name"] = "upload_m3u8_video"
        
        # print(queueData)
        response = rabbitMq.postQueue(queueData)
        print(response)
        if response:
            print("#################### PUSHED TO UPLOAD #######################")
        else:
            print("####ERROR IN CONNECTION#####")
        return response

    def GenerateM3u8AndProcessData(data, servers_credentials):
        src_docid=data["docid"]
        src_src_video=data["src_video"]
        src_bitrate=data["bitrate"]
        src_width=data["width"]
        src_height=data["height"]
        src_video_codec=data["video_codec"]
        src_video_profile=data["video_profile"]
        src_video_level=data["video_level"]
        src_avg_frame_rate=data["avg_frame_rate"]
        src_color_range=data["color_range"]
        src_color_space=data["color_space"]
        src_color_primaries=data["color_primaries"]
        src_color_transfer=data["color_transfer"]
        src_matrix_coefficients=data["matrix_coefficients"]
        src_audio_profile=data["audio_profile"]
        src_random_key=data["reference_id"]
        src_video_thumb=data["video_thumb"]

        # Prepare HLS Transcoding
        # resolution_arr = ["1080", "720", "540", "480", "360", "240", "180"]
        if int(src_bitrate) >= 2500000:
            resolution_arr = ["720", "540", "360", "240"]
            bitrate_dict = {
                "720": 3300000,  # 3.3 Mbps
                "540": 1400000,  # 1.4 Mbps
                "480": 1200000,  # 1.2 Mbps
                "360": 700000,  # 700 kbps
                "240": 240000,  # 240 kbps
            }
        elif int(src_bitrate) >= 1500000 and int(src_bitrate) < 2500000:
            resolution_arr = ["720", "540", "360", "240"]
            bitrate_dict = {
                "720": 3300000,  # 3.3 Mbps
                "540": 1200000,  # 1.2 Mbps
                "480": 1200000,  # 1.2 Mbps
                "360": 600000,  # 600 kbps
                "240": 240000,  # 240 kbps
            }
        elif int(src_bitrate) >= 900000 and int(src_bitrate) < 1500000:
            resolution_arr = ["720", "360", "240"]
            bitrate_dict = {
                "720": 3300000,  # 3.3 Mbps
                "540": 1200000,  # 1.2 Mbps
                "480": 1200000,  # 1.2 Mbps
                "360": 600000,  # 600 kbps
                "240": 240000,  # 240 kbps
            }
        elif int(src_bitrate) >= 500000 and int(src_bitrate) < 900000:
            resolution_arr = ["540", "360", "240"]
            bitrate_dict = {
                "720": 3300000,  # 3.3 Mbps
                "540": 1200000,  # 1.2 Mbps
                "480": 1200000,  # 1.2 Mbps
                "360": 400000,  # 400 kbps
                "240": 180000,  # 180 kbps
            }
        elif int(src_bitrate) < 500000:
            resolution_arr = ["360", "240"]
            bitrate_dict = {
                "720": 3300000,  # 3.3 Mbps
                "540": 1200000,  # 1.2 Mbps
                "480": 1200000,  # 1.2 Mbps
                "360": 500000,  # 500 kbps
                "240": 200000,  # 200 kbps
            }
        else:
            print(f"NO CONDITION MATCHED {src_bitrate}")

        closest_res = StreamConversionProcess.GetResolution(src_width,src_height)
        avg_motion = StreamConversionProcess.GetAvgMotion(f"{src_src_video}")
        
        NFS_in_path = getConfigInfo('NFS_path.video_input')
        if src_docid != "":
            input_dir = f"{NFS_in_path}/{src_docid}/{src_random_key}"
        else:
            input_dir = f"{NFS_in_path}/{src_random_key}"
        
        input_file = src_src_video

        NFS_out_path = getConfigInfo('NFS_path.video_output')
        
        if src_docid != "":
            # output_dir = f"{NFS_out_path}/{src_docid}/{src_random_key}"
            output_dir = f"{NFS_out_path}/{str.lower(src_docid)}_{src_random_key}"
            master_filename = str.lower(src_docid) + "_" + src_random_key + ".m3u8"
            master_playlist = os.path.join(NFS_out_path, master_filename)
        else:
            randomUniqueIdent = str.lower(generateRandom(10))
            output_dir = f"{NFS_out_path}/{randomUniqueIdent}_{src_random_key}"
            master_filename = randomUniqueIdent + "_" + src_random_key + ".m3u8"
            master_playlist = os.path.join(NFS_out_path, master_filename)
        
        # master_playlist = os.path.join(output_dir, "master.m3u8")
        parent_path = output_dir.split("/")[-1]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(master_playlist), exist_ok=True)

        with open(master_playlist, "w") as f:
            f.write("#EXTM3U\n")

        for resolution in resolution_arr:
            print(f" ############# >>>>>> Process for {resolution}")

            if src_width > src_height:
                target_width = (int(resolution) * src_width) // src_height
                target_height = int(resolution)
            else:
                target_width = int(resolution)
                target_height = (int(resolution) * src_height) // src_width

            # Ensure target dimensions are even
            target_width = (target_width + 1) // 2 * 2
            target_height = (target_height + 1) // 2 * 2

            # calcualted_bitrate = int(int(src_bitrate) * target_width * target_height / (src_width * src_height))
            if int(src_bitrate) > bitrate_dict[resolution]:
                target_bitrate = bitrate_dict[resolution]
            else:
                target_bitrate = int(src_bitrate)
            # target_bitrate = bitrate_dict.get(resolution, 1000000)  # Default to 1 Mbps if not found
            print(f"################# Target bitrate for {resolution}p: {target_bitrate} bps #################")
            maxrate = int(target_bitrate * 1.5)
            bufsize = int(target_bitrate * 2)
            
            # fps = round(float(src_avg_frame_rate.split('/')[0]) / 
            #     float(src_avg_frame_rate.split('/')[1]) if '/' in str(src_avg_frame_rate) else float(src_avg_frame_rate or 30))
            fps = 30

            #gop_size = fps * 2 if avg_motion > 0.2 else fps * 5
            # gop_size = StreamConversionProcess.CalculateGop(fps, avg_motion)
            gop_size = 120
            crf, preset, audio_bitrate, audio_sample_rate, profile_level_id, codecs, process_value = StreamConversionProcess.GetVideoSetting(resolution, src_video_profile, src_audio_profile, src_video_level)

            print(target_width,target_height)
            print(f"⚠️ ✅ processing ({resolution}) --- target width ({target_width})  --- target height ({target_height})")
            
            # Convert to integers for comparison (skip if closest_res > resolution)
            try:
                if int(resolution) > int(closest_res):
                    print(f"⚠️ Skipping: resolution ({resolution}) is greater than closest_res ({closest_res})")
                    #return None, None, None, None, None, None, None
                    continue
            except ValueError:
                print(f"⚠️ Invalid numeric values: resolution={resolution}, closest_res={closest_res}")
                return None, None, None, None, None, None, None

            if(src_video_codec and src_video_codec in {"h264", "hevc", "vp9", "av1"}):

                print("video is fragmented")
                process, output_ts, output_m3u8, input_file_1, video_codec, audio_codec, video_profile = StreamConversionProcess.GetProcessData(resolution, closest_res, src_video_codec, input_dir, input_file, output_dir)

                if StreamConversionProcess.IsValid(input_dir, input_file, output_dir, closest_res):
                    print("Inside validation")
                    print("### Process ",process)
                    # exit(1)
                    if None in [process, output_ts, output_m3u8, input_file]:
                        print("Skipping due to missing values.")
                        continue
                    else:
                        ffmpeg_params = {
                            "resolution": resolution,
                            "closest_resolution":closest_res,
                            "video_codec":src_video_codec,
                            "target_width":target_width,
                            "target_height":target_height,
                            "target_bitrate":target_bitrate,
                            "maxrate":maxrate,
                            "bufsize":bufsize,
                            "fps":fps,
                            "gop":gop_size,
                            "crf":crf,
                            "preset":preset,
                            "audio_bitrate":audio_bitrate,
                            "audio_sample_rate":audio_sample_rate,
                            "video_profile":src_video_profile,
                            "process":process,
                            "output_ts":output_ts,
                            "output_m3u8":output_m3u8,
                            "input_file":input_file,
                            "master_playlist":master_playlist,
                            "codecs":codecs,
                            "video_codec":video_codec,
                            "audio_codec":audio_codec
                        }
                        
                        print(ffmpeg_params)
                        
                    # Check for missing values
                    if not output_ts or not output_m3u8 or not input_file:
                        raise ValueError("Error: output_ts or output_m3u8 is missing!")
                    
                    # Define FFmpeg command based on process type
                    if process == "copy_codec":
                        print("Processing: Copy Codec (No Re-encoding)")
                        ffmpeg_cmd = [
                            # "/usr/local/bin/ffmpeg", "-i", input_file,
                            "ffmpeg", "-i", input_file,
                            "-c:v", "copy",
                            "-metadata:s:v", f"color_primaries={src_color_primaries}",
                            "-metadata:s:v", f"transfer_characteristics={src_color_transfer}",
                            "-metadata:s:v", f"matrix_coefficients={src_matrix_coefficients}",
                            "-metadata:s:v", f"color_range={src_color_range}",
                            "-c:a", "copy",
                            "-hls_time", "4",
                            "-hls_playlist_type", "vod",
                            # "-hls_flags", "append_list+independent_segments+single_file",
                            "-hls_segment_type", "mpegts",
                            "-hls_segment_filename", output_ts,
                            "-hls_list_size", "0",
                            "-f", "hls",
                            output_m3u8
                        ]
            
                    elif process == "resize_encode":
                        print("Processing: Resize & Re-encode")
                        
                        if src_color_primaries == "reserved" or src_color_primaries == "unspecified":
                            src_color_primaries = "bt709"

                        if src_color_transfer == "reserved" or src_color_transfer == "unspecified":
                            src_color_transfer = "bt709"

                        vf_filter = (
                            f"scale=trunc(iw*min({target_width}/iw\\,{target_height}/ih)/4)*4:"
                            f"trunc(ih*min({target_width}/iw\\,{target_height}/ih)/4)*4,"
                            # f"colorspace=primaries={src_color_primaries}:trc={src_color_transfer}:space={src_color_space}:format=yuv420p,setsar=1"
                            f"colorspace=iall={src_color_primaries}:all={src_color_transfer}:format=yuv420p,setsar=1"
                            
                        ).format(target_width, target_height, target_width, target_height)


                        ffmpeg_cmd = [
                            # "/usr/local/bin/ffmpeg", "-i", input_file,
                            "ffmpeg", "-i", input_file,
                            "-vf", vf_filter,
                            "-c:v", "libx264",
                            "-b:v", str(target_bitrate),
                            "-maxrate", str(maxrate),
                            "-bufsize", str(bufsize),
                            "-profile:v", video_profile,
                            "-crf", str(crf),
                            "-preset", preset,
                            "-g", str(gop_size),
                            "-keyint_min", str(gop_size),
                            "-force_key_frames", f"expr:gte(t,n_forced*{gop_size})",
                            "-r", str(fps)
                        ]

                        # Handle audio settings dynamically
                        if audio_bitrate:
                            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate, "-ar", audio_sample_rate, "-ac", "2"])
                        else:
                            ffmpeg_cmd.append("-an")  # No audio

                        # HLS options
                        ffmpeg_cmd.extend([
                            "-hls_time", "4",
                            "-hls_playlist_type", "vod",
                            # "-hls_flags", "append_list+independent_segments+single_file",
                            "-hls_segment_type", "mpegts",
                            "-hls_segment_filename", output_ts,
                            "-hls_list_size", "0",
                            "-f", "hls",
                            output_m3u8
                        ])

                    elif process == "re_encode":
                        print("Processing: Re-encode")
                        ffmpeg_cmd = [
                            # "/usr/local/bin/ffmpeg", "-i", input_file,
                            "ffmpeg", "-i", input_file,
                            "-c:v", "libx264",
                            "-b:v", str(target_bitrate),
                            "-maxrate", str(maxrate),
                            "-bufsize", str(bufsize),
                            "-profile:v", video_profile,
                            "-crf", str(crf),
                            "-preset", preset,
                            "-g", str(gop_size),
                            "-keyint_min", str(gop_size),
                            "-force_key_frames", "expr:gte(t,n_forced*2)",
                            "-r", str(fps),
                            "-c:a", "aac", "-b:a", audio_bitrate, "-ar", audio_sample_rate, "-ac", "2",
                            output_m3u8
                        ]

                else:
                    print("process : ",process)
                    print("Unknown process type! Exiting.")
                    return
            
                # Execute FFmpeg command
                try:
                    print("Generated FFmpeg Command:", " ".join(ffmpeg_cmd))
                    
                    result = subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    # if resolution=="720":
                    metadata_lower = StreamConversionProcess.GetMetaData(output_ts)
                    print(json.dumps(metadata_lower,indent=4))
                    #sys.exit(1)

                    with open(master_playlist, "a") as f:
                        f.write(
                            f"#EXT-X-STREAM-INF:BANDWIDTH={target_bitrate},"
                            f"CODECS=\"{video_codec},{audio_codec}\","
                            f"RESOLUTION={target_width}x{target_height}\n"
                            f"{parent_path}/{resolution}p.m3u8\n"
                        )
                    
                    db_connection = DbConnection()
                    dbconn = db_connection.db_connect_live()
                    try:
                        cursor = dbconn.cursor(prepared=True)
                        
                        total_ts_file_size = sum(
                                os.path.getsize(os.path.join(output_dir, f))
                                for f in os.listdir(output_dir)
                                if f.endswith(".ts") and resolution in f
                            )
                        print(f"Total size of resolution {resolution} having .ts files : {total_ts_file_size} bytes")
                        
                        sql = """
                        INSERT INTO tbl_video_process_details SET 
                            docid = %s, 
                            ref_id = %s,
                            resolution = %s, 
                            width = %s, 
                            height = %s, 
                            bitrate = %s, 
                            maxrate = %s, 
                            buffsize = %s, 
                            fps = %s, 
                            gop = %s, 
                            audio_bitrate = %s, 
                            audio_sample_rate = %s, 
                            preset = %s,
                            crf = %s, 
                            process = %s,
                            input_dir= %s,
                            input_file = %s, 
                            size = %s
                        ON DUPLICATE KEY UPDATE
                            width = VALUES(width), 
                            height = VALUES(height),
                            bitrate = VALUES(bitrate),
                            maxrate = VALUES(maxrate),
                            buffsize = VALUES(buffsize),
                            fps = VALUES(fps),
                            gop = VALUES(gop),
                            audio_bitrate = VALUES(audio_bitrate),
                            audio_sample_rate = VALUES(audio_sample_rate),
                            preset = VALUES(preset),
                            crf = VALUES(crf),
                            process = VALUES(process),
                            input_file = VALUES(input_file),
                            size = VALUES(size);
                        """

                        values = (
                            src_docid,
                            src_random_key,
                            resolution, 
                            target_width, 
                            target_height,
                            target_bitrate,
                            maxrate, 
                            bufsize, 
                            fps, 
                            gop_size, 
                            audio_bitrate,
                            audio_sample_rate, 
                            preset, 
                            crf,
                            process,
                            input_dir,
                            input_file,
                            total_ts_file_size
                        )

                        cursor.execute(sql, values)
                        formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
                        print("Generated SQL Query:", formatted_sql)
                        dbconn.commit()
                    except Exception as err:
                        print(f"Database Insert Error: {err}")

                    print(f"✅ Master playlist generated: {master_playlist}")
                    #return result.stdout
                
                except subprocess.CalledProcessError as e:
                    print("FFmpeg Execution Failed!")
                    print(e.stderr)
                    upd_res = StreamConversionProcess.UpdateStatusLog(src_random_key, "FAILED")
                    return None
                    # exit(1)                
            else:
                print("Conditions not met. Check your inputs.")
        
        upload_data = dict()
        upload_data = data
        upload_data["docid"] = src_docid
        upload_data["random_key"] = src_random_key
        upload_data["src_dir"] = input_dir
        upload_data["dest_dir"] = output_dir
        upload_data["thumbnail"] = src_video_thumb
        upload_data["resolution"] = resolution
        
        # push_to_rabbitmq(data)
        rabbitMq = RabbitMQ()
        queueData = dict()
        # queueHost = getConfigInfo('rabbitmq_server1')
        # queueHost["ho"]
        queueData["credentials"] = servers_credentials
        queueData["message"] = upload_data
        queueData["queue_name"] = "video_upload"
        
        # print(queueData)
        response = rabbitMq.postQueue(queueData)
        print(response)
        if response:
            print("#################### PUSHED TO UPLOAD #######################")
        else:
            print("####ERROR IN CONNECTION#####")
        return response

    def UpdateCallbackData(self, msg):
        reference_id = msg["ref_id"] if "ref_id" in msg and msg["ref_id"] != "" else ""
        print("Reference Id :",reference_id)
        status, jsonObj = StreamConversionProcess.GetVideoLogData(reference_id)
        logData = json.loads(jsonObj)
        if status and reference_id != "":
            upload_module = logData[0]['module']
            print(f"#### Upload Source :{upload_module} #######")
            m3u8_url    = logData[0]["dest_video_url"]
            thumb_url   = logData[0]["thumb_url"]
            random_key  = logData[0]["random_key"]
            video_url   = logData[0]["video_url"]
            file_size   = logData[0]["size"]
            duration    = logData[0]["duration"]
            width       = logData[0]["width"]
            height      = logData[0]["height"]
            src_video   = logData[0]["video_src"]
                
            aspect_ratio = calculateAspectRatio(width,height)
            postData  = json.loads(logData[0]["post_data"])
            print(f"POSTDATA {postData}")
            delete_flag = int(postData["delete_flag"]) if "delete_flag" in postData else 0
            post_video_url = postData["video_url"] if "video_url" in postData else ""
            if delete_flag == 1 and post_video_url!="":
                print("Delete video url")
                self.DeleteS3Data(delete_flag, post_video_url)
            
            contract_video_mod = ["catalogue", "mcatalogue", "editlisting", "image_convert"]
            if upload_module in contract_video_mod:
                # print(postData)
                utility = Utility()
                bitflag_cnt  = 0
                docid        = postData["docid"] if "docid" in postData else ""
                random_key   = postData["random_key"] if "random_key" in postData else ""
                source       = postData["source"] if "source" in postData else "jd_backend"
                company_name = postData["company_name"] if "company_name" in postData else ""
                city         = postData["city"] if "city" in postData else ""
                paid_status  = int(postData["paid_status"]) if "paid_status" in postData else 0
                platform     = postData["platform"] if "platform" in postData else ""
                
                upload_by    = postData["upload_by"] if "upload_by" in postData else "backend_video_process"
                module_type  = int(postData["module_type"]) if "module_type" in postData else 3
                video_source = postData["video_source"] if "video_source" in postData else "1"
                video_tag    = int(postData["video_tag"]) if "video_tag" in postData else 1
                approved     = int(postData["approved"]) if "approved" in postData else 0
                # video_tag    = int(postData["video_tag"]) if "video_tag" in postData else 1
                service_id   = int(postData["service_id"]) if "service_id" in postData else ""
                collection   = postData["collection"] if "collection" in postData else ""
                video_src    = postData["video_src"] if "video_src" in postData else ""
                video_link   = postData["video_link"] if "video_link" in postData else ""
                ip_address   = postData["ip_address"] if "ip_address" in postData else ""
                catalogue_id = postData["catalogue_id"] if "catalogue_id" in postData else 0
                reprocess    = int(postData["reprocess"]) if "reprocess" in postData else 0
                video_id     = int(postData["video_id"]) if "video_id" in postData else 0
                modified_by  = postData["modified_by"] if "modified_by" in postData else ""
                ref_id       = msg["ref_id"]
                id           = postData["id"] if "id" in postData else ""
                entity_id    = postData["entity_id"] if "entity_id" in postData else ""
                
                catalog_source_content = utility.getContractCityCircle(docid, city)
                
                if module_type == 3:
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
                
                if video_src != "":
                    if video_src == "user_story":
                        video_tag = 8
                    elif video_src == "instagram_reels":
                        video_tag = 11
                    elif video_src == "short_videos":
                        video_tag = 14
                else:
                    if video_link != "":
                        if video_link == 1:
                            video_tag = 5
                    else:
                        video_tag = 1

                if docid != "":
                    # get_video_mediainfo_json
                    media_info, err_msg = VideoMeta.get_video_mediainfo_json(src_video)
                    hash_value = media_info["sha256"] if "sha256" in media_info and err_msg == "" else ""
                    print("\n\nhash_value :=", hash_value)
                    #duplicate video check
                    dup_res = self.duplicate_video_check(docid, hash_value)
                    print(f"Duplicate Video Check Result: {dup_res}")
                    if dup_res:
                        print("Duplicate video found, skipping further processing.")
                        # return False
                        additional_duplicate_qry = ",delete_flag=1,deleted_by='Exact Duplicate Video- Auto_Reject', moderator_comment = 17"
                    else:
                        additional_duplicate_qry = ""
                    
                    if video_id == 0:
                        vidStatus, foundVideoId = self.checkRandomKeyExist(postData)
                        if vidStatus:
                            video_id = foundVideoId
                    
                    if catalogue_id == 0:
                        found_cid = utility.FetchCatalogueId(docid, company_name)
                        catalogue_id = found_cid if found_cid is not None else 0
                        
                    if upload_module == "image_convert":
                        additional_duplicate_qry = ""
                    
                    if reprocess == 1:
                        remarks = "reprocessed"
                    else:
                        remarks = "updated"
                    
                    if upload_by == "utube_backend":
                        if id != "" and entity_id != "":
                            upd_stat = self.updateYoutubeChannelMaster(id, entity_id, random_key)
                            print(upd_stat)
                    
                    company_res = utility.GetCompanyDetails(docid)
                    # flag = company_res.get('error')

                    if company_res != None:
                        company_details = company_res
                    else:
                        print(f"Error fetching company details for docid {docid}")
                        company_details = {}
                    
                    print(f"Company Details: {company_details}")
                    
                    # Check Restricted category
                    restricted_qry = ""
                    if module_type != 3 and upload_module != "image_convert" and reprocess == 0:
                        restricted_arr = {}
                        restricted_arr["new_catidlineage"] = company_details["new_catidlineage"] if "new_catidlineage" in company_details else ""
                        restricted_arr["docid"] = docid
                        restricted_arr["rand"] = random_key
                        restricted_arr["content"] = "video"
                        restricted_arr["video_tag"] = video_tag
                        
                        restrict_flag = utility.check_restricted_category(restricted_arr)
                        print(f"Restricted Category Check Flag: {restrict_flag}")
                        if restrict_flag == 1:
                            approved = 0
                            restricted_qry = ", modified_by='restriction_cat', modified_date=NOW()"
                     
                    # service catalogue monogdb
                    if module_type == 27 and reprocess == 0:
                        vdo_add = {
                            "id": random_key,
                            "url": m3u8_url,
                            "timg": thumb_url,
                            "asp": aspect_ratio,
                            "dur": duration,
                            "status": 0
                        }
                        vdo_data_add = {
                            "docid": docid,
                            "p_id": service_id,
                            "up_by": upload_by,
                            "ctlg_type": "serv",
                            "media_type": "vdo",
                            "add": vdo_add,
                            "key": "id",
                            "collection": collection
                        }
                        vdo_data_add_res = CurlPostJson(
                            SERVICE_CATALOGUE_CENTRALIZED + 'update_media', vdo_data_add
                        )
                        print(f"Service Catalogue Response: {vdo_data_add_res}")
                    
                    db_connection = DbConnection()
                    dbconn = db_connection.db_connect_live()
                    try:
                        cursor = dbconn.cursor(prepared=True)
                        if reprocess == 0:
                            approved1 = 0
                        else:
                            approved1 = approved
                        if video_id != 0:
                            # Update existing record
                            update_qry = """
                                UPDATE tbl_video_details SET
                                    ref_id = %s, approved = %s, video_url = %s, video_url_image = %s, aspect_ratio = %s, bit_flag = %s, video_tag = %s, upload_by = %s, cityname = %s, company_name = %s,
                                    module_type = %s, catalog_source_content = %s,
                                    file_size=%s, width=%s, height=%s, duration=%s, hash=%s, modified_date = NOW(),
                                    modified_by = %s, old_video_url = %s """  + additional_duplicate_qry + restricted_qry + """
                                WHERE docid = %s AND video_id = %s
                            """
                            
                            values = (
                                ref_id, approved1, m3u8_url, thumb_url, aspect_ratio, 
                                bitflag_cnt, video_tag, upload_by, city, company_name, 
                                module_type, catalog_source_content, 
                                file_size, width, height, duration, hash_value, 
                                modified_by, remarks,
                                docid, video_id
                            )
                            
                            cursor.execute(update_qry, values)
                            formatted_sql = update_qry % tuple(map(repr, values)) # Converts to a string
                            print("Generated Update SQL Query:", formatted_sql)
                            dbconn.commit()
                            cursor.close()
                            # Remove mp4 file from local
                            RemoveFile(src_video)
                            
                            # Check Blocked Contract status
                            self.checkBlockedContractStatus(docid, company_details)
                            
                            # Pass in fraud video check
                            if reprocess == 0:
                                que_push = self.PushToFraudVideoDetect(video_id, docid, random_key, upload_by, video_url, approved)
                            return True
                        else:
                            insert_qry = """
                                INSERT INTO tbl_video_details SET 
                                    docid = %s,
                                    cityname = %s, 
                                    company_name = %s, 
                                    create_date = NOW(),
                                    approved = %s, 
                                    upload_by = %s, 
                                    ip_address = %s, 
                                    module_type = %s, 
                                    paid_status = %s, 
                                    process_flag = '1',
                                    random_key = %s,
                                    ref_id = %s,
                                    video_tag = %s,
                                    catalog_source_content = %s,
                                    video_url = %s,
                                    video_url_image = %s,
                                    aspect_ratio = %s,
                                    bit_flag = %s,
                                    file_size=%s,
                                    width=%s,
                                    height=%s,
                                    catalogue_id=%s,
                                    duration=%s,
                                    hash=%s """ + additional_duplicate_qry + restricted_qry + """
                                """
                        
                            values = (
                                docid,
                                city,
                                company_name, 
                                approved1, 
                                upload_by,
                                ip_address,
                                module_type, 
                                paid_status, 
                                random_key, 
                                ref_id,
                                video_tag, 
                                catalog_source_content,
                                m3u8_url, 
                                thumb_url, 
                                aspect_ratio,
                                bitflag_cnt,
                                file_size,
                                width,
                                height,
                                catalogue_id,
                                duration,
                                hash_value
                            )
                            cursor.execute(insert_qry, values)
                            formatted_sql = insert_qry % tuple(map(repr, values))  # Converts to a string
                            print("Generated Insert SQL Query:", formatted_sql)
                            dbconn.commit()
                            cursor.close()
                            # Remove mp4 file from local
                            # RemoveFile(src_video)
                            # Check Blocked Contract status
                            self.checkBlockedContractStatus(docid, company_details)
                            # Pass in fraud video check
                            if reprocess == 0:
                                que_push = self.PushToFraudVideoDetect(video_id, docid, random_key, upload_by, video_url, approved)
                            return True
                    except dbconn.Error as e:
                        print(f"Error found in insert {e}")
                        traceback.print_exc()
                        return False
                else:
                    print("No action now")
                    return True
            elif upload_module == "gojd":
                # callback API from gojd team
                post_params = dict()
                post_params["ref_id"]   = random_key
                post_params["m3u8"]     = m3u8_url
                post_params["thumb"]    = thumb_url
                post_params["duration"] = str(duration)
                post_params["width"]    = str(width)
                post_params["height"]   = str(height)
                post_params["aspect_ratio"] = aspect_ratio
                
                callback_req = dict()
                callback_req["img_data"]   = post_params
                callback_req["post_data"]  = postData
                
                callback_data = dict()
                callback_data["data"] = callback_req
                
                # API call curl request hit below
                RemoveFile(src_video)
                file_src = postData["file_src"] if "file_src" in postData else ""
                pid_id = postData["pid_id"] if "pid_id" in postData else ""
                content_type = postData["content_type"] if "content_type" in postData else ""
                
                if file_src == "gojd_content_process":
                    # print("Update in mysql 13.90")
                    db_connection = DbConnection()
                    # extract_col = ExtractColor
                    dest_downloaded_path = "/var/log/images/common_upload/tmp/"+generateRandom(20) + ".jpg"
                    downloaded_img = DownloadImageFromUrl(thumb_url, dest_downloaded_path)
                    color_hex_val = ""
                    if downloaded_img != None:
                        color_output, ab, xy = ExtractColor.extractColor(downloaded_img)
                        print(f"####### Color Output : {color_output} && {ab} && {xy} #######")
                        print(f"####### Color Type : {type(color_output)} && {type(ab)} && {type(xy)} #######")
                        RemoveFile(downloaded_img)
                        if ab == 'True':
                            color_hex_val = ""
                        else:
                            color_hex_val = color_output[0].get("hex")
                            perce_max = color_output[0].get("percentage")
                            print(f"Hex value of max percentage {perce_max}  have color : {color_hex_val}")
                    dbconn = db_connection.db_connect_dev()
                    try:
                        cursor = dbconn.cursor()
                        
                        sql = """
                        UPDATE meta_ad_keywords_normalize SET 
                            stream_video_url = %s, image_url = %s, bg_color = %s, asp = %s, process_flag = 1
                        WHERE pid_id = %s AND content_type = %s
                        """
                        
                        values = (
                            m3u8_url, thumb_url, color_hex_val, aspect_ratio, pid_id, content_type
                        )

                        cursor.execute(sql, values)
                        formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
                        print("Generated SQL Query:", formatted_sql)
                        dbconn.commit()

                        cursor.close()
                        dbconn.close()
                        print(f"Image Url Updated for {pid_id}")
                        return True
                    except Exception as err:
                        print(f"Database Insert Error: {err}")
                        return False
                else:
                    go_jd_url = "http://192.168.24.125:5036/ratings/api/v1/gojd/callback"
                    curlResponse = CurlPostJson(go_jd_url, callback_data)
                    curl_output = json.loads(json.dumps(curlResponse))
                    if(curl_output['errorCode'] == 0):
                        print("Success as 0 errorCode")
                        return True
                    else:
                        return False
            elif upload_module == "category":
                print(f"Category upload_module : {upload_module}")
                catid        = postData["docid"] if "docid" in postData else ""
                random_key   = postData["random_key"] if "random_key" in postData else ""
                source       = postData["source"] if "source" in postData else "jd_backend"
                catname      = postData["catname"] if "catname" in postData else ""
                feed_url     = postData["feed_url"] if "feed_url" in postData else ""
                product_position = int(postData["product_position"]) if "product_position" in postData else 0
                image_label  = postData["image_label"] if "image_label" in postData else ""
                image_description = postData["image_description"] if "image_description" in postData else ""
                
                upload_by    = postData["upload_by"] if "upload_by" in postData else "backend_video_process"
                img_scope    = int(postData["img_scope"]) if "img_scope" in postData else 1
                priority_flag= int(postData["priority_flag"]) if "priority_flag" in postData else 0
                video_tag    = int(postData["video_tag"]) if "video_tag" in postData else 1
                reprocess    = int(postData["reprocess"]) if "reprocess" in postData else 0
                video_id     = int(postData["video_id"]) if "video_id" in postData else 0
                modified_by  = postData["modified_by"] if "modified_by" in postData else "backend_video_reprocess"
                ref_id       = msg["ref_id"]
                
                # fetch max id for category in new insertion
                max_id = self.FetchMaxImgId(catid)
                img_id = max_id + 1 if max_id is not None else 1
                
                if reprocess == 1:
                    process_flag = 9
                else:
                    process_flag = 1
                
                db_connection = DbConnection()
                dbconn = db_connection.db_connect_live()
                try:
                    cursor = dbconn.cursor(prepared=True)
                    if video_id == 0:
                        #check random key exist in tbl_category_video_details
                        vidStatus, foundVideoId = self.checkRandomKeyCategoryVideoExist(postData)
                        if vidStatus:
                            video_id = foundVideoId
                            
                    if video_id != 0:
                        # Update existing record
                        update_qry = """
                            UPDATE tbl_category_video_details SET
                                ref_id = %s, product_url = %s, product_image_url = %s, aspect_ratio = %s,
                                file_size=%s, duration=%s, updated_date = NOW(), updated_by = %s, process_flag = %s
                            WHERE national_catid = %s AND id = %s
                        """
                        
                        values = (
                            ref_id, m3u8_url, thumb_url, aspect_ratio,
                            file_size, duration, modified_by, process_flag,
                            catid, video_id
                        )
                        
                        cursor.execute(update_qry, values)
                        formatted_sql = update_qry % tuple(map(repr, values)) # Converts to a string
                        print("Generated Update SQL Query:", formatted_sql)
                        dbconn.commit()
                        cursor.close()
                        # Remove mp4 file from local
                        RemoveFile(src_video)
                        
                        return True
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
                            video_tag, ref_id, random_key,
                            m3u8_url, thumb_url, aspect_ratio,
                            duration, file_size
                        )
                        cursor.execute(insert_qry, values)
                        formatted_sql = insert_qry % tuple(map(repr, values))  # Converts to a string
                        print("Generated Insert SQL Query:", formatted_sql)
                        dbconn.commit()
                        cursor.close()
                        # Remove mp4 file from local
                        RemoveFile(src_video)
                        return True
                except Exception as e:
                    print(f"Error found in insert {e}")
                    traceback.print_exc()
                    return False
            else:
                print(f"Something else upload_module {upload_module}")
                # Remove mp4 file from local
                # RemoveFile(src_video)
                return True
    
    def GetVideoLogData(ref_id):
        db_connection = DbConnection()
        try:
            if ref_id != "":
                dbconn = db_connection.db_connect_live()
                dbcursor = dbconn.cursor(prepared=True)
                query = "SELECT post_data,video_url,video_src,dest_video_url,thumb_url,duration,width,height,size,module,random_key FROM tbl_video_process_log WHERE ref_id = %s LIMIT 1"
                dbcursor.execute(query, (ref_id,))
                columns = dbcursor.description
                data  = []
                row = dbcursor.fetchone()
                if row:
                    row_data = {}
                    for idx, col in enumerate(columns):
                        row_data[col[0]] = row[idx]
                    data.append(row_data)
                dbcursor.close()

                json_object = json.dumps(data)
                # print(json_object)
                return True, json_object
            else:
                return False, {}
        except Exception as e:
            print(f"Error found {e}")
            return False, {}
    
    def UpdateStatusLog(ref_id, msg):
        db_connection = DbConnection()
        dbconn = db_connection.db_connect_live()
        try:
            cursor = dbconn.cursor()
            
            sql = """
            UPDATE tbl_video_process_log SET 
                status = %s, process_flag=9
            WHERE ref_id = %s
            """

            values = (
                msg, ref_id
            )

            cursor.execute(sql, values)
            formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
            print("Generated SQL Query:", formatted_sql)
            dbconn.commit()

            cursor.close()
            dbconn.close()
            print(f"Status Updated for {ref_id}")
            return True
        except Exception as err:
            print(f"Database Insert Error: {err}")
            return False
        
    def DeleteS3Data(self, delete_flag, s3url):
        if s3url != "" and delete_flag == 1:
            if "sourcestream.jdmagicbox.com" in s3url:
                try:
                    m3u8url = s3url.split("//")[1]
                    final_m3u8_url = str.replace(str.replace(str.replace(m3u8url, "sourcestream.jdmagicbox", "stream.jdmagicbox"),"/input/", "/hls/"),".mp4",".m3u8")
                    
                    m3u8_base_name = f"output/hls/{os.path.basename(final_m3u8_url)}".replace(".m3u8","")
                    m3u8_with_ext = f"output/hls/{os.path.basename(final_m3u8_url)}"
                    mp4_base_name = f"input/{os.path.basename(s3url)}"
                    thumb_base_name = f"output/thumbnail/{os.path.basename(final_m3u8_url)}".split(".")[0]
                    
                    print(f"m3u8 directory : {m3u8_base_name} && mp4 directory : {mp4_base_name}")
                    print(f"m3u8 File : {m3u8_with_ext} && thumb directory : {thumb_base_name}")
                    
                    print("###########################################")
                    # print(f"Deleted S3 object: {s3url} KEY : {key}")
                    del_res = UploadVideo.DeleteMultipleFilesFromS3(m3u8_base_name)
                    del_thumb_res = UploadVideo.DeleteMultipleFilesFromS3(thumb_base_name)
                    del_mp4_res = UploadVideo.DeleteSingleFileFromS3(mp4_base_name)
                    del_m3u8_res = UploadVideo.DeleteSingleFileFromS3(m3u8_with_ext)
                    print(f"M3U8 FILES : {del_res}")
                    print(f"THUMB FILES : {del_thumb_res}")
                    print(f"MP4 File : {del_mp4_res}")
                    print(f"M3U8 File :{del_m3u8_res}")
                    return True
                except Exception as err:
                    print(f"Error deleting S3 object from {s3url}: {err}")
            else:
                print("s3url does not contain target domain; skipping deletion.")
        else:
            print("Invalid data found")

    def duplicate_video_check(self, docid, hash_code=None):
        duplicate_val = False
        
        if hash_code:
            check_url = (CHECK_VIDEO_DUPLICACY +
                        "?docid=" + docid +
                        "&content_type=catalogue&hashcode=" + hash_code)
            try:
                check_response = CurlGetJson(check_url)
                if check_response.get('exists') is True:
                    duplicate_val = True
            except Exception as e:
                print("Error checking video duplicacy:", e)
                duplicate_val = False
                # return data
        return duplicate_val
    
    def checkRandomKeyExist(self, postData):
        db_connection = DbConnection()
        docid = postData.get("docid", "")
        random_key = postData.get("random_key", "")
        try:
            if docid != "" and random_key != "":
                dbconn = db_connection.db_connect_live()
                cursor = dbconn.cursor(prepared=True)
                query = "SELECT video_id FROM tbl_video_details WHERE docid = %s AND random_key = %s LIMIT 1"
                cursor.execute(query, (docid, random_key))
                row = cursor.fetchone()
                cursor.close()
                if row:
                    return True, row[0]
            return False, 0
        except Exception as e:
            print(f"Error found {e}")
            return False, 0
    
    def checkRandomKeyCategoryVideoExist(self, postData):
        db_connection = DbConnection()
        catid = postData.get("docid", "")
        random_key = postData.get("random_key", "")
        try:
            if catid != "" and random_key != "":
                dbconn = db_connection.db_connect_live()
                cursor = dbconn.cursor(prepared=True)
                query = "SELECT id FROM tbl_category_video_details WHERE national_catid = %s AND random_key = %s LIMIT 1"
                cursor.execute(query, (catid, random_key))
                row = cursor.fetchone()
                cursor.close()
                if row:
                    return True, row[0]
            return False, 0
        except Exception as e:
            print(f"Error found {e}")
            return False, 0
    
    def updateYoutubeChannelMaster(self, id, entity_id, random_key):
        db_connection = DbConnection()
        try:
            dbconn = db_connection.db_connect_live()
            cursor = dbconn.cursor(prepared=True)
            update_sql = """
                UPDATE Utube_channel_master 
                SET process_flag=1, random_catalouge_key=%s, process_date=NOW() 
                WHERE id = %s AND entity_id = %s
            """
            cursor.execute(update_sql, (random_key, id, entity_id))
            formatted_sql = update_sql % tuple(map(repr, (random_key, id, entity_id)))  # Converts to a string
            print("Generated Update SQL Query:", formatted_sql)
            dbconn.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error found in updateYoutubeChannelMaster: {e}")
            return False
    
    def checkBlockedContractStatus(self, data, comp_details):
        # // check blocked contract
        print("########### Checking Blocked Contract Status ###########")
        if 'tag_info' in comp_details and comp_details['tag_info'] != "":
            if 'bfe' in comp_details['tag_info'] and isinstance(comp_details['tag_info']['bfe'], list) and "PHOTO" in comp_details['tag_info']['bfe']:
                # blocked
                upd_video_qry = """
                    UPDATE tbl_video_details 
                    SET delete_flag=2, modified_by='owner_restricted', deleted_by='owner_restricted', 
                        modified_date=NOW(), deleted_date=NOW(), moderator_comment=24  
                    WHERE docid=%s AND random_key = %s
                """
                values = (data['docid'], data['random_key'])
                db_connection = DbConnection()
                dbconn = db_connection.db_connect_live()
                cursor = dbconn.cursor(prepared=True)
                cursor.execute(upd_video_qry, values)
                formatted_sql = upd_video_qry % tuple(map(repr, values))
                print("Generated Update SQL Query:", formatted_sql)
                
                # Update in Logs
                logs_data = {
                        "id": data['docid'],
                        "publish": "MEDIA",
                        "route": "BLOCKED_CONTRACT",
                        "user_id": data['upload_by'],
                        "critical": 1,
                        "msg": "Blocked Contract Video Upload",
                        "query": json.dumps({
                            "post_data": data,
                            "query": formatted_sql
                        })
                    }
                sendMediaLogs(data['docid'], logs_data, 'BLOCKED_CONTRACT', 'Blocked Contract Video Upload', data['upload_by'])
    
    def FetchMaxImgId(self, catid):
        db_connection = DbConnection()
        try:
            dbconn = db_connection.db_connect_slave()
            cursor = dbconn.cursor(prepared=True)
            query = "SELECT IFNULL(MAX(img_id),0) as max_img_id FROM tbl_category_video_details WHERE national_catid = %s"
            cursor.execute(query, (catid,))
            row = cursor.fetchone()
            cursor.close()
            if row and row[0] is not None:
                return int(row[0])
            else:
                return 0
        except Exception as e:
            print(f"Error found {e}")
            return 0
    
    def PushToFraudVideoDetect(self, video_id, docid, random_key, upload_by, mp4_url, approved):
        db_connection = DbConnection()
        rabbitMq = RabbitMQ()
        try:
            if video_id == 0:
                dbconn = db_connection.db_connect_live()
                cursor = dbconn.cursor(prepared=True)
                query = "SELECT video_id FROM tbl_video_details WHERE docid = %s AND random_key = %s LIMIT 1"
                cursor.execute(query, (docid, random_key))
                row = cursor.fetchone()
                cursor.close()
                if row and row[0] is not None:
                    video_id = row[0]
                else:
                    video_id = 0
                    
            
            if video_id != 0:
                current_date = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                queue_data = {
                    "server": "192.168.24.103",
                    "username": "mqadmin",
                    "password": "mqadmin",
                    "port": "5672",
                    "queue_name": "FRAUD_NUMBER_VIDEO_REQUEST"
                }
                process_data = {
                    "video_url" : mp4_url,
                    "docid" : docid,
                    "product_id" : str(video_id),
                    "product_url" : mp4_url,
                    "upload_by" : upload_by,
                    "content_type" : "video",
                    "upload_date": current_date,
                    "credentials": queue_data,
                    "approved" : approved,
                    "queue_name": "FRAUD_NUMBER_VIDEO_REQUEST"
                }
                response = rabbitMq.postQueue(process_data)
                return response
        except Exception as e:
            print(f"Error found in push to Fraud {e}")
            return 0
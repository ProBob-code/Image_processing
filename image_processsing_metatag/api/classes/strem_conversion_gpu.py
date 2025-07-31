import os
import json
import subprocess
from helper import getConfigInfo
from rabbitmq import RabbitMQ
from classes.dbconnection import DbConnection
from classes.imagemeta_tag import ImageMeta

class StreamConversionProcessGPU():
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
            preset = "faster"
        elif resolution == "240":
            preset = "veryfast"
        elif resolution == "180":
            preset = "ultrafast"
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
        output_ts   = process_data[closest_res][resolution]["output_ts"]
        output_m3u8 = process_data[closest_res][resolution]["output_m3u8"]
        input_file  = process_data[closest_res][resolution]["input_file"]
        input_file = os.path.abspath(input_file)
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
    
    def GetMetaDataHLS(file_path):
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


    def GenerateAndProcessDataHLS(self, data):

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
        closest_res = StreamConversionProcessGPU.GetResolution(src_width,src_height)
        avg_motion = StreamConversionProcessGPU.GetAvgMotion(f"{src_src_video}")
        
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
                gop_size = StreamConversionProcessGPU.CalculateGop(fps, avg_motion)
                crf, preset, audio_bitrate, audio_sample_rate, profile_level_id, codecs, process_value = StreamConversionProcessGPU.GetVideoSetting(resolution, src_video_profile, src_audio_profile, src_video_level)

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
                    process, output_ts, output_m3u8, input_file, video_codec, audio_codec, video_profile = StreamConversionProcessGPU.GetProcessData(resolution, closest_res, src_video_codec, input_dir, input_file, output_dir)

                    if StreamConversionProcessGPU.IsValid(input_dir, input_file, output_dir, closest_res):

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
                                self.ffmpeg_path, 
                                "-hwaccel", "cuda", "-i", input_file,
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
                                f"trunc(ih*min({target_width}/iw\\,{target_height}/ih)/4)*4,format=yuv420p,setsar=1"
                                # f"trunc(ih*min({target_width}/iw\\,{target_height}/ih)/4)*4,"
                                # f"colorspace=primaries={src_color_primaries}:trc={src_color_transfer}:space={src_color_space}:format=yuv420p,setsar=1"
                                
                            )


                            ffmpeg_cmd = [
                                self.ffmpeg_path, 
                                "-hwaccel",  "cuda", "-i", input_file,
                                "-vf", vf_filter,
                                "-c:v", "h264_nvenc",
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
                            print(f"2. ffmpeg_cmd = {ffmpeg_cmd}")

                        elif process == "re_encode":
                            print("Processing: Re-encode")
                            ffmpeg_cmd = [
                                self.ffmpeg_path, "-y", 
                                "-hwaccel cuda", "-i", input_file,
                                "-c:v", "h264_nvenc",
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
                            print(f"3. ffmpeg_cmd = {ffmpeg_cmd}")
                    else:
                        print("process : ",process)
                        print("Unknown process type! Exiting.")
                        return
                
                    # Execute FFmpeg command
                    try:
                        
                        print("Generated FFmpeg Command:", " ".join(ffmpeg_cmd))
                        
                        result = subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                        # if resolution=="720":
                        metadata_lower = StreamConversionProcessGPU.get_metadata_hls(output_ts)
                        print(json.dumps(metadata_lower,indent=4))
                        #sys.exit(1)

                        with open(master_playlist, "a") as f:
                            f.write(
                                f"#EXT-X-STREAM-INF:BANDWIDTH={target_bitrate},"
                                f"CODECS=\"{video_codec},{audio_codec}\","
                                f"RESOLUTION={target_width}x{target_height}\n"
                                f"{resolution}p.m3u8\n"
                            )

                        try:
                            connection = mysql.connector.connect(**DB_CONFIG)
                            cursor = connection.cursor(prepared=True)
                            
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
                            connection.commit()
                        except mysql.connector.Error as err:
                            print(f"Database Insert Error: {err}")

                        print(f"✅ Master playlist generated: {master_playlist}")
                        #return result.stdout
                    
                    except subprocess.CalledProcessError as e:
                        print("❌ FFmpeg Execution Failed!")
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
        src_random_key=data["random_key"]
        src_video_thumb=data["video_thumb"]

        # Prepare HLS Transcoding
        resolution_arr = ["1080", "720", "540", "480", "360", "240", "180"]
        closest_res = StreamConversionProcessGPU.GetResolution(src_width,src_height)
        avg_motion = StreamConversionProcessGPU.GetAvgMotion(f"{src_src_video}")
        
        NFS_in_path = getConfigInfo('NFS_path.video_input')
        if src_docid != "":
            input_dir = f"{NFS_in_path}/{src_docid}/{src_random_key}"
        else:
            input_dir = f"{NFS_in_path}/{src_random_key}"
        
        input_file = src_src_video

        NFS_out_path = getConfigInfo('NFS_path.video_output')
        
        if src_docid != "":
            output_dir = f"{NFS_out_path}/{src_docid}/{src_random_key}"
        else:
            output_dir = f"{NFS_out_path}/{src_random_key}"
        
        master_playlist = os.path.join(output_dir, "master.m3u8")

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

            target_bitrate = int(int(src_bitrate) * target_width * target_height / (src_width * src_height))
            maxrate = int(target_bitrate * 1.5)
            bufsize = int(target_bitrate * 2)
            
            fps = round(float(src_avg_frame_rate.split('/')[0]) / 
                float(src_avg_frame_rate.split('/')[1]) if '/' in str(src_avg_frame_rate) else float(src_avg_frame_rate or 30))

            #gop_size = fps * 2 if avg_motion > 0.2 else fps * 5
            gop_size = StreamConversionProcessGPU.CalculateGop(fps, avg_motion)
            crf, preset, audio_bitrate, audio_sample_rate, profile_level_id, codecs, process_value = StreamConversionProcessGPU.GetVideoSetting(resolution, src_video_profile, src_audio_profile, src_video_level)

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
                process, output_ts, output_m3u8, input_file, video_codec, audio_codec, video_profile = StreamConversionProcessGPU.GetProcessData(resolution, closest_res, src_video_codec, input_dir, input_file, output_dir)

                if StreamConversionProcessGPU.IsValid(input_dir, input_file, output_dir, closest_res):
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
                    metadata_lower = StreamConversionProcessGPU.GetMetaData(output_ts)
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
                    dbconn = db_connection.db_connect_live()
                    try:
                        cursor = dbconn.cursor(prepared=True)
                        
                        sql = """
                        UPDATE tbl_video_process_log SET 
                            random_key = %s,
                            width = %s, 
                            height = %s, 
                            bitrate = %s,
                            fps = %s, 
                            audio_bitrate = %s, 
                            audio_sample_rate = %s
                        WHERE ref_id = %s
                        """

                        values = (
                            src_random_key,
                            target_width, 
                            target_height,
                            target_bitrate,
                            fps, 
                            audio_bitrate,
                            audio_sample_rate,
                            src_random_key
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
                    # return None
                    exit(1)                
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

    def UpdateCallbackData(msg):
        if "ref_id" in msg and msg["ref_id"] != "":
            status, jsonObj = StreamConversionProcessGPU.GetVideoLogData(msg["ref_id"])
            print(status)
            if status:
                logData = json.loads(jsonObj)
                print(logData[0])
                
                postData  = json.loads(logData[0]["post_data"])
                video_url = logData[0]["video_url"]
                m3u8_url  = logData[0]["dest_video_url"]
                thumb_url = logData[0]["thumb_url"]
                file_size = logData[0]["size"]
                duration  = logData[0]["duration"]
                width     = logData[0]["width"]
                height    = logData[0]["height"]
                
                aspect_ratio = ImageMeta.calculate_aspect(width,height)
                print(postData)
                bitflag_cnt = 0
                docid = postData["docid"] if "docid" in postData else ""
                random_key = postData["random_key"] if "random_key" in postData else ""
                source = postData["source"] if "source" in postData else "jd_backend"
                company_name = postData["company_name"] if "company_name" in postData else ""
                city = postData["city"] if "city" in postData else ""
                paid_status = int(postData["paid_status"]) if "paid_status" in postData else 0
                platform = postData["platform"] if "platform" in postData else ""
                
                upload_by = postData["upload_by"] if "upload_by" in postData else "backend_video_process"
                module_type = int(postData["module_type"]) if "module_type" in postData else 3
                video_tag = int(postData["video_tag"]) if "video_tag" in postData else 1
                approved = int(postData["approved"]) if "approved" in postData else 0
                # video_tag = int(postData["video_tag"]) if "video_tag" in postData else 1
                service_id = int(postData["service_id"]) if "service_id" in postData else ""
                ip_address = postData["ip_address"] if "ip_address" in postData else ""
                hash = postData["hash"] if "hash" in postData else ""
                catalogue_id = postData["catalogue_id"] if "catalogue_id" in postData else 0
                ref_id = msg["ref_id"]
                
                catalog_source_content = ""
                
                if module_type == 3:
                    approved = 1
                
                if approved == 0:
                    bitflag_cnt  = bitflag_cnt + 1
                elif approved == 1:
                    bitflag_cnt  = bitflag_cnt + 2
                elif approved == 2:
                    bitflag_cnt  = bitflag_cnt + 4

                
                if platform != '':
                    if platform == 1:
                        bitflag_cnt  = bitflag_cnt + 32
                    elif platform == 2:
                        bitflag_cnt  = bitflag_cnt + 64
                    elif platform == 3:
                        bitflag_cnt  = bitflag_cnt + 128
                    elif platform == 4:
                        bitflag_cnt  = bitflag_cnt + 16384
                
                if docid != "":
                    # insert into tbl_video_details
                    db_connection = DbConnection()
                    dbconn = db_connection.db_connect_live()
                    try:
                        cursor = dbconn.cursor(prepared=True)
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
                                    hash=%s
                                """
                        
                        values = (
                                docid,
                                city,
                                company_name, 
                                approved, 
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
                                hash
                            )
                        
                        cursor.execute(insert_qry, values)
                        formatted_sql = insert_qry % tuple(map(repr, values))  # Converts to a string
                        print("Generated Insert SQL Query:", formatted_sql)
                        dbconn.commit()
                        cursor.close()
                        return True
                    except dbconn.Error as e:
                        print(f"Error found in insert {e}")
                        return False
                else:
                    print("No action now")
                    return True
    
    def GetVideoLogData(ref_id):
        db_connection = DbConnection()
        try:
            dbconn = db_connection.db_connect_live()
            dbcursor = dbconn.cursor(prepared=True)
            query = "SELECT post_data,video_url,dest_video_url,thumb_url,duration,width,height,size,random_key FROM tbl_video_process_log WHERE ref_id = %s LIMIT 1"
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
        except Exception as e:
            print(f"Error found {e}")
            return False, {}
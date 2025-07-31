import boto3, magic
import os
import concurrent.futures
import mysql.connector
from os.path import join
from classes.dbconnection import DbConnection
import logging
from botocore.exceptions import ClientError
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import generateRandom, RemoveDirectory, RemoveDirectoryContents, RemoveFile
# import threading
from PIL import Image
from helper import getConfigInfo


AWS_ACCESS_KEY = getConfigInfo('aws.access_key')
AWS_SECRET_KEY = getConfigInfo('aws.secret_key')
S3_BUCKET_NAME = getConfigInfo('aws.s3_bucket_name')

class UploadVideo():
    def __init__():
        pass
    
    def upload_single_file(s3_client, file_path, bucket_name, s3_key):
        """
        Upload a single file to S3
        """
        try:
            print(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(str(file_path), bucket_name, s3_key)
            return True
        except ClientError as e:
            print(f"Error uploading {file_path}: {str(e)}")
            return False

    def ParallelUploadFilesToS3(queue_data, s3_prefix="", max_workers=5):
        """
        Upload multiple files to S3 in parallel using threads
        
        Args:
            directory_path (str): Local directory containing files to upload
            bucket_name (str): Name of the S3 bucket
            s3_prefix (str): Optional prefix (folder) in S3 bucket
            max_workers (int): Maximum number of parallel uploads
        """
        bucket_name = S3_BUCKET_NAME
        directory_path = queue_data["dest_dir"]
        # Create S3 client
        # s3_client = boto3.client('s3')
        s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
        
        try:
            directory = Path(directory_path)
            if not directory.is_dir():
                print(f"Error: Directory {directory_path} does not exist")
                return
            
            # Prepare list of upload tasks
            upload_tasks = []
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(directory)
                    s3_key = f"{s3_prefix}{relative_path}".replace("\\", "/")
                    upload_tasks.append((file_path, s3_key))
            
            # Execute uploads in parallel
            success_count = 0
            s3_folder = str.replace(queue_data['dest_dir'], "/var/log/images/video_output/", "video_output/")
            s3_up_folder = os.path.join("output", s3_folder)
            print("################")
            print(s3_up_folder)
            print("################")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        UploadVideo.upload_single_file,
                        s3_client,
                        file_path,
                        bucket_name,
                        os.path.join(s3_up_folder, s3_key)
                    ): file_path for file_path, s3_key in upload_tasks
                }
                all_uploaded_success = True
                for future in future_to_file:
                    if future.result():
                        success_count += 1
                    else:
                        all_uploaded_success = False
            
            # Print summary
            print(f"\nUpload Summary:")
            print(f"Files successfully uploaded: {success_count}")
            print(f"Files failed: {len(upload_tasks) - success_count}")
            
            if all_uploaded_success:
                print("Upload Success")
                try:
                    db_connection = DbConnection()
                    dbconn = db_connection.db_connect_dev()

                    dest_video = os.path.join("https://stream.jdmagicbox.com/",s3_folder,"master.m3u8")
                    
                    sql = """
                    UPDATE lg_videos_29L SET 
                        dest_video = %s, 
                        status = 'uploaded', process_flag=4
                    WHERE docid = %s AND random_key = %s
                    """
                    values = (
                        dest_video, 
                        queue_data["docid"], 
                        queue_data["random_key"]
                    )
                    upd_dbcursor = dbconn.cursor()
                    upd_dbcursor.execute(sql, values)
                    dbconn.commit()
                    upd_dbcursor.close()

                    formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
                    print("Generated SQL Query:", formatted_sql)

                except mysql.connector.Error as err:
                    print(f"Database Insert Error: {err}")
            
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
    def UploadThumbnailToS3(queue_data):
        """
        Upload a single file to S3
        """
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
            thumbnail_source = queue_data['thumbnail']
            if not os.path.isfile(thumbnail_source):
                print(f"Error: Thumbnail File {thumbnail_source} does not exist")
                return False
            
            thumbnail_dest = "thumbnail/"+queue_data["docid"]+"/"+queue_data["docid"]+"_"+queue_data["random_key"]+".jpg"
            thumbnail_path = os.path.join("output", thumbnail_dest)
            print("################")
            print(thumbnail_source)
            print(thumbnail_path)
            print("################")
            
            mime = magic.Magic(mime=True)
            mime_type_val = mime.from_file(thumbnail_source)
            print(mime_type_val)
            if mime_type_val!="image/jpeg":
                img = Image.open(thumbnail_source)
                rgb_im = img.convert('RGB')
                rgb_im.save(thumbnail_path)
                
            queue_data['thumbnail_key'] = thumbnail_path
            print(f"Uploading {thumbnail_source} to s3://{S3_BUCKET_NAME}/{queue_data['thumbnail_key']}")
            # s3_client.upload_file(str(thumbnail_source), S3_BUCKET_NAME, queue_data['thumbnail_key'])
            s3_client.upload_file(
                thumbnail_source,
                S3_BUCKET_NAME,
                queue_data['thumbnail_key'],
                ExtraArgs={
                    'ContentType': 'image/jpeg',  # Explicitly set for JPEG
                    'ACL': 'public-read'          # Make it publicly accessible
                }
            )
            
            db_connection = DbConnection()
            dbconn = db_connection.db_connect_dev()

            dest_thumbnail = os.path.join("https://stream.jdmagicbox.com/",thumbnail_dest)
            
            sql = """
            UPDATE lg_videos_29L SET 
                thumb_url = %s
            WHERE docid = %s AND random_key = %s
            """
            values = (
                dest_thumbnail, 
                queue_data["docid"], 
                queue_data["random_key"]
            )
            upd_dbcursor = dbconn.cursor()
            upd_dbcursor.execute(sql, values)
            dbconn.commit()
            upd_dbcursor.close()
            print("Thumbnail uploaded successfully and url updated in database", dest_thumbnail)
            return True
        except ClientError as e:
            print(f"Error uploading {queue_data['thumbnail']}: {str(e)}")
            return False
        
    def UploadParallelFilesToS3(queue_data, s3_prefix="", max_workers=5):
        """
        Upload multiple files to S3 in parallel using threads
        
        Args:
            directory_path (str): Local directory containing files to upload
            bucket_name (str): Name of the S3 bucket
            s3_prefix (str): Optional prefix (folder) in S3 bucket
            max_workers (int): Maximum number of parallel uploads
        """
        bucket_name = S3_BUCKET_NAME
        
        # inside below directory path all files should be upload
        directory_path = queue_data["dest_dir"]
        # Create S3 client
        s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
        
        try:
            new_format_m3u8_status = False
            new_format_m3u8_file = Path(queue_data["dest_dir"] + ".m3u8")
            directory = Path(directory_path)
            if not directory.is_dir():
                print(f"Error: Directory {directory_path} does not exist")
                return False, ""
            
            # Prepare list of upload tasks
            upload_tasks = []
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(directory)
                    s3_key = f"{s3_prefix}{relative_path}".replace("\\", "/")
                    upload_tasks.append((file_path, s3_key))
            
            # Execute uploads in parallel
            success_count = 0
            s3_folder = str.replace(queue_data['dest_dir'], "/var/log/images/video_output/", "video_output/")

            # contract video
            # print(upload_tasks)
            if "docid" in queue_data and queue_data["docid"] != "":
                if queue_data["source"] == "gojd":
                    s3_parent_file = "gojd/hls/" + str.lower(queue_data["docid"]) + "_" + queue_data["random_key"]+".m3u8"
                    s3_folder = "gojd/hls/" + str.lower(queue_data["docid"]) + "_" + queue_data["random_key"] + "/"
                else:
                    s3_parent_file = "comp/hls/" + str.lower(queue_data["docid"]) + "_" + queue_data["random_key"]+".m3u8"
                    s3_folder = "comp/hls/" + str.lower(queue_data["docid"]) + "_" + queue_data["random_key"] + "/"
            # Standard video
            elif "type" in queue_data and queue_data["type"] != "" and queue_data["source"] == "standard":
                s3_parent_file = "justdial/hls/" + str.lower(queue_data["type"]) + "_" + queue_data["random_key"] + ".m3u8"
                s3_folder = "justdial/hls/" + str.lower(queue_data["type"]) + "_" + queue_data["random_key"] + "/"
            # Marketplace or custom
            elif "source" in queue_data and (queue_data["source"] == "mp" or queue_data["source"] == "mp_custom"):
                if "supplier_id" in queue_data and queue_data["supplier_id"]!="":
                    s3_parent_file = "mp/hls/" + str.lower(queue_data["supplier_id"]) + "_" + queue_data["random_key"] + ".m3u8"
                    s3_folder = "mp/hls/" + str.lower(queue_data["supplier_id"]) + "_" + queue_data["random_key"] + "/"
                elif "product_id" in queue_data and queue_data["product_id"]!="":
                    s3_parent_file = "mp/hls/" + str.lower(queue_data["product_id"]) + "_" + queue_data["random_key"] + ".m3u8"
                    s3_folder = "mp/hls/" + str.lower(queue_data["product_id"]) + "_" + queue_data["random_key"] + "/"
                else:
                    uniqueIdentify = str.lower(generateRandom(20))
                    s3_parent_file = "mp/hls/" + uniqueIdentify + "_" + queue_data["random_key"] + ".m3u8"
                    s3_folder = "mp/hls/" + uniqueIdentify + "_" + queue_data["random_key"] + "/"
                
                
            s3_up_folder = os.path.join("output", s3_folder)
            print(new_format_m3u8_file)
            if new_format_m3u8_file.is_file():
                s3_parent_folder = "output/" + s3_parent_file
                print("Yes parent file exist")
                if UploadVideo.upload_single_file(s3_client,new_format_m3u8_file,bucket_name,s3_parent_folder):
                    new_format_m3u8_status = True
                else:
                    print("Error in parent file upload")
            print("################")
            print(s3_up_folder)
            print("################")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        UploadVideo.upload_single_file,
                        s3_client,
                        file_path,
                        bucket_name,
                        os.path.join(s3_up_folder, s3_key)
                    ): file_path for file_path, s3_key in upload_tasks
                }
                all_uploaded_success = True
                for future in future_to_file:
                    if future.result():
                        success_count += 1
                    else:
                        all_uploaded_success = False
            
            # Print summary
            print(f"\nUpload Summary:")
            print(f"Files successfully uploaded: {success_count}")
            print(f"Files failed: {len(upload_tasks) - success_count}")
            
            if all_uploaded_success:
                print("Upload Success")
                try:
                    db_connection = DbConnection()
                    dbconn = db_connection.db_connect_live()

                    if new_format_m3u8_status:
                        dest_video = "https://stream.jdmagicbox.com/"+s3_parent_file
                    else:
                        dest_video = os.path.join("https://stream.jdmagicbox.com/",s3_folder,"master.m3u8")
                    
                    sql = """
                    UPDATE tbl_video_process_log SET 
                        dest_video_url = %s, 
                        status = 'COMPLETED', process_flag=4
                    WHERE ref_id = %s
                    """
                    values = (
                        dest_video, 
                        queue_data["random_key"]
                    )
                    upd_dbcursor = dbconn.cursor()
                    upd_dbcursor.execute(sql, values)
                    dbconn.commit()
                    upd_dbcursor.close()

                    formatted_sql = sql % tuple(map(repr, values))  # Converts to a string
                    print("Generated SQL Query:", formatted_sql)
                    #Remove directory contents
                    RemoveDirectoryContents(queue_data["dest_dir"])
                    RemoveFile(new_format_m3u8_file)
                    RemoveDirectory(queue_data["dest_dir"])
                    #Remove directory
                    return True, dest_video

                except Exception as err:
                    print(f"Database Insert Error: {err}")
                    return False, ""
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False, ""

    def UploadThumbnailFileToS3(queue_data):
        """
        Upload a single file to S3
        """
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
            thumbnail_source = queue_data['thumbnail']
            if not os.path.isfile(thumbnail_source):
                print(f"Error: Thumbnail File {thumbnail_source} does not exist")
                return False, ""
            
            if "docid" in queue_data:
                thumbnail_dest = "thumbnail/"+str.lower(queue_data["docid"])+"/"+str.lower(queue_data["docid"])+"_"+queue_data["random_key"]+".jpg"
            elif "source" in queue_data and queue_data["source"] != "":
                thumbnail_dest = "thumbnail/"+queue_data["source"]+"/"+str.lower(generateRandom(15))+"_"+queue_data["random_key"]+".jpg"
            else:
                thumbnail_dest = "thumbnail/"+queue_data["random_key"]+"/"+str.lower(generateRandom(15))+"_"+queue_data["random_key"]+".jpg"
            
            thumbnail_path = os.path.join("output", thumbnail_dest)
            print("################")
            print(thumbnail_source)
            print(thumbnail_path)
            print("################")
            
            mime = magic.Magic(mime=True)
            mime_type_val = mime.from_file(thumbnail_source)
            print(mime_type_val)
            if mime_type_val!="image/jpeg":
                img = Image.open(thumbnail_source)
                rgb_im = img.convert('RGB')
                rgb_im.save(thumbnail_path)
                
            queue_data['thumbnail_key'] = thumbnail_path
            print(f"Uploading {thumbnail_source} to s3://{S3_BUCKET_NAME}/{queue_data['thumbnail_key']}")
            # s3_client.upload_file(str(thumbnail_source), S3_BUCKET_NAME, queue_data['thumbnail_key'])
            s3_client.upload_file(
                thumbnail_source,
                S3_BUCKET_NAME,
                queue_data['thumbnail_key'],
                ExtraArgs={
                    'ContentType': 'image/jpeg',  # Explicitly set for JPEG
                    'ACL': 'public-read'          # Make it publicly accessible
                }
            )
            
            db_connection = DbConnection()
            dbconn = db_connection.db_connect_live()

            dest_thumbnail = os.path.join("https://stream.jdmagicbox.com/",thumbnail_dest)
            
            sql = """
            UPDATE tbl_video_process_log SET 
                thumb_url = %s
            WHERE ref_id = %s
            """
            values = (
                dest_thumbnail,
                queue_data["random_key"]
            )
            upd_dbcursor = dbconn.cursor()
            upd_dbcursor.execute(sql, values)
            dbconn.commit()
            upd_dbcursor.close()
            print("Thumbnail uploaded successfully and url updated in database", dest_thumbnail)
            #Remove thumbnail file from local
            RemoveFile(thumbnail_source)
            return True, dest_thumbnail
        except ClientError as e:
            print(f"Error uploading {queue_data['thumbnail']}: {str(e)}")
            return False, ""
    
    def UploadMp4FileToS3(queue_data):
        """
        Upload a single file to S3 with storage class set as ONEZONE_IA.
        """
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
            video_mp4_source = queue_data['src_video']
            if not os.path.isfile(video_mp4_source):
                print(f"Error: Video File {video_mp4_source} does not exist")
                return False
            
            if "docid" in queue_data and queue_data["docid"] != "":
                if queue_data["source"] == "gojd":
                    mp4_video_dest = "gojd/" + str.lower(queue_data["docid"]) + "/" + queue_data["random_key"] + ".mp4"
                else:
                    mp4_video_dest = "comp/" + str.lower(queue_data["docid"]) + "/" + queue_data["random_key"] + ".mp4"
            elif "type" in queue_data and queue_data["type"] != "" and queue_data["source"] == "standard":
                mp4_video_dest = "justdial/" + str.lower(queue_data["type"]) + "/" + queue_data["random_key"] + ".mp4"
            elif "source" in queue_data and (queue_data["source"] == "mp" or queue_data["source"] == "mp_custom"):
                if "supplier_id" in queue_data and queue_data["supplier_id"] != "":
                    mp4_video_dest = "mp/" + str.lower(queue_data["supplier_id"]) + "/" + queue_data["random_key"] + ".mp4"
                elif "product_id" in queue_data and queue_data["product_id"] != "":
                    mp4_video_dest = "mp/" + str.lower(queue_data["product_id"]) + "/" + queue_data["random_key"] + ".mp4"
                else:
                    mp4_video_dest = "mp/" + str.lower(generateRandom(20)) + "/" + queue_data["random_key"] + ".mp4"
            else:
                mp4_video_dest = "video_output/" + queue_data["random_key"] + ".mp4"
            
            mp4_video_path = os.path.join("output", mp4_video_dest)
            print("################")
            print(video_mp4_source)
            print(mp4_video_path)
            print("################")
            
            print(f"Uploading {video_mp4_source} to s3://{S3_BUCKET_NAME}/{mp4_video_path}")
            
            s3_client.upload_file(
                video_mp4_source,
                S3_BUCKET_NAME,
                mp4_video_path,
                ExtraArgs={
                    'ContentType': 'video/mp4',  # Explicitly set for mp4
                    'ACL': 'public-read',         # Make it publicly accessible
                    'StorageClass': 'ONEZONE_IA'    # Set storage class as ONEZONE_IA
                }
            )
            
            db_connection = DbConnection()
            dbconn = db_connection.db_connect_live()

            dest_mp4_video_url = os.path.join("https://stream.jdmagicbox.com/",mp4_video_dest)
            
            sql = """
            UPDATE tbl_video_process_log SET 
                video_url = %s
            WHERE ref_id = %s
            """
            values = (
                dest_mp4_video_url,
                queue_data["random_key"]
            )
            upd_dbcursor = dbconn.cursor()
            upd_dbcursor.execute(sql, values)
            dbconn.commit()
            upd_dbcursor.close()
            print("Mp4 uploaded successfully and url updated in database", dest_mp4_video_url)
            return True, dest_mp4_video_url
        except ClientError as e:
            print(f"Error uploading {queue_data['src_video']}: {str(e)}")
            return False, ""
    
    def DeleteMultipleFilesFromS3(directory_path):
        # Initialize the S3 client
        # s3_client = boto3.client('s3')
        s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
        bucket_name = S3_BUCKET_NAME
        
        # Ensure directory path ends with a slash
        if not directory_path.endswith('/'):
            directory_path += '/'
        
        # List all objects with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=directory_path)
        
        # Collect objects to delete
        objects_to_delete = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects_to_delete.append({'Key': obj['Key']})
        
        # Delete objects in batches (up to 1000 per request)
        if objects_to_delete:
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': objects_to_delete}
            )
            return f"Deleted {len(objects_to_delete)} objects from {directory_path}"
        else:
            return f"No objects found in {directory_path}"
        
    def DeleteSingleFileFromS3(file_path):
        # s3_client = boto3.client('s3')
        s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
        try:
            s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=file_path)
            return f"File {file_path} deleted successfully from bucket {S3_BUCKET_NAME}."
        except Exception as e:
            print(f"Error deleting file: {e}")
            return str(e)
        
import json
import shutil
from urllib.parse import urlparse
import pandas as pd
# import urllib.request
import time
import random, string
import os
import re, requests
from helper import getConfigData, getConfigInfo
from classes.imagemeta_tag import ImageMeta
import cv2
import numpy as np
from classes.manage_logs import Manage_Logs
import logging
from moviepy.editor import VideoFileClip
import mimetypes


VIDEO_MIME_TYPES = [
    "video/mp4",
    "video/x-msvideo",
    "video/mpeg",
    "video/ogg",
    "video/webm",
    "video/3gpp",
    "video/3gpp2",
    "video/x-flv",
    "video/mp4",
    "application/x-mpegURL",
    "video/MP2T",
    "video/quicktime",
    "video/x-ms-wmv",
    "video/x-quicktime",
    "video/avi",
    "video/x-matroska",
    "video/x-ms-asf",
    "application/octet-stream"
]

def generateRandom(val):
    range_val = int(val)
    random_str = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(range_val))
    return random_str

def download_image_requests(url, path):
  """
  Downloads an image from the URL and saves it to the path using the requests library.

  Args:
      url: The URL of the image to download.
      path: The path where the image should be saved.
  """
  try:
    response = requests.get(url, timeout = (10,15))
    response.raise_for_status()  # Raise an exception for non-2xx status codes

    # Download entire content at once (not recommended for large files)
    image_data = response.content

    with open(path, 'wb') as file:
      file.write(image_data)

    print(f"Image downloaded successfully: {path}")
  except requests.exceptions.RequestException as e:
    logging.info(f"Error downloading image using requests: {e}")

def initialData1_Video(data1):
    url = data1['url']
    product = generateRandom(6)
    if url.endswith(('.mp4', '.mov', '.avi', '.wmv', '.mkv', '.webm', '.flv', '.avchd', '.mpeg', '.ogv', '.m3u8')):
        name = re.search(r'([a-zA-Z0-9-]*)\.(?:mp4|mov|avi|wmv|mkv|webm|flv|avchd|mpeg|ogv)', url) #using regular expression to extract out the image name from url
        if name:
            filename = name.group()
        else:
            filename_1 = url.split('/')[-1]
            filename = product+'-'+filename_1
        try:
            # validate video exist on url
            response = requests.head(url)
            if response.status_code == 200:
                df1 = {
                    'error_code' : 0,
                    'filename' : filename,
                    'msg' : 'success'
                }
            else:
                df1 = {
                    'error_code' : 1,
                    'filename' : '',
                    'msg' : 'Invaid Video URL'
                }
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            df1 = {
                'error_code' : 1,
                'filename' : '',
                'exception' : e,
                'msg' : 'Error Processing in URL'
            }
    else:
        df1 = {
            'error_code' : 1,
            'filename' : 0,
            'msg' : 'Invalid Video URL or Extension'
        }
    

    return df1
    

def initialData1(data1, force=0, is_resize=False):

    NFS_path = getConfigData('NFS_path.path')
    # if 'docid' in data1 and data1['docid']!='':
    #     desire_dir = str(NFS_path) + '/common_upload/metadata/' + data1['docid']
    # else:
    #     desire_dir = str(NFS_path) + '/common_upload/metadata/new_images'
    desire_dir = str(NFS_path) + '/common_upload/meta'
    if not os.path.exists(desire_dir):
        os.makedirs(desire_dir) #this will create a new folder if it doesn't exist and start maintaining all the downloads

    if force == 1:
        url = data1['product_url_ori']
    else:
        url = data1['product_url']
    
    # additional generate Missing Image -- start
    url = re.sub(r"\\\/", "/", url)
    if (url != "" and url.endswith(('.JPEG', '.jpg', '.png', '.jpeg', '.JPG', '.gif', '.PNG', '.avif', '.webp', '.heif', '.heic'))):
        response = requests.get(url)
    else:
        return {
            'error_code' : 1,
            'filename' : 0,
            'local_path_flag' : 0,
            'msg' : 'Download Failed!'
        }
    if response.status_code != 200 and '-w.jp' in url and force == 1:
        print('INSIDE GENERATE IMAGE')
        product_url = url.replace("-w.jp",".jp")
        generateMissingImage(product_url)
    # additional generate Missing Image -- end
    product = generateRandom(6)
    local_path_flag = 0

    if url.endswith(('.JPEG', '.jpg', '.png', '.jpeg', '.JPG', '.gif', '.PNG', '.avif', '.webp', '.heif', '.heic')) and url != "":
        name = re.search(r'([a-zA-Z0-9-]*)\.(?:JPEG|jpg|png|jpeg|JPG|gif|PNG|avif|webp|heif|heic)', url) #using regular expression to extract out the image name from url
        if name:
            filename = name.group()
        else:
            filename_1 = url.split('/')[-1]
            filename = product+'-'+filename_1

        try:
            # # Set a valid user agent header
            # opener = urllib.request.build_opener()
            # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            # urllib.request.install_opener(opener)

            # validate image exist on url
            response = requests.get(url)
            if response.status_code == 200:
                path = os.path.join(desire_dir, filename)
                # urllib.request.urlretrieve(url, path)
                download_image_requests(url, path)
                size = os.path.getsize(path)
                if(is_resize):
                    from PIL import Image

                    # Define path, width, and height for resizing (adjust as needed)
                    path = os.path.join(desire_dir, filename)

                    # Open the image
                    try:
                        img = Image.open(path)
                    except FileNotFoundError:
                        print("Error: File not found at", path)
                        exit()
                    
                    height = img.height
                    width = img.width

                    new_width = 1600
                    new_height = int((height / width) * new_width)
                    # Resize the image
                    resized_img = img.resize((new_width, new_height))

                    # Save the resized image (overwrites original file by default)
                    resized_img.save(path)

                    # print("Image resized and saved successfully!")
                df1 = {
                    'error_code' : 0,
                    'filename' : filename,
                    'local_path_flag' : local_path_flag,
                    'dir' : desire_dir,
                    'msg' : 'success',
                    'size': size
                }
            else:
                df1 = {
                    'error_code' : 1,
                    'filename' : '',
                    'local_path_flag' : local_path_flag,
                    'dir' : '',
                    'msg' : 'Download Failed!'
                }

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            url = 0
            df1 = {
                'error_code' : 1,
                'filename' : 0,
                'local_path_flag' : local_path_flag,
                'dir' : desire_dir,
                'exception' : e,
                'msg' : 'Download Failed!'
            }
    else:
        df1 = {
            'error_code' : 1,
            'filename' : 0,
            'local_path_flag' : local_path_flag,
            'msg' : 'Download Failed!'
        }
    
    return df1


def initialData2(data2):

    column_names = ['image_name', 'product_id', 'docid', 'path_flag', 'business_tag', 'product_url_ori']
    df1 = pd.DataFrame(columns=column_names)

    url = data2['url']
    product = data2['product']
    docid = data2['docid']
    local_path_flag = 1
    business_tag = data2['business_tag']
    product_url_ori = str(data2['product_url_ori'])
    

    if url.endswith(('.JPEG', '.jpg', '.png', '.jpeg', '.JPG','.gif','.PNG','.avif','.webp','.heif','.heic')):
        name = re.search(r'([a-zA-Z0-9-]*)\.(?:JPEG|jpg|png|jpeg|JPG|gif|PNG|avif|webp|heif|heic)', url) #using regular expression to extract out the image name from url
        if name:
            filename = name.group()
        else:
            filename = url.split('/')[-1]

        df1 = [filename, product, docid, local_path_flag, business_tag, product_url_ori]
    
    # return tuple(df1.values.tolist())
    return df1

def sendLogs(data,route):
    print(data)
    logs_url = getConfigData('LOGS.url')

    request = {
        'PUBLISH': 'META',
        'ROUTE': route,
        'CRITICAL_FLAG': '1',
        'ID': data['request']['docid'],
        'USER_ID': 'Python API Endpoint',
        'MESSAGE': 'Meta Logs Check',
        'DATA[RESPONSE]': json.dumps(data)
    }
    files=[]
    headers = {}
    print(logs_url)

    response = requests.request("POST", logs_url, headers=headers, data=request, files=files)
    print(response.text)

def generateMissingImage(product_url):
    fupload_url = getConfigData('FUPLOAD_API.url')

    request = {
        'source': 'imagemissing',
        'insta': 0,
        'missing_img': 'all',
        'url': product_url
    }
    files=[]
    headers = {}
    # print(fupload_url)
    response = requests.request("POST", fupload_url, headers=headers, data=request, files=files)
    print(response.text)

def getCompanyDetails(docid):
    company_details_api_url = getConfigData('COMPANY_DETAILS.url')
    url = company_details_api_url + '?case=content_service&docid=' + docid
    payload={}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    return json.loads(response.text)

def is_corrupted(image_path, product_url=None):
  if(product_url):
    product_url = re.sub(r"\\\/", "/", product_url)
  if not os.path.exists(image_path):
    logging.info(f"Error: Image file not found: {image_path}")
    return False
  
  size_local = os.path.getsize(image_path)
#   if(product_url):
#     size_url = ImageMeta.fetchImageHeightWidthV2(product_url).get('size', -1)
#     if(size_url != size_local):
#         logging.info(f"product_url={product_url}\nimage_path={image_path}\nsize(local)={size_local}, size(url)={size_url}")
  if size_local == 0:
    logging.info(f"Error: Empty image file: {image_path}")
    return False

  if(not image_path.endswith(('.gif'))):
    try:
      with open(image_path, 'rb') as f:
        # Read the entire file in one go (might not be ideal for large images)
        image_data = f.read()
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logging.info(f"Error: Failed to read image: {image_path}")
            return False
    except Exception as e:
        logging.info(f"Error: Unknown error while processing image: {image_path} - {e}")
        return False
  # - Use libraries like `pyvips` for more advanced image validation

  return True  # Image seems valid based on basic checks
  
def downloadFile(data):
    url = data["video_url"]
    NFS_path = getConfigInfo('NFS_path.video_input')
    # parsed_url = urlparse(url)
    # parsed_path = parsed_url.path
    # _, extension = os.path.splitext(parsed_path)
    # file_extension = extension.lstrip('.')
    file_extension = os.path.splitext(url)[1]
    if "docid" in data and data["docid"]!="":
        dest_folder = NFS_path+"/"+data["docid"]
    else:
        dest_folder = NFS_path
    
    if "random_key" in data and data["random_key"]!="":
        random_str = data["random_key"]
    else:
        random_str = generateRandom(12)
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    # filename = url.split('/')[-1][:25].replace(" ", "").replace("-","")
    filename = generateRandom(20)
    file_path = os.path.join(dest_folder, random_str+"_"+filename)

    r = requests.get(url, stream=True)
    if r.ok:
        content_type = r.headers.get('Content-Type', '').lower()
        # if 'video/mp4' not in content_type:
        print("Content-Type:", content_type)
        print("File Extension:", file_extension)
        if content_type not in VIDEO_MIME_TYPES:
            raise ValueError(f"Unexpected MIME type: {content_type}. Expected 'video/mp4'.")
        else:
            if file_extension != ".mp4":
                file_path += ".mp4"
            else:
                file_path += file_extension
        print("saving to", os.path.abspath(file_path))
        data["downloaded_path"] = file_path
        data["status"] = True
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        return data
    else:
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))
        data["status"] = False
        data['error_msg'] = str(r.text)
        return data
    

def CheckVideoFileMimeType(input_path):
    print("Checking Video Mime type")
    if os.path.exists(input_path):
        filesize = os.path.getsize(input_path)
        if filesize == 0:
            print(f"File {input_path} is empty (filesize: 0).")
            return False, input_path, "file_size_zero"
        mime_type, _ = mimetypes.guess_type(input_path)
        # if mime_check_status:
        if mime_type is not None and str.lower(mime_type) == "video/mp4":
            print(f"Valid file format {input_path}")
            return True, input_path, "video/mp4"
        else:            
            output_path = os.path.splitext(input_path)[0] + ".mp4"
            print(f"Expected file path after covert {output_path}")
            try:
                # Load the video file
                video = VideoFileClip(input_path)
                # Write the video to MP4 format
                video.write_videofile(output_path, codec="libx264", audio_codec="aac")
                print(f"Video successfully converted and saved to {output_path}")
                os.remove(input_path)
                return True, output_path, "video/mp4"
            except Exception as e:
                print(f"An error occurred: {e}")
                return False, output_path, str(e)
        
    else:
        return None, input_path, "file_not_found"

def RemoveFile(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} removed successfully")
        else:
            print(f"File {file_path} does not exist")
    except Exception as e:
        print(f"Error removing file {file_path}: {e}")
        
def RemoveDirectory(directory_path):
    try:
        if os.path.exists(directory_path):
            os.rmdir(directory_path)
            print(f"Directory {directory_path} removed successfully")
        else:
            print(f"Directory {directory_path} does not exist")
    except Exception as e:
        print(f"Error removing directory {directory_path}: {e}")

def RemoveDirectoryContents(directory_path):
    try:
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
        else:
            print(f"Directory {directory_path} does not exist")
    except Exception as e:
        print(f"Error removing directory contents {directory_path}: {e}")

def CurlPostJson(url, post_request):
    payload = json.dumps(post_request)
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    return json.loads(response.text)

def CurlGetJson(url):
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("GET", url, headers=headers)
    
    return json.loads(response.text)

def CurlPostPlain(url, post_data = dict()):
    payload = post_data
    files=[]
    headers = {}
    try:
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
    except Exception as e:
        print(f"Error in CurlPostPlain: {e}")
        return {'error': str(e)}
    
    return json.loads(response.text)

def sendMediaLogs(docid, data, route, message, user_id):
    # print(data)
    logs_url = getConfigData('LOGS.url')

    request = {
        'PUBLISH': 'MEDIA',
        'ROUTE': route,
        'CRITICAL_FLAG': '1',
        'ID': str.upper(docid),
        'USER_ID': user_id,
        'MESSAGE': message,
        'DATA[RESPONSE]': json.dumps(data)
    }
    files=[]
    headers = {}
    print(logs_url)

    response = requests.request("POST", logs_url, headers=headers, data=request, files=files)
    print(response.text)

def calculateAspectRatio(width, height):
    def gcd(a, b):
        return a if b == 0 else gcd(b, a % b)

    r = gcd(width, height)
    x = int(width / r)
    y = int(height / r)
    return("%d:%d" % (x, y))

def DownloadImageFromUrl(url, file_path):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Send a GET request to the image URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the image content to a file at the specified path
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Image successfully downloaded to {file_path}")
            return file_path
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
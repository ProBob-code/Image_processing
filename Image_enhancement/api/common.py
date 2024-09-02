import json
import pandas as pd
import urllib.request
import time
import random, string
import os
import re, requests
from helper import getConfigData
from classes.imagemeta_tag import ImageMeta
import cv2
import numpy as np

def generateRandom(val):
    range_val = int(val)
    random_str = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(range_val))
    return random_str


def initialData1_Video(data1):
    url = data1['url']
    product = generateRandom(6)
    if url.endswith(('.mp4', '.mov', '.avi', '.wmv', '.mkv', '.webm', '.flv', '.avchd', '.mpeg', '.ogv')):
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
    if 'docid' in data1 and data1['docid']!='':
        desire_dir = str(NFS_path) + '/common_upload/metadata/' + data1['docid']
    else:
        desire_dir = str(NFS_path) + '/common_upload/metadata/new_images'
    
    if not os.path.exists(desire_dir):
        os.makedirs(desire_dir) #this will create a new folder if it doesn't exist and start maintaining all the downloads

    if force == 1:
        url = data1['product_url_ori']
    else:
        url = data1['product_url']
    
    # additional generate Missing Image -- start
    url = re.sub(r"\\\/", "/", url)
    response = requests.get(url)
    if response.status_code != 200 and '-w.jp' in url and force == 1:
        print('INSIDE GENERATE IMAGE')
        product_url = url.replace("-w.jp",".jp")
        generateMissingImage(product_url)
    # additional generate Missing Image -- end
    product = generateRandom(6)
    local_path_flag = 0

    if url.endswith(('.JPEG', '.jpg', '.png', '.jpeg', '.JPG', '.gif', '.PNG', '.avif', '.webp', '.heif', '.heic')):
        name = re.search(r'([a-zA-Z0-9-]*)\.(?:JPEG|jpg|png|jpeg|JPG|gif|PNG|avif|webp|heif|heic)', url) #using regular expression to extract out the image name from url
        if name:
            filename = name.group()
        else:
            filename_1 = url.split('/')[-1]
            filename = product+'-'+filename_1

        try:
            # Set a valid user agent header
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)

            # validate image exist on url
            response = requests.get(url)
            if response.status_code == 200:
                urllib.request.urlretrieve(url, os.path.join(desire_dir, filename))
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
                    'msg' : 'success'
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
                'msg' : 'Error Processing in url'
            }
    else:
        df1 = {
            'error_code' : 1,
            'filename' : 0,
            'local_path_flag' : local_path_flag,
            'msg' : 'Invalid URL'
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
    print(f"Error: Image file not found: {image_path}")
    return False
  
  size_local = os.path.getsize(image_path)
  if(product_url):
    size_url = ImageMeta.fetchImageHeightWidthV2(product_url).get('size', -1)
    if(size_url != size_local):
        print("##########")
        print(product_url)
        print(f"size(local)={size_local}, size(url)={size_url}")
        print("##########")
  if size_local == 0:
    print(f"Error: Empty image file: {image_path}")
    return False

  try:
    with open(image_path, 'rb') as f:
      # Read the entire file in one go (might not be ideal for large images)
      image_data = f.read()
      img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
      if img is None:
        print(f"Error: Failed to read image: {image_path}")
        return False
  except Exception as e:
    print(f"Error: Unknown error while processing image: {image_path} - {e}")
    return False
  # - Use libraries like `pyvips` for more advanced image validation

  return True  # Image seems valid based on basic checks
  
#import necessary libraries
import PIL
from PIL import Image
import os
import numpy as np
import pandas as pd
import re
import shutil
from pathlib import Path
import wget
import cv2 as cv2
# import urllib
from flask import request as req
import json
from PIL.ExifTags import TAGS
# import urllib.request
import time
from src.libs.helpers import getConfigData
import requests, logging

class ImageUtils:
    def __init__(self):
        pass
        
    
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

    @staticmethod
    def initialData1(data1):
        #maintain two lists for appending data into it
        U = []
        y = []
        
        #create a dataframe with certain columns
        column_names = ['image_name', 'product_id', 'docid', 'path_flag', 'business_tag', 'product_url_ori']
        df1 = pd.DataFrame(columns=column_names)

        NFS_path = getConfigData('NFS_path.path')
        desire_dir = str(NFS_path) + '/new_images'
        
        if not os.path.exists(desire_dir):
            os.makedirs(desire_dir) #this will create a new folder if it doesn't exist and start maintaining all the downloads

        for i in range(len(data1)):
            url = data1.iloc[i, 1]
            product = data1.iloc[i, 0]
            docid = data1.iloc[i, 2]
            local_path_flag = 0
            business_tag = str(data1.iloc[i, 3])
            product_url_ori = str(data1.iloc[i, 4])

            if url.endswith(('.JPEG', '.jpg', '.png', '.jpeg', '.JPG', '.gif', '.PNG', '.avif', '.webp', '.heif', '.heic')):
                name = re.search(r'([a-zA-Z0-9-]*)\.(?:JPEG|jpg|png|jpeg|JPG|gif|PNG|avif|webp|heif|heic)', url) #using regular expression to extract out the image name from url
                if name:
                    filename_1 = name.group()
                    filename = product+filename_1
                else:
                    filename_1 = url.split('/')[-1]
                    filename = product+filename_1

                try:
                    # # Set a valid user agent header
                    # opener = urllib.request.build_opener()
                    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    # urllib.request.install_opener(opener)

                    # Download the image with retries
                    retries = 2
                    while retries > 0:
                        path = os.path.join(desire_dir, filename)
                        try:
                            # urllib.request.urlretrieve(url, path)
                            ImageUtils.download_image_requests(url, path)
                            break
                        # except (urllib.error.HTTPError, urllib.error.URLError) as e:
                        except Exception as e:
                            retries -= 1
                            print(f"Error downloading {url}. Retrying in 2 seconds... ({retries} retries left)")
                            time.sleep(2)
                            if retries == 0:
                                raise e

                    df1.loc[i] = [filename, product, docid, local_path_flag, business_tag, product_url_ori]
                    U.append(url)

                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    url = 0
                    df1.loc[i] = [0, product, docid, local_path_flag, business_tag, product_url_ori]
                    y.append(url)
                    continue
            else:
                df1.loc[i] = [0, product, docid, local_path_flag, business_tag, product_url_ori]

        return tuple(df1.values.tolist())

    def initialData2(data2):

        column_names = ['image_name', 'product_id', 'docid', 'path_flag', 'business_tag', 'product_url_ori']
        df1 = pd.DataFrame(columns=column_names)

        for i in range(len(data2)):
            url = data2.iloc[i, 1]
            product = data2.iloc[i, 0]
            docid = data2.iloc[i, 2]
            local_path_flag = 1
            business_tag = data2.iloc[i, 4]
            product_url_ori = str(data2.iloc[i, 5])
            

            if url.endswith(('.JPEG', '.jpg', '.png', '.jpeg', '.JPG','.gif','.PNG','.avif','.webp','.heif','.heic')):
                name = re.search(r'([a-zA-Z0-9-]*)\.(?:JPEG|jpg|png|jpeg|JPG|gif|PNG|avif|webp|heif|heic)', url) #using regular expression to extract out the image name from url
                if name:
                    filename_1 = name.group()
                    filename = product+filename_1
                else:
                    filename_1 = url.split('/')[-1]
                    filename = product+filename_1
                    
        
                df1.loc[i] = [filename, product, docid, local_path_flag, business_tag, product_url_ori]
        
        return tuple(df1.values.tolist())
                
          
        

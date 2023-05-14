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
import urllib
from flask import request as req
import json
from PIL.ExifTags import TAGS
import urllib.request
import time


class ImageUtils:
    def __init__(self):
        pass
        

    @staticmethod
    def initialData(data1):
        #maintain two lists for appending data into it
        U = []
        y = []
        
        #create a dataframe with certain columns
        column_names = ['image_name', 'product_id']
        df1 = pd.DataFrame(columns=column_names)

        NFS_path = '/var/log/images_bkp'
        desire_dir = str(NFS_path) + '/new_images'
        
        if not os.path.exists(desire_dir):
            os.makedirs(desire_dir) #this will create a new folder if it doesn't exist and start maintaining all the downloads

        for i in range(len(data1)):
            url = data1.iloc[i, 1]
            product = data1.iloc[i, 0]

            if url.endswith(('.JPEG', '.jpg', '.png', '.jpeg', '.JPG')):
                name = re.search(r'[a-zA-Z0-9-]*\.jpg', url) #using regular expression to extract out the image name from url
                if name:
                    filename = name.group()
                else:
                    filename = url.split('/')[-1]

                try:
                    # Set a valid user agent header
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)

                    # Download the image with retries
                    retries = 3
                    while retries > 0:
                        try:
                            urllib.request.urlretrieve(url, os.path.join(desire_dir, filename))
                            break
                        except (urllib.error.HTTPError, urllib.error.URLError) as e:
                            retries -= 1
                            print(f"Error downloading {url}. Retrying in 3 seconds... ({retries} retries left)")
                            time.sleep(3)
                            if retries == 0:
                                raise e

                    df1.loc[i] = [filename, product]
                    U.append(url)

                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    url = 0
                    df1.loc[i] = [0, product]
                    y.append(url)
                    continue
            else:
                df1.loc[i] = [0, product]

        return tuple(df1.values.tolist()),desire_dir


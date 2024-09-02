import PIL
import cv2 as cv2
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
import requests
import json, imagehash
from PIL import Image
from PIL.ExifTags import TAGS
import io
from helper import getConfigData
import os
import mysql.connector
from helper import getConfigData
import subprocess
import hashlib
from classes.mediainfo_video import VideoMeta
import torch
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import Dataset, DataLoader
from classes.manage_logs import Manage_Logs
import re
import logging

None_details = {
        "class": "0",
        "type": "0",
        "channel_statistics": {
            "red": {
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "standard deviation": 0.0,
                "kurtosis": 0.0,
                "skewness": 0.0,
                "entropy": 0.0,
            },
            "green": {
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "standard deviation": 0.0,
                "kurtosis": 0.0,
                "skewness": 0.0,
                "entropy": 0.0,
            },
            "blue": {
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "standard deviation": 0.0,
                "kurtosis": 0.0,
                "skewness": 0.0,
                "entropy": 0.0,
            }
        },
        "image_statistics": {
            "overall": {
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "standard deviation": 0.0,
                "kurtosis": 0.0,
                "skewness": 0.0,
                "entropy": 0.0,
            },
            "rendering_intent":"0",
            "gamma": 0.0,
            "background_color":"0",
            "border_color":"0",
            "matte_color":"0",
            "transparent_color":"0",
            "compression": "0",
            "quality": 0,
            "orientation": "0"
        },
        "properties": {
            "signature": "0"
        },
        "tainted": "0",
        "pixels_per_second": 0  # Assuming this value is static as it's not provided in the JSON
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_file_path = os.path.abspath(__file__)  # Get absolute path to current file
# Get parent's parent directory path:
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
print("parent_parent_dir = ", parent_parent_dir)
trained_model = torch.load(parent_parent_dir + '/models/trained_model-Kaggle_dataset')
trained_model = trained_model['model_state']

class ImageMeta():
    def __init__(self):
        pass
    
    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    def getSize(file_path, unit='bytes'):
        file_size = os.path.getsize(file_path)
        exponents_map = {'bytes':0,'kb':1,'mb':2,'gb':3}
        if unit not in exponents_map:
            raise ValueError("Must select from ['bytes','kb','mb','gb']")
        else:
            size = file_size / 1000 ** exponents_map[unit]
            return round(size, 3)
        
    def convert_pixels_per_second(pixels_per_second):
        match = None
        if(pixels_per_second is not None and isinstance(pixels_per_second, str)):
            match = re.match(r'(\d+(\.\d+)?)([KMB])?', pixels_per_second)
        elif(pixels_per_second is not None and isinstance(pixels_per_second, int)):
            return pixels_per_second
        if not match:
            raise ValueError("Invalid format of pixels_per_second")
        
        number = round(float(match.group(1)))
        unit = match.group(3)
        
        if unit == 'K':
            return number * 10**3
        elif unit == 'M':
            return number * 10**6
        elif unit == 'B':
            return number
        else:
            return number

    def is_image_blurry(img, threshold=0.5):
        feature_extractor = featureExtractor()
        accumulator = []

        # Resize the image by the downsampling factor
        feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])
        # print(np.shape(img)[0])
        # print(np.shape(img)[1])
        # compute the image ROI using local entropy filter
        feature_extractor.compute_roi()

        # extract the blur features using DCT transform coefficients
        extracted_features = feature_extractor.extract_feature()
        extracted_features = np.array(extracted_features)

        if(len(extracted_features) == 0):
            return True
        test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

        # trained_model.test()
        for batch_num, input_data in enumerate(test_data_loader):
            x = input_data
            x = x.to(device).float()

            output = trained_model(x)
            _, predicted_label = torch.max(output, 1)
            accumulator.append(predicted_label.item())
        
        # prediction= np.mean(accumulator) < threshold
        prediction= np.mean(accumulator)
        return(prediction)
    
    def avgHashValueExtract(local_path):
        # data = df1
        # product_id = data['product_id']
        # docid = data['docid']

        if local_path==0 or local_path == '':
            result = int(0)
        else:
            image = Image.open(local_path)
            ahash_value = imagehash.average_hash(image)
            dhash_value = imagehash.dhash(image)
            phash_value = imagehash.phash(image)
            # chash_value = imagehash.colorhash(image)
            chash_value = imagehash.colorhash(image)
            # print(ahash_value)
            
            aresult = str(ahash_value)
            dresult = str(dhash_value)
            presult = str(phash_value)
            cresult = str(chash_value)

            metadata_ImageMagick = ImageMeta.image_sys_details(local_path)
            signature = metadata_ImageMagick["properties"]["signature"]

            plaintext = aresult + dresult + presult + cresult + signature
            dup_signature = VideoMeta.shake_128(plaintext)
        return aresult, dresult, presult, cresult, dup_signature
        # , cresult
        
    def hashValueExtract(data):
        product_id = data['produc4t_id']
        print("product_id:",product_id)
        docid = data['docid']
        local_path = data['image_path']
        # image_name = data['image_name_1'].iloc[0]

        if local_path==0:
            result = int(0)

        else:
            hash_url = getConfigData('content_processing_api.hash_url')
            url = str(hash_url)+str(product_id)+'&docid='+str(docid)+'&localpath='+str(local_path)

            payload={}
            headers = {}

            response = requests.request("GET", url, headers=headers, data=payload)

            # Parse the JSON data from the API response
            json_data = json.loads(response.text)
            
            # Extract the "result" value
            result = json_data.get('result')

        return result

    def dictConvert(final_data,path):
        #print(final_data.columns)
        data = final_data
        # color = color_data
        local_path = path
        data_dict = data.to_dict(orient='records')

        parent_dict = {}
        derived_dict = {}
        meta_data = {}

        for i in range(len(data_dict)):
            # meta = {parent_dict,derived_dict}
            parent_dict[i] = {
                'image_name': data_dict[i]['image_name_1'],
                'height': data_dict[i]['height_1'],
                'height_ori':data_dict[i]['height_ori_1'],
                'width': data_dict[i]['width_1'],
                'width_ori': data_dict[i]['width_ori_1'],
                'size': data_dict[i]['size_1'],
                'size_ori': data_dict[i]['size_ori_1'],
                'image_format': data_dict[i]['img_format_1'],
                'image_mode': data_dict[i]['img_mode_1'],
                'description': data_dict[i]['description_1'],
                'author': data_dict[i]['author_1'],
                'keywords': data_dict[i]['keywords_1'],
                'copyright': data_dict[i]['copyright_1'],
                'location': data_dict[i]['location_1'],
                'image_shape':str(data_dict[i]['image_shape_1']),
                # 'data_from':data_dict[i]['flag']
                }
            derived_dict[i] = {
                'pixels': float(data_dict[i]['resolution_1']),
                'megapixels': data_dict[i]['megapixels_1'],
                'ppi': data_dict[i]['ppi_1'],
                'exif_dict': str(data_dict[i]['exif_dict_1']),
                'red': data_dict[i]['red_1'],
                'green': data_dict[i]['green_1'],
                'blue': data_dict[i]['blue_1'],
                # 'laplacian_variance_blur': data_dict[i]['laplacian_variance_blur_1'],
                # 'fourier_transform_blur': data_dict[i]['fourier_transform_blur_1'],
                # 'gradient_magnitude_blur': data_dict[i]['gradient_magnitude_blur_1'],
                # 'brightness_score':data_dict[i]['brightness_score_1'],
                'colourfulness':data_dict[i]['colourfulness_1'],
                'sharpness_score':data_dict[i]['sharpness_score_1'],
                'image_metric': data_dict[i]['image_metric_1'],
                # 'hash_value':data_dict[i]['hash_value'],
                'duplicate': data_dict[i]['duplicate_1']
                }
            
            #this is the part where the process flag is set and if this is not set properly then the dataflow into mongo won't be done properly
            if data_dict[i]['status']=='error':
                meta_data[i] = {'docid': data_dict[i]['docid'],'product_id': data_dict[i]['product_id'],'process_flag':str(32), 'local_path': str(local_path),'business_tag': data_dict[i]['business_tag'], 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i]}}

            elif data_dict[i]['status']=='corrupt':
                meta_data[i] = {'docid': data_dict[i]['docid'],'product_id': data_dict[i]['product_id'],"process_flag":str(32), 'local_path': str(local_path),'business_tag': data_dict[i]['business_tag'], 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i]}}
            
            elif data_dict[i]['status']=='empty':
                meta_data[i] = {'docid': data_dict[i]['docid'],'product_id': data_dict[i]['product_id'],"process_flag":str(32), 'local_path': str(local_path),'business_tag': data_dict[i]['business_tag'], 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i]}}
            
            elif data_dict[i]['status']=='good':
                meta_data[i] = {'docid': data_dict[i]['docid'],'product_id': data_dict[i]['product_id'],"process_flag":str(2),  'local_path': str(local_path),'business_tag': data_dict[i]['business_tag'], 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i]}}

        return meta_data

    def imageMetatag(df1):
        # #df1 is dict
        queue_data = df1
        docid = queue_data['docid']
        product_id = queue_data['product_id']
        business_tag = queue_data['business_tag']

        # if queue_data['path_flag'] == '1':
        #     image_filename = str(queue_data['local_path'])
        #     file_name = os.path.basename(queue_data['local_path'])
        #     NFS_path = getConfigData('NFS_path.image_folder')
        #     # print(NFS_path)
        #     # image_path = str(NFS_path) + image_filename
        #     image_path = str(image_filename)
        #     #print('data1\n',image_path)
        # else:
        image_filename = str(queue_data['local_path'])
        file_name = os.path.basename(queue_data['local_path'])
        # print(image_filename)
        NFS_path = getConfigData('NFS_path.path')
        # image_path = str(NFS_path) + '/new_images/' + image_filename
        image_path = str(image_filename)
            #print('data2\n',image_path)


        # image_filename = str(queue_data.iloc[0,0])
        # print(image_filename)
        # NFS_path = getConfigData('NFS_path_sandbox.path')
        # image_path = str(NFS_path) + '/new_images/' + image_filename

        column_names = ['image_name_1','height_1','width_1','resolution_1','megapixels_1','ppi_1','size_1','img_format_1',
        'img_mode_1','exif_dict_1','description_1','keywords_1','author_1','copyright_1','location_1','red_1','green_1','blue_1','image_shape_1','colourfulness_1','sharpness_score_1','size_ori_1','width_ori_1','height_ori_1', 'duplicate_1', 'image_metric_1','status','docid','product_id','business_tag','flag']
        df2 = pd.DataFrame(columns=column_names)

        # ori_url = queue_data.iloc[0, 5]
        ori_url = queue_data['product_url_ori']

        mydb = mysql.connector.connect(
        host= getConfigData('mysql_17_132.host'),
        user= getConfigData('mysql_17_132.username'),
        password= getConfigData('mysql_17_132.password')
        )

        mycursor = mydb.cursor()
        mycursor.execute("SELECT product_id,ori_width,ori_height,ori_size FROM db_product.tbl_catalogue_image_info WHERE product_id = %s",[product_id])

        myresult = mycursor.fetchall()

        mydb = mysql.connector.connect(
            host= getConfigData('mysql_17_132.host_master'),
            user= getConfigData('mysql_17_132.username'),
            password= getConfigData('mysql_17_132.password')
        )

        mycursor = mydb.cursor()
        if myresult:
        # Data is present in the database
            product_id_db, width_ori_db, height_ori_db, size_ori_db = myresult[0]
            # print(product_id_db)

            # Check if any of the values are missing
            if width_ori_db is None and height_ori_db is None and size_ori_db is None:
                # Fetch missing data from the URL
                try:
                    response = requests.get(ori_url)
                    if response.status_code == 200:
                        headers = response.headers
                        size_ori = float(headers.get('content-length', ''))
                        image_data = io.BytesIO(response.content)

                        try:
                            # Attempt to open the image using Pillow
                            image = Image.open(image_data)
                            print(image)
                            # Extract the height and width of the image using Pillow
                            width_ori, height_ori = image.size

                        except (OSError, PIL.UnidentifiedImageError):
                            # If Pillow fails to open the image, try using cv2 method
                            image_np = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
                            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                            # Extract the height and width of the image using cv2
                            height_ori, width_ori = img.shape

                        # Update the database with the missing values
                        update_query = "UPDATE db_product.tbl_catalogue_image_info SET ori_width = %s, ori_height = %s, ori_size = %s WHERE product_id = %s"
                        mycursor.execute(update_query, (width_ori, height_ori, size_ori, product_id))
                        mydb.commit()

                    else:
                        print(f"Failed to fetch data from the URL. Status code: {response.status_code}")
                        size_ori = 0
                        width_ori = 0
                        height_ori = 0

                except requests.exceptions.RequestException as e:
                    print(f"Error occurred during the request: {e}")
                    size_ori = 0
                    width_ori = 0
                    height_ori = 0

                except OSError as e:
                    print(f"Error occurred while processing the image: {e}")
                    size_ori = 0
                    width_ori = 0
                    height_ori = 0
                flag = 0

            else:
                
                # All values are present, use them
                width_ori = width_ori_db
                height_ori = height_ori_db
                size_ori = size_ori_db
                flag = 1
            
        else:   
            try:
                response = requests.get(ori_url)
                if response.status_code == 200:
                    headers = response.headers
                    size_ori = float(headers.get('content-length', ''))
                    image_data = io.BytesIO(response.content)

                    try:
                        # Attempt to open the image using Pillow
                        image = Image.open(image_data)
                        print(image)
                        # Extract the height and width of the image using Pillow
                        width_ori, height_ori = image.size

                    except (OSError, PIL.UnidentifiedImageError):
                        # If Pillow fails to open the image, try using cv2 method
                        image_np = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
                        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                        # Extract the height and width of the image using cv2
                        height_ori, width_ori = img.shape

                    # Update the database with the missing values
                    update_query = "UPDATE db_product.tbl_catalogue_image_info SET ori_width = %s, ori_height = %s, ori_size = %s WHERE product_id = %s"
                    mycursor.execute(update_query, (width_ori, height_ori, size_ori, product_id))
                    mydb.commit()

                else:
                    print(f"Failed to fetch data from the URL. Status code: {response.status_code}")
                    size_ori = 0
                    width_ori = 0
                    height_ori = 0

            except requests.exceptions.RequestException as e:
                print(f"Error occurred during the request: {e}")
                size_ori = 0
                width_ori = 0
                height_ori = 0

            except OSError as e:
                print(f"Error occurred while processing the image: {e}")
                size_ori = 0
                width_ori = 0
                height_ori = 0
            flag = 0


        try:
            # cv2 method used to extract the color vectors, image matrix, and blur values
            with open(image_path, 'rb') as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    Manage_Logs.input_request(local_path=image_path, product_url=None, api_name = 'blur_v3', e="image corrupted or absent")
                    raise ValueError(f"Error: Failed to read image: {image_path}, image is Nonetype, type(img)={type(img)}") 


                r, g, b = cv2.split(img)
                # print('inside with')

                if len(img.shape) == 2:  # grayscale image
                    r_vector = img.ravel()
                    g_vector = img.ravel()
                    b_vector = img.ravel()
                else:  # image color
                    b, g, r = cv2.split(img)
                    r_vector = r.ravel()
                    g_vector = g.ravel()
                    b_vector = b.ravel()
                mean_r = np.mean(r_vector)
                mean_g = np.mean(g_vector)
                mean_b = np.mean(b_vector)
                img_array = np.array(img)
                # print(img_array.shape) 


                matrix = img_array[:,:,0]
                matrix_shape = matrix.shape
                # print(len(matrix))

                size = ImageMeta.getSize(image_path, 'bytes')

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Calculate gradients using Sobel operator
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate Tenengrad (gradient magnitude)
                tenengrad = np.sqrt(grad_x**2 + grad_y**2)

                # Calculate the sharpness score as the average of Tenengrad values
                sharpness_score = float(np.mean(tenengrad))
                # print('Sharpness_score:', sharpness_score)

                # Pil method used to extract remaining metatags
                with open(image_path, 'rb') as f:
                    image = PIL.Image.open(io.BytesIO(f.read()))

                    image_np = np.array(image).astype(np.float32) / 255.0

                    if image_np.ndim == 2:  # Grayscale image
                        rg = image_np - image_np
                        yb = 0.5 * (image_np + image_np) - image_np
                    else:  # Color image
                        rg = image_np[:, :, 0] - image_np[:, :, 1]
                        yb = 0.5 * (image_np[:, :, 0] + image_np[:, :, 1]) - image_np[:, :, 2]

                    std_rg = np.std(rg)
                    std_yb = np.std(yb)

                    mean_rg = np.mean(rg)
                    mean_yb = np.mean(yb)

                    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
                    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)

                    colourfulness = (std_root + 0.3 * mean_root) * 100

                    wid, hgt = image.size
                    ppi_formula = (hgt*wid)**0.5/(3*4)
                    ppi = round(ppi_formula)
                    resolution = (wid*hgt)
                    megapixels = resolution/1000000
                    
                    img_format = str(image.format)
                    # Get the image mode (e.g. RGB, CMYK)
                    img_mode = str(image.mode)

                    # Get the EXIF data (if available)
                    exif_data = image.getexif()
                    exif_dict = {}
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            exif_dict[tag] = value

                    # Extract additional metadata using the IPTC profile
                    iptc = image.info.get("iptc", {})
                    description = str(iptc.get((2, 120), None))
                    keywords = str(iptc.get((2, 25), None))
                    author = str(iptc.get((2, 80), None))
                    copyright = str(iptc.get((2, 116), None))
                    location = str(iptc.get((2, 92), None))


                    # df2, image_name = meta_tag.imageMetatag(df1)
                    # d = getConfigData('NFS_path.image_folder') + str(image_name)
                    # print(df2, d)
                
                    # the process to calculate image_metric using the below mentioned columns
                    try:
                        image_metric_1 = np.sum(np.array(matrix.flatten())) + size + mean_b + mean_g + mean_r + ppi
                        duplicate_1 = 0
                    except:
                        image_metric_1 = 0.0
                        duplicate_1 = 0

                    df2 = pd.DataFrame([[
                    image_filename, hgt, wid, resolution, megapixels, ppi, size, img_format, img_mode, exif_dict, description, keywords, author, copyright, location, mean_r, mean_g, mean_b, matrix_shape, colourfulness, sharpness_score, size_ori, width_ori, height_ori, duplicate_1,image_metric_1,'good',docid,product_id,business_tag,flag]], columns=column_names)
        
        except (AttributeError, TypeError) as e:
            print(f"Error while processing {image_filename}: {e}")
            df2 = pd.DataFrame([[image_filename, 0, 0, 0, 0.0, 0, 0.0, '0', '0', '0', '0', '0', '0', '0', '0', 0.0, 0.0, 0.0, '0', 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 'error', docid,product_id, business_tag,0]], columns=column_names)
            raise ValueError(f"Error: [{e}]: {image_path}")
        
        except (cv2.error, ValueError) as e:
            print(f"Error while processing {image_filename}: {e}")
            df2 = pd.DataFrame([[image_filename, 0, 0, 0, 0.0, 0, 0.0, '0', '0', '0', '0', '0', '0', '0', '0', 0.0, 0.0, 0.0, '0', 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 'corrupt', docid,product_id, business_tag,0]], columns=column_names)
            raise ValueError(f"Error: [{e}]: {image_path}")




        df2 = ImageMeta.dictConvert(df2,image_path)
        # print(df2)
        data22 =ImageMeta.safeSerialize(df2[0])

        data_dict = json.loads(data22)
        # print("aa: ", data22["meta"]["parent"]["image_statistics"])
        # print("data_dict: ", data_dict)
        payload = json.dumps(data_dict)
        return payload


    def get_image_extension(path):
        _, extension = os.path.splitext(path)
        if extension.lower() in (".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp", ".avif", ".tif", ".webp"):
            return extension
        else:
            return None
    
    def check_image_corruption(file_path):
        try:
            img = Image.open(file_path)
            img.verify()  # Raises an error if the image is corrupted
            return False  # Not corrupted
        except (IOError, OSError, Image.DecompressionBombError) as error:
            print("@@@Image corrupted: ", error)
            return True  # Corrupted
        
    def color_channel_validation(extracted_details, ext_key1, ext_key2, all_details, key1, key2):
        if(key1 in all_details and key2 in all_details[key1]):
            parameters = ["min", "max", "mean", "standard deviation", "kurtosis", "skewness", "entropy"]
            for parameter in parameters:
                if(parameter in all_details[key1][key2]):
                    if(parameter in parameters[:2]):
                        extracted_details[ext_key1][ext_key2][parameter] = int(all_details[key1][key2][parameter].split(' ')[0])
                    elif(parameter in parameters[2:4]):
                        extracted_details[ext_key1][ext_key2][parameter] = float(all_details[key1][key2][parameter].split(' ')[0])
                    else:
                        extracted_details[ext_key1][ext_key2][parameter] = float(all_details[key1][key2][parameter])

    def image_sys_details(path):
        # print(path)
        extension = ImageMeta.get_image_extension(path)
        # print("file Extension: ", extension)
        if(extension in [".avif", ".bmp", ".tiff", ".ico", ".gif", ".svg", ".ai", ".eps", ".pdf", ".psd", ".raw", ".heic"]):
            logging.info(f"extension not supported {extension}")
            return None_details
        
        if ImageMeta.check_image_corruption(path):
            logging.info("image corrupted")
            return None_details

        if(os.path.exists(path)):
            # print("path does exists!")
            try:
                command = ["identify", "-verbose", path]
                details = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)

            except: 
                logging.info("Problem with ImageMagick library")
                return None_details
        else:
            logging.info("path not exists!")
            return None
        
        lines = details.splitlines()
        # temp = [(len(line) - len(line.lstrip(' ')), line.split(':', 1)[0].strip(), line.split(':', 1)[1].strip()) for line in lines[1:]]
        try:
            temp = []
            for line in lines[1:]:
                if(len(line)==0):
                    continue
                stripped_line = line.lstrip(' ')
                word_count = len(line) - len(stripped_line)
                key, value = stripped_line.split(':', 1)
                temp.append((word_count, key.strip(), value.strip()))
            keys = []
            all_details ={}
            for zeros, string_a, string_b in temp:
                # print(zeros, string_a, string_b)
                if(len(string_b)!=0):
                    while(len(keys) and keys[-1][1]>=zeros):
                        keys = keys[:-1]
                    keys.append([string_a, zeros])
                    consecutive_keys = keys
                    curr_dict = all_details
                    for i in range(len(consecutive_keys) - 1):
                        key = consecutive_keys[i][0]
                        if not isinstance( curr_dict, str):
                            curr_dict = curr_dict[key]
                    # Assign the value to the final key in the nth nested dictionary
                    if not isinstance( curr_dict, str):
                        curr_dict[consecutive_keys[-1][0]] = string_b
                else:
                    while(len(keys) and keys[-1][1]>=zeros):
                        # keys = keys.pop()
                        keys = keys[:-1]
                    keys.append([string_a, zeros])
                    curr_dict = all_details
                    for i in range(len(keys) - 1):
                        key = keys[i][0]
                        curr_dict = curr_dict[key]
                    # Assign the value to the final key in the nth nested dictionary
                    if not isinstance( curr_dict, str):
                        curr_dict[keys[-1][0]] = {}
        except Exception as e:
            return None_details
        # print("all_details:\n", json.dumps(all_details, indent=4))
        number_pixels = all_details.get("Number pixels", 0)
        number_pixels = ImageMeta.convert_pixels_per_second(number_pixels)
        pixels_per_second = all_details.get("Pixels per second", 0)
        pixels_per_second = ImageMeta.convert_pixels_per_second(pixels_per_second)
        try:
            # Extract the required details
            extracted_details = {
            "class": all_details.get("Class", "0"),
            "type": all_details.get("Type", "0"),
            "channel_statistics": {
                "red": {},
                "green": {},
                "blue": {}
            },
            "image_statistics": {
                "overall": {},
                "rendering_intent": all_details.get("Rendering intent", "0"),
                "gamma": float(all_details.get("Gamma", )),
                "background_color": all_details.get("Background color", "0"),
                "border_color": all_details.get("Border color", "0"),
                "matte_color": all_details.get("Matte color", "0"),
                "transparent_color": all_details.get("Transparent color", "0"),
                # "Interlace": all_details["Interlace"],
                # "Intensity": all_details["Intensity"],
                # "Compose": all_details["Compose"],
                # "Compression": all_details["Compression"],
                # "Compression": all_details["Compression"] if(extension in ['.webp', '.png', '.jpg', '.jpeg', '.tif'] and all_details["Compression"] != "Undefined") else None,
                # "Quality": int(all_details["Quality"]) if(extension in ['.jpeg', '.jpg']) else None,
                # "Orientation": all_details["Orientation"] if(extension in [ '.tif']) else None

                "compression": all_details.get("Compression", "0") if(all_details["Compression"] != "Undefined") else "0",
                "quality": int(all_details.get("Quality", 0)) if(all_details.get("Quality", 0)!= "Undefined" and all_details.get("Quality", 0) != 0) else 0,
                "orientation": all_details.get("Orientation", "0") if(all_details["Orientation"] != "Undefined") else "0"
            },
            "properties": {
                # "date:create": all_details["Properties"]["date"],
                # "date:modify": all_details["Properties"]["date:modify"],
                # "jpeg:colorspace": all_details["Properties"]["jpeg"],
                "signature": VideoMeta.shake_128(all_details.get("Properties", "0").get("signature", "0"))
            },
            "tainted": all_details.get("Tainted", "0"),
            "number_pixels": number_pixels,
            "pixels_per_second": pixels_per_second
            }
            # int(all_details.get("Number pixels", 0))  # Assuming this value is static as it's not provided in the JSON
            ImageMeta.color_channel_validation(extracted_details, "channel_statistics", "red", all_details, "Channel statistics", "Red")
            ImageMeta.color_channel_validation(extracted_details, "channel_statistics", "green", all_details, "Channel statistics", "Green")
            ImageMeta.color_channel_validation(extracted_details, "channel_statistics", "blue", all_details, "Channel statistics", "Blue")
            ImageMeta.color_channel_validation(extracted_details, "image_statistics", "overall", all_details, "Image statistics", "Overall")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None_details
        # extracted_details = ImageMeta.remove_null_keys(extracted_details)
        return extracted_details

    def calculate_aspect(width, height):
        def gcd(a, b):
            return a if b == 0 else gcd(b, a % b)

        r = gcd(width, height)
        x = int(width / r)
        y = int(height / r)
        return("%d:%d" % (x, y))
    
    def fetchImageHeightWidth(image_path):
        with open(image_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        image_height, image_width, _ = img.shape # get image height and width
        size = ImageMeta.getSize(image_path)
        os.remove(image_path)
        out = {
            'size': int(size),
            'height': image_height,
            'width': image_width,
            'aspect_ratio': ImageMeta.calculate_aspect(image_width,image_height)
        }
        return out
    
    def fetchImageHeightWidthV2(image_url):
        response = requests.get(image_url, stream=True)
        chunk_size = 1024  # You can adjust this based on the image file's format
        if response.status_code == 200:
            # Open the image using PIL (Pillow)
            with Image.open(io.BytesIO(response.content)) as img:
                width, height = img.size
                size = int(response.headers.get('Content-Length', 0))
        else:
            print("Failed to fetch the image")
            output = {
                'error_code': 1,
                'url' : image_url,
                'msg' : 'Invalid url or not exist'
            }
            return output
        # Close the response
        out = {
            'error_code': 0,
            'size': size,
            'height': height,
            'width': width,
            'aspect_ratio': ImageMeta.calculate_aspect(width,height)
        }
        return out
    
    def calculateVideoInfo(video_url):
        cap = cv2.VideoCapture(video_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
        duration = frame_count/fps
        minutes = int(duration/60)
        seconds = int(duration%60)
        # print('duration in (M:S) = ' + str(minutes) + ':' + str(seconds))
        cap.release()
        out = {
            'duration_in_s': str(round(duration,2)),
            'duration(M:S)': str(minutes) + ':' + str(seconds),
            'width': str(width),
            'height': str(height)
        }
        return out

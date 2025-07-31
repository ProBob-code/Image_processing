import PIL
import os
import numpy as np
import pandas as pd
import re
import cv2 as cv2
from flask import request as req
import json
from PIL.ExifTags import TAGS
from flask import Flask, render_template, request
from flask import Flask
import requests
import json
from PIL import Image
import io
from src.libs.helpers import getConfigData
import os

import extcolors
from colormap import rgb2hex

from classes.manage_logs import Manage_Logs
from PIL import Image
from io import BytesIO
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MetaTag():
    def __init__(self):
        pass


    @staticmethod
    def hash_value_extract(df1,d):
        data = df1

        print(data)

        product_id = data['product_id_1'].iloc[0]
        print("product_id:",product_id)
        docid = data['docid_1'].iloc[0]
        local_path = d
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

    @staticmethod
    def extract_brightness_matrix(image_path):
    # Open the image
        image = Image.open(image_path)

        # Convert the image to grayscale
        grayscale_image = image.convert("L")

        # Extract the pixel values as a 2D matrix
        brightness_matrix = list(grayscale_image.getdata())
        width, height = grayscale_image.size
        brightness_matrix = [brightness_matrix[i:i+width] for i in range(0, len(brightness_matrix), width)]
        
        return brightness_matrix

    @staticmethod
    def calculate_overall_brightness(brightness_matrix):
        total_pixels = 0
        brightness_sum = 0

        # Iterate over each pixel in the brightness matrix
        for row in brightness_matrix:
            for pixel in row:
                brightness_sum += pixel
                total_pixels += 1

        # Calculate the average brightness
        overall_brightness = brightness_sum / total_pixels

        return overall_brightness


    # Function to extract the size of an image
    @staticmethod
    def getSize(file_path, unit='bytes'):
        file_size = os.path.getsize(file_path)
        exponents_map = {'bytes':0,'kb':1,'mb':2,'gb':3}
        if unit not in exponents_map:
            raise ValueError("Must select from ['bytes','kb','mb','gb']")
        else:
            size = file_size / 1000 ** exponents_map[unit]
            return round(size, 3)

    
    # Function to extract blur values like - Laplacian Blur, Fourier Blur and Gradient Blur
    @staticmethod
    def blurValues(img):
        
        np.random.seed(0)

        lap = []
        fourier = []
        gradient = []
        # Define the grid parameters
        grid_size = 4 # number of rows and columns in the grid
        image_height, image_width, _ = img.shape # get image height and width
        grid_height = int(image_height / grid_size) # height of each grid box
        grid_width = int(image_width / grid_size) # width of each grid box

        # Draw the grid on the image
        for i in range(0, image_height, grid_height):
            cv2.line(img, (0, i), (image_width, i), (0, 255, 0), 1)
        for j in range(0, image_width, grid_width):
            cv2.line(img, (j, 0), (j, image_height), (0, 255, 0), 1)


        # Calculate the blur level for randomly selected grid boxes
        selected_boxes = np.random.choice(grid_size * grid_size, size=10, replace=False)

        for box_index in selected_boxes:
            i, j = divmod(box_index, grid_size)
            # get the ROI for the current grid box
            x1, y1 = j * grid_width, i * grid_height
            x2, y2 = x1 + grid_width, y1 + grid_height
            roi = img[y1:y2, x1:x2]
            # calculate the Laplacian variance
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            lap.append(laplacian_var)

            # calculate the Fourier transform
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dft_roi = cv2.dft(np.float32(gray_roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            mag_dft_roi = 20 * np.log(cv2.magnitude(dft_roi[:, :, 0], dft_roi[:, :, 1]))
            blur = np.mean(mag_dft_roi)
            fourier.append(blur)

            # calculate the gradient magnitude
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)

            # calculate a blur metric from the gradient magnitude values
            blur_metric = np.mean(mag)
            gradient.append(blur_metric)

        avg_blur1 = np.std(lap)
        Laplacian_variance_blur = avg_blur1

        avg_blur2 = np.std(fourier)
        Fourier_Transform_blur = avg_blur2

        avg_blur3 = np.std(gradient)
        Gradient_magnitude_blur = avg_blur3

        return Laplacian_variance_blur,Fourier_Transform_blur,Gradient_magnitude_blur


    # Function to extract metatag values
    @staticmethod
    def imageMetatag(df1):
        queue_data = df1


        if queue_data['path_flag_1'].iloc[0].item() == 1:
            image_filename = str(df1.iloc[0,0])
            print(image_filename)
            NFS_path = getConfigData('NFS_path.image_folder')
            image_path = str(NFS_path) + image_filename
            #print('data1\n',image_path)
        else:
            image_filename = str(df1.iloc[0,0])
            print(image_filename)
            NFS_path = getConfigData('NFS_path.path')
            image_path = str(NFS_path) + '/new_images/' + image_filename
            #print('data2\n',image_path)


        # image_filename = str(queue_data.iloc[0,0])
        # print(image_filename)
        # NFS_path = getConfigData('NFS_path_sandbox.path')
        # image_path = str(NFS_path) + '/new_images/' + image_filename

        column_names = ['image_name_1','height_1','width_1','resolution_1','megapixels_1','ppi_1','size_1','img_format_1',
        'img_mode_1','exif_dict_1','description_1','keywords_1','author_1','copyright_1','location_1','laplacian_variance_blur_1',
        'fourier_transform_blur_1','gradient_magnitude_blur_1','red_1','green_1','blue_1','image_shape_1','matrix_1','brightness_score_1','colourfulness_1','sharpness_score_1','size_ori_1','width_ori_1','height_ori_1','status']
        df2 = pd.DataFrame(columns=column_names)

        ori_url = queue_data.iloc[0, 5]

        try:
            response = requests.get(ori_url)
            if response.status_code == 200:
                headers = response.headers
                size_ori = float(headers.get('content-length', ''))
                image_data = BytesIO(response.content)

                try:
                    # Attempt to open the image using Pillow
                    image = Image.open(image_data)

                    # Extract the height and width of the image using Pillow
                    width_ori, height_ori = image.size

                except (OSError, PIL.UnidentifiedImageError):
                    # If Pillow fails to open the image, try using cv2 method
                    image_np = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
                    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

                    # Extract the height and width of the image using cv2
                    height_ori, width_ori = img.shape

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

        try:
            # cv2 method used to extract the color vectors, image matrix, and blur values
            with open(image_path, 'rb') as f:
                img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    Manage_Logs.input_request(local_path=image_path, product_url=None, api_name = 'blur_v3', e="image corrupted or absent")
                    raise ValueError(f"Error: Failed to read image: {image_path}, image is Nonetype, type(img)={type(img)}") 

  
                r, g, b = cv2.split(img)

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
                print(img_array.shape) 


                matrix = img_array[:,:,0]
                matrix_shape = matrix.shape

                Laplacian_variance_blur, Fourier_Transform_blur, Gradient_magnitude_blur = MetaTag().blurValues(img)
                H = image_path
                size = MetaTag().getSize(H, 'bytes')

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Calculate gradients using Sobel operator
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

                # Calculate Tenengrad (gradient magnitude)
                tenengrad = np.sqrt(grad_x**2 + grad_y**2)

                # Calculate the sharpness score as the average of Tenengrad values
                sharpness_score = float(np.mean(tenengrad))
                print('Sharpness_score:', sharpness_score)

                # Pil method used to extract remaining metatags
                with open(H, 'rb') as f:
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
                    print('Colorfulness:', colourfulness)

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

                    brightness_matrix = MetaTag().extract_brightness_matrix(H)
                    overall_brightness = MetaTag().calculate_overall_brightness(brightness_matrix)

                    print('overall_brightness:',overall_brightness)


                    df2 = pd.DataFrame([[
    image_filename, hgt, wid, resolution, megapixels, ppi, size, img_format, img_mode, exif_dict,
    description, keywords, author, copyright, location, Laplacian_variance_blur, Fourier_Transform_blur,
    Gradient_magnitude_blur, mean_r, mean_g, mean_b, matrix_shape, matrix, overall_brightness, colourfulness, sharpness_score, size_ori, width_ori, height_ori,'good']], columns=column_names)
        
        except (AttributeError, TypeError) as e:
            print(f"Error while processing {image_filename}: {e}")
            df2 = pd.DataFrame([[image_filename, 0, 0, 0, 0.0, 0, 0.0, '0', '0', '0', '0', '0', '0', '0', '0', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '0', '0', 0.0, 0.0, 0.0, 0.0, 0, 0, 'error']], columns=column_names)

        except (cv2.error, ValueError) as e:
            print(f"Error while processing {image_filename}: {e}")
            df2 = pd.DataFrame([[image_filename, 0, 0, 0, 0.0, 0, 0.0, '0', '0', '0', '0', '0', '0', '0', '0', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '0', '0', 0.0, 0.0, 0.0, 0.0, 0, 0, 'corrupt']], columns=column_names)
    
        # print(df2['gradient_magnitude_blur_1'],image_filename)

        return df2,image_filename

    @staticmethod
    def color_to_df(input):
        colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
        df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
        df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

        # Convert RGB to HEX code
        df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                                int(i.split(", ")[1]),
                                int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

        df = pd.DataFrame(zip(df_color_up, df_percent),
                            columns=['c_code', 'occurence'])
        return df

    @staticmethod
    def hex_to_rgb(hex_code):
        hex_code = str(hex_code).strip('#')  # Remove '#' symbol
        r = int(hex_code[0:2], 16)  # Convert red component from hex to decimal
        g = int(hex_code[2:4], 16)  # Convert green component from hex to decimal
        b = int(hex_code[4:6], 16)  # Convert blue component from hex to decimal
        return r, g, b



    @staticmethod
    def extract_color(folder_path, path, tolerance, final_data):
        result_rows = []  # List to store the result rows
        # output_folder = getConfigData('NFS_path.path')
        path = path

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.JPEG', '.jpg', '.png', '.jpeg', '.JPG', '.gif', '.PNG', '.avif', '.webp', '.heif', '.heic'))]

        if len(image_files) == 0:
            return [{
                "hex": "0",
                "percentage": 0.0,
                "rbg_values": "0"
            }]

        elif path == 0 or path is None:
            return [{
                "hex": "0",
                "percentage": 0.0,
                "rbg_values": "0"
            }]

        elif final_data['status'][0]=='corrupt':
            return [{
                "hex": "0",
                "percentage": 0.0,
                "rbg_values": "0"
            }]


        # Create dataframe
        colors_x = extcolors.extract_from_path(path, tolerance=tolerance, limit=13)
        df_color = MetaTag().color_to_df(colors_x)

        list_color = list(df_color['c_code'])
        rgb_values = [MetaTag().hex_to_rgb(hex_code) for hex_code in list_color]
        list_precent = [int(i) for i in list(df_color['occurence'])]
        text_c = [{'hex': str(c), 'percentage': round(p * 100 / sum(list_precent), 1), 'rbg_values': str(r)} for c, p, r in
                zip(list_color, list_precent, rgb_values)]

        result_row = text_c
        result_rows.append(result_row)

        # Create final dictionary with the desired format
        output_dict = result_rows
        rearranged_dict = output_dict[0]

        return rearranged_dict



    # Convert the data from Dataframe to dictionary format
    @staticmethod
    def dictConvert(final_data,color_data,path):
        
        #print(final_data.columns)
        data = final_data
        color = color_data
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
                'image_shape':data_dict[i]['image_shape_1']
                }
            derived_dict[i] = {
                'pixels': data_dict[i]['resolution_1'],
                'megapixels': data_dict[i]['megapixels_1'],
                'ppi': data_dict[i]['ppi_1'],
                'exif_dict': data_dict[i]['exif_dict_1'],
                'red': data_dict[i]['red_1'],
                'green': data_dict[i]['green_1'],
                'blue': data_dict[i]['blue_1'],
                'laplacian_variance_blur': data_dict[i]['laplacian_variance_blur_1'],
                'fourier_transform_blur': data_dict[i]['fourier_transform_blur_1'],
                'gradient_magnitude_blur': data_dict[i]['gradient_magnitude_blur_1'],
                'brightness_score':data_dict[i]['brightness_score_1'],
                'colourfulness':data_dict[i]['colourfulness_1'],
                'sharpness_score':data_dict[i]['sharpness_score_1'],
                'image_metric': data_dict[i]['image_metric_1'],
                'hash_value':data_dict[i]['hash_value'],
                'duplicate': data_dict[i]['duplicate_1']
                }
            
            #this is the part where the process flag is set and if this is not set properly then the dataflow into mongo won't be done properly
            if data_dict[i]['status']=='error':
                meta_data[i] = {'docid': data_dict[i]['docid_1'],'product_id': data_dict[i]['product_id_1'],'process_flag':str(32), 'localpath': str(local_path),'business_tag': data_dict[i]['business_tag_1'], 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i], 'color': color}}

            elif data_dict[i]['status']=='corrupt':
                meta_data[i] = {'docid': data_dict[i]['docid_1'],'product_id': data_dict[i]['product_id_1'],"process_flag":str(32), 'localpath': str(local_path),'business_tag': data_dict[i]['business_tag_1'], 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i], 'color': color}}
            
            elif data_dict[i]['status']=='empty':
                meta_data[i] = {'docid': data_dict[i]['docid_1'],'product_id': data_dict[i]['product_id_1'],"process_flag":str(32), 'localpath': str(local_path),'business_tag': data_dict[i]['business_tag_1'], 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i], 'color': color}}
            
            elif data_dict[i]['status']=='good':
                meta_data[i] = {'docid': data_dict[i]['docid_1'],'product_id': data_dict[i]['product_id_1'],"process_flag":str(2),  'localpath': str(local_path),'business_tag': data_dict[i]['business_tag_1'], 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i], 'color': color}}

        return meta_data


    #convert the json to a non serializable data this is done to push into api easily
    @staticmethod
    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    #convert all the json values to string object
    @staticmethod
    def convertValuesToStr(obj):
        if isinstance(obj, dict):
            for key in obj:
                if isinstance(obj[key], (int, float)):
                    obj[key] = str(obj[key])
                elif isinstance(obj[key], dict):
                    MetaTag().convertValuesToStr(obj[key])
        else:
            raise TypeError("Input object must be a dictionary.")


    
    # Function to push the final data to api
    @staticmethod
    def apiPush(meta):
        meta_data = meta[0]

        #  convert image_shape to string format
        meta_data['meta']['parent']['image_shape'] = str(meta_data['meta']['parent']['image_shape'])

        # convert exif_dict to string format
        meta_data['meta']['derived']['exif_dict'] = str(meta_data['meta']['derived']['exif_dict'])

        url = getConfigData('content_processing_api.url')

        data22 = MetaTag().safeSerialize(meta_data)
        data_dict = json.loads(data22)
        # MetaTag().convertValuesToStr(data_dict)
        payload = json.dumps(data_dict)
        print(payload)
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("PUT", url, headers=headers, data=payload)
        print(response.text)
    


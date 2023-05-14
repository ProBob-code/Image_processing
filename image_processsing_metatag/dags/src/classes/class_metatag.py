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

class MetaTag():
    def __init__(self):
        pass

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
        
        lap = []
        fourier = []
        gradient = []
        # Define the grid parameters
        grid_size = 10 # number of rows and columns in the grid
        image_height, image_width, _ = img.shape # get image height and width
        grid_height = int(image_height / grid_size) # height of each grid box
        grid_width = int(image_width / grid_size) # width of each grid box

        # Draw the grid on the image
        for i in range(0, image_height, grid_height):
            cv2.line(img, (0, i), (image_width, i), (0, 255, 0), 1)
        for j in range(0, image_width, grid_width):
            cv2.line(img, (j, 0), (j, image_height), (0, 255, 0), 1)


        # Calculate the blur level for randomly selected grid boxes
        selected_boxes = np.random.choice(grid_size * grid_size, size=7, replace=False)

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
    def imageMetatag():
        NFS_path = '/var/log/images_bkp'
        list_images = str(NFS_path) + '/new_images'
        # D = os.listdir(list_images)

        images = [f for f in os.listdir(list_images) if f.lower().endswith(('.jpg', '.png', '.jpeg','.JPG','.gif'))]
        
        column_names = ['image_name_1','height_1','width_1','resolution_1','megapixels_1','ppi_1','size_1','img_format_1',
        'img_mode_1','exif_dict_1','description_1','keywords_1','author_1','copyright_1','location_1','laplacian_variance_blur_1',
        'fourier_transform_blur_1','gradient_magnitude_blur_1','red_1','green_1','blue_1','image_shape_1','matrix_1','status']
        df2 = pd.DataFrame(columns=column_names)

        for i, image_filename in enumerate(images):
            try:
                # cv2 method used to extract the color vectors, image matrix and blur values
                with open(os.path.join(str(NFS_path), 'new_images', images[i]), 'rb') as f:
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError('Failed to read image')    
                    r, g, b = cv2.split(img)
                    
                    # Flatten the image into a 1D array
                    flat_img = np.ravel(img)

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
                    matrix = img_array[:,:,0]
                    matrix_shape = matrix.shape

                    Laplacian_variance_blur, Fourier_Transform_blur, Gradient_magnitude_blur = MetaTag().blurValues(img)

                    H = list_images + '/' + images[i]
                    size = MetaTag().getSize(H, 'kb')

                    # Pil method used to extract remaining metatags
                    with open(H, 'rb') as f:
                        image = PIL.Image.open(io.BytesIO(f.read()))
                        wid, hgt = image.size
                        ppi_formula = (hgt*wid)**0.5/(3*4)
                        ppi = round(ppi_formula)
                        resolution = (wid*hgt)
                        megapixels = resolution/1000000
                        img_format = image.format

                        # Get the image mode (e.g. RGB, CMYK)
                        img_mode = image.mode

                        # Get the EXIF data (if available)
                        exif_data = image.getexif()
                        exif_dict = {}
                        if exif_data:
                            for tag_id, value in exif_data.items():
                                tag = TAGS.get(tag_id, tag_id)
                                exif_dict[tag] = value

                        # Extract additional metadata using the IPTC profile
                        iptc = image.info.get("iptc", {})
                        description = iptc.get((2, 120), None)
                        keywords = iptc.get((2, 25), None)
                        author = iptc.get((2, 80), None)
                        copyright = iptc.get((2, 116), None)
                        location = iptc.get((2, 92), None)

                        df2.loc[i] = [image_filename,hgt,wid,resolution,megapixels,ppi,size,img_format,img_mode,exif_dict,description,keywords,author,copyright,location,Laplacian_variance_blur,Fourier_Transform_blur,Gradient_magnitude_blur,mean_r,mean_g,mean_b,matrix_shape,matrix,'good']

            
            except (AttributeError, TypeError) as e:
                print(f"Error while processing {image_filename}: {e}") 
                # print('name:',image_filename,'\nsize:',size,'\nLaplacian_variance_blur:',Laplacian_variance_blur,'\nFourier_Transform_blur:',Fourier_Transform_blur,'\nGradient_magnitude_blur:',Gradient_magnitude_blur,'\nimg_format:',img_format,'\nimg_mode:',img_mode,'\nexif_dict:',exif_dict,'\ndescription:',description,'\nkeywords:',keywords,'\nauthor:',author,'\ncopyright:',copyright,
                #         '\nlocation:',location,'\nred:',mean_r,'\ngreen:',mean_g,'\nblue:',mean_b,'\nwidth:',wid,'\nheight:',hgt,'\nppi:',ppi,'\nresolution:',resolution,'\nmegapixels:',megapixels,'\n\n')
               
                df2.loc[i] = [image_filename,'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','error']

            except (cv2.error, ValueError) as e:
                print(f"Error while processing {image_filename}: {e}")
                
                # Or mark the image as corrupt in the output dataframe
                df2.loc[i] = [image_filename,'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','corrupt'] 

            return df2

    
    # Convert the data from Dataframe to dictionary format
    @staticmethod
    def dictConvert(final_data):

        data = final_data

        data_dict = data.to_dict(orient='records')

        parent_dict = {}
        derived_dict = {}
        meta_data = {}

        for i in range(len(data_dict)):
            # meta = {parent_dict,derived_dict}
            parent_dict[i] = {
                'image_name': data_dict[i]['image_name_1'],
                'height': data_dict[i]['height_1'],
                'width': data_dict[i]['width_1'],
                'size': data_dict[i]['size_1'],
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
                'image_metric': data_dict[i]['image_metric_1'],
                'duplicate': data_dict[i]['duplicate_1']
                }
            
            #this is the part where the process flag is set and if this is not set properly then the dataflow into mongo won't be done properly
            if data_dict[i]['status']=='error':
                meta_data[i] = {'product_id': data_dict[i]['product_id_1'],"process_flag":32, 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i]}}

            elif data_dict[i]['status']=='corrupt':
                meta_data[i] = {'product_id': data_dict[i]['product_id_1'],"process_flag":32, 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i]}}
            
            elif data_dict[i]['status']=='empty':
                meta_data[i] = {'product_id': data_dict[i]['product_id_1'],"process_flag":32, 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i]}}
            
            else:
                meta_data[i] = {'product_id': data_dict[i]['product_id_1'],"process_flag":2, 'meta':{'parent': parent_dict[i], 'derived': derived_dict[i]}}

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
        MetaTag().convertValuesToStr(data_dict)
        payload = json.dumps(data_dict)
        print(payload)
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("PUT", url, headers=headers, data=payload)
        print(response.text)
    


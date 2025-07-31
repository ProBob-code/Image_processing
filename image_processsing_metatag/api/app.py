# Import libraries
from crypt import methods
from flask import Flask, request, jsonify, flash,redirect
import pandas as pd
import numpy as np
import traceback
from route_predictions.routes import prediction_bp
from main.push_to_queue import push_to_queue_bp
from main.redis_purge import redis_bp
from main.dominant_color_bp import dominant_color_bp
from class_api_response import ApiResponse
import requests
from PIL import Image
from io import BytesIO
# import classes.blur_detection
from classes.blur_detection import BlurDetect
from classes.blur_identification_v2 import BlurIdentify
from classes.blur_identification_v3 import BlurIdentify_V3
# from classes.blur_identification_v4 import BlurIdentify_V4
from classes.calculate_brightness import CalculateBrighntess
from classes.extract_color import ExtractColor
from classes.normalized_data import CatalogueScore
from classes.normalized_data import CategoryScore
from classes.normalized_data import MetaCatalogueScore
from classes.imagemeta_tag import ImageMeta
from classes.duplicate_score import ImageLSH
from classes.extract_exif import Extract_exif
from classes.mediainfo_video import VideoMeta
from classes.partial_blur import Partial_Blur
from classes.manage_logs import Manage_Logs
from classes.star_performance import CatalogueStarScore
from classes.N_detection import NDetection
from classes.N_detection_v2 import NDetection_V2
# from classes.dominant_color import DominantColorProcess
from common import initialData1, getCompanyDetails, initialData1_Video, is_corrupted
from helper import getConfigData 
from rabbitmq import RabbitMQ
# from dags.src.libs.helpers import getConfigData
import os
import json
import numpy as np
import validators
from classes.mthreading import MultiThreading
import cv2
from common import initialData1,getCompanyDetails
from helper import getConfigData
# from dags.src.libs.helpers import getConfigData
import os
import json
import threading
from queue import Queue
from datetime import datetime
import re
import logging
import time
import mysql.connector

app = Flask(__name__)
app.secret_key = getConfigData('secret.value')
app.register_blueprint(prediction_bp, url_prefix='/api/predictions')

#Register Blueprint
app.register_blueprint(push_to_queue_bp)
app.register_blueprint(redis_bp)
app.register_blueprint(dominant_color_bp)

logging_enabled = True
logs = []

def log(message):
    global logging_enabled, logs
    if logging_enabled:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(f"[{current_time}] {message}")
        logs.append(f"[{current_time}] {message}")


@app.route('/',methods=['GET'])
def index():
    apiResponse = ApiResponse()
    return apiResponse.responseSuccess("Hello World!")

@app.route('/cp',methods=['GET'])
def index_cp():
    apiResponse = ApiResponse()
    return apiResponse.responseSuccess("Hello World!")

@app.route('/cp/health-check/stat',methods=['GET'])
def health_check():
    apiResponse = ApiResponse()
    return apiResponse.responseSuccess("Okay")
    

@app.route('/cp/api/v1/calculatesize',methods=['POST'])
def calculateSize():
    if request.method == 'POST':
        post_data = request.form.to_dict()
        if 'product_url' in post_data and post_data['product_url'] != '':
            # print(url)
            download_res = initialData1(request.form)
            # print(download_res)
            if download_res['error_code'] == 0:
                image_path = download_res['dir'] + '/' + download_res['filename']
                image_path = re.sub(r"\\\/", "/", image_path)
                if not is_corrupted(image_path):
                    api_name = "cal_size"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
        # return apiResponse.responseSuccess("Okay")
        else:
            badresponse = ApiResponse()
            return badresponse.responseNotFound('Error Found')
        
        response = ImageMeta.fetchImageHeightWidth(image_path)
        # return response
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Success",response)
    
@app.route('/cp/api/v2/calculatesize',methods=['POST','GET'])
def calculateSizeV2():
    if request.method == 'POST':
        product_url = request.form['product_url']
        product_url = re.sub(r"\\\/", "/", product_url)
        response = ImageMeta.fetchImageHeightWidthV2(product_url)
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Success",response)
    if request.method == 'GET':
        if 'img' in request.args and request.args['img']!='':
            response = ImageMeta.fetchImageHeightWidthV2(request.args['img'])
            apiResponse = ApiResponse()
            return apiResponse.responseSuccess("Success",response)
        else:
            badresponse = ApiResponse()
            return badresponse.responseNotFound('No valid Parameter')

@app.route('/cp/api/v1/videoinfo',methods=['POST'])
def calculateVideoInfo():
    if request.method == 'POST':
        response = ImageMeta.calculateVideoInfo(request.form['video_url'])
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Success",response)

@app.route('/cp/api/v1/extract-blur',methods=['GET', 'POST'])
def extract_blur_detection():
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
            # print(post_data)
        try:
            check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
            if check_file_exist:
                image_path = post_data['local_path']
                image_path = re.sub(r"\\\/", "/", image_path)
                if not is_corrupted(image_path):
                    api_name = "blur_detection"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound("image corrupted or deleted")
            elif 'product_url' in post_data and post_data['product_url'] != '':
                # print(url)
                download_res = initialData1(request.form)
                # print(download_res)
                if download_res['error_code'] == 0:
                    image_path = download_res['dir'] + '/' + download_res['filename']
                    if not is_corrupted(image_path, post_data['product_url']):
                        api_name = "blur_detection"
                        Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                        badresponse = ApiResponse()
                        return badresponse.responseNotFound(download_res['msg'])
                else:
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
                # print(image_path)
            else:
                badresponse = ApiResponse()
                return badresponse.responseBadRequest('Please post valid parameters')
            # image_path = request.form['local_file']
            l,f,g = BlurDetect.blurValues(image_path)
            response = {
                'local_path' : image_path,
                'laplacian_variance_blur' : str(l),
                'fourier_transform_blur'  : str(f),
                'gradient_magnitude_blur' : str(g)
            }
            data = {}
            data['request'] = dict(request.form)
            data['response'] = response
            apiResponse = ApiResponse()
            return apiResponse.responseSuccess("Blur extracted sucessfully",data)
        except Exception as e:
            api_name = "blur_detection"
            Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
            apiResponse = ApiResponse()
            return apiResponse.responseBadRequest("Bad request!")


@app.route('/cp/api/v1/extract-brightness', methods=['GET', 'POST'])
def extract_brightness():
    global logs
    logs = []
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
            # print(post_data)
        # try:
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
    # start_time_f = datetime.now()
    # log(f"Start Time: {start_time}")
    # Clear logs for each request
    # if request.method == 'GET':
    #     apiResponse = ApiResponse()
    #     return apiResponse.responseSuccess("Please select valid method")
    # elif request.method == 'POST':

    #     post_data = request.form.to_dict()
    #     if 'local_path' in post_data:
            start_time_i = datetime.now()
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "brightness"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image corrupted or deleted")
            log(f"Image exists: {image_path}")
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            log(f"Time_taken: {time_taken}")
        elif 'product_url' in post_data and post_data['product_url'] != '':
            start_time_d = datetime.now()
            log(f"Download from image URL: {post_data['product_url']}")
            download_res = initialData1(request.form)
            if download_res['error_code'] == 0:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, post_data['product_url']):
                    api_name = "brightness"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            log(f"Time_taken_to_download: {time_taken}")
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')
        start_time_fn = datetime.now()
        brightness_val = CalculateBrighntess.calculateOverallBrightness(image_path)
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        log(f"Time_taken_to_extract_from_function: {time_taken_function}")

        # end_time_f = datetime.now()
        time_taken_f = time_taken + time_taken_function
        # log(f"End Time: {end_time}")
        log(f"Overall_Time_Taken: {time_taken_f}")
        response = {
            'brightness': str(brightness_val),
            'local_path': image_path
        }
        data = {
            'request': dict(request.form),
            'response': response,
        }
        if logging_enabled:
            data['logs_brightness_' + post_data.get('product_id', '')] = logs
            name = ["logs_brightness_"+post_data.get('product_id', '')]
            print(name)
            for log_entry in logs:
                print(log_entry)
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Brightness extracted successfully", data)
        # except Exception as e:
        #     api_name = "brightness"
        #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
        #     apiResponse = ApiResponse()
        #     return apiResponse.responseBadRequest("Bad request!")


@app.route('/cp/api/v1/extract-color', methods=['GET', 'POST'])
def extract_color():
    global logs
    logs = []
    # start_time = datetime.now()
    # log(f"Start Time: {start_time}")
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # print(post_data)
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            start_time_i = datetime.now()
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "image_score"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image corrupted or deleted")
            log(f"Image exists: {image_path}")
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            log(f"Time_taken: {time_taken}")
        elif 'product_url' in post_data and post_data['product_url'] != '':
            start_time_d = datetime.now()
            download_res = initialData1(request.form)
            log(f"Download from image URL: {post_data['product_url']}")
            if download_res['error_code'] == 0:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, post_data['product_url']):
                    api_name = "image_score"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            log(f"Time_taken_to_download: {time_taken}")
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')
        
        # try:
        path = image_path
        # final_data = request.form['final_data']
        response = {}
        start_time_fn = datetime.now()
        color_nodes, verdict, modified_path = ExtractColor.extractColor(path)
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        log(f"Time_taken_to_extract_from_function: {time_taken_function}")
        # log(f"Time Taken: {time_taken}")

        time_taken_f = time_taken + time_taken_function
        # log(f"End Time: {end_time}")
        log(f"Overall_Time_Taken: {time_taken_f}")

        response = {
        'color': color_nodes,
        'single_color': verdict,
        'local_path': modified_path
    }

        data = {}
        data['request'] = post_data
        data['response'] = response

        if logging_enabled:
            data['logs_color_' + post_data.get('product_id', '')] = logs
            name = ["logs_color_"+post_data.get('product_id', '')]
            print(name)
            for log_entry in logs:
                print(log_entry)


        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Color extracted sucessfully",data)
        # except Exception as e:
        #     api_name = ""
        #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
        #     apiResponse = ApiResponse()
        #     return apiResponse.responseBadRequest("Bad request! Please check post parameter")


@app.route('/cp/api/v1/image-score', methods=['GET', 'POST'])
def image_score():
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select a valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        
        try:
            docid = post_data.get('docid', '')
            national_catid = post_data.get('national_catid', '')
            contentstatus = post_data.get('content_status', 'approved')  # Default to 'approved' if not provided

            if contentstatus == 'category' and national_catid:
                # national_catids = list(str(national_catid).split(","))
                response = CategoryScore.dataNormalizedCategoryScore(national_catid)
                if 'error_code' in response and response['error_code'] == 1:
                    meta = response
                    success_msg = response['msg']
                else:
                    meta = CategoryScore.dictConvert(response)
                    success_msg = "Image score fetched sucessfully"
            elif contentstatus == 'meta' and docid:
                company_res = getCompanyDetails(docid)
                flag = company_res.get('error')

                if flag == None and company_res[docid].get('gdocids') != '':
                    gdocids = company_res[docid].get('gdocids') 
                    docid = gdocids
                else:
                    docid = docid
                docids = list(str(docid).split(","))
                response = MetaCatalogueScore.dataNormalizedMetaCatalogueScore(docids)
                if 'error_code' in response and response['error_code'] == 1:
                    meta = response
                    success_msg = response['msg']
                else:
                    meta = CatalogueScore.dictConvert(response)
                    success_msg = "Image score fetched sucessfully"
            elif docid:
                company_res = getCompanyDetails(docid)
                flag = company_res.get('error')

                if flag == None and company_res[docid].get('gdocids') != '':
                    gdocids = company_res[docid].get('gdocids') 
                    docid = gdocids
                else:
                    docid = docid
                docids = list(str(docid).split(","))
                response = CatalogueScore.dataNormalizedCatalogueScore(docids)
                if 'error_code' in response and response['error_code'] == 1:
                    meta = response
                    success_msg = response['msg']
                else:
                    meta = CatalogueScore.dictConvert(response)
                    success_msg = "Image score fetched sucessfully"
            else:
                badresponse = ApiResponse()
                return badresponse.responseBadRequest('Please post valid parameters')

            # return 
            data = {}
            data['request'] = dict(request.form)
            data['response'] = meta
            apiResponse = ApiResponse()
            return apiResponse.responseSuccess(success_msg,data)
        except Exception as e:
            api_name = "image_score"
            Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
            apiResponse = ApiResponse()
            return apiResponse.responseBadRequest("Bad request!")
        
@app.route('/cp/api/v1/extract-meta', methods=['GET','POST'])
def extract_imagemeta():
    global logs
    logs = []
    # start_time = datetime.now()
    # log(f"Start Time: {start_time}")
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # print(post_data)
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            start_time_i = datetime.now()
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "imagemeta"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image corrupted or deleted")
            log(f"Image exists: {image_path}")
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            log(f"Time_taken: {time_taken}")

        elif ('product_url' not in post_data or post_data['product_url'] == ''):
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('local path is invalid or file is absent')
        elif 'product_url' in post_data:
            start_time_d = datetime.now()
            # print(url)
            log(f"Download from image URL: {post_data['product_url']}")
            download_res = initialData1(request.form)
            if download_res['error_code'] == 0:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, post_data['product_url']):
                    api_name = "imagemeta"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            log(f"Time_taken_to_download: {time_taken}")
        else: 
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')

        # image_path = request.form['local_file']
        # try:
        business_tag_check = post_data['business_tag'] if 'business_tag' in post_data and post_data['business_tag'] != '' else '1'
        docid = request.form['docid']
        product_id = request.form['product_id']
        product_url = request.form['product_url']
        product_url = re.sub(r"\\\/", "/", product_url)

        if product_url.find('amazonaws.com') != -1:
            product_url_ori = post_data['product_url'] if 'product_url' in post_data and post_data['product_url'] != '' else product_url
        else:
            product_url_ori = request.form['product_url_ori']
        product_url_ori = re.sub(r"\\\/", "/", product_url_ori)
        input = {
            'docid'          : docid,
            'product_id'     : product_id,
            'local_path'     : image_path,
            'product_url'    : product_url,
            'product_url_ori': product_url_ori,
            'business_tag'   : business_tag_check
        }
        metadata = ImageMeta.imageMetatag(input)
        start_time_fn = datetime.now()
        metadata_ImageMagick = ImageMeta.image_sys_details(input["local_path"])
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        log(f"Time_taken_to_extract_from_function: {time_taken_function}")



        time_taken_f = time_taken + time_taken_function
        # log(f"End Time: {end_time}")
        log(f"Overall_Time_Taken: {time_taken_f}")

        data = {}
        data['request'] = post_data
        data['response'] = json.loads(metadata)
        if(isinstance(metadata_ImageMagick.get("channel_statistics", "0"), dict)):
            data['response']['meta']['derived']["channel_statistics"] = metadata_ImageMagick.get("channel_statistics", "0").copy()
        else:
            data['response']['meta']['derived']["channel_statistics"] = "0"
        data['response']['meta']['derived']["image_statistics"] = {}
        if(isinstance(metadata_ImageMagick.get("image_statistics", "0"), dict)):
            data['response']['meta']['derived']["image_statistics"]["overall"] = metadata_ImageMagick.get("image_statistics", "0").get("overall", "0").copy()
        else:
            if(isinstance(metadata_ImageMagick.get("image_statistics", "0"), dict)):
                data['response']['meta']['derived']["image_statistics"] = metadata_ImageMagick.get("image_statistics", "0").copy()
            else:
                data['response']['meta']['derived']["image_statistics"] = "0"

        if(metadata_ImageMagick.get("channel_statistics") is not None):
            metadata_ImageMagick.pop("channel_statistics")
        if(metadata_ImageMagick.get("image_statistics") is not None and metadata_ImageMagick["image_statistics"].get("overall")):
            metadata_ImageMagick["image_statistics"].pop("overall")
        # changing keys of dictionary
        metadata_ImageMagick['image_details'] = metadata_ImageMagick.get('image_statistics', "0")
        if(metadata_ImageMagick.get('image_statistics') is not None):
            del metadata_ImageMagick['image_statistics']

        # metadata_ImageMagick.pop("image_statistics")
        data['response']['meta']['parent'].update(metadata_ImageMagick)

        if logging_enabled:
            data['logs_meta_' + post_data.get('product_id', '')] = logs
            name = ["logs_meta_"+post_data.get('product_id', '')]
            print(name)
            for log_entry in logs:
                print(log_entry)
        apiResponse = ApiResponse()

        return apiResponse.responseSuccess("Meta success", data)
        # except Exception as e:
        #     api_name = "imagemeta"
        #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
        #     apiResponse = ApiResponse()
        #     return apiResponse.responseBadRequest("Bad request! Please check post parameter")



@app.route('/cp/api/v1/extract-hash', methods=['GET', 'POST'])
def extract_hash():
    
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # print(post_data)
        try:
            if 'local_path' in post_data and post_data['local_path'] != '':
                image_path = post_data['local_path']
                image_path = re.sub(r"\\\/", "/", image_path)
                if not is_corrupted(image_path):
                    api_name = "hash"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound("image is corrupted or deleted")
                
            elif 'product_url' in post_data:
                # print(url)
                download_res = initialData1(request.form)
                
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, post_data['product_url']):
                    api_name = "hash"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
                # print(download_res)
            
            post_data['image_path'] = image_path
            hash_response = ImageMeta.hashValueExtract(post_data)
            data = {}
            data['request'] = post_data
            data['response'] = json.loads(hash_response)
            apiResponse = ApiResponse()
            return apiResponse.responseSuccess("Success",data)
        except Exception as e:
            api_name = "hash"
            Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
            apiResponse = ApiResponse()
            return apiResponse.responseBadRequest("Bad request!")
    
@app.route('/cp/api/v1/extract-avg-hash',methods=['GET','POST'])
def extract_avg_hash():
    global logs
    logs = []
    # start_time = datetime.now()
    # log(f"Start Time: {start_time}")

    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # print(post_data)
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            start_time_i = datetime.now()
            # print('image exist')
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "avg_hash"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image is corrupted or deleted")
            log(f"Image exists: {image_path}")
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            log(f"Time_taken: {time_taken}")
        elif 'product_url' in post_data and post_data['product_url'] != '':
            start_time_d = datetime.now()
            # print(type(post_data))
            download_res = initialData1(request.form)
            log(f"Download from image URL: {post_data['product_url']}")
            # print(download_res)
            if download_res['error_code'] == 0:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, post_data['product_url']):
                    api_name = "avg_hash"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            log(f"Time_taken_to_download: {time_taken}")
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')

        # try: 
        start_time_fn = datetime.now()
        avg_hash, d_hash, p_hash, c_hash, dup_signature = ImageMeta.avgHashValueExtract(image_path)
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        log(f"Time_taken_to_extract_from_function: {time_taken_function}")

        time_taken_f = time_taken + time_taken_function
        # log(f"End Time: {end_time}")
        log(f"Overall_Time_Taken: {time_taken_f}")

        output = {}
        output['local_path'] = image_path
        output['average_hash'] = avg_hash
        output['difference_hash'] = d_hash
        output['perceptual_hash'] = p_hash
        output['color_hash'] = c_hash
        output['dup_signature'] = dup_signature
        
        data = {}
        data['request'] = post_data
        data['response'] = output

        if logging_enabled:
            data['logs_hash_' + post_data.get('product_id', '')] = logs
            name = ["logs_hash_"+post_data.get('product_id', '')]
            print(name)
            for log_entry in logs:
                print(log_entry)

        apiResponse = ApiResponse()
            
        return apiResponse.responseSuccess("Success",data)
        # except Exception as e:
        #     api_name = "avg_hash"
        #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
        #     apiResponse = ApiResponse()
        #     return apiResponse.responseBadRequest("Bad request! Please check post parameter")
    

@app.route('/cp/api/v1/extract-duplicate-score', methods=['GET', 'POST'])
def extract_duplicate_score():
        if request.method == 'GET':
            apiResponse = ApiResponse()
            return apiResponse.responseSuccess("Please select a valid method")
        elif request.method == 'POST':
            post_data = request.form.to_dict()
            
            try:
                docid = post_data.get('docid', '')
                national_catid = post_data.get('national_catid', '')    
                contentstatus = post_data.get('content_status', 'approved')  # Default to 'approved' if not provided




                if contentstatus == 'category' and national_catid:
                    # national_catids = list(str(national_catid).split(","))
                    # national_catids = list(map(int, national_catid.split(",")))
                    response = ImageLSH.data_conversion(national_catid, True, 'category')
                elif docid:
                    company_res = getCompanyDetails(docid)
                    flag = company_res.get('error')

                    if flag == None and company_res[docid].get('gdocids') != '':
                        gdocids = company_res[docid].get('gdocids') 
                        docid = gdocids
                    else:
                        docid = docid
                    docids = list(str(docid).split(","))
                    response = ImageLSH.data_conversion(docids, False, 'approved')
                else:
                    badresponse = ApiResponse()
                    return badresponse.responseBadRequest('Please post valid parameters')

                data = {}
                if 'error_code' in response and response['error_code'] == 1:
                    output = response
                    success_msg = response['msg']
                else:
                    output = json.loads(response)
                    success_msg = "Duplicate score fetched successfully"
                data['request'] = post_data
                data['response'] = output
                apiResponse = ApiResponse()
                return apiResponse.responseSuccess(success_msg, data)
            except Exception as e:
                api_name = "ex_dup_score"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
                apiResponse = ApiResponse()
                return apiResponse.responseBadRequest("Bad request!")
            
@app.route('/cp/api/v1/extract-exif', methods=['GET', 'POST'])
def extract_exif():
    global logs
    logs = []
    # start_time = datetime.now()
    # log(f"Start Time: {start_time}")

    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # print(post_data)
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            start_time_i = datetime.now()
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "extract_exif"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image is corrupted or deleted")
            log(f"Image exists: {image_path}")
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            log(f"Time_Taken: {time_taken}")
        elif 'product_url' in post_data and post_data['product_url']!='':
            start_time_d = datetime.now()
            download_res = initialData1(request.form)
            log(f"Download from image URL: {post_data['product_url']}")
            if download_res['error_code'] == 0:
                image_path = download_res['dir'] + '/' + download_res['filename']
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            log(f"Time_taken_to_download: {time_taken}")
            if not is_corrupted(image_path, post_data['product_url']):
                api_name = "extract_exif"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')
    # try:
    path = image_path
    pid = request.form['product_id']
    # final_data = request.form['final_data']
    response = {}
    start_time_fn = datetime.now()

    # old_exif = Extract_exif.precheck(pid)	
    # print('old exif: ',old_exif)
    old_exif_final = {} # Changed from old_exif to {}
    new_exif = Extract_exif.exif_generate(path)
    # print('new exif:',new_exif)

    old_exif_final = Extract_exif.final_parsed(old_exif_final)
    # print('old_exif_final: ',old_exif_final)

    img = Image.open(path)
    width = img.width
    height = img.height
    if width > 250:
        new_width = 250
        new_height = int((height/width)*new_width)

        # Resize the image to 250x(calculated height) pixels
        img = img.resize((new_width, new_height))

        # Save the resized image back to the ORIGINAL path (overwrite the original)
        img.convert('RGB').save(path)
        path = path #resized path
    else:
        path = image_path #old path

    response['old_exif'] = old_exif_final
    response['new_exif'] = new_exif
    response['local_path'] = path

    # print('response: ',response)

    # Check if new_exif is not an integer before calling combine_json
    if not isinstance(new_exif, int):
        response['final_exif_generated'] = Extract_exif.combine_json(old_exif_final, new_exif[0])
    else:
        # Handle the case when new_exif is an integer (0)
        response['final_exif_generated'] = old_exif_final  

    end_time_fn = datetime.now()
    time_taken_function = end_time_fn - start_time_fn
    log(f"Time_taken_to_extract_from_function: {time_taken_function}")

    time_taken_f = time_taken + time_taken_function
    # log(f"End Time: {end_time}")
    log(f"Overall_Time_Taken: {time_taken_f}")

    # You can choose what to do in this case
    data = {}
    data['request'] = post_data
    data['response'] = response

    if logging_enabled:
        data['logs_exif_' + post_data.get('product_id', '')] = logs
        name = ["logs_exif_"+post_data.get('product_id', '')]
        print(name)
        for log_entry in logs:
            print(log_entry)

    apiResponse = ApiResponse()
    return apiResponse.responseSuccess("Exif extracted sucessfully",data)
    # except Exception as e:
    #     api_name = "extract_exif"
    #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
    #     apiResponse = ApiResponse()
    #     return apiResponse.responseBadRequest("Bad request!")

        
@app.route('/cp/api/v2/extract-blur', methods=['GET', 'POST'])
def extract_blur_identification():
    global logs
    logs = []
    # start_time = datetime.now()
    # log(f"Start Time: {start_time}")

    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # print(post_data)
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            # print('image exist')
            start_time_i = datetime.now()
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "blur_v2"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image is corrupted or deleted")
            log(f"Image exists: {image_path}")
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            log(f"Time_taken: {time_taken}")
        elif 'product_url' in post_data and post_data['product_url'] != '':
            start_time_d = datetime.now()
            # print(url)
            download_res = initialData1(request.form)
            # print(download_res)
            log(f"Download from image URL: {post_data['product_url']}")
            if download_res['error_code'] == 0:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, post_data['product_url']):
                    api_name = "blur_v2"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            log(f"Time_taken_to_download: {time_taken}")
            # print(image_path)
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')
        # image_path = request.form['local_file']
        


        start_time_fn = datetime.now()
        ground_truth, variance = BlurIdentify.is_blurry(image_path)
        # print('ground_truth:',ground_truth)

        
        if ground_truth==True:
            flag = 1
        else:
            flag = 0
        

        earlier_blur_result,grid_count,grid_values, diff = BlurIdentify.blurValues(image_path)

        earlier_blur_result['lvb_main'] = variance
        
        # print('grid_count',grid_count)

        # Calculate the total grid count
        total_grids = grid_count["blur"] + grid_count["partial_blur"]+ grid_count["clear"]

        # Calculate the percentage of each category
        blur_percentage = (grid_count["blur"] / total_grids) * 100
        partial_blur_percentage = (grid_count["partial_blur"] / total_grids) * 100
        clear_percentage = (grid_count["clear"] / total_grids) * 100

        # Make a final decision
        if blur_percentage >= 50:
            final_decision = "blur"
        elif partial_blur_percentage >= 50:
            final_decision = "partial_blur"
        elif clear_percentage >= 50:
            final_decision = "clear"
        elif partial_blur_percentage == clear_percentage:
            final_decision = "partial_blur"
        elif partial_blur_percentage == blur_percentage:
            final_decision = "blur"
        else:
            final_decision = "send to moderation"

        # Convert float32 values to float
        for key in earlier_blur_result:
            earlier_blur_result[key] = float(earlier_blur_result[key])

        # Convert float32 values in grid_values
        for category in grid_values:
            for item in grid_values[category]:
                item["lvb"] = float(item["lvb"])
                item["ftb"] = float(item["ftb"])
                item["gmb"] = float(item["gmb"])

        response = {
            'local_path' : image_path,
            # 'image_blur' : int(flag),
            'blur_type' : str(final_decision),
            # 'difference_in_overal_vs_grid' : int(diff),
            'laplacian_variance_blur' : str(earlier_blur_result['lvb_grid']),
            'fourier_transform_blur' : str(earlier_blur_result['ftb_grid']),
            'gradient_magnitude_blur' : str(earlier_blur_result['gmb_grid']),
            'laplacian_variance_main' : str(earlier_blur_result['lvb_main'])
        }
        # print(post_data)
        # if 'blur_details' in post_data and post_data['blur_details'] == '1':
        #     response['blur_details'] = earlier_blur_result

        if 'grid_count' in post_data and post_data['grid_count'] == '1':
            response['grid_count'] = grid_count

        if 'grid_values' in post_data and post_data['grid_values'] == '1':
            response['grid_values'] = grid_values

        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        log(f"Time_taken_to_extract_from_function: {time_taken_function}")

        time_taken_f = time_taken + time_taken_function
        # log(f"End Time: {end_time}")
        log(f"Overall_Time_Taken: {time_taken_f}")

        data = {}
        data['request'] = post_data
        data['response'] = response
        # print(data)


        if logging_enabled:
            data['logs_blur_' + post_data.get('product_id', '')] = logs
            name = ["logs_blur_"+post_data.get('product_id', '')]
            print(name)
            for log_entry in logs:
                print(log_entry)

        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Blur extracted sucessfully",data)
        

@app.route('/cp/api/v1/meta_video',methods=['GET', 'POST'])
def meta_video():
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # print(post_data)
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
        elif ('url' not in post_data or post_data['url'] == ''):
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('local path is invalid or file is absent')
        elif 'url' in post_data and post_data['url'] != '' and validators.url(post_data['url']):
            # print(url)
            download_res = initialData1_Video(request.form)
            # print("download_res:\n", download_res)
            if download_res['error_code'] == 0:
                image_path = post_data['url']
                # print("image_url:\n", image_path)
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            # print(image_path)
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')
    # try:
    response, error_message_sha256 = VideoMeta.get_video_mediainfo_json(image_path)
    data = {}
    data['request'] = post_data
    data['response'] = response
    # print(response)
    # print(data)
    apiResponse = ApiResponse()
    return apiResponse.responseSuccess("Video Meta Extraction Successfull, " + error_message_sha256, data)
    # except Exception as e:
    #     api_name = "meta_video"
    #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
    #     apiResponse = ApiResponse()
    #     return apiResponse.responseBadRequest("Bad request!")




def timeout_handler(input_data):
    # Retry logic in case the thread takes too long
    print("Timeout reached, restarting API call...")
    thread = MultiThreading.main_thread(input_data)
    
    # Handle response
    response = {
        'data': json.loads(thread)
    }
    
    data = {
        'request': input_data,
        'response': response
    }
    
    apiResponse = ApiResponse()
    return apiResponse.responseSuccess("Multi thread re-executed after timeout", data)

@app.route('/cp/api/v1/multi-thread', methods=['GET', 'POST'])
def multi_thread():
    if request.method == 'POST':
        post_data = request.form.to_dict()
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "multi_thread"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image is corrupted or deleted")
        elif 'product_url' in post_data and post_data['product_url'] != '':
            download_res = initialData1(request.form)
            if download_res['error_code'] == 0:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, post_data['product_url']):
                    api_name = "multi_thread"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')

        business_tag_check = post_data['business_tag'] if 'business_tag' in post_data and post_data['business_tag'] != '' else '1'
        docid = request.form['docid']
        product_id = request.form['product_id']
        product_url = request.form['product_url']
        product_url = re.sub(r"\\\/", "/", product_url)

        if product_url.find('amazonaws.com') != -1:
            product_url_ori = post_data['product_url'] if 'product_url' in post_data and post_data['product_url'] != '' else product_url
        else:
            product_url_ori = request.form['product_url_ori']
        product_url_ori = re.sub(r"\\\/", "/", product_url_ori)
        
        input_data = {
            'docid': docid,
            'product_id': product_id,
            'local_path': image_path,
            'product_url': product_url,
            'product_url_ori': product_url_ori,
            'business_tag': business_tag_check
        }

        # Set a 10-second timeout for the thread
        thread_result = [None]
        def worker_thread():
            thread_result[0] = MultiThreading.main_thread(input_data)
        
        worker = threading.Thread(target=worker_thread)
        worker.start()
        
        worker.join(timeout=10)  # Wait 10 seconds for the thread to complete

        if worker.is_alive():
            # Thread has not finished within 10 seconds, restart the API call
            worker.join(timeout=1)  # Give a small grace period before restarting
            response = timeout_handler(input_data)  # Retry handling
            return response

        # If the thread completes within the timeout
        response = {
            'data': json.loads(thread_result[0])
        }

        data = {}
        data['request'] = post_data
        data['response'] = response
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Multi thread executed successfully", data)
    
        # except Exception as e:
        #     api_name = "multi_thread"
        #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
        #     print(f'Exception: {e}')
        #     apiResponse = ApiResponse()
        #     return apiResponse.responseBadRequest("Bad request!")



@app.route('/cp/api/v1/contract-star-performance', methods=['GET', 'POST'])
def contract_star_performance():
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select a valid method")
    
    elif request.method == 'POST':
        post_data = request.form.to_dict()  # Convert POST data into a dictionary
        
        try:
            # Log the incoming POST data for debugging
            print("Received POST data:", post_data)

            # Extract necessary fields from the POST request
            docid = post_data.get('docid', '')
            include_scr = post_data.get('scr', '0')  # Ensure include_scr is a string

            # Check if docid is provided
            if not docid:
                print("Missing 'docid' in POST request")
                apiResponse = ApiResponse()
                return apiResponse.responseBadRequest('Please post valid parameters')

            # Fetch star performance score for the given `docid`
            response = CatalogueStarScore.starDataNormalizedCatalogueScore(docid)
            print("docid_response:",response)

            # Determine if response is a tuple or single DataFrame
            if isinstance(response, tuple):
                df = response[0]  # Extract DataFrame from the tuple
                overall_score = response[1]
            else:
                df = response
                overall_score = response

            # Convert `overall_score` to a float if it's a numpy.float64
            if isinstance(overall_score, (float, np.float64)):
                overall_score = float(overall_score)
            else:
                overall_score = 0.0

            # Prepare response data
            if isinstance(df, pd.DataFrame):
                meta = CatalogueStarScore.dictConvert(df, include_scr)
                # Extract the first item from the `meta` dictionary if it's a list
                meta = meta[0] if isinstance(meta, list) else meta
                # Create a separate dictionary for the overall_score
                meta2 = {'docid_overall_star': float(overall_score)}
                # Merge both dictionaries
                meta.update(meta2)
                success_msg = "Star performance score fetched successfully"
            else:
                meta = {'docid_overall_star': float(overall_score)}
                success_msg = "No data available"

            # Convert integer keys in meta to strings (fixes the JSON serialization issue)
            meta = {str(k): v for k, v in meta.items()}

            # Construct the response data
            data = {
                'request': dict(request.form),
                'response': meta
            }

            # Debugging: Print the data to check its content and types
            print("Response data:", data)
            print("Data types in response:", {k: type(v) for k, v in meta.items()})

            # Send the success response
            apiResponse = ApiResponse()
            return apiResponse.responseSuccess(success_msg, data)

        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            traceback.print_exc()  # Print full stack trace for debugging

            api_name = "contract_star_performance"
            Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, str(e))

            apiResponse = ApiResponse()
            return apiResponse.responseBadRequest("Bad request!")



@app.route('/cp/api/v3/extract-blur', methods=['GET', 'POST'])
def extract_blur_identification_v3():
    global logs
    logs = []
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # try:
        # print(post_data)
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            start_time_i = datetime.now()
            # print('image exist')
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "blur_v3"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image is corrupted or deleted")
            size = os.path.getsize(image_path)
            # Open the image
            try:
                img = Image.open(image_path)
            except FileNotFoundError:
                print("Error: File not found at", image_path)
                badresponse = ApiResponse()
                return badresponse.responseBadRequest('Please post valid parameters')
            
            height = img.height
            width = img.width

            new_width = 1600
            new_height = int((height / width) * new_width)
            # Resize the image
            resized_img = img.resize((new_width, new_height))

            # Save the resized image (overwrites original file by default)
            resized_img.save(image_path)
            resized_img = np.array(resized_img)
            log(f"Image exists: {image_path}")
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            log(f"Time_taken: {time_taken}")

        elif 'product_url' in post_data and post_data['product_url'] != '':
            start_time_d = datetime.now()
            log(f"Download from image URL: {post_data['product_url']}")
            product_url = request.form['product_url']
            product_url = re.sub(r"\\\/", "/", product_url)
            download_res = initialData1(request.form, is_resize=True)
            size = download_res.get('size', -1)
            if download_res['error_code'] == 0 and size!=-1:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, product_url):
                    api_name = "blur_v3"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            # print("image_path:", image_path)
            # Read the image using cv2.imread()
            resized_img = cv2.imread(image_path)
            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            log(f"Time_taken_to_download: {time_taken}")
            if(resized_img is None):
                logging.info(f"Blur_V3: resized_img is None")
                badresponse = ApiResponse()
                return badresponse.responseBadRequest('Please post valid parameters')
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')
        
        start_time_fn = datetime.now()

        response = BlurIdentify_V3.is_blurr_v3(image_path, size)
        _, _, response["partial_blur_status"] = Partial_Blur.create_grid(size, resized_img, grid_pixel = 120, p_score_min=5.0, p_score_max=30.0, size_cutoff=30000,sharpness_score_cutoff=40, top_percent=0.05)
        
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        log(f"Time_taken_to_extract_from_function: {time_taken_function}")

        time_taken_f = time_taken + time_taken_function
        # log(f"End Time: {end_time}")
        log(f"Overall_Time_Taken: {time_taken_f}")

        data = {}
        data['request'] = post_data
        data['response'] = response

        if logging_enabled:
            data['logs_blurV3_' + post_data.get('product_id', '')] = logs
            name = ["logs_blurV3_"+post_data.get('product_id', '')]
            print(name)
            for log_entry in logs:
                print(log_entry)

        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Blur extracted sucessfully",data)
        # except Exception as e:
        #     api_name = "blur_v3"
        #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
        #     apiResponse = ApiResponse()
        #     return apiResponse.responseBadRequest("Bad request!")

# Database configuration
db_config = {
    "host": "192.168.131.29",
    "user": "web_app",
    "password": "!5@uGuST1($7FrEe1Ndi@",
    "database": "db_product"
}

# Ad-hoc API URL
ADHOC_API_URL = "http://192.168.40.172:7060/cs/v1/metaprocess/api/adhoc-process"
# Content-Service API URL
CONTENT_SERVICE_API_URL = "http://192.168.131.170/cs/v1/metaprocess/api/multi-thread"

@app.route('/cp/api/v2/category_processing', methods=['POST'])
def categoryProcessing_v2():
    if request.method != 'POST':
        return jsonify({"status": "error", "message": "Invalid request method"}), 405

    # Retrieve parameters from the request
    random_key = request.form.get('random_key')
    content_status = request.form.get('content_status')
    local_path = request.form.get('local_path')
    national_catid_param = request.form.get('national_catid')

    # Ensure required parameters are provided
    if not all([random_key, content_status, local_path, national_catid_param]):
        return jsonify({"status": "error", "message": "Missing required parameters"}), 400

    try:
        # Establish database connection
        with mysql.connector.connect(**db_config) as connection:
            with connection.cursor(dictionary=True) as cursor:
                # Query to fetch category details
                query = "SELECT national_catid, id, product_url FROM tbl_category_catalogue_details WHERE random_key = %s"
                cursor.execute(query, (random_key,))
                result = cursor.fetchone()

                if not result:
                    return jsonify({"status": "error", "message": "No records found for the given random_key"}), 404

                # Validate national_catid
                try:
                    national_catid_param = int(national_catid_param)
                except ValueError:
                    return jsonify({"status": "error", "message": "Invalid national_catid format"}), 400

                if result['national_catid'] != national_catid_param:
                    return jsonify({
                        "status": "error",
                        "message": "Provided national_catid does not match the database record"
                    }), 400

                product_id = result['id']
                national_catid = result['national_catid']
                product_url = result['product_url']

                # Fetch image metadata
                image_metadata = ImageMeta.fetchImageHeightWidthV2(product_url)
                if image_metadata['error_code'] != 0:
                    return jsonify({
                        "status": "error",
                        "message": image_metadata.get("msg", "Failed to retrieve image metadata")
                    }), 500

                aspect_ratio = ImageMeta.calculate_aspect(image_metadata['width'], image_metadata['height'])

                # Update database with image metadata
                update_query = """
                    UPDATE tbl_category_catalogue_details
                    SET size = %s, height = %s, width = %s, aspect_ratio = %s
                    WHERE id = %s
                """
                cursor.execute(update_query, (
                    image_metadata['size'],
                    image_metadata['height'],
                    image_metadata['width'],
                    aspect_ratio,
                    product_id
                ))
                connection.commit()

                # Call Ad-hoc processing API (**fix here**)
                adhoc_payload = {
                    "product_id": product_id,
                    "content_type": 1,
                    "docid": "",
                    "national_catid": national_catid,
                    "id": product_id,
                    "content_status": content_status
                }
                headers = {'Content-Type': 'application/json'}
                try:
                    adhoc_response = requests.post(ADHOC_API_URL, headers=headers, json=adhoc_payload, timeout=10)
                    adhoc_response.raise_for_status()
                    # Try JSON, else plain text "Success"
                    try:
                        adhoc_response_data = adhoc_response.json()
                    except ValueError:
                        body = adhoc_response.text.strip()
                        if body == "Success":
                            adhoc_response_data = {"status": "Success"}
                        else:
                            raise RuntimeError(f"Ad-hoc API returned unparseable response: {body}")
                    if adhoc_response_data.get("status") != "Success":
                        raise ValueError(f"Ad-hoc API returned an error: {adhoc_response_data}")
                except requests.exceptions.RequestException as e:
                    return jsonify({"status": "error", "message": f"Ad-hoc API request failed: {str(e)}"}), 500
                except Exception as e:
                    return jsonify({"status": "error", "message": str(e)}), 500

                # Call Content-Service API
                content_service_payload = {
                    'docid': national_catid,
                    'product_id': product_id,
                    'product_url': product_url,
                    'product_url_ori': product_url.replace('.jpg', '-w.jpg'),
                    'content_status': content_status
                }
                try:
                    content_service_response = requests.post(CONTENT_SERVICE_API_URL, data=content_service_payload, timeout=10)
                    content_service_response.raise_for_status()
                    content_service_data = content_service_response.json()

                    # Accept Content-Service API response if it contains 'local_path' and 'message' keys
                    if (
                        isinstance(content_service_data, dict)
                        and 'local_path' in content_service_data
                        and 'message' in content_service_data
                    ):
                        # Accept this as a valid response, do not treat as error
                        pass
                    # Improved error checking for Content-Service API
                    elif 'error' in content_service_data:
                        raise ValueError(f"Content-Service API returned an error: {content_service_data['error']}")
                    elif 'message' in content_service_data and isinstance(content_service_data['message'], dict) and 'error' in content_service_data['message']:
                        raise ValueError(f"Content-Service API returned an error: {content_service_data['message']['error']}")
                    elif 'message' in content_service_data and isinstance(content_service_data['message'], dict) and 'data' in content_service_data['message'] and 'response' in content_service_data['message']['data'] and 'error' in content_service_data['message']['data']['response']:
                        raise ValueError(f"Content-Service API returned an error: {content_service_data['message']['data']['response']['error']}")
                    elif content_service_data.get("status_code") != 200 and content_service_data.get("message") != "Multi thread executed successfully":
                        raise ValueError(f"Content-Service API returned an unexpected response: {content_service_data}")
                    elif 'message' in content_service_data and isinstance(content_service_data['message'], str) and 'error' in content_service_data['message'].lower():
                        raise ValueError(f"Content-Service API returned an error in message: {content_service_data['message']}")
                except requests.exceptions.RequestException as e:
                    return jsonify({"status": "error", "message": f"Content-Service API request failed: {str(e)}"}), 500
                except ValueError as e:
                    return jsonify({"status": "error", "message": str(e)}), 500

                # Construct success response
                data = {
                    "message": "Category processing completed successfully",
                    "status": "success",
                    "response": {
                        "error_code": 0,
                        "national_catid": national_catid,
                        "id": product_id,
                        "aspect_ratio": aspect_ratio,
                        "image_metadata": image_metadata,
                        "adhoc_response": adhoc_response_data,
                        "content_service_response": content_service_data
                    }
                }
                return jsonify(data), 200

    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": f"Database error: {str(err)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500


@app.route('/cp/api/v1/n-detect', methods=['GET', 'POST'])
def n_detect():
    logs = []
    time_taken = None  # Initialize with a default value to avoid UnboundLocalError

    if request.method == 'GET':
        # Handle GET request: just return a success message indicating POST is expected
        api_response = ApiResponse()
        return api_response.responseSuccess("Please select valid method")

    elif request.method == 'POST':
        post_data = request.form.to_dict()

        # Handle local path image case
        if 'local_path' in post_data:
            image_path = post_data['local_path']
            logs.append(f"Received image path: {image_path}")
            if not os.path.isfile(image_path):
                api_response = ApiResponse()
                return api_response.responseNotFound("File not found at the provided path")

            # Process image
            start_time_i = datetime.now()
            image_path = re.sub(r"\\\/", "/", image_path)

            if not is_corrupted(image_path):
                api_response = ApiResponse()
                return api_response.responseNotFound("Image is corrupted or deleted")

            # No resizing here, just pass image path to the detection function
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            logs.append(f"Time taken to process image: {time_taken}")

        # Handle image URL case
        elif 'product_url' in post_data and post_data['product_url']:
            start_time_d = datetime.now()
            product_url = post_data['product_url']
            logs.append(f"Received product URL: {product_url}")
            product_url = re.sub(r"\\\/", "/", product_url)

            download_res = initialData1(request.form, is_resize=True)
            size = download_res.get('size', -1)
            if download_res['error_code'] == 0 and size != -1:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, product_url):
                    api_response = ApiResponse()
                    return api_response.responseNotFound(download_res['msg'])
            else:
                api_response = ApiResponse()
                return api_response.responseNotFound(download_res['msg'])

            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            logs.append(f"Time taken to download image: {time_taken}")

        else:
            api_response = ApiResponse()
            return api_response.responseBadRequest('Missing or invalid parameters')

        # Process the image for detection (without resizing here)
        start_time_fn = datetime.now()
        n_detection = NDetection()
        detection_result = n_detection.process_image(image_path, product_url)
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        logs.append(f"Time taken for detection function: {time_taken_function}")

        # Calculate overall time taken
        if time_taken is None:
            time_taken = time_taken_function
        else:
            time_taken += time_taken_function
        logs.append(f"Overall Time Taken: {time_taken}")

        # Include predictions if specified
        include_predictions = post_data.get('predictions', '0') == '1'
        response_data = {
            'detection_result': {
                'classification': detection_result['classification']
            },
            'local_path': image_path
        }

        if include_predictions:
            response_data['detection_result']['predictions'] = detection_result.get('predictions', [])

        # Prepare the response data
        data = {
            'request': dict(request.form),
            'response': response_data,
        }

        # Add logs to response if logging is enabled
        if logging_enabled:
            data['logs_n_detect_' + post_data.get('product_id', '')] = logs
            for log_entry in logs:
                print(log_entry)

        # Send the final response
        api_response = ApiResponse()
        return api_response.responseSuccess("Detection completed successfully", data)

def convert_to_serializable(obj):
    """Recursively convert objects to JSON serializable formats."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):  # Convert NumPy floats to Python floats
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):  # Convert NumPy ints to Python ints
        return int(obj)
    return obj

@app.route('/cp/api/v2/n-detect', methods=['GET', 'POST'])
def n_detect_v2():
    logs = []
    time_taken = None  # Initialize with a default value to avoid UnboundLocalError

    if request.method == 'GET':
        # Handle GET request: just return a success message indicating POST is expected
        api_response = ApiResponse()
        return api_response.responseSuccess("Please select valid method")

    elif request.method == 'POST':
        post_data = request.form.to_dict()

        # Handle local path image case
        if 'local_path' in post_data:
            image_path = post_data['local_path']
            logs.append(f"Received image path: {image_path}")
            if not os.path.isfile(image_path):
                api_response = ApiResponse()
                return api_response.responseNotFound("File not found at the provided path")

            # Process image
            start_time_i = datetime.now()
            image_path = re.sub(r"\\\/", "/", image_path)

            if not is_corrupted(image_path):
                api_response = ApiResponse()
                return api_response.responseNotFound("Image is corrupted or deleted")

            # No resizing here, just pass image path to the detection function
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            logs.append(f"Time taken to process image: {time_taken}")

        # Handle image URL case
        elif 'product_url' in post_data and post_data['product_url']:
            start_time_d = datetime.now()
            product_url = post_data['product_url']
            logs.append(f"Received product URL: {product_url}")
            product_url = re.sub(r"\\\/", "/", product_url)

            download_res = initialData1(request.form, is_resize=True)
            size = download_res.get('size', -1)
            if download_res['error_code'] == 0 and size != -1:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, product_url):
                    api_response = ApiResponse()
                    return api_response.responseNotFound(download_res['msg'])
            else:
                api_response = ApiResponse()
                return api_response.responseNotFound(download_res['msg'])

            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            logs.append(f"Time taken to download image: {time_taken}")

        else:
            api_response = ApiResponse()
            return api_response.responseBadRequest('Missing or invalid parameters')

        # Process the image for detection (without resizing here)
        start_time_fn = datetime.now()
        n_detection = NDetection_V2()
        detection_result = n_detection.process_image(image_path, product_url)
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        logs.append(f"Time taken for detection function: {time_taken_function}")

        # Calculate overall time taken
        if time_taken is None:
            time_taken = time_taken_function
        else:
            time_taken += time_taken_function
        logs.append(f"Overall Time Taken: {time_taken}")

        # Check if predictions should be included
        include_predictions = post_data.get('predictions', '0') == '1'

        response_data = {
            'detection_result': {
                'classification': detection_result.get('classification', 'Send to moderation'),
            },
            'local_path': image_path
        }

        # Log the contents of detection_result to confirm if detected_objects and predictions exist
        # logs.append(f"detection_result contains: {detection_result}")

        # Include detected_objects and predictions only if prediction == '1'
        if include_predictions:
            response_data['detection_result']['detected_objects'] = detection_result.get('detected_objects', [])
            response_data['detection_result']['image_tag'] = detection_result.get('image_tag', [])
            response_data['detection_result']['predictions'] = detection_result.get('predictions', [])

        # Ensure the response is JSON serializable
        response_data = convert_to_serializable(response_data)

        # Prepare the response data
        data = {
            'request': dict(request.form),
            'response': response_data,
        }

        # Add logs to response if logging is enabled
        if logging_enabled:
            data['logs_n_detect_v2' + post_data.get('product_id', '')] = logs
            for log_entry in logs:
                print(log_entry)

        # Ensure a valid response is returned
        api_response = ApiResponse()
        return api_response.responseSuccess("Detection completed successfully", data)



@app.route('/cp/api/v3/n-detect', methods=['GET', 'POST'])
def n_detect_v3():
    logs = []
    time_taken = None  # Initialize with a default value to avoid UnboundLocalError

    if request.method == 'GET':
        # Handle GET request: just return a success message indicating POST is expected
        api_response = ApiResponse()
        return api_response.responseSuccess("Please select valid method")

    elif request.method == 'POST':
        post_data = request.form.to_dict()

        # Handle local path image case
        if 'local_path' in post_data:
            image_path = post_data['local_path']
            logs.append(f"Received image path: {image_path}")
            if not os.path.isfile(image_path):
                api_response = ApiResponse()
                return api_response.responseNotFound("File not found at the provided path")

            # Process image
            start_time_i = datetime.now()
            image_path = re.sub(r"\\\/", "/", image_path)

            if not is_corrupted(image_path):
                api_response = ApiResponse()
                return api_response.responseNotFound("Image is corrupted or deleted")

            # No resizing here, just pass image path to the detection function
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            logs.append(f"Time taken to process image: {time_taken}")

        # Handle image URL case
        elif 'product_url' in post_data and post_data['product_url']:
            start_time_d = datetime.now()
            product_url = post_data['product_url']
            logs.append(f"Received product URL: {product_url}")
            product_url = re.sub(r"\\\/", "/", product_url)

            download_res = initialData1(request.form, is_resize=True)
            size = download_res.get('size', -1)
            if download_res['error_code'] == 0 and size != -1:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, product_url):
                    api_response = ApiResponse()
                    return api_response.responseNotFound(download_res['msg'])
            else:
                api_response = ApiResponse()
                return api_response.responseNotFound(download_res['msg'])

            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            logs.append(f"Time taken to download image: {time_taken}")

        else:
            api_response = ApiResponse()
            return api_response.responseBadRequest('Missing or invalid parameters')

        # Process the image for detection (without resizing here)
        start_time_fn = datetime.now()
        n_detection = NDetection_V2()
        detection_result = n_detection.process_image(image_path, product_url)
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        logs.append(f"Time taken for detection function: {time_taken_function}")

        # Calculate overall time taken
        if time_taken is None:
            time_taken = time_taken_function
        else:
            time_taken += time_taken_function
        logs.append(f"Overall Time Taken: {time_taken}")

        # Check if predictions should be included
        include_predictions = post_data.get('predictions', '0') == '1'

        response_data = {
            'detection_result': {
                'classification': detection_result.get('classification', 'Send to moderation'),
            },
            'local_path': image_path
        }

        # Log the contents of detection_result to confirm if detected_objects and predictions exist
        # logs.append(f"detection_result contains: {detection_result}")

        # Include detected_objects and predictions only if prediction == '1'
        if include_predictions:
            response_data['detection_result']['detected_objects'] = detection_result.get('detected_objects', [])
            response_data['detection_result']['image_tag'] = detection_result.get('image_tag', [])
            response_data['detection_result']['predictions'] = detection_result.get('predictions', [])

        # Ensure the response is JSON serializable
        response_data = convert_to_serializable(response_data)

        # Prepare the response data
        data = {
            'request': dict(request.form),
            'response': response_data,
        }

        # Add logs to response if logging is enabled
        if logging_enabled:
            data['logs_n_detect_v3' + post_data.get('product_id', '')] = logs
            for log_entry in logs:
                print(log_entry)

        # Ensure a valid response is returned
        api_response = ApiResponse()
        return api_response.responseSuccess("Detection completed successfully", data)
    


@app.route('/cp/api/v4/extract-blur', methods=['GET', 'POST'])
def extract_blur_identification_v4():
    global logs
    logs = []
    if request.method == 'GET':
        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Please select valid method")
    elif request.method == 'POST':
        post_data = request.form.to_dict()
        # try:
        # print(post_data)
        check_file_exist = os.path.isfile(post_data['local_path']) if 'local_path' in post_data else False
        if check_file_exist:
            start_time_i = datetime.now()
            # print('image exist')
            image_path = post_data['local_path']
            image_path = re.sub(r"\\\/", "/", image_path)
            if not is_corrupted(image_path):
                api_name = "blur_v4"
                Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                badresponse = ApiResponse()
                return badresponse.responseNotFound("image is corrupted or deleted")
            size = os.path.getsize(image_path)
            # Open the image
            try:
                img = Image.open(image_path)
            except FileNotFoundError:
                print("Error: File not found at", image_path)
                badresponse = ApiResponse()
                return badresponse.responseBadRequest('Please post valid parameters')
            
            height = img.height
            width = img.width

            new_width = 1600
            new_height = int((height / width) * new_width)
            # Resize the image
            resized_img = img.resize((new_width, new_height))

            # Save the resized image (overwrites original file by default)
            resized_img.save(image_path)
            resized_img = np.array(resized_img)
            log(f"Image exists: {image_path}")
            end_time_i = datetime.now()
            time_taken = end_time_i - start_time_i
            log(f"Time_taken: {time_taken}")

        elif 'product_url' in post_data and post_data['product_url'] != '':
            start_time_d = datetime.now()
            log(f"Download from image URL: {post_data['product_url']}")
            product_url = request.form['product_url']
            product_url = re.sub(r"\\\/", "/", product_url)
            download_res = initialData1(request.form, is_resize=True)
            size = download_res.get('size', -1)
            if download_res['error_code'] == 0 and size!=-1:
                image_path = download_res['dir'] + '/' + download_res['filename']
                if not is_corrupted(image_path, product_url):
                    api_name = "blur_v4"
                    Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name)
                    badresponse = ApiResponse()
                    return badresponse.responseNotFound(download_res['msg'])
            else:
                badresponse = ApiResponse()
                return badresponse.responseNotFound(download_res['msg'])
            # print("image_path:", image_path)
            # Read the image using cv2.imread()
            resized_img = cv2.imread(image_path)
            end_time_d = datetime.now()
            time_taken = end_time_d - start_time_d
            log(f"Time_taken_to_download: {time_taken}")
            if(resized_img is None):
                logging.info(f"Blur_V4: resized_img is None")
                badresponse = ApiResponse()
                return badresponse.responseBadRequest('Please post valid parameters')
        else:
            badresponse = ApiResponse()
            return badresponse.responseBadRequest('Please post valid parameters')
        
        start_time_fn = datetime.now()

        response = BlurIdentify_V4.is_blurr_v4(image_path, size)
        _, _, response["partial_blur_status"] = Partial_Blur.create_grid(size, resized_img, grid_pixel = 120, p_score_min=5.0, p_score_max=30.0, size_cutoff=30000,sharpness_score_cutoff=40, top_percent=0.05)
        
        end_time_fn = datetime.now()
        time_taken_function = end_time_fn - start_time_fn
        log(f"Time_taken_to_extract_from_function: {time_taken_function}")

        time_taken_f = time_taken + time_taken_function
        # log(f"End Time: {end_time}")
        log(f"Overall_Time_Taken: {time_taken_f}")

        data = {}
        data['request'] = post_data
        data['response'] = response

        if logging_enabled:
            data['logs_blurV4_' + post_data.get('product_id', '')] = logs
            name = ["logs_blurV4_"+post_data.get('product_id', '')]
            print(name)
            for log_entry in logs:
                print(log_entry)

        apiResponse = ApiResponse()
        return apiResponse.responseSuccess("Blur extracted sucessfully",data)
        # except Exception as e:
        #     api_name = "blur_v4"
        #     Manage_Logs.input_request(post_data.get("local_path", 0), post_data.get("product_url", 0), api_name, e)
        #     apiResponse = ApiResponse()
        #     return apiResponse.responseBadRequest("Bad request!")

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8081, debug=True)
    except (KeyboardInterrupt, OSError):
        print("Server exited unexpectedly. Please contact the server admin.")




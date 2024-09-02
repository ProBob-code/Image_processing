import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
import matplotlib.pyplot as plt
from sklearn import preprocessing
import re
import json
import time
import numpy as np
from datetime import timedelta
from datetime import datetime
from helper import getConfigData

from pymongo import MongoClient
import pandas as pd
import math


class CatalogueStarScore():
    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)
    
    def get_score(value, ranges, scores):
        try:
            value = float(value)  # Convert the value to float
        except ValueError:
            # Handle the case where conversion fails (e.g., if the value is non-numeric)
            return float(0)  # Return NaN or any default value indicating an error

        for (low, high), score in zip(ranges, scores):
            if low <= value < high:
                return score

        return float(0)  # Return NaN or a default score if no range matches
  # Return the highest score if value exceeds the last range

    def quantize_and_score(column, ranges, scores):
        return column.apply(lambda x: CatalogueStarScore.get_score(x, ranges, scores)).astype(float)

    def custom_bscore_quantize_and_score(column, ranges, scores):
        column = pd.to_numeric(column, errors='coerce')  # Convert entire column to numeric, coercing errors to NaN
        return column.apply(lambda x: CatalogueStarScore.get_score(x, ranges, scores)).astype(float)

    
    # def custom_bscore_quantize_and_score(column, n_bins):
    #     bins = np.linspace(column.min(), column.max(), n_bins + 1)
    #     # Categorize the data into bins
    #     quantized_column = pd.cut(column, bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True)
    #     # Map the bins to the custom scores
    #     score_map = {1: 1, 2: 3, 3: 5, 4: 3, 5: 1}
    #     return quantized_column.map(score_map).astype(float)

    # # General quantization and scoring function for other columns
    # def quantize_and_score(column, n_bins):
    #     bins = np.linspace(column.min(), column.max(), n_bins + 1)
    #     return pd.cut(column, bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True).astype(float)


    def starDataNormalizedCatalogueScore(docid):
        start_time = time.time()
        DocId = docid
        docids = [docid.lower() for docid in DocId]

        # Connect to MongoDB server
        username = getConfigData("mongo.username")
        password = getConfigData("mongo.password")
        auth_db = getConfigData("mongo.auth_db")

        with MongoClient(getConfigData('mongo.host'), username=username, password=password, authSource=auth_db) as client:
            # Access database
            db = client[getConfigData('mongo.db')]

            # Access collection
            collection = db[getConfigData('mongo.collection')]

            # Query data
            query = {
                "$or": [
                    {"did": {"$in": docids}, "df": 1, "dlf": 0, "ap": 1},
                    {"did": {"$in": docids}, "df": 1, "dlf": 0, "ap": 2},
                    {"did": {"$in": docids}, "df": 1, "dlf": 0, "ap": 0}
                ]
            }

            # Fetch data and create a DataFrame
            document_count = collection.count_documents(query)

            if document_count > 0:
            # if cursor.count() > 0:
                print('docid {0} exist in collection'.format(DocId))

                cursor = collection.find(query)
                document_list = list(cursor)
                df = pd.DataFrame(document_list)
            else:
                output = {"msg": "Docid not found", "error_code": 1}
                return output

        # except Exception as e:
        #     print(f"An error occurred: {e}")

        # Close the MongoDB connection (if needed)
        client.close()

        # Data preprocessing
        df.fillna(0, inplace=True)

        try:
            df['keyword_ct'] = df['kwd'].apply(lambda x: x.get('ct', ''))
            df['keyword_cc'] = df['kwd'].apply(lambda x: x.get('cc', ''))
            df['keyword_cm'] = df['kwd'].apply(lambda x: x.get('cm', ''))
            df['keyword_dt'] = df['kwd'].apply(lambda x: x.get('dt', ''))
            df['keyword_cf'] = df['kwd'].apply(lambda x: x.get('cf', ''))
            df['quality'] = df['si'].apply(lambda x: x.get('qlty', ''))
            df['source'] = df['si'].apply(lambda x: x.get('src', ''))
            df['classification'] = df['si'].apply(lambda x: x.get('clf', ''))
            df['like'] = df['si'].apply(lambda x: x.get('like', ''))
            df['report'] = df['si'].apply(lambda x: x.get('rep', ''))

            # For 'parent' keys
            extract_height = lambda x: x.get('parent', {}).get('hgt', '') if isinstance(x, dict) else ''
            extract_width = lambda x: x.get('parent', {}).get('wid', '') if isinstance(x, dict) else ''
            extract_size = lambda x: x.get('parent', {}).get('siz', '') if isinstance(x, dict) else ''
            extract_image_format = lambda x: x.get('parent', {}).get('img_fmt', '') if isinstance(x, dict) else ''
            extract_image_mode = lambda x: x.get('parent', {}).get('img_mod', '') if isinstance(x, dict) else ''
            extract_description = lambda x: x.get('parent', {}).get('desc', '') if isinstance(x, dict) else ''
            extract_author = lambda x: x.get('parent', {}).get('aut', '') if isinstance(x, dict) else ''
            extract_copyright = lambda x: x.get('parent', {}).get('cpr', '') if isinstance(x, dict) else ''
            extract_location = lambda x: x.get('parent', {}).get('loc', '') if isinstance(x, dict) else ''
            extract_image_shape = lambda x: x.get('parent', {}).get('img_shp', '') if isinstance(x, dict) else ''
            extract_image_detail_quality = lambda x: x.get('parent', {}).get('image_details', {}).get('quality', '') if isinstance(x, dict) else ''

            # For 'derived' keys
            extract_pixel_1 = lambda x: x.get('derived', {}).get('pxl', '') if isinstance(x, dict) else ''
            extract_megapixels = lambda x: x.get('derived', {}).get('mgp', '') if isinstance(x, dict) else ''
            extract_ppi = lambda x: x.get('derived', {}).get('ppi', '') if isinstance(x, dict) else ''
            extract_exif_dict = lambda x: x.get('derived', {}).get('exif_dict', '') if isinstance(x, dict) else ''
            extract_red = lambda x: x.get('derived', {}).get('red', '') if isinstance(x, dict) else ''
            extract_green = lambda x: x.get('derived', {}).get('green', '') if isinstance(x, dict) else ''
            extract_blue = lambda x: x.get('derived', {}).get('blue', '') if isinstance(x, dict) else ''
            extract_laplacian_variance_blur = lambda x: x.get('derived', {}).get('lvb', '') if isinstance(x, dict) else ''
            extract_fourier_transform_blur = lambda x: x.get('derived', {}).get('ftb', '') if isinstance(x, dict) else ''
            extract_gradient_magnitude_blur = lambda x: x.get('derived', {}).get('gmb', '') if isinstance(x, dict) else ''
            extract_brightness_score = lambda x: x.get('derived', {}).get('bscore', '') if isinstance(x, dict) else ''
            extract_colourfulness = lambda x: x.get('derived', {}).get('clrfln', '') if isinstance(x, dict) else ''
            extract_sharpness_score = lambda x: x.get('derived', {}).get('shpscr', '') if isinstance(x, dict) else ''
            extract_image_metric = lambda x: x.get('derived', {}).get('imgmet', '') if isinstance(x, dict) else ''
            extract_blur_type = lambda x: x.get('derived', {}).get('blur_type', '') if isinstance(x, dict) else ''
            extract_color = lambda x: x.get('color', '') if isinstance(x, dict) else ''
            extract_artifact = lambda x: x.get('derived', {}).get('artifact_verdict', '') if isinstance(x, dict) else ''
            extract_partial_blur = lambda x: x.get('derived', {}).get('partial_blur_status', '') if isinstance(x, dict) else ''
            

            df['height'] = df['meta'].apply(extract_height)
            df['width'] = df['meta'].apply(extract_width)
            df['size'] = df['meta'].apply(extract_size)
            df['image_format'] = df['meta'].apply(extract_image_format)
            df['image_mode'] = df['meta'].apply(extract_image_mode)
            df['description'] = df['meta'].apply(extract_description)
            df['author'] = df['meta'].apply(extract_author)
            df['copyright'] = df['meta'].apply(extract_copyright)
            df['location'] = df['meta'].apply(extract_location)
            df['image_shape'] = df['meta'].apply(extract_image_shape)

            df['pixels'] = df['meta'].apply(extract_pixel_1)
            df['megapixels'] = df['meta'].apply(extract_megapixels)
            df['ppi'] = df['meta'].apply(extract_ppi)
            df['exif_dict'] = df['meta'].apply(extract_exif_dict)
            df['red'] = df['meta'].apply(extract_red)
            df['green'] = df['meta'].apply(extract_green)
            df['blue'] = df['meta'].apply(extract_blue)
            df['laplacian_variance_blur'] = df['meta'].apply(extract_laplacian_variance_blur)
            df['fourier_transform_blur'] = df['meta'].apply(extract_fourier_transform_blur)
            df['gradient_magnitude_blur'] = df['meta'].apply(extract_gradient_magnitude_blur)
            df['brightness_score'] = df['meta'].apply(extract_brightness_score)
            df['colourfulness'] = df['meta'].apply(extract_colourfulness)
            df['sharpness_score'] = df['meta'].apply(extract_sharpness_score)
            df['color'] = df['meta'].apply(extract_color)
            df['image_metric'] = df['meta'].apply(extract_image_metric)
            df['blur_type'] = df['meta'].apply(extract_blur_type)
            df['artifact_verdict'] = df['meta'].apply(extract_artifact)
            df['partial_blur_status'] = df['meta'].apply(extract_partial_blur)
            df['imagemagic_quality'] = df['meta'].apply(extract_image_detail_quality)

        except (AttributeError) as e:
            df['keyword_ct'] = 0
            df['keyword_cc'] = 0
            df['keyword_cm'] = 0
            df['keyword_dt'] = 0
            df['keyword_cf'] = 0 
            df['source'] = 0 
            df['classification'] = 0 
            df['like'] = 0 
            df['report'] = 0   
            df['image_metric'] = 0   

            # For 'parent' keys
            extract_height = lambda x: x.get('parent', {}).get('hgt', '') if isinstance(x, dict) else ''
            extract_width = lambda x: x.get('parent', {}).get('wid', '') if isinstance(x, dict) else ''
            extract_size = lambda x: x.get('parent', {}).get('siz', '') if isinstance(x, dict) else ''
            extract_image_format = lambda x: x.get('parent', {}).get('img_fmt', '') if isinstance(x, dict) else ''
            extract_image_mode = lambda x: x.get('parent', {}).get('img_mod', '') if isinstance(x, dict) else ''
            extract_description = lambda x: x.get('parent', {}).get('desc', '') if isinstance(x, dict) else ''
            extract_author = lambda x: x.get('parent', {}).get('aut', '') if isinstance(x, dict) else ''
            extract_copyright = lambda x: x.get('parent', {}).get('cpr', '') if isinstance(x, dict) else ''
            extract_location = lambda x: x.get('parent', {}).get('loc', '') if isinstance(x, dict) else ''
            extract_image_shape = lambda x: x.get('parent', {}).get('img_shp', '') if isinstance(x, dict) else ''

            # For 'derived' keys
            extract_pixel_1 = lambda x: x.get('derived', {}).get('pxl', '') if isinstance(x, dict) else ''
            extract_megapixels = lambda x: x.get('derived', {}).get('mgp', '') if isinstance(x, dict) else ''
            extract_ppi = lambda x: x.get('derived', {}).get('ppi', '') if isinstance(x, dict) else ''
            extract_exif_dict = lambda x: x.get('derived', {}).get('exif_dict', '') if isinstance(x, dict) else ''
            extract_red = lambda x: x.get('derived', {}).get('red', '') if isinstance(x, dict) else ''
            extract_green = lambda x: x.get('derived', {}).get('green', '') if isinstance(x, dict) else ''
            extract_blue = lambda x: x.get('derived', {}).get('blue', '') if isinstance(x, dict) else ''
            extract_laplacian_variance_blur = lambda x: x.get('derived', {}).get('lvb', '') if isinstance(x, dict) else ''
            extract_fourier_transform_blur = lambda x: x.get('derived', {}).get('ftb', '') if isinstance(x, dict) else ''
            extract_gradient_magnitude_blur = lambda x: x.get('derived', {}).get('gmb', '') if isinstance(x, dict) else ''
            extract_brightness_score = lambda x: x.get('derived', {}).get('bscore', '') if isinstance(x, dict) else ''
            extract_colourfulness = lambda x: x.get('derived', {}).get('clrfln', '') if isinstance(x, dict) else ''
            extract_sharpness_score = lambda x: x.get('derived', {}).get('shpscr', '') if isinstance(x, dict) else ''
            extract_image_metric = lambda x: x.get('derived', {}).get('imgmet', '') if isinstance(x, dict) else ''
            extract_blur_type = lambda x: x.get('derived', {}).get('blur_type', '') if isinstance(x, dict) else ''
            extract_color = lambda x: x.get('color', '') if isinstance(x, dict) else ''
            extract_artifact = lambda x: x.get('derived', {}).get('artifact_verdict', '') if isinstance(x, dict) else ''
            extract_partial_blur = lambda x: x.get('derived', {}).get('partial_blur_status', '') if isinstance(x, dict) else ''
            extract_image_detail_quality = lambda x: x.get('parent', {}).get('image_details', {}).get('quality', '') if isinstance(x, dict) else ''

            # Apply these modified lambda functions to your DataFrame
            df['height'] = df['meta'].apply(extract_height)
            df['width'] = df['meta'].apply(extract_width)
            df['size'] = df['meta'].apply(extract_size)
            df['image_format'] = df['meta'].apply(extract_image_format)
            df['image_mode'] = df['meta'].apply(extract_image_mode)
            df['description'] = df['meta'].apply(extract_description)
            df['author'] = df['meta'].apply(extract_author)
            df['copyright'] = df['meta'].apply(extract_copyright)
            df['location'] = df['meta'].apply(extract_location)
            df['image_shape'] = df['meta'].apply(extract_image_shape)

            df['pixels'] = df['meta'].apply(extract_pixel_1)
            df['megapixels'] = df['meta'].apply(extract_megapixels)
            df['ppi'] = df['meta'].apply(extract_ppi)
            df['exif_dict'] = df['meta'].apply(extract_exif_dict)
            df['red'] = df['meta'].apply(extract_red)
            df['green'] = df['meta'].apply(extract_green)
            df['blue'] = df['meta'].apply(extract_blue)
            df['laplacian_variance_blur'] = df['meta'].apply(extract_laplacian_variance_blur)
            df['fourier_transform_blur'] = df['meta'].apply(extract_fourier_transform_blur)
            df['gradient_magnitude_blur'] = df['meta'].apply(extract_gradient_magnitude_blur)
            df['brightness_score'] = df['meta'].apply(extract_brightness_score)
            df['colourfulness'] = df['meta'].apply(extract_colourfulness)
            df['sharpness_score'] = df['meta'].apply(extract_sharpness_score)
            df['color'] = df['meta'].apply(extract_color)
            df['image_metric'] = df['meta'].apply(extract_image_metric)
            df['blur_type'] = df['meta'].apply(extract_blur_type)
            df['artifact_verdict'] = df['meta'].apply(extract_artifact)
            df['partial_blur_status'] = df['meta'].apply(extract_partial_blur)
            df['imagemagic_quality'] = df['meta'].apply(extract_image_detail_quality)


        # Drop unnecessary columns
        df = df.drop(columns=['meta'], axis=1)
        df.reset_index(inplace=True, drop=True)

        # Convert data types
        df = df.convert_dtypes()
        df['size'] = pd.to_numeric(df['size'], errors='ignore')
        df['ppi'] = pd.to_numeric(df['ppi'], errors='ignore')
        df['width'] = pd.to_numeric(df['width'], errors='coerce').fillna(0).astype(int)
        df['height'] = pd.to_numeric(df['height'], errors='coerce').fillna(0).astype(int)

        # Calculate additional features
        df['size_div_ppi'] = df['size'] / df['ppi']
        df['size2_div_ppi'] = df['size'] ** 2 / df['ppi']
        df['aspect_ratio'] = df['width'] / df['height']
        df.fillna(0, inplace=True)

        # Apply custom scoring for categorical columns
        blur_type_score_map = {"clear": 5.0, "blur": 0.0, "partial_blur": 2.0, ' ': 1.0, "send to moderation": 0.0}
        artifact_verdict_score_map = {' ': 2.0, "Not Artifact": 5.0, "Artifact": 0.0}
        partial_blur_status_score_map = {' ': 2.0, "False": 5.0, "True": 0.0}

        df['blur_type_score'] = df['blur_type'].map(blur_type_score_map).fillna(0)
        df['artifact_verdict_score'] = df['artifact_verdict'].map(artifact_verdict_score_map).fillna(0)
        df['partial_blur_status_score'] = df['partial_blur_status'].map(partial_blur_status_score_map).fillna(0)

        # Define ranges and scores for each column
        bscore_ranges = [(0.0, 50.9909), (50.9909, 101.9819), (101.9819, 152.9728), (152.9728, 203.9637), (203.9637, 254.9546)]
        bscore_scores = [1, 3, 5, 3, 1]

        gmb_ranges = [(0.0, 28.9526), (28.9526, 57.9051), (57.9051, 86.8577), (86.8577, 115.8103), (115.8103, 144.7628)]
        gmb_scores = [1, 2, 3, 4, 5]

        clrfln_ranges = [(0.0, 18.3482), (18.3482, 36.6965), (36.6965, 55.0447), (55.0447, 73.3929), (73.3929, 91.7411)]
        clrfln_scores = [1, 2, 3, 4, 5]

        shpscr_ranges = [(0.0, 85.7273), (85.7273, 171.4545), (171.4545, 257.1818), (257.1818, 342.9090), (342.9090, 428.6363)]
        shpscr_scores = [1, 2, 3, 4, 5]

        imagemagic_quality_ranges = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 100.0)]
        imagemagic_quality_scores = [1, 2, 3, 4, 5]

        size2_div_ppi_ranges = [(177863.14285714287, 58624666653.90252), (58624666653.90252, 117249155444.66217), (117249155444.66217, 175873644235.42184), (175873644235.42184, 234498133026.1815), (234498133026.1815, 293122621816.94116)]
        size2_div_ppi_scores = [1, 2, 3, 4, 5]

        def custom_bscore_quantize_and_score(column, ranges, scores):
            return column.apply(lambda x: CatalogueStarScore.get_score(x, ranges, scores)).astype(float)

        def quantize_and_score(column, ranges, scores):
            return column.apply(lambda x: CatalogueStarScore.get_score(x, ranges, scores)).astype(float)

        # Apply quantization and scoring to each column
        columns_to_quantize = {
            'brightness_score': (bscore_ranges, bscore_scores),
            'gradient_magnitude_blur': (gmb_ranges, gmb_scores),
            'colourfulness': (clrfln_ranges, clrfln_scores),
            'sharpness_score': (shpscr_ranges, shpscr_scores),
            'imagemagic_quality': (imagemagic_quality_ranges, imagemagic_quality_scores),
            'size2_div_ppi': (size2_div_ppi_ranges, size2_div_ppi_scores)
        }

        for column, (ranges, scores) in columns_to_quantize.items():
            if column == 'brightness_score':  # Example for custom bscore
                df[f'{column}_score'] = custom_bscore_quantize_and_score(df[column], ranges, scores)
            else:
                df[f'{column}_score'] = quantize_and_score(df[column], ranges, scores)

        all_scores = [f'{column}_score' for column in columns_to_quantize.keys()] + ['blur_type_score', 'artifact_verdict_score', 'partial_blur_status_score']
        df['star_score'] = df[all_scores].mean(axis=1)

        quality_data = df.copy()
        quality_data.sort_values(by=['star_score'], ascending=False, inplace=True)
        quality_data['pid'] = quality_data['pid'].astype(str)

        d_overall_star = quality_data['star_score'].mean()

        return quality_data, d_overall_star


    def dictConvert(quality_data,include_scr='0'):
        data = quality_data
        data_dict = data.to_dict(orient='records')
        include_scr = include_scr
        score = {}
        meta_data = {}

        for i in range(len(data_dict)):
            score[i] = {
                'bscore_score': data_dict[i]['brightness_score_score'],
                'bscore': data_dict[i]['brightness_score'],
                'gmb_score': data_dict[i]['gradient_magnitude_blur_score'],
                'gmb': data_dict[i]['gradient_magnitude_blur'],
                'clrfln_score': data_dict[i]['colourfulness_score'],
                'clrfln':data_dict[i]['colourfulness'],
                'shpscr_score': data_dict[i]['sharpness_score_score'],
                'shpscr':data_dict[i]['sharpness_score'],
                'imagemagic_quality_score': data_dict[i]['imagemagic_quality_score'],
                'imagemagic_quality':data_dict[i]['imagemagic_quality'],
                'size2_div_ppi_score': data_dict[i]['size2_div_ppi_score'],
                'size2_div_ppi':data_dict[i]['size2_div_ppi'],
                'blur_type_score': data_dict[i]['blur_type_score'],
                'blur_type':data_dict[i]['blur_type'],
                'artifact_verdict_score': data_dict[i]['artifact_verdict_score'],
                'artifact_verdict':data_dict[i]['artifact_verdict'],
                'partial_blur_status_score': data_dict[i]['partial_blur_status_score'],
                'partial_blur_status':data_dict[i]['partial_blur_status']
            }

            meta_data[i] = {
            'product_id': int(data_dict[i]['pid']),
            'product_url': str(data_dict[i]['pul']),
            'pid_score': str(data_dict[i]['star_score'])
            }

            # Conditionally include 'scr' in the response
            if include_scr=='1':
                meta_data[i]['scr'] = score[i]

        return meta_data

class CategoryStarScore():
    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    def starDataNormalizedCategoryScore(docid):

        start_time = time.time()
        # this is to record the date and time of the process to cross check how much time it took to process the whole queue
        # DocId = docid

        docids = int(docid)
        print("DocID:",docids)
        # DocId = [int(docid) for docid in docid]

        # Connect to MongoDB server
        username = getConfigData("mongo.username")
        password = getConfigData("mongo.password")
        auth_db = getConfigData("mongo.auth_db")

        with MongoClient(getConfigData('mongo.host'), username=username, password=password, authSource=auth_db) as client:
            # Access database
            db = client[getConfigData('mongo.db')]

            # Access collection
            collection = db[getConfigData('mongo.collection_category')]

            # Query data
            query = {"national_catid": docids}

            # Fetch data and create a DataFrame
            document_count = collection.count_documents(query)

            if document_count > 0:
            # if cursor.count() > 0:
                print('docid {0} exist in collection'.format(docids))

                cursor = collection.find(query)
                document_list = list(cursor)
                df = pd.DataFrame(document_list)
            else:
                output = {"msg": "Docid not found", "error_code": 1}
                return output

        # Close the MongoDB connection (if needed)
        client.close()

        df.fillna(0,inplace=True)
        # print(df)

        try:
            df['keyword_ct'] = df['keywords'].apply(lambda x: x.get('ct', ''))
            df['keyword_cc'] = df['keywords'].apply(lambda x: x.get('cc', ''))
            df['keyword_cm'] = df['keywords'].apply(lambda x: x.get('cm', ''))
            df['keyword_dt'] = df['keywords'].apply(lambda x: x.get('dt', ''))
            df['keyword_cf'] = df['keywords'].apply(lambda x: x.get('cf', ''))
            df['quality'] = df['si'].apply(lambda x: x.get('qlty', ''))
            df['source'] = df['si'].apply(lambda x: x.get('src', ''))
            df['classification'] = df['si'].apply(lambda x: x.get('clf', ''))
            df['like'] = df['si'].apply(lambda x: x.get('like', ''))
            df['report'] = df['si'].apply(lambda x: x.get('rep', ''))

            # For 'parent' keys
            extract_height = lambda x: x.get('parent', {}).get('hgt', '') if isinstance(x, dict) else ''
            extract_width = lambda x: x.get('parent', {}).get('wid', '') if isinstance(x, dict) else ''
            extract_size = lambda x: x.get('parent', {}).get('siz', '') if isinstance(x, dict) else ''
            extract_image_format = lambda x: x.get('parent', {}).get('img_fmt', '') if isinstance(x, dict) else ''
            extract_image_mode = lambda x: x.get('parent', {}).get('img_mod', '') if isinstance(x, dict) else ''
            extract_description = lambda x: x.get('parent', {}).get('desc', '') if isinstance(x, dict) else ''
            extract_author = lambda x: x.get('parent', {}).get('aut', '') if isinstance(x, dict) else ''
            extract_copyright = lambda x: x.get('parent', {}).get('cpr', '') if isinstance(x, dict) else ''
            extract_location = lambda x: x.get('parent', {}).get('loc', '') if isinstance(x, dict) else ''
            extract_image_shape = lambda x: x.get('parent', {}).get('img_shp', '') if isinstance(x, dict) else ''
            extract_image_detail_quality = lambda x: x.get('parent', {}).get('image_details', {}).get('quality', '') if isinstance(x, dict) else ''

            # For 'derived' keys
            extract_pixel_1 = lambda x: x.get('derived', {}).get('pxl', '') if isinstance(x, dict) else ''
            extract_megapixels = lambda x: x.get('derived', {}).get('mgp', '') if isinstance(x, dict) else ''
            extract_ppi = lambda x: x.get('derived', {}).get('ppi', '') if isinstance(x, dict) else ''
            extract_exif_dict = lambda x: x.get('derived', {}).get('exif_dict', '') if isinstance(x, dict) else ''
            extract_red = lambda x: x.get('derived', {}).get('red', '') if isinstance(x, dict) else ''
            extract_green = lambda x: x.get('derived', {}).get('green', '') if isinstance(x, dict) else ''
            extract_blue = lambda x: x.get('derived', {}).get('blue', '') if isinstance(x, dict) else ''
            extract_laplacian_variance_blur = lambda x: x.get('derived', {}).get('lvb', '') if isinstance(x, dict) else ''
            extract_fourier_transform_blur = lambda x: x.get('derived', {}).get('ftb', '') if isinstance(x, dict) else ''
            extract_gradient_magnitude_blur = lambda x: x.get('derived', {}).get('gmb', '') if isinstance(x, dict) else ''
            extract_brightness_score = lambda x: x.get('derived', {}).get('bscore', '') if isinstance(x, dict) else ''
            extract_colourfulness = lambda x: x.get('derived', {}).get('clrfln', '') if isinstance(x, dict) else ''
            extract_sharpness_score = lambda x: x.get('derived', {}).get('shpscr', '') if isinstance(x, dict) else ''
            extract_image_metric = lambda x: x.get('derived', {}).get('imgmet', '') if isinstance(x, dict) else ''
            extract_blur_type = lambda x: x.get('derived', {}).get('blur_type', '') if isinstance(x, dict) else ''
            extract_color = lambda x: x.get('color', '') if isinstance(x, dict) else ''
            extract_artifact = lambda x: x.get('derived', {}).get('artifact_verdict', '') if isinstance(x, dict) else ''
            extract_partial_blur = lambda x: x.get('derived', {}).get('partial_blur_status', '') if isinstance(x, dict) else ''

            df['height'] = df['meta'].apply(extract_height)
            df['width'] = df['meta'].apply(extract_width)
            df['size'] = df['meta'].apply(extract_size)
            df['image_format'] = df['meta'].apply(extract_image_format)
            df['image_mode'] = df['meta'].apply(extract_image_mode)
            df['description'] = df['meta'].apply(extract_description)
            df['author'] = df['meta'].apply(extract_author)
            df['copyright'] = df['meta'].apply(extract_copyright)
            df['location'] = df['meta'].apply(extract_location)
            df['image_shape'] = df['meta'].apply(extract_image_shape)

            df['pixels'] = df['meta'].apply(extract_pixel_1)
            df['megapixels'] = df['meta'].apply(extract_megapixels)
            df['ppi'] = df['meta'].apply(extract_ppi)
            df['exif_dict'] = df['meta'].apply(extract_exif_dict)
            df['red'] = df['meta'].apply(extract_red)
            df['green'] = df['meta'].apply(extract_green)
            df['blue'] = df['meta'].apply(extract_blue)
            df['laplacian_variance_blur'] = df['meta'].apply(extract_laplacian_variance_blur)
            df['fourier_transform_blur'] = df['meta'].apply(extract_fourier_transform_blur)
            df['gradient_magnitude_blur'] = df['meta'].apply(extract_gradient_magnitude_blur)
            df['brightness_score'] = df['meta'].apply(extract_brightness_score)
            df['colourfulness'] = df['meta'].apply(extract_colourfulness)
            df['sharpness_score'] = df['meta'].apply(extract_sharpness_score)
            df['color'] = df['meta'].apply(extract_color)
            df['image_metric'] = df['meta'].apply(extract_image_metric)
            df['blur_type'] = df['meta'].apply(extract_blur_type)
            df['artifact_verdict'] = df['meta'].apply(extract_artifact)
            df['partial_blur_status'] = df['meta'].apply(extract_partial_blur)
            df['imagemagic_quality'] = df['meta'].apply(extract_image_detail_quality)

        except (AttributeError) as e:
            df['keyword_ct'] = 0
            df['keyword_cc'] = 0
            df['keyword_cm'] = 0
            df['keyword_dt'] = 0
            df['keyword_cf'] = 0 
            df['source'] = 0 
            df['classification'] = 0 
            df['like'] = 0 
            df['report'] = 0   
            df['image_metric'] = 0   

            # For 'parent' keys
            extract_height = lambda x: x.get('parent', {}).get('hgt', '') if isinstance(x, dict) else ''
            extract_width = lambda x: x.get('parent', {}).get('wid', '') if isinstance(x, dict) else ''
            extract_size = lambda x: x.get('parent', {}).get('siz', '') if isinstance(x, dict) else ''
            extract_image_format = lambda x: x.get('parent', {}).get('img_fmt', '') if isinstance(x, dict) else ''
            extract_image_mode = lambda x: x.get('parent', {}).get('img_mod', '') if isinstance(x, dict) else ''
            extract_description = lambda x: x.get('parent', {}).get('desc', '') if isinstance(x, dict) else ''
            extract_author = lambda x: x.get('parent', {}).get('aut', '') if isinstance(x, dict) else ''
            extract_copyright = lambda x: x.get('parent', {}).get('cpr', '') if isinstance(x, dict) else ''
            extract_location = lambda x: x.get('parent', {}).get('loc', '') if isinstance(x, dict) else ''
            extract_image_shape = lambda x: x.get('parent', {}).get('img_shp', '') if isinstance(x, dict) else ''
            extract_image_detail_quality = lambda x: x.get('parent', {}).get('image_details', {}).get('quality', '') if isinstance(x, dict) else ''

            # For 'derived' keys
            extract_pixel_1 = lambda x: x.get('derived', {}).get('pxl', '') if isinstance(x, dict) else ''
            extract_megapixels = lambda x: x.get('derived', {}).get('mgp', '') if isinstance(x, dict) else ''
            extract_ppi = lambda x: x.get('derived', {}).get('ppi', '') if isinstance(x, dict) else ''
            extract_exif_dict = lambda x: x.get('derived', {}).get('exif_dict', '') if isinstance(x, dict) else ''
            extract_red = lambda x: x.get('derived', {}).get('red', '') if isinstance(x, dict) else ''
            extract_green = lambda x: x.get('derived', {}).get('green', '') if isinstance(x, dict) else ''
            extract_blue = lambda x: x.get('derived', {}).get('blue', '') if isinstance(x, dict) else ''
            extract_laplacian_variance_blur = lambda x: x.get('derived', {}).get('lvb', '') if isinstance(x, dict) else ''
            extract_fourier_transform_blur = lambda x: x.get('derived', {}).get('ftb', '') if isinstance(x, dict) else ''
            extract_gradient_magnitude_blur = lambda x: x.get('derived', {}).get('gmb', '') if isinstance(x, dict) else ''
            extract_brightness_score = lambda x: x.get('derived', {}).get('bscore', '') if isinstance(x, dict) else ''
            extract_colourfulness = lambda x: x.get('derived', {}).get('clrfln', '') if isinstance(x, dict) else ''
            extract_sharpness_score = lambda x: x.get('derived', {}).get('shpscr', '') if isinstance(x, dict) else ''
            extract_image_metric = lambda x: x.get('derived', {}).get('imgmet', '') if isinstance(x, dict) else ''
            extract_blur_type = lambda x: x.get('derived', {}).get('blur_type', '') if isinstance(x, dict) else ''
            extract_color = lambda x: x.get('color', '') if isinstance(x, dict) else ''
            extract_artifact = lambda x: x.get('derived', {}).get('artifact_verdict', '') if isinstance(x, dict) else ''
            extract_partial_blur = lambda x: x.get('derived', {}).get('partial_blur_status', '') if isinstance(x, dict) else ''

            # Apply these modified lambda functions to your DataFrame
            df['height'] = df['meta'].apply(extract_height)
            df['width'] = df['meta'].apply(extract_width)
            df['size'] = df['meta'].apply(extract_size)
            df['image_format'] = df['meta'].apply(extract_image_format)
            df['image_mode'] = df['meta'].apply(extract_image_mode)
            df['description'] = df['meta'].apply(extract_description)
            df['author'] = df['meta'].apply(extract_author)
            df['copyright'] = df['meta'].apply(extract_copyright)
            df['location'] = df['meta'].apply(extract_location)
            df['image_shape'] = df['meta'].apply(extract_image_shape)

            df['pixels'] = df['meta'].apply(extract_pixel_1)
            df['megapixels'] = df['meta'].apply(extract_megapixels)
            df['ppi'] = df['meta'].apply(extract_ppi)
            df['exif_dict'] = df['meta'].apply(extract_exif_dict)
            df['red'] = df['meta'].apply(extract_red)
            df['green'] = df['meta'].apply(extract_green)
            df['blue'] = df['meta'].apply(extract_blue)
            df['laplacian_variance_blur'] = df['meta'].apply(extract_laplacian_variance_blur)
            df['fourier_transform_blur'] = df['meta'].apply(extract_fourier_transform_blur)
            df['gradient_magnitude_blur'] = df['meta'].apply(extract_gradient_magnitude_blur)
            df['brightness'] = df['meta'].apply(extract_brightness_score)
            df['colourfulness'] = df['meta'].apply(extract_colourfulness)
            df['sharpness'] = df['meta'].apply(extract_sharpness_score)
            df['color'] = df['meta'].apply(extract_color)
            df['image_metric'] = df['meta'].apply(extract_image_metric)
            df['blur_type'] = df['meta'].apply(extract_blur_type)
            df['artifact_verdict'] = df['meta'].apply(extract_artifact)
            df['partial_blur_status'] = df['meta'].apply(extract_partial_blur)

            df['artifact_verdict'] = 0
            df['partial_blur_status'] = 0
            df['imagemagic_quality'] = df['meta'].apply(extract_image_detail_quality)


        # Drop unnecessary columns
        df = df.drop(columns=['meta'], axis=1)
        df.reset_index(inplace=True, drop=True)

        # Convert data types
        df = df.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True)
        # df['blur_type'] = df['blur_type'].astype(str).replace({'blur': '0', 'clear': '20'})
        df['size'] = pd.to_numeric(df['size'], errors='ignore')
        df['ppi'] = pd.to_numeric(df['ppi'], errors='ignore')
        df['width'] = pd.to_numeric(df['width'], errors='coerce').fillna(0).astype(int)
        df['height'] = pd.to_numeric(df['height'], errors='coerce').fillna(0).astype(int)

        # Calculate additional features
        df['size_div_ppi'] = df['size'] / df['ppi']
        df['size_div_ppi2'] = df['size'] ** 2 / df['ppi']
        df['aspect_ratio'] = df['width'] / df['height']
        df.fillna(0, inplace=True)

        # Apply custom scoring for 'blur_type', 'artifact_verdict', and 'partial_blur_status'
        blur_type_score_map = {"CLEAR": 5, "BLUR": 0, "PARTIAL_BLUR": 2, ' ': 1, "SEND TO MODERATION": 0}
        artifact_verdict_score_map = {' ': 2, "NOT ARTIFACT": 5, "ARTIFACT": 0}
        partial_blur_status_score_map = {' ': 2, "FALSE": 5, "TRUE": 0}

        df['blur_type_score'] = df['blur_type'].map(blur_type_score_map).fillna(0)
        df['artifact_verdict_score'] = df['artifact_verdict'].map(artifact_verdict_score_map).fillna(0)
        df['partial_blur_status_score'] = df['partial_blur_status'].map(partial_blur_status_score_map).fillna(0)
        

        # Quality calculation
        D = [str(docids)]

        # Define ranges and scores for each column
        bscore_ranges = [(0.0, 50.9909), (50.9909, 101.9819), (101.9819, 152.9728), (152.9728, 203.9637), (203.9637, 254.9546)]
        bscore_scores = [1, 3, 5, 3, 1]

        gmb_ranges = [(0.0, 28.9526), (28.9526, 57.9051), (57.9051, 86.8577), (86.8577, 115.8103), (115.8103, 144.7628)]
        gmb_scores = [1, 2, 3, 4, 5]

        clrfln_ranges = [(0.0, 18.3482), (18.3482, 36.6965), (36.6965, 55.0447), (55.0447, 73.3929), (73.3929, 91.7411)]
        clrfln_scores = [1, 2, 3, 4, 5]

        shpscr_ranges = [(0.0, 85.7273), (85.7273, 171.4545), (171.4545, 257.1818), (257.1818, 342.9090), (342.9090, 428.6363)]
        shpscr_scores = [1, 2, 3, 4, 5]

        imagemagic_quality_ranges = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 100.0)]
        imagemagic_quality_scores = [1, 2, 3, 4, 5]

        size2_div_ppi_ranges = [(177863.14285714287, 58624666653.90252), (58624666653.90252, 117249155444.66217), (117249155444.66217, 175873644235.42184), (175873644235.42184, 234498133026.1815), (234498133026.1815, 293122621816.94116)]
        size2_div_ppi_scores = [1, 2, 3, 4, 5]


        def custom_bscore_quantize_and_score(column):
            return column.apply(lambda x: CatalogueStarScore.get_score(x, bscore_ranges, bscore_scores)).astype(float)

        def quantize_and_score(column, ranges, scores):
            return column.apply(lambda x: CatalogueStarScore.get_score(x, ranges, scores)).astype(float)


        for i in range(len(D)):
            df = df.iloc[:, :]
            df.reset_index(drop=True, inplace=True)
            gf = df

            # List of columns to quantize and assign scores
            columns_to_quantize = {
                'brightness': (bscore_ranges, bscore_scores),
                'gradient_magnitude_blur': (gmb_ranges, gmb_scores),
                'colourfulness': (clrfln_ranges, clrfln_scores),
                'sharpness': (shpscr_ranges, shpscr_scores),
                'imagemagic_quality': (imagemagic_quality_ranges, imagemagic_quality_scores),
                'size2_div_ppi': (size2_div_ppi_ranges, size2_div_ppi_scores)
            }

            # Apply quantization and scoring to each column
            for column, (ranges, scores) in columns_to_quantize.items():
                if column == 'brightness':  # Example for custom bscore
                    gf[f'{column}_score'] = custom_bscore_quantize_and_score(gf[column])
                else:
                    gf[f'{column}_score'] = quantize_and_score(gf[column], ranges, scores)

            all_scores = [f'{column}_score' for column in columns_to_quantize.keys()] + ['blur_type_score', 'artifact_verdict_score', 'partial_blur_status_score']
            gf['star_score'] = gf[all_scores].mean(axis=1)

            quality_data = pd.concat([gf, df], axis=1)

            quality_data.sort_values(by=['star_score'], ascending=False, inplace=True)
            quality_data['pid'] = quality_data['pid'].astype(str)

            d_overall_star = quality_data['star_score'].mean()

        return quality_data, d_overall_star


    def dictConvert(quality_data,d_overall_star):
        data = quality_data
        data_dict = data.to_dict(orient='records')

        docid_overall_star = d_overall_star

        score = {}
        meta_data = {}

        for i in range(len(data_dict)):
            score[i] = {
                'bscore_score': round(data_dict[i]['brightness_score']),
                'gmb_score': data_dict[i]['gradient_magnitude_blur_score'],
                'clrfln_score': data_dict[i]['colourfulness_score'],
                'shpscr_score': data_dict[i]['sharpness_score'],
                'imagemagic_quality_score': data_dict[i]['imagemagic_quality_score'],
                'size2_div_ppi_score': data_dict[i]['size2_div_ppi_score'],
                'blur_type_score': data_dict[i]['blur_type_score'],
                'artifact_verdict_score': data_dict[i]['artifact_verdict_score'],
                'partial_blur_status_score': data_dict[i]['partial_blur_status_score']
            }

            meta_data[i] = {
                'docid': data_dict[i]['did'],
                'product_id': int(data_dict[i]['pid']),
                'score': score[i],
                'docid_overall_score': str(docid_overall_star),
                'pid_score': str(data_dict[i]['star_score'])

            }
            
        return meta_data


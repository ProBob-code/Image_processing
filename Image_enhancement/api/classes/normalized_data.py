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


class CatalogueScore():
    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    def dataNormalizedCatalogueScore(docid):
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


        # Drop unnecessary columns
        df = df.drop(columns=['meta'], axis=1)
        df.reset_index(inplace=True, drop=True)

        # Convert data types
        df = df.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True)
        df['blur_type'] = df['blur_type'].astype(str).replace({'blur': '0', 'clear': '20'})
        df['size'] = pd.to_numeric(df['size'], errors='ignore')
        df['ppi'] = pd.to_numeric(df['ppi'], errors='ignore')
        df['width'] = pd.to_numeric(df['width'], errors='coerce').fillna(0).astype(int)
        df['height'] = pd.to_numeric(df['height'], errors='coerce').fillna(0).astype(int)

        # Check if 'artifact_verdict' and 'partial_blur_status' columns exist
        if 'artifact_verdict' in df.columns and 'partial_blur_status' in df.columns:
            df['artifact_verdict'] = df['artifact_verdict'].astype(str).replace({'Artifact': '0', 'Not Artifact': '10'})
            df['partial_blur_status'] = df['partial_blur_status'].astype(str).replace({'True': '0', 'False': '10'})
        else:
            if 'blur_type' in df.columns and df['blur_type'].iloc[0] == 'partial_blur':
                df['artifact_verdict'] = '0'
                df['partial_blur_status'] = '0'

        # Handle non-numeric values in 'blur_type' column
        df['blur_type'] = pd.to_numeric(df['blur_type'], errors='coerce').fillna(0)

        # Calculate additional features
        df['size_div_ppi'] = df['size'] / df['ppi']
        df['size_div_ppi2'] = df['size'] ** 2 / df['ppi']
        df['aspect_ratio'] = df['width'] / df['height']
        df.fillna(0, inplace=True)
        


        # Quality calculation
        D = [str(DocId)]

        for i in range(len(D)):
            df = df.iloc[:, :]
            df.reset_index(drop=True, inplace=True)
            gf = df

            Norm_data = gf[['size_div_ppi2', 'ppi', 'blur_type', 'brightness_score', 'colourfulness', 'sharpness_score', 'artifact_verdict', 'partial_blur_status']]
            
            # Replace empty strings with NaN
            Norm_data.replace('', np.nan, inplace=True)

            # Fill NaN values with 0
            Norm_data.fillna(0, inplace=True)
            
            if len(Norm_data) > 0:
                Norm_data.replace('', '0', inplace=True)
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(Norm_data)
                norm_df = pd.DataFrame(x_scaled)
                
                a = norm_df[0] * 20
                b = norm_df[1] * 20
                c = norm_df[2] * 80
                d = norm_df[3] * 10
                e = norm_df[4] * 15
                f = norm_df[5] * 0
                g = norm_df[6] * 30
                h = norm_df[7] * 10
                
                Performance = a + b + c + d + e + f + g + h

                # Initialize the adjusted performance array
                adjusted_performance = Performance.copy()

                # Apply aspect ratio adjustment for each row
                for idx in range(len(gf)):
                    aspect_ratio = gf['aspect_ratio'].iloc[idx]
                    if aspect_ratio >= 2.0 and aspect_ratio <= 4.0:
                        adjusted_performance.iloc[idx] -= 30
                    elif aspect_ratio > 4.0:
                        adjusted_performance.iloc[idx] -= 50
                    
                    # # Debugging each row's aspect ratio and performance
                    # print(f"Index: {idx}, Aspect Ratio: {aspect_ratio}, Initial Performance: {Performance.iloc[idx]}, Adjusted Performance: {adjusted_performance.iloc[idx]}")

                y = pd.DataFrame(adjusted_performance)
                bf = pd.concat([norm_df, y], axis=1, ignore_index=True)
                y_scaled = min_max_scaler.fit_transform(y)
                normalized_df = pd.DataFrame(y_scaled) * 60
                nf = pd.concat([bf, normalized_df], axis=1, ignore_index=True)
                nf.rename(columns={0: 'N_sdp2', 1: 'N_ppi', 2: 'N_bt', 3: 'N_b', 4: 'N_c', 5: 'N_sc', 6: 'N_av', 7: 'N_pbs', 8: 'performance', 9: 'quality_score'}, inplace=True)

                quality_data = pd.concat([gf, nf], axis=1)

                # Source
                L = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 69, '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '69']

                source_scores = []

                for module_type in L:
                    module_df = quality_data[quality_data['mt'] == module_type].copy()
                    # module_df[0:2]
                    lcontract = quality_data['lc'].apply(lambda x: x.get('ut'))
                    upload_by = quality_data['ui'].apply(lambda x:x.get('ub'))
                    # print(upload_by)
                    source_score = []  

                    if module_type in [3,10,'3','10']:
                        for index, row in module_df.iterrows():
                            ub_value = upload_by[index]
                            if ub_value == 'lg_owner_feed':
                                score = 30
                            elif ub_value == 'instagram_feed':
                                score = 21
                            elif ub_value == 'facebook_feed':
                                score = 18
                            else:
                                score = 27
                            source_score.append(score)


                    elif module_type in [5,2,'5','2']:
                        for index, row in module_df.iterrows():
                            lc_value = lcontract[index]  # Use the corresponding lcontract value
                            if lc_value == '':
                            # lc_value = lcontract[index]  # Use the corresponding lcontract value
                            # if lc_value == '':
                            # if row['lc'] == '':
                                score = 24
                            else:
                                a = int(lc_value)
                                b_values = [1, 2, 4] or ['1','2','4']
                                found_matching_b = False
                                score = 24
                                for b in b_values:
                                    c = a & b
                                    if c == b:
                                        score = 24 if b in [1, 4,'1','4'] else 24
                                        found_matching_b = True
                                        break
                            source_score.append(score)   

                    
                    elif module_type in [6, 12, 26, 42,'6','12','26','42']:
                        for index, row in module_df.iterrows():
                            lc_value = lcontract[index]  # Use the corresponding lcontract value
                            if lc_value == '':
                            # if row['lc'] == '':
                                score = 9
                            else:
                                a = int(lc_value)
                                b_values = [1, 2, 4] or ['1','2','4']
                                found_matching_b = False
                                score = 24
                                for b in b_values:
                                    c = a & b
                                    if c == b:
                                        score = 9 if b in [1, 4,'1','4'] else 24
                                        found_matching_b = True
                                        break
                            source_score.append(score)

                    
                    elif module_type in [8,9,21,'8','9','21']:
                        for index, row in module_df.iterrows():
                            lc_value = lcontract[index]  # Use the corresponding lcontract value
                            if lc_value == '':
                            # if row['lc'] == '':
                                score = 12
                            else:
                                a = int(lc_value)
                                b_values = [1, 2, 4] or ['1','2','4']
                                found_matching_b = False
                                score = 12
                                for b in b_values:
                                    c = a & b
                                    if c == b:
                                        score = 12 if b in [1, 4,'1','4'] else 12
                                        found_matching_b = True
                                        break
                            source_score.append(score)
                    
                    elif module_type in [11,'11']:
                        for index, row in module_df.iterrows():
                            lc_value = lcontract[index]  # Use the corresponding lcontract value
                            if lc_value == '':
                            # if row['lc'] == '':
                                score = 15
                            else:
                                a = int(lc_value)
                                b_values = [1, 2, 4] or ['1','2','4']
                                found_matching_b = False
                                score = 15
                                for b in b_values:
                                    c = a & b
                                    if c == b:
                                        score = 15 if b in [1, 4] else 15
                                        found_matching_b = True
                                        break
                            source_score.append(score)


                    elif module_type in [1,7,'1','7']:
                        for index, row in module_df.iterrows():
                            lc_value = lcontract[index]  # Use the corresponding lcontract value
                            if lc_value == '':
                            # if row['lc'] == '':
                                score = 9
                            else:
                                a = int(lc_value)
                                b_values = [1, 2, 4] or ['1','2','4']
                                found_matching_b = False
                                score = 15
                                for b in b_values:
                                    c = a & b
                                    if c == b:
                                        score = 9 if b in [1, 4,'1','4'] else 15
                                        found_matching_b = True
                                        break
                            source_score.append(score)


                    elif module_type in [4,13,16,17,18,19,20,22,23,24,25,27,30,'4','13','16','17','18','19','20','22','23','24','25','27','30']:
                        for index, row in module_df.iterrows():
                            if row['mt'] == 4:
                                score = 3
                            else:
                                score = 3
                            source_score.append(score)


                    elif module_type in [14,15,28,29,31,32,33,34,35,36,37,38,39,40,41,69,'14','15','28','29','31','32','33','34','35','36','37','38','39','40','41','69']:
                        for index, row in module_df.iterrows():
                            if row['mt'] == 14:
                                score = 0
                            else:
                                score = 0
                            source_score.append(score)


                    module_df['source_score'] = source_score
                    source_scores.append(module_df)

                # Combine all DataFrames into a single DataFrame
                merged_df = pd.concat(source_scores, ignore_index=True)
                source_data = quality_data.merge(merged_df[['pid','source_score']], on='pid', how='left')

                # Likes  
                likes_score = []
                for index, row in quality_data.iterrows():
                    likes = 0  # Default value
                    like_value = row['like']
                    if like_value:
                        like_value = int(like_value)
                        if like_value == 5 or like_value >= 5:
                            likes = 10
                        elif like_value == 4:
                            likes = 8
                        elif like_value == 3:
                            likes = 6
                        elif like_value == 2:
                            likes = 4
                        elif like_value == 1:
                            likes = 2
                        elif like_value == 0:
                            likes = 0
                    likes_score.append(likes)

                quality_data['likes_score'] = likes_score

                # Create a DataFrame containing 'pid' and 'likes_score'
                likes_score_df = pd.DataFrame({'pid': quality_data['pid'], 'likes_score': likes_score})

                # Merge the 'likes_score_df' with 'source_data'
                likes_data = source_data.merge(likes_score_df, on='pid', how='left')

                # #Report
                report_score = []
                for index, row in quality_data.iterrows():
                    if row['report'] == 1 or row['report'] == '1':
                        report = -90
                    else:
                        report = 10
                    report_score.append(report)

                quality_data['report_score'] = report_score

                # Create a DataFrame containing 'pid' and 'report_score'
                report_score_df = pd.DataFrame({'pid': quality_data['pid'], 'report_score': report_score})

                # Merge the 'report_score_df' with 'source_data'
                report_data = source_data.merge(report_score_df, on='pid', how='left')

                dataframes_list = [source_data, likes_data, report_data]

                # Combine DataFrames using pd.concat()
                combined_df = pd.concat(dataframes_list, axis=1)

                # You might want to remove duplicate columns, as the same column names can occur in the merged DataFrames
                final_data = combined_df.loc[:, ~combined_df.columns.duplicated()]
                final_data.fillna(0,inplace=True)

                #Final scoring
                final_data['Score'] = final_data['classification'] + final_data['quality_score'] + final_data['likes_score'] + final_data['report_score']
                
                final_data.sort_values(by=['Score'],ascending=False,inplace=True)

                end_time = time.time()
                total_time = end_time - start_time

                hours, rem = divmod(total_time, 3600)
                minutes, seconds = divmod(rem, 60)

                total_time_formatted = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

                # print(f"Start Time: {start_time1}")
                # print(f'\n Time taken to process this image: {total_time_formatted}')

                final_data['pid'] = final_data['pid'].astype(str)

            return final_data


    def dictConvert(final_data):
        data = final_data
        data_dict = data.to_dict(orient='records')

        score = {}
        meta_data = {}

        for i in range(len(data_dict)):
            score[i] = {
                'quality': round(data_dict[i]['quality_score'],3),
                'source': data_dict[i]['source_score'],
                'classification': data_dict[i]['classification'],
                'like': data_dict[i]['likes_score'],
                'report': data_dict[i]['report_score'],
                'final_score': round(data_dict[i]['Score'],3)
            }

            meta_data[i] = {
                'docid': data_dict[i]['did'],
                'product_id': int(data_dict[i]['pid']),
                'score': score[i]
            }

        return meta_data

class CategoryScore():
    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    def dataNormalizedCategoryScore(docid):

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

            df['artifact_verdict'] = 0
            df['partial_blur_status'] = 0

        # Drop unnecessary columns
        df = df.drop(columns=['meta'], axis=1)
        df.reset_index(inplace=True, drop=True)
        
        
        # Convert data types
        df = df.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True)
        df['blur_type'] = df['blur_type'].astype(str).replace({'blur': '0', 'clear': '20'})
        df['size'] = pd.to_numeric(df['size'], errors='ignore')
        df['ppi'] = pd.to_numeric(df['ppi'], errors='ignore')
        df['width'] = pd.to_numeric(df['width'], errors='coerce').fillna(0).astype(int)
        df['height'] = pd.to_numeric(df['height'], errors='coerce').fillna(0).astype(int)

        # Check if 'artifact_verdict' and 'partial_blur_status' columns exist
        if 'artifact_verdict' in df.columns and 'partial_blur_status' in df.columns:
            df['artifact_verdict'] = df['artifact_verdict'].astype(str).replace({'Artifact': '0', 'Not Artifact': '10'})
            df['partial_blur_status'] = df['partial_blur_status'].astype(str).replace({'True': '0', 'False': '10'})
        else:
            if 'blur_type' in df.columns and df['blur_type'].iloc[0] == 'partial_blur':
                df['artifact_verdict'] = '0'
                df['partial_blur_status'] = '0'

        # Handle non-numeric values in 'blur_type' column
        df['blur_type'] = pd.to_numeric(df['blur_type'], errors='coerce').fillna(0)

        # Calculate additional features
        df['size_div_ppi'] = df['size'] / df['ppi']
        df['size_div_ppi2'] = df['size'] ** 2 / df['ppi']
        df['aspect_ratio'] = df['width'] / df['height']
        df.fillna(0, inplace=True)
        
        D = [str(docids)]
        # print(D)

        # Quality 
        for i in range(len(D)):
            df = df.iloc[:, :]
            df.reset_index(drop=True, inplace=True)
            gf = df


            Norm_data = gf[['size_div_ppi2', 'ppi', 'blur_type', 'brightness_score', 'colourfulness', 'sharpness_score', 'artifact_verdict', 'partial_blur_status']]
            
            # Replace empty strings with NaN
            Norm_data.replace('', np.nan, inplace=True)

            # Fill NaN values with 0
            Norm_data.fillna(0, inplace=True)
            
            if len(Norm_data) > 0:
                Norm_data.replace('', '0', inplace=True)
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(Norm_data)
                norm_df = pd.DataFrame(x_scaled)
                
                a = norm_df[0] * 20
                b = norm_df[1] * 20
                c = norm_df[2] * 80
                d = norm_df[3] * 10
                e = norm_df[4] * 15
                f = norm_df[5] * 0
                g = norm_df[6] * 30
                h = norm_df[7] * 10
                
                Performance = a + b + c + d + e + f + g + h

                # Initialize the adjusted performance array
                adjusted_performance = Performance.copy()

                # Apply aspect ratio adjustment for each row
                for idx in range(len(gf)):
                    aspect_ratio = gf['aspect_ratio'].iloc[idx]
                    if aspect_ratio >= 2.0 and aspect_ratio <= 4.0:
                        adjusted_performance.iloc[idx] -= 30
                    elif aspect_ratio > 4.0:
                        adjusted_performance.iloc[idx] -= 50
                    
                    # # Debugging each row's aspect ratio and performance
                    # print(f"Index: {idx}, Aspect Ratio: {aspect_ratio}, Initial Performance: {Performance.iloc[idx]}, Adjusted Performance: {adjusted_performance.iloc[idx]}")

                y = pd.DataFrame(adjusted_performance)
                bf = pd.concat([norm_df, y], axis=1, ignore_index=True)
                y_scaled = min_max_scaler.fit_transform(y)
                normalized_df = pd.DataFrame(y_scaled) * 60
                nf = pd.concat([bf, normalized_df], axis=1, ignore_index=True)
                nf.rename(columns={0: 'N_sdp2', 1: 'N_ppi', 2: 'N_bt', 3: 'N_b', 4: 'N_c', 5: 'N_sc', 6: 'N_av', 7: 'N_pbs', 8: 'performance', 9: 'quality_score'}, inplace=True)

                quality_data = pd.concat([gf, nf], axis=1)

                likes_score = []
                for index, row in quality_data.iterrows():
                    likes = 0  # Default value
                    like_value = row['like']
                    if like_value:
                        like_value = int(like_value)
                        if like_value == 5 or like_value >= 5:
                            likes = 10
                        elif like_value == 4:
                            likes = 8
                        elif like_value == 3:
                            likes = 6
                        elif like_value == 2:
                            likes = 4
                        elif like_value == 1:
                            likes = 2
                        elif like_value == 0:
                            likes = 0
                    likes_score.append(likes)

                quality_data['likes_score'] = likes_score

                # Drop the original 'likes_score' column from 'quality_data'
                quality_data = quality_data.drop('likes_score', axis=1)

                # Create a DataFrame containing 'pid' and 'likes_score'
                likes_score_df = pd.DataFrame({'id': quality_data['id'], 'likes_score': likes_score})

                # Merge the 'likes_score_df' with 'source_data' with explicit suffixes
                likes_data = quality_data.merge(likes_score_df, on='id', how='left')

                ####### 

                # #Report
                # # Report
                report_score = []
                for index, row in quality_data.iterrows():
                    if row['report'] == 1 or row['report'] == '1':
                        report = -90
                    else:
                        report = 10
                    report_score.append(report)

                quality_data['report_score'] = report_score

                # Create a DataFrame containing 'id' and 'report_score'
                report_score_df = pd.DataFrame({'id': quality_data['id'], 'report_score': report_score})

                # Use concat to combine DataFrames along columns
                report_data = pd.concat([quality_data[['id']], report_score_df[['report_score']]], axis=1)


                dataframes_list = [likes_data, report_data]

                # Combine DataFrames using pd.concat()
                combined_df = pd.concat(dataframes_list, axis=1)

                # You might want to remove duplicate columns, as the same column names can occur in the merged DataFrames
                final_data = combined_df.loc[:, ~combined_df.columns.duplicated()]
                final_data.fillna(0,inplace=True)

                # print(final_data.columns)

                #Final scoring
                final_data['Score'] = final_data['classification'] + final_data['quality_score'] + final_data['likes_score'] + final_data['report_score']
                
                final_data.sort_values(by=['Score'],ascending=False,inplace=True)

                end_time = time.time()
                total_time = end_time - start_time

                hours, rem = divmod(total_time, 3600)
                minutes, seconds = divmod(rem, 60)

                total_time_formatted = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

                # print(f"Start Time: {start_time1}")
                # print(f'\n Time taken to process this image: {total_time_formatted}')

                final_data['id'] = final_data['id'].astype(str)

            return final_data


    def dictConvert(final_data):
        data = final_data
        data_dict = data.to_dict(orient='records')

        score = {}
        meta_data = {}

        for i in range(len(data_dict)):
            score[i] = {
                'quality': round(data_dict[i]['quality_score'],3),
                'source': int(0),
                'classification': data_dict[i]['classification'],
                'like': data_dict[i]['likes_score'],
                'report': data_dict[i]['report_score'],
                'final_score': round(data_dict[i]['Score'],3)
            }

            meta_data[i] = {
                'docid': str(data_dict[i]['national_catid']),
                'product_id': int(data_dict[i]['id']),
                'score': score[i]
            }

        return meta_data


#callling necessary libraries
import sys
import os
import pandas as pd
import requests
import time
import numpy as np
from datetime import timedelta

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent1 = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent1)

#import the necessary classes from src folder
from src.classes.class_image_utils import ImageUtils
from src.classes.class_metatag import MetaTag
from src.classes.class_cleaning import Cleaning
from src.libs.rabbitmq import RabbitMQ
from src.libs.helpers import getConfigData, clean_latin1
from datetime import datetime
import json

try:
    #creating a variable for the imported classes
    image_utils = ImageUtils()  
    meta_tag    = MetaTag()
    cleaning    = Cleaning()

    #mentioning the rabbitmq queue details to extract necessary information
    subscribe_queue = 'content_service1'
    rabbit_mq       = RabbitMQ()
    connection      = rabbit_mq.createConnection(getConfigData('rabbitmq'))
    channel         = connection.channel()
    queue           = channel.queue_declare(queue=subscribe_queue, passive=False, durable=False, exclusive=False, auto_delete=False)
    message_count   = queue.method.message_count

    #extract only if the message count is greater than 0
    # if(message_count > 0):
    if True:

        def callback(ch, method, properties, body):
            
            is_queue_data_processed = True
            queue_data              = json.loads(clean_latin1(body))
            
            #create a new variable for the data that has been extracted from queue
            data = queue_data

            print('\n -------------- Image Processing -------------- ')

            #this is to record the time for processing each image
            start_time = time.time()
            #this is to record the date and time of the process to cross check how much time it took to process the whole queue
            start_time1 = datetime.now()

            print(type(data)) #identify the type of data i.e is it Dataframe or list or array etc

            #Extract the relevant fields
            product_id = data["main"]["product_id"]
            image_url = data["main"]["product_url"]
            data_1 = pd.DataFrame({"product_id": [product_id], "product_url": [image_url]})
            print(data_1)

            print('\n Get data from queue -- done')

# ----------------------------------------------------------------------------------------------- #


            # # TASK 1 : GET DATA
            #the process to download image urls and save it in a folder - new_images
            df1_tuple,desire_dir = image_utils.initialData(data_1)
            df1 = pd.DataFrame(df1_tuple, columns=['image_name_1', 'product_id_1']) #this is used to convert tuple to Dataframe datatype
            print(type(df1),desire_dir)

            print('\n Task 1 -- done','\n Start with Task 2')


# ------------------------------------------------------------------------------------------------ #


            # TASK 2 : GET RULESET [VALIDATE DATA]
            #create a dataframe with necessary columns
            df2 = pd.DataFrame(columns=['image_name_1','height_1','width_1','resolution_1','megapixels_1','ppi_1','size_1','img_format_1','img_mode_1','exif_dict_1','description_1','keywords_1','author_1','copyright_1','location_1','laplacian_variance_blur_1','fourier_transform_blur_1','gradient_magnitude_blur_1','red_1','green_1','blue_1','image_shape_1','matrix_1','status'])
            print(type(df2))

            #this is to preprocess and skip those images no url in queue
            if df1.loc[0, 'image_name_1'] == 0:
                df2.loc[0] = [0] * (len(df2.columns)-1) + ['empty']
            
            #the main process to extract metatags
            else:
                df2 = meta_tag.imageMetatag()
                print(df2)
            
            #the process to calculate image_metric using the below mentioned columns
            try:
                df2['image_metric_1'] = df2.apply(lambda row: np.sum(np.array(row['matrix_1']).flatten()) + row['size_1'] + row['blue_1'] + row['green_1'] + row['red_1'] + row['ppi_1'], axis=1)
                df2['duplicate_1'] = 0
            except:
                df2['image_metric_1'] = 0
                df2['duplicate_1'] = 0

            #clubbing two dataframes to final_data
            final_data = pd.concat([df1, df2], axis=1)
            final_data.fillna(0, inplace=True)
            final_data.drop(['matrix_1'], axis=1, inplace=True) #this column is not needed while sending to mongo, this was only needed to calculate image_metric

            #this is to get the dataframe into a dictionary format - meta:{product_id:,{parent:,derived:}}
            meta = meta_tag.dictConvert(final_data)
            #this is to push the dictionary into mongodb
            meta_tag.apiPush(meta)

            c = '/var/log/images_bkp/new_images/' #the path where the images are getting downloaded


# --------------------------------------------------------------------------------------------------- #


            # TASK 3 : SET PROCESS
            cleaning.removeImages(c)
            print('Done with all steps')

            #recording the endtime to check the processtime
            end_time = time.time()
            total_time = end_time - start_time

            hours, rem = divmod(total_time, 3600)
            minutes, seconds = divmod(rem, 60)

            total_time_formatted = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
            
            print(f"Start Time: {start_time1}")
            print(f'\n Time taken to process this image: {total_time_formatted}')
            

# --------------------------------------------------------------------------------------------------- #


            #this process is to check the values in queue and remove it from queue after processing it
            if method.delivery_tag == queue.method.message_count:
                channel.stop_consuming(consumer_tag=method.consumer_tag)

            if(is_queue_data_processed == True):
                channel.basic_ack(delivery_tag = method.delivery_tag)
                
            else:
                print('#|#|')
                # NEED TO ADD SOME FUNCTIONALITY WHEN TASK FAILED TO SUCCESSFULLY PROCESS DATA
                # NOTIFY OWNER
                # PUSH TO GENERAL QUEUE

                print('###########################')
                print('Something went wrong! Failed to publish data in next queue')
                print('###########################')

                channel.basic_ack(delivery_tag = method.delivery_tag)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=subscribe_queue, on_message_callback=callback)
        
        print('[SUBSCRIBERS (EditListing)] [*] Waiting for messages. To exit press CTRL+C')
        channel.start_consuming()

    else:
        print(f"SUBSCRIBERS (EditListing)] No data available in queue : {subscribe_queue}")      

except Exception as e:

    print("[SUBSCRIBERS (EditListing)] failed")
    print(str(e))
    exit(1)

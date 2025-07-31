import sys
import os   
import pandas as pd
import requests
import time
import numpy as np
from datetime import timedelta
import mysql.connector

# Import other necessary libraries

try:
    # Import necessary classes and set up connections
    
    # calling necessary libraries
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

    # import the necessary classes from src folder
    from src.classes.class_image_utils import ImageUtils
    from src.classes.class_metatag import MetaTag
    from src.classes.class_cleaning import Cleaning	
    from src.libs.rabbitmq import RabbitMQ
    from src.libs.helpers import getConfigData, clean_latin1
    from datetime import datetime
    import json
    import pika

    def process_queue_data():
        # creating a variable for the imported classes  
        image_utils = ImageUtils()  
        meta_tag    = MetaTag()
        cleaning    = Cleaning()

        # mention the rabbitmq queue details to extract necessary information
        subscribe_queue = 'meta_process'
        rabbit_mq       = RabbitMQ()
        connection      = rabbit_mq.createConnection(getConfigData('rabbitmq'))
        channel         = connection.channel()
        queue           = channel.queue_declare(queue=subscribe_queue, passive=False, durable=True, exclusive=False, auto_delete=False)
        message_count   = queue.method.message_count

        # extract only if the message count is greater than 0
        if True:
            try:
                def callback(ch, method, properties, body):

                    is_queue_data_processed = True
                    queue_data              = json.loads(clean_latin1(body))
                    
                    # create a new variable for the data that has been extracted from queue
                    data = queue_data
                    c = getConfigData('NFS_path.image_folder') # the path where the images are getting downloaded

                                      
                    print('\n -------------- Image Processing -------------- ')

                    # this is to record the time for processing each image
                    start_time = time.time()
                    # this is to record the date and time of the process to cross check how much time it took to process the whole queue
                    start_time1 = datetime.now()

                    # Extract the relevant fields
                    product_id = data["main"]["product_id"]
                    image_url = data["main"]["product_url"]
                    docid = data["main"]["docid"]
                    local_path = data["main"]["localpath"]
                    business_tag = data["main"]["business_tag"]
                    product_url_ori = data["main"]["product_url_ori"]
                    data_from_queue = pd.DataFrame({"product_id": [product_id], "product_url": [image_url], "docid": [docid], "localpath": [local_path], "business_tag":[business_tag], "product_url_ori":[product_url_ori]})

                    print('\n Get data from queue -- done')

                    print(data_from_queue)

                    mydb = mysql.connector.connect(
                    host= getConfigData('mysql.host'),
                    user= getConfigData('mysql.username'),
                    password= getConfigData('mysql.password')
                    )

                    mycursor = mydb.cursor()
                    mycursor.execute("SELECT process_flag FROM db_product.tbl_catalogue_details_meta_process WHERE product_id = %s",[product_id])

                    myresult = mycursor.fetchall()

                    if myresult[0][0] == 2:
                        print("The image is processed and the process flag is: ", myresult[0][0])

                        

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

                    else:
                        print("This needs processing because the process flag is: ", myresult[0][0])  

                        if data_from_queue['localpath'].iloc[[0]].item() == None:
                            data_1 = pd.DataFrame({"product_id": [product_id], "product_url": [image_url], "docid": [docid], "business_tag":[business_tag], "product_url_ori":[product_url_ori]})
                            print('data1\n',data_1)
                            df1_tuple = image_utils.initialData1(data_1)
                        else:
                            data_2 = pd.DataFrame({"product_id": [product_id], "product_url": [image_url], "docid": [docid], "localpath": [local_path], "business_tag":[business_tag], "product_url_ori":[product_url_ori]})
                            print('data2\n',data_2)
                            df1_tuple = image_utils.initialData2(data_2)
                        
                        # Task 1: Get data
                        # the process to download image urls and save it in a folder - new_images
                        df1 = pd.DataFrame(df1_tuple, columns=['image_name_1', 'product_id_1', 'docid_1', 'path_flag_1', 'business_tag_1', 'product_url_ori_1'])
                        # this is used to convert tuple to Dataframe datatype
                        print(type(df1))

                        print('\n Task 1 -- done', '\n Start with Task 2')

                        # print(df1)
                        print(df1.loc[0])

                        # Task 2: Get ruleset [Validate data]
                        # create a dataframe with necessary columns

                        column_names = ['image_name_1', 'height_1', 'width_1', 'resolution_1', 'megapixels_1', 'ppi_1', 'size_1', 'img_format_1', 'img_mode_1', 'exif_dict_1', 'description_1', 'keywords_1', 'author_1', 'copyright_1', 'location_1', 'laplacian_variance_blur_1', 'fourier_transform_blur_1', 'gradient_magnitude_blur_1', 'red_1', 'green_1', 'blue_1', 'image_shape_1', 'matrix_1', 'brightness_score_1', 'colourfulness_1', 'sharpness_score_1', 'size_ori_1', 'width_ori_1', 'height_ori_1', 'status']
                        df2 = pd.DataFrame(columns=column_names)

                        # Add a new row with default values
                        # Add a new row with default values and 'corrupt' status
                        if df1.loc[0, 'image_name_1'] == 0:
                            df2.loc[0] = ['0',0,0,0,0.0,0,0.0,'0','0','0','0','0','0','0','0',0.0,0.0,0.0,0.0,0.0,0.0,'0','0',0.0,0.0,0.0,0.0,0,0,'corrupt']
                            d = 0
                            df2['hash_value'] = str(0)

                        # the main process to extract metatags
                        else:
                            df2, image_name = meta_tag.imageMetatag(df1)
                            d = getConfigData('NFS_path.image_folder') + str(image_name)
                            print(df2, d)
                            if df2['image_name_1'].iloc[0] == 0:
                                df2['hash_value'] = str(0)
                            else:
                                result = meta_tag.hash_value_extract(df1,d)
                                df2['hash_value'] = str(result)
                        
                        # the process to calculate image_metric using the below mentioned columns
                        try:
                            df2['image_metric_1'] = df2.apply(lambda row: np.sum(np.array(row['matrix_1']).flatten()) + row['size_1'] + row['blue_1'] + row['green_1'] + row['red_1'] + row['ppi_1'], axis=1)
                            df2['duplicate_1'] = 0
                        except:
                            df2['image_metric_1'] = 0.0
                            df2['duplicate_1'] = 0


                        # clubbing two dataframes to final_data
                        final_data = pd.concat([df1, df2], axis=1)
                        final_data.fillna(0, inplace=True)
                        final_data.drop(['matrix_1'], axis=1, inplace=True) # this column is not needed while sending to mongo, this was only needed to calculate image_metric
                        
                        print(type(final_data))


                        ## here will come the color hex extract function and then we'll have to merge after derived node into the main meta
                        color_data = MetaTag.extract_color(c, d, 11, final_data)
                        #color_data.dtypes
                        # print("the_resized_image is :", resize_path)
                        ##

                        # this is to get the dataframe into a dictionary format - meta:{product_id:,{parent:,derived:}}
                        meta = meta_tag.dictConvert(final_data, color_data, d)
                        print(meta[0])
                        
                        # this is to push the dictionary into mongodb
                        meta_tag.apiPush(meta)

                        # Task 3: Set process
                        #cleaning.removeImages(resize_path)
                        print('Done with all steps')

                        #cleaning.removeImages(d)

                        # recording the end time to check the process time
                        end_time = time.time()
                        total_time = end_time - start_time

                        hours, rem = divmod(total_time, 3600)
                        minutes, seconds = divmod(rem, 60)

                        total_time_formatted = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

                        print(f"Start Time: {start_time1}")
                        print(f'\n Time taken to process this image: {total_time_formatted}')

                        #cleaning.remove_resize_images()

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

            except Exception as e:

                print("[SUBSCRIBERS (EditListing)] failed")
                print(str(e))
                exit(1)
    
except Exception as e:

    print("[SUBSCRIBERS (EditListing)] failed")
    print(str(e))
    exit(1)

process_queue_data()


#import necessary libraries
import os
from PIL import Image
from pathlib import Path
import cv2 as cv2


class Cleaning():

    @staticmethod
    def removeImages(c):
        #identify the image path
        NFS_path = '/var/log/images_bkp'
        list_images = str(NFS_path) + '/new_images'
        new_list_images = c
        
        #identify only the files with the below mentioned extensions
        images = [f for f in os.listdir(list_images) if f.lower().endswith(('.jpg', '.png', '.jpeg','.JPG','.gif'))]
        new_images = [f for f in os.listdir(new_list_images) if f.lower().endswith(('.jpg', '.png', '.jpeg','.JPG','.gif'))]
        print(new_images)

        #remove the files
        for i in range(len(images)):
            file_to_remove = list_images+'/'+images[i]
            try:
                os.remove(str(file_to_remove))
            except:
                print('All working fine')

        #if the images are moved to new file path then this cleaning loop will work
        for i in range(len(new_images)):
            new_file_to_remove = new_list_images+new_images[i]
            try:
                os.remove(str(new_file_to_remove))
                
            except:
                print('Thank you, all working fine')

    

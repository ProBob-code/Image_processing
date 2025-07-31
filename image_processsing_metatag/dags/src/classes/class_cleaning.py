#import necessary libraries
import os
from PIL import Image
from pathlib import Path
import cv2 as cv2
from src.libs.helpers import getConfigData


class Cleaning():

    # @staticmethod
    # def removeImages(file_path):
    #     if file_path:
    #         try:
    #             os.remove(file_path)
    #         except OSError as e:
    #             print(f"Error while removing the file: {e}")
    #     else:
    #         print("File was corrupt. Skipping removal.")

    @staticmethod
    def removeImages(d):
        new_list_images = d
        file_to_remove = new_list_images
        if file_to_remove:
            try:
                os.remove(str(file_to_remove))
            except OSError as e:
                print(f"This image got deleted already: {e}")
        else:
            print("File was corrupt. Skipping removal.")


    
    @staticmethod
    def remove_resize_images():
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        path = getConfigData('NFS_path.path')
        # new_list_images = c
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in image_extensions):
                os.remove(file_path)     

    

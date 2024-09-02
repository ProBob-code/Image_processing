import pandas as pd
import numpy as np
import json
from flask import Flask
from colormap import rgb2hex
import extcolors
from PIL import Image  # Import the Pillow library for image resizing
import re

class ExtractColor():
    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    def colorToDf(input):
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

    def hexToRgb(hex_code):
        hex_code = str(hex_code).strip('#')  # Remove '#' symbol
        r = int(hex_code[0:2], 16)  # Convert red component from hex to decimal
        g = int(hex_code[2:4], 16)  # Convert green component from hex to decimal
        b = int(hex_code[4:6], 16)  # Convert blue component from hex to decimal
        return r, g, b

    def is_single_tone(image_path, colors_x, threshold=5):
        # colors_x = extcolors.extract_from_path(image_path, tolerance=11, limit=13)
        
        _, total_pixels = colors_x
        df_color = ExtractColor.colorToDf(colors_x)
        colors_count = list(df_color['occurence'])

        percentage_list =[]
              
        for color_pixels in colors_count:
            color_pixels = int(color_pixels)
            color_percentage = color_pixels * 100  /  total_pixels 
            percentage_list.append(color_percentage)

        max_percentage =0
        percentage_list.sort()
 
             
        if len(percentage_list) >= 1:
            max_percentage += percentage_list[-1]
        elif len(percentage_list) >= 2: 
            max_percentage += percentage_list[-2] 
        elif len(percentage_list) >= 3:    
            max_percentage += percentage_list[-3]
        
       
        distinct_colors = len(df_color)
        
        if distinct_colors <= threshold:
                return "True"
        elif (max_percentage >= 75): 
              return "True"
        else:
             return "False"

    def extractColor(path):
        result_rows = []  # List to store the result rows
        # output_folder = getConfigData('NFS_path.path')
        tolerance = 11

        # Load the image using Pillow
        img = Image.open(path)
        width = img.width
        height = img.height
        if width > 250:
            new_width = 250
            new_height = int((height/width)*new_width)

            # Resize the image to 250x(calculated height) pixels
            img = img.resize((new_width, new_height))

            # Save the resized image back to the ORIGINAL path (overwrite the original)
            img.convert('RGB').save(path)  # Use the original path here
            path = path #resized path

        else:
            path = path #old path

        # Create dataframe
        colors_x = extcolors.extract_from_path(path, tolerance=tolerance, limit=13)
        df_color = ExtractColor.colorToDf(colors_x)

        list_color = list(df_color['c_code'])
        rgb_values = [ExtractColor.hexToRgb(hex_code) for hex_code in list_color]
        list_precent = [int(i) for i in list(df_color['occurence'])]
        text_c = [{'hex': str(c), 'percentage': round(p * 100 / sum(list_precent), 1), 'rbg_values': str(r)} for c, p, r in
                zip(list_color, list_precent, rgb_values)]

        result_row = text_c
        result_rows.append(result_row)

        # Create final dictionary with the desired format
        output_dict = result_rows
        rearranged_dict = output_dict[0]
	
        verdict = ExtractColor.is_single_tone(path, colors_x)
       
        #print("VERDICT : " , verdict)
        return rearranged_dict, verdict, path


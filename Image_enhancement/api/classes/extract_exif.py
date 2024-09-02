import PIL
import PIL.Image
from PIL.ExifTags import TAGS
import pandas as pd
import io
import mysql.connector
import re
import json
from phpserialize import loads
from helper import getConfigData

class Extract_exif:
    
    def __init__(self):
        pass

    @staticmethod
    def preprocess_php_serialized_data(data):
        """
        Preprocesses PHP serialized data to correct string length specifiers.
        """
        def correct_string_length(match):
            string_content = match.group(2)
            correct_length = len(string_content.encode('utf-8'))
            return 's:{0}:"{1}";'.format(correct_length, string_content)

        corrected_data = re.sub(r's:(\d+):"((?:[^"\\]|\\.)*)";', correct_string_length, data)
        return corrected_data

    @staticmethod
    def extract_php_serialized_data(php_data):
        """
        Extracts PHP serialized data using regex pattern.
        """
        pattern = re.compile(r's:(\d+):"(.*?)";|i:(\d+);', re.DOTALL)
        extracted_data = {}
        key = None

        for match in pattern.finditer(php_data):
            if match.group(2):  # String value found
                if key is None:
                    key = match.group(2)  # First occurrence is key
                else:
                    extracted_data[key] = match.group(2)  # Second occurrence is value
                    key = None  # Reset for next key-value pair
            elif match.group(3) and key:  # Integer value found and a key exists
                extracted_data[key] = int(match.group(3))
                key = None

        return extracted_data

    @staticmethod
    def parse_data(data):
        """
        Parses data from PHP serialized or JSON format.
        """
        preprocessed_data = Extract_exif.preprocess_php_serialized_data(data)
        #print(preprocessed_data)
        extracted_data = {}

        # Define the EXIF attributes of interest
        exif_list = ['FileName', 'FileDateTime', 'FileType', 'MimeType', 'Orientation', 'Exif_IFD_Pointer', 'ColorSpace', 'Make', 'Model', 'XResolution', 'YResolution', 'ResolutionUnit', 'Software', 'DateTime', 'ExifOffset', 'ShutterSpeedValue', 'DateTimeOriginal', 'DateTimeDigitized', 'ApertureValue', 'BrightnessValue', 'ExposureBiasValue', 'MaxApertureValue', 'MeteringMode', 'LightSource', 'Flash', 'FocalLength', 'SubsecTime', 'SensingMethod', 'ExposureTime', 'FNumber', 'ExposureProgram', 'ISOSpeedRatings', 'ExposureMode', 'WhiteBalance', 'DigitalZoomRatio', 'FocalLengthIn35mmFilm', 'SceneCaptureType']

        try:
            if data.startswith("a:"):
                exif_bytes = preprocessed_data.encode('utf-8')
                php_data = loads(exif_bytes, decode_strings=True)
                extracted_data = {key: php_data[key] for key in exif_list[0:30] if key in php_data}
            else:
                exif_data_json = json.loads(preprocessed_data)
                extracted_data = {key: exif_data_json[key] for key in exif_list if key in exif_data_json}
        except Exception as e:
            print(f"Exception during deserialization: {e}")
            extracted_data = Extract_exif.extract_php_serialized_data(preprocessed_data)

        return extracted_data


    def precheck(pid):
        product_id = pid

        try:
            mydb = mysql.connector.connect(
                host=getConfigData("mysql_17_132.host"),
                user=getConfigData("mysql_17_132.username"),
                password=getConfigData("mysql_17_132.password")
            )

            mycursor = mydb.cursor()
            mycursor.execute("SELECT exif_data FROM db_product.tbl_catalogue_image_info WHERE product_id = %s", [product_id])

            myresult = mycursor.fetchall()

            # Check if myresult contains data
            if myresult and myresult[0][0]:
                # print('pre data: ',myresult[0][0])
                extracted_data = Extract_exif.parse_data(myresult[0][0])
                # print("post extracted data: ",extracted_data)
            else:
                extracted_data = str(0)

        except Exception as e:
            # Handle the exception case
            print("Exception:", e)
            extracted_data = str(0)

        # Convert extracted_data to JSON
        extracted_data_json = json.dumps(extracted_data)

        return json.loads(extracted_data_json)


    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)


    def exif_generate(path):
        image_path = path

        exif_list = ['Make', 'Model', 'XResolution', 'YResolution', 'ResolutionUnit', 'Software', 'DateTime', 'ExifOffset', 'ShutterSpeedValue', 'DateTimeOriginal', 'DateTimeDigitized', 'ApertureValue', 'BrightnessValue', 'ExposureBiasValue', 'MaxApertureValue', 'MeteringMode', 'LightSource', 'Flash', 'FocalLength', 'SubsecTime', 'SensingMethod', 'ExposureTime', 'FNumber', 'ExposureProgram', 'ISOSpeedRatings', 'ExposureMode', 'WhiteBalance', 'DigitalZoomRatio', 'FocalLengthIn35mmFilm', 'SceneCaptureType']

        # with open(image_path, 'rb') as f:
        #     image = PIL.Image.open(io.BytesIO(f.read()))
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image = PIL.Image.open(io.BytesIO(image_data))
                exif_data = image._getexif()
        except PIL.UnidentifiedImageError:
            # Handle the case where the image format is not recognized
            print(f"Error: Could not identify image format for {image_path}")
            raise PIL.UnidentifiedImageError(f"Error: Could not identify image format for {image_path}")
        if exif_data:
            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in exif_list and value is not None and value != '':
                    # Remove null ('\x00') values
                    if not isinstance(value, bytes) or b'\x00' not in value:
                        exif_dict[tag] = value

            # Create a DataFrame from the filtered EXIF data
            exif_dataframe = pd.DataFrame.from_dict([exif_dict])

            # Convert the DataFrame to JSON
            exif_data_json = exif_dataframe.to_json(orient='records', default_handler=str, indent=4)

        else:
            # print("No EXIF metadata found in the image.")
            exif_dataframe = pd.DataFrame()
            exif_data_json = "0"  # Empty JSON object

        return json.loads(exif_data_json)

    
    def final_parsed(old_exif):

        parsed_data = old_exif

        # Define a function to convert to the desired data type
        def convert_to_str(value, default="0"):
            return str(value) if value is not None else default

        def convert_to_int(value, default=0):
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        def convert_to_float(value, default=0.0):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        # Create a list of keys and data type conversion functions
        key_data_type_mapping = [
            ('Orientation', convert_to_int),
            ('ExifIFDPointer', convert_to_int),
            ('Exif_IFD_Pointer', convert_to_int),
            ('ColorSpace', convert_to_int),
            ('Make', convert_to_str),
            ('Model', convert_to_str),
            ('XResolution', convert_to_int),
            ('YResolution', convert_to_int),
            ('ResolutionUnit', convert_to_int),
            ('Software', convert_to_str ),
            ('DateTime', convert_to_str ),
            ('ShutterSpeedValue', convert_to_float),
            ('DateTimeOriginal', convert_to_str ),
            ('DateTimeDigitized', convert_to_str, ),
            ('ApertureValue', convert_to_float),
            ('BrightnessValue', convert_to_int ),
            ('MaxApertureValue', convert_to_int ),
            ('MeteringMode', convert_to_str),
            ('LightSource', convert_to_str),
            ('Flash', convert_to_str),
            ('ISOSpeedRatings', convert_to_int),
            ('ExposureMode', convert_to_int),
            ('FNumber', convert_to_str),
            ('ExposureProgram', convert_to_str),
            ('ISOSpeedRatings', convert_to_int),
            ('ExposureMode', convert_to_str),
            ('WhiteBalance', convert_to_str),
            ('FocalLengthIn35mmFilm', convert_to_int),
            ('DigitalZoomRatio', convert_to_str),
            ('SceneCaptureType', convert_to_str)
        ]

          # Check data type before conversion
        if not isinstance(parsed_data, dict):
            return {}  # Return empty dictionary if not a dictionary

        # Create the new dictionary with modified data types
        new_dict = {key: data_type(parsed_data.get(key)) for key, data_type in key_data_type_mapping}
        #print('new dict: ', new_dict)
        # Specify keys with constant data type
        constant_data_type_keys = ['FileName', 'FileDateTime', 'FileType', 'MimeType', 'DateTimeDigitized', 'SceneCaptureType']
        for key in constant_data_type_keys:
            new_dict[key] = str(parsed_data.get(key, 0))

        # Convert the combined dictionary back to JSON
        parsed_data_str = json.dumps(new_dict, indent=4)

        return json.loads(parsed_data_str)


    def combine_json(parsed_data, exif_data):
        # Initialize an empty dictionary to combine data
        combined_json = {}
        
        # print('old_exif:',parsed_data)
        # print('new_exif:',exif_data)

        # Define data types for each key
        data_types = {
            'FileName': str,
            'FileDateTime': str,
            'FileType': str,
            'MimeType': str,
            'Orientation': int,
            'Exif_IFD_Pointer': int,
            'ExifIFDPointer': int,
            'ColorSpace': int,
            'Make': str,
            'Model': str,
            'XResolution': str,
            'YResolution': str,
            'ResolutionUnit': int,
            'Software': str,
            'DateTime': str,
            'ExifOffset': int,
            'ShutterSpeedValue': float,
            'DateTimeOriginal': str,
            'DateTimeDigitized': str,
            'ApertureValue': float,
            'BrightnessValue': float,
            'ExposureBiasValue': str,
            'MaxApertureValue': str,
            'MeteringMode': str,
            'LightSource': str,
            'Flash': str,
            'FocalLength': str,
            'SubsecTime': str,
            'SensingMethod': int,
            'ExposureTime': str,
            'FNumber': str,
            'ExposureProgram': str,
            'ISOSpeedRatings': int,
            'ExposureMode': int,
            'WhiteBalance': str,
            'DigitalZoomRatio': str,
            'FocalLengthIn35mmFilm': int,
            'SceneCaptureType': str
        }

        # Check if parsed_data is a dictionary
        if isinstance(parsed_data, dict):
            # Add parsed_data to combined_json with the specified data types only if data is present
            combined_json.update({key: data_types[key](parsed_data[key]) for key in parsed_data if parsed_data[key] is not None and parsed_data[key] != ''})

        # Check if exif_data is a dictionary
        if isinstance(exif_data, dict):
            # Add exif_data to combined_json with the specified data types only if data is present
            combined_json.update({key: data_types[key](exif_data[key]) for key in exif_data if exif_data[key] is not None and exif_data[key] != ''})
            
        # print('final_exif:',combined_json)

        # Convert the combined dictionary back to JSON
        combined_json_str = json.dumps(combined_json, indent=4)

        return json.loads(combined_json_str)



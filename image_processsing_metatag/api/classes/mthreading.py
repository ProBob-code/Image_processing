from classes.blur_identification_v3 import BlurIdentify_V3
from classes.calculate_brightness import CalculateBrighntess
from classes.extract_color import ExtractColor
from classes.imagemeta_tag import ImageMeta
from classes.extract_exif import Extract_exif
from classes.partial_blur import Partial_Blur
import json
import threading
from queue import Queue
from PIL import Image
import numpy as np
import cv2
import re
import os
import logging

logging.basicConfig(level=logging.INFO)

class MultiThreading():

    def __init__(self):
        pass    

    @staticmethod
    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    @staticmethod
    def brightness_thread(image_path):
        response = {}
        # try:
        brightness_val = CalculateBrighntess.calculateOverallBrightness(image_path)
        response = {
            'brightness_thread': {
                'brightness_score': float(brightness_val)
            }
        }
        # except Exception as e:
        #     response = {'brightness_thread': str(e)}
        return response

    @staticmethod
    def color_thread(image_path):
        response = {}
        # try:
        color_nodes, verdict, modified_path = ExtractColor.extractColor(image_path)
        response = {
            'color_thread': {
                'color': color_nodes,
                'single_color': verdict,
                'path': modified_path
            }
        }
        # except Exception as e:
        #     response = {'color_thread': str(e)}
        return response

    @staticmethod
    def hash_thread(image_path):
        response = {}
        # try:
        avg_hash, d_hash, p_hash, c_hash, dup_signature = ImageMeta.avgHashValueExtract(image_path)
        response = {
            'hash_thread': {
                'average_hash': avg_hash,
                'difference_hash': d_hash,
                'perceptual_hash': p_hash,
                'color_hash': c_hash,
                'dup_signature': dup_signature
            }
        }
        # except Exception as e:
        #     response = {'hash_thread': str(e)}
        return response

    @staticmethod
    def exif_thread(image_path, pid):
        response = {}
        try:
            new_exif = Extract_exif.exif_generate(image_path)
            old_exif_final = {}

            if not isinstance(new_exif, int):
                final_exif_generated = Extract_exif.combine_json(old_exif_final, new_exif[0])
                response = { 
                    'exif_thread':{
                        'final_exif_generated': final_exif_generated
                    }
                }
            else:
                final_exif_generated = old_exif_final
                response = { 
                    'exif_thread':{
                        'final_exif_generated': final_exif_generated
                    }
                }
        except Exception as e:
            response = {'exif_thread': str(e)}
        return response

    @staticmethod
    def meta_thread(input):
        response = {}
        # try:
        metadata = ImageMeta.imageMetatag(input)
        metadata_ImageMagick = ImageMeta.image_sys_details(input["local_path"])
        response['meta_thread'] = json.loads(metadata)
        response['meta_thread']['meta']['derived']["channel_statistics"] = metadata_ImageMagick["channel_statistics"]
        response['meta_thread']['meta']['derived']["image_statistics"] = {}
        response['meta_thread']['meta']['derived']["image_statistics"]["overall"] = metadata_ImageMagick["image_statistics"]["overall"].copy()
        metadata_ImageMagick.pop("channel_statistics")
        metadata_ImageMagick["image_statistics"].pop("overall")
        # changing keys of dictionary
        metadata_ImageMagick['image_details'] = metadata_ImageMagick['image_statistics']
        del metadata_ImageMagick['image_statistics']

        # metadata_ImageMagick.pop("image_statistics")
        response['meta_thread']['meta']['parent'].update(metadata_ImageMagick)
        # except Exception as e:
        #     response = {'meta_thread': str(e)}
        return response
    
    @staticmethod
    def blur_thread(image_path, product_url):
        response = {}
        try:
            # Step 1: Open the image using PIL to handle different formats
            img = Image.open(image_path)
            
            # Log the initial mode and format of the image for debugging
            logging.info(f"Initial image mode: {img.mode}, format: {img.format}")
            
            # Step 2: Convert the image to RGB if it's grayscale or has a transparency channel (e.g., PNG)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
                logging.info("Image converted to RGB mode")
            elif img.mode == "RGBA":
                img = img.convert("RGB")  # Remove alpha channel
                logging.info("RGBA image converted to RGB by dropping alpha channel")
            
            # Step 3: Resize the image
            height, width = img.height, img.width
            new_width = 1600
            new_height = int((height / width) * new_width)
            resized_img = img.resize((new_width, new_height))
            resized_img.save(image_path)  # Save the resized image
            
            # Convert the resized image to a NumPy array
            resized_img = np.array(resized_img)
            
            # Step 4: Modify product URL
            product_url = re.sub(r"\\\/", "/", product_url)
            
            # Fetch image metadata
            size = ImageMeta.fetchImageHeightWidthV2(product_url).get('size', -1)
            
            # Step 5: Use OpenCV to read the image again for further processing
            resized_img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Force 3-channel image
            
            # Check if the image was read successfully
            if resized_img_cv is None:
                raise ValueError(f"Failed to read the image at {image_path} using OpenCV")
            
            # Step 6: Ensure that the image is BGR with 3 channels
            if len(resized_img_cv.shape) != 3 or resized_img_cv.shape[2] != 3:
                raise ValueError(f"Invalid image format, expected a 3-channel image, but got {resized_img_cv.shape}")
            
            # Step 7: Call the Blur identification function
            blur_response = BlurIdentify_V3.is_blurr_v3(image_path, size)
            
            # Partial blur detection
            _, _, partial_blur_status = Partial_Blur.create_grid(
                size, resized_img_cv, grid_pixel=120, p_score_min=5.0, 
                p_score_max=30.0, size_cutoff=30000, 
                sharpness_score_cutoff=40, top_percent=0.05
            )
            
            # Prepare the response with blur information
            response = {
                'blur_thread': {
                    'blur_type': str(blur_response['blur_type']),
                    'laplacian_variance_blur': float(blur_response['laplacian_variance_blur']),
                    'fourier_transform_blur': float(blur_response['fourier_transform_blur']),
                    'gradient_magnitude_blur': float(blur_response['gradient_magnitude_blur']),
                    'laplacian_variance_main': float(blur_response['laplacian_variance_main']),
                    'partial_blur_status': str(partial_blur_status),
                    'artifact_verdict': str(blur_response['artifact_verdict'])
                }
            }
        except Exception as e:
            logging.error(f"An error occurred in blur_thread: {str(e)}")
            response = {'blur_thread': str(e)}
        
        return response


    @staticmethod
    def main_thread(input):
        que1 = Queue()
        que2 = Queue()
        que3 = Queue()
        que4 = Queue()
        que5 = Queue()
        que6 = Queue()

        input = {
            'docid': str(input['docid']),
            'product_id': str(input['product_id']),
            'local_path': str(input['local_path']),
            'product_url': str(input['product_url']),
            'product_url_ori': str(input['product_url_ori']),
            'business_tag': str(input['business_tag'])
        }

        threads = [
            threading.Thread(target=lambda q, arg1: q.put(MultiThreading.brightness_thread(arg1)), args=(que1, input['local_path'])),
            threading.Thread(target=lambda q, arg1: q.put(MultiThreading.hash_thread(arg1)), args=(que2, input['local_path'])),
            threading.Thread(target=lambda q, arg1: q.put(MultiThreading.meta_thread(arg1)), args=(que5, input))
        ]

        # Start and join all threads except blur_thread
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Start the blur_thread after others have finished
        blur_thread = threading.Thread(target=lambda q, arg1, arg2: q.put(MultiThreading.blur_thread(arg1, arg2)), args=(que6, input['local_path'], input['product_url']))
        blur_thread.start()
        blur_thread.join()

        new_thread = [
            threading.Thread(target=lambda q, arg1, arg2: q.put(MultiThreading.exif_thread(arg1, arg2)), args=(que3, input['local_path'], input['product_id'])),
            threading.Thread(target=lambda q, arg1: q.put(MultiThreading.color_thread(arg1)), args=(que4, input['local_path']))]
        
        for thread in new_thread:
            thread.start()
        for thread in new_thread:
            thread.join()

        result1 = que1.get()
        result2 = que2.get()
        result3 = que3.get()
        result4 = que4.get()
        result5 = que5.get()
        result6 = que6.get()

        result = {**result1, **result2, **result3, **result4, **result5, **result6}

        data = json.dumps(result)

        return data

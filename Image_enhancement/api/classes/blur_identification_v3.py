from flask import jsonify, make_response
import numpy as np
import cv2
import json
import pandas as pd
import pickle
import requests
from classes.imagemeta_tag import ImageMeta
from classes.manage_logs import Manage_Logs
from helper import getConfigData
import logging
# import requests
# from io import BytesIO

# Load DT trained model
NFS_models_path = getConfigData('NFS_path.models_path')
with open(NFS_models_path + '/dt4_manually_3.pkl', 'rb') as f:
    model_dt_3 = pickle.load(f)
class BlurIdentify_V3():

    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    # Function to check if an image is blurry
    def is_blurry(image_path, img, threshold=100):
        # with open(image_path, 'rb') as f:
        #     img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) #
        # img = cv2.imread(image_path)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            # Check if image has 3 channels and is it not None before conversion
            if img is not None and img.size!=0 and len(img.shape)==3 and (img.shape[2] == 3 or img.shape[2] == 4):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img is not None:
                # Handle unexpected shape
                if(len(img.shape)==2):
                    gray = img
                    logging.info(f"Blur_V3: Image already grayscale, img.shape = {img.shape}")
                elif(len(img.shape)==3 and img.shape[2]==1):
                    # Remove the last dimension if it has only 1 element
                    gray = np.squeeze(img)
                else:
                    logging.info(f"Blur_V3: Image has unexpected number of channels, img.shape={img.shape}")
                    raise AttributeError(f"Blur_V3: Image has unexpected number of channels, img.shape={img.shape}")
            else:
                # logging.info(f"Blur_V3: img={img}")
                raise AttributeError(f"Blur_V3: Image is of NoneType, img={img}")
        except Exception as e:
            # Handle cases where img might not have a shape attribute
            logging.info(f"Blur_V3: Image loading failed or has unexpected format, type(img) = {type(img)}")
            raise AttributeError(e)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # std_lap = np.std(laplacian)
        variance = laplacian.var()
        ground_truth = variance < threshold
        # print(ground_truth)
        return ground_truth, variance


    def all_blur_std(image_path, img):
        # with open(image_path, 'rb') as f:
        #     img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.imread(image_path)

        np.random.seed(0)
        lap = []
        fourier = []
        gradient = []

        grid_size = 4
        # image_height, image_width, _ = img.shape
        try:
            if img is not None and img.size!=0 and len(img.shape)==3 and (img.shape[2] == 3 or img.shape[2] == 4):
                image_height, image_width, _ = img.shape
            elif img is not None:
                # Handle unexpected shape
                if(len(img.shape)==2):
                    image_height, image_width = img.shape
                    logging.info(f"Blur_V3: Image already grayscale, img.shape = {img.shape}")
                elif(len(img.shape)==3 and img.shape[2]==1):
                    image_height, image_width, _ = img.shape
                    logging.info(f"Blur_V3: Image already grayscale with img.shape[2]==1")
                else:
                    logging.info(f"Blur_V3: Image has unexpected number of channels, img.shape={img.shape}")
                    raise AttributeError(f"Blur_V3: Image has unexpected number of channels, img.shape={img.shape}")
            else:
                # logging.info(f"Blur_V3: img={img}")
                raise AttributeError(f"Blur_V3: Image is of NoneType, img={img}")
        except Exception as e:
            # Handle cases where img might not have a shape attribute
            logging.info(f"Blur_V3: Image loading failed or has unexpected format, type(img) = {type(img)}")
            raise AttributeError(e)
        grid_height = int(image_height / grid_size)
        grid_width = int(image_width / grid_size)

        for i in range(0, image_height, grid_height):
            cv2.line(img, (0, i), (image_width, i), (0, 255, 0), 1)
        for j in range(0, image_width, grid_width):
            cv2.line(img, (j, 0), (j, image_height), (0, 255, 0), 1)

        #selected_boxes = np.random.choice(grid_size * grid_size, size=10, replace=False)
        selected_indexes = [1, 2, 4, 5, 6, 9, 10, 11, 13, 14]

        for box_index in selected_indexes:
            i, j = divmod(box_index, grid_size)
            x1, y1 = j * grid_width, i * grid_height
            x2, y2 = x1 + grid_width, y1 + grid_height
            roi = img[y1:y2, x1:x2]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            lap.append(laplacian_var)

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dft_roi = cv2.dft(np.float32(gray_roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            magnitude_values = cv2.magnitude(dft_roi[:, :, 0], dft_roi[:, :, 1]) + 1e-6
            mag_dft_roi = 20 * np.log(magnitude_values)
            blur = np.mean(mag_dft_roi)
            fourier.append(blur)

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            blur_metric = np.mean(mag)
            gradient.append(blur_metric)

            avg_blur1 = np.std(lap)
            Laplacian_variance_blur = avg_blur1

            avg_blur2 = np.std(fourier)
            Fourier_Transform_blur = avg_blur2

            avg_blur3 = np.std(gradient)
            Gradient_magnitude_blur = avg_blur3

        return Laplacian_variance_blur,Fourier_Transform_blur,Gradient_magnitude_blur


    def blurValues(image_path, img, overall_variance):
        # with open(image_path, 'rb') as f:
        #     img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.imread(image_path)
        np.random.seed(0)
        
        # print(img)
        # is_blurry_result, overall_variance = BlurIdentify_V3.is_blurry(image_path, img) 
        # print('overall_variance:', overall_variance)

        lvb,ftb,gmb = BlurIdentify_V3.all_blur_std(image_path, img)
        Laplacian_variance_blur = lvb

        # print('overall_variance:',overall_variance)
        # print('Laplacian_variance_blur:',Laplacian_variance_blur)

        if (overall_variance > Laplacian_variance_blur).all():
            diff = (overall_variance) - (Laplacian_variance_blur)
            # print('diff:', diff)
        else:
            diff = -((overall_variance) - (Laplacian_variance_blur))
            # print('diff:', diff)

        lap = []
        fourier = []
        gradient = []
        grid_values = []  # Store blur values for each grid

        laplacian_thresholds = [(0, 150), (150, 700), (700, float('inf'))]
        fourier_thresholds = [(0, 139), (139, 147), (147, float('inf'))]
        gradient_thresholds = [(0, 25), (25, 35), (35, float('inf'))]

        blur_grid_values = []
        partial_blur_grid_values = []
        clear_grid_values = []

        
        grid_size = 4
        image_height, image_width, _ = img.shape
        grid_height = int(image_height / grid_size)
        grid_width = int(image_width / grid_size)

        for i in range(0, image_height, grid_height):
            cv2.line(img, (0, i), (image_width, i), (0, 255, 0), 1)
        for j in range(0, image_width, grid_width):
            cv2.line(img, (j, 0), (j, image_height), (0, 255, 0), 1)

        #selected_boxes = np.random.choice(grid_size * grid_size, size=16, replace=False)
        selected_indexes = [1, 2, 4, 5, 6, 9, 10, 11, 13, 14]

        for box_index in selected_indexes:
            i, j = divmod(box_index, grid_size)
            x1, y1 = j * grid_width, i * grid_height
            x2, y2 = x1 + grid_width, y1 + grid_height
            roi = img[y1:y2, x1:x2]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            lap.append(laplacian_var)

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dft_roi = cv2.dft(np.float32(gray_roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            magnitude_values = cv2.magnitude(dft_roi[:, :, 0], dft_roi[:, :, 1]) + 1e-6
            mag_dft_roi = 20 * np.log(magnitude_values)
            blur = np.mean(mag_dft_roi)
            fourier.append(blur)

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            blur_metric = np.mean(mag)
            gradient.append(blur_metric)

            # Categorize grid values based on the threshold ranges
            laplacian_category = None
            fourier_category = None
            # gradient_category = None
            
            
            

            for start, end in laplacian_thresholds:
                if start <= laplacian_var <= end:
                    laplacian_category = f"{start}-{end}"

            for start, end in fourier_thresholds:
                if start <= blur <= end:
                    fourier_category = f"{start}-{end}"

            for start, end in gradient_thresholds:
                if start <= blur_metric <= end:
                    gradient_category = f"{start}-{end}"

            lb = "0-150"
            lpb = "150-700"
            fb = "0-139"
            fpb = "139-147"
            gb = "0-25"
            gpb = "25-35"
            a = (0 <= diff <= 13)
            b = (14 <= diff <= 23)
            c = (23 <= diff <= 32)
            d = (32 <= diff <= 78)
            e = (78 <= diff <= 107)
            f = (107 <= diff <= 113)
            g = (113 <= diff <= 131)
            h = (131 <= diff <= 172)
            i = (141 <= diff <= 172)
            
            if a.all():
                
                if laplacian_category == lb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif laplacian_category == lpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Laplacian category")

            elif b.all(): 
                if fourier_category == fb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif fourier_category == fpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Fourier category")

            elif c.all():
                if gradient_category == gb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif gradient_category == gpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Gradient category")

            elif d.all():
                if fourier_category == fb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif fourier_category == fpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Fourier category")

            elif e.all(): 
                if fourier_category == fb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif fourier_category == fpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Fourier category")

            elif f.all():
                if gradient_category == gb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif gradient_category == gpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Gradient category")

            elif g.all(): 
                if laplacian_category == lb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif laplacian_category == lpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Laplacian category")
            
            elif h.all():
                if fourier_category == fb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif fourier_category == fpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Fourier category")
            
            elif i.all():
                if gradient_category == gb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif gradient_category == gpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Gradient category")

            else:
                if fourier_category == fb:
                    blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                elif fourier_category == fpb:
                    partial_blur_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                else:
                    clear_grid_values.append({
                        "lvb": laplacian_var,
                        "ftb": blur,
                        "gmb": blur_metric
                    })
                # print("Use Fourier category")

        avg_blur2 = np.std(fourier)
        Fourier_Transform_blur = avg_blur2

        avg_blur3 = np.std(gradient)
        Gradient_magnitude_blur = avg_blur3


        #Count the number of grids in each category
        num_blur_grids = len(blur_grid_values)
        num_partial_blur_grids = len(partial_blur_grid_values)
        num_clear_grids = len(clear_grid_values)


        earlier_blur_values = {
            "lvb_grid": Laplacian_variance_blur, 
            "ftb_grid" : Fourier_Transform_blur , 
            "gmb_grid": Gradient_magnitude_blur
        }

        grid_count_result = {
            "blur": num_blur_grids,
            "partial_blur": num_partial_blur_grids,
            "clear": num_clear_grids,
        }

        grid_values = {
                "blur": blur_grid_values,
                "partial_blur": partial_blur_grid_values,
                "clear": clear_grid_values
        }

        return earlier_blur_values, grid_count_result, grid_values, diff

    #### Artifact #####
    def analyze_artifacts(image_path, image, visualize=False, threshold=0.1, max_width=250):
        """
        Analyzes an image for artifacts based on multiple defined color ranges
        and performs pixel-wise analysis.

        Args:
            image_path (str): Path to the image file.
            color_ranges (list of tuples, optional): A list of tuples defining Hsl color ranges
                for artifacts. Defaults to None (black pixels).
            visualize (bool, optional): Flag to enable visualization of artifact regions with red contours. Defaults to False.
            threshold (float, optional): Threshold for considering a pixel an artifact (0.0 to 1.0). Defaults to 0.1.
            max_width (int, optional): Maximum width for resizing the image. Defaults to 250.

        Returns:
            tuple: A tuple containing:
            tuple: A tuple containing:
    tuple containing:
                - artifact_percentage (float): Percentage of pixels classified as artifacts.
                - artifact_mask (np.ndarray, optional): Binary mask image (white: artifact, black: non-artifact) if visualization is enabled, None otherwise.
                - artifact_pixels (list, optional): List of artifact pixel coordinates if thresholding is applied, None otherwise.
        """

        color_ranges = [
            ((0, 0, 0), (5, 255, 255)),  # Black pixels
            ((160, 50, 50), (180, 255, 255)),  # Red pixels
            ((140, 50, 10), (165, 255, 255)),  # Brownish red range
            ((10, 100, 100), (30, 255, 255)), # Adjust Hue value if necessary (bright yellow/orange)
            ((107, 224, 237),(145, 255, 255)), # light blueish grey
            ((9, 71, 23),(28, 255, 255)), #greenish black
            ((99, 99, 92),(140, 255, 255)), #grey yellow
            ((99, 189, 107),(140, 255, 255)), #light greenish grey
            ((12, 18, 13),(31,255,255)), #blackish grey
            ((184, 177, 147),(200,255,255)) #beige grey
            # ((135, 115, 201),(28, 255, 255)),
            # Add more color ranges for other artifact colors here
            ]

        try:
            # Load image
            # image = cv2.imread(image_path)
            if image is None:
                return -1.0  # Error loading image

            # Resize the image if its width exceeds the maximum width
            width = image.shape[1]
            if width > 600:
                new_width = 250
                new_height = int((image.shape[0] / width) * new_width)
                image = cv2.resize(image, (new_width, new_height))

            # Convert image to Hsl color space
            hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)

            # Initialize variables
            total_pixels = image.shape[0] * image.shape[1]
            artifact_pixels = []  # List for artifact pixel coordinates (optional)

            # Create a copy for visualization (optional)
            if visualize:
                image_copy = image.copy()

            # Analyze artifacts for each color range
            artifact_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)  # Initialize mask (black)
            for color_range in color_ranges:
                hsl_min = np.array(color_range[0], np.uint8)
                hsl_max = np.array(color_range[1], np.uint8)

                # Detect artifact pixels within the defined range
                mask = cv2.inRange(hsl, hsl_min, hsl_max)
                artifact_mask = cv2.bitwise_or(artifact_mask, mask)  # Accumulate mask across ranges

                # Pixel-wise analysis with thresholding (optional)
                if threshold > 0:
                    # Find artifact pixel coordinates exceeding the threshold
                    artifact_pixels.extend(np.argwhere(mask > 0).tolist())

                # Draw contours around artifact regions (optional)
                if visualize:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image_copy, contours, -1, (0, 0, 255), 2)  # Draw red contours

            # Calculate overall artifact percentage
            artifact_count = len(artifact_pixels) if threshold > 0 else cv2.countNonZero(artifact_mask)
            artifact_percentage = artifact_count / total_pixels
            if(isinstance(artifact_percentage, tuple)):
                logging.INFO(f"type(artifact_pixels) = {type(artifact_pixels)}\ntype(len(artifact_pixels)) = {type(len(artifact_pixels))}\nlen(artifact_pixels) = {len(artifact_pixels)}")

            # Return results
            return artifact_percentage
            # , artifact_mask, mask if visualize else None, artifact_pixels if threshold > 0 else None

        except Exception as e:
            print(f"An error occurred: {e}")
            return -1.0


    def is_blurr_v3(image_path, size):    
        # cv2 method used to extract the color vectors, image matrix, and blur values
        # with open(image_path, 'rb') as f:
            # img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.imread(image_path) 
        ground_truth, variance = BlurIdentify_V3.is_blurry(image_path, img)
        
        if ground_truth==True:
            flag = 1
        else:
            flag = 0
        
        earlier_blur_result,grid_count,grid_values, diff = BlurIdentify_V3.blurValues(image_path, img, variance)
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

        new_sharpness_score = None
        gmb = earlier_blur_result['gmb_grid']
        lvb = earlier_blur_result['lvb_grid']
        ftb = earlier_blur_result['ftb_grid']
        lvm = earlier_blur_result['lvb_main']
        blur_score = None

        if img is None:
            Manage_Logs.input_request(local_path=image_path, product_url=None, api_name = 'blur_v3', e="image corrupted or absent")
            raise ValueError(f"Error: Failed to read image: {image_path}, image is Nonetype, type(img)={type(img)}") 

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate gradients using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate Tenengrad (gradient magnitude)
        tenengrad = np.sqrt(grad_x**2 + grad_y**2)

        # Calculate the sharpness score as the average of Tenengrad values
        new_sharpness_score = float(np.mean(tenengrad))
        # print('Sharpness_score:', sharpness_score)

        # Find blur_score
        img.resize(img.shape[0], img.shape[1])
        blur_score = ImageMeta.is_image_blurry(img)

        # DT model prediction
        val = [new_sharpness_score, blur_score, size, gmb, lvm]
        val_df = pd.DataFrame([val], columns=['new_sharpness_score', 'blur_score', 'size', 'gmb', 'lvm'])
        out = model_dt_3.predict(val_df)[0]

        artifact_percentage = BlurIdentify_V3.analyze_artifacts(image_path, img)
        x = 0.40
        # print("artifact_percentage:", "{:.2f}".format(artifact_percentage*100), "%")
        # print("size:", size)
        # Check if at least 2 out of 3 conditions are true
        if (size < 150000 and artifact_percentage > float(x) and out == "Bad") or \
        (size < 150000 and artifact_percentage > float(x) and out == "Good") or \
        (size > 150000 and artifact_percentage > float(x) and out == "Bad") or \
        (size < 150000 and artifact_percentage < float(x) and out == "Bad"):
            artifact_verdict = "Artifact"
        else:
            artifact_verdict = "Not Artifact"

        response = {
            'local_path' : image_path,
            'blur_type' : "clear" if out=="Good" else "blur",
            # 'new_sharpness_score': new_sharpness_score,
            # 'blur_score': blur_score,
            # 'size': size,
            'gradient_magnitude_blur' : str(gmb),
            'laplacian_variance_main' : str(lvm),
            'laplacian_variance_blur' : str(lvb),
            'fourier_transform_blur' : str(ftb),
            'artifact_verdict'     : str(artifact_verdict),
            # 'artifact_percentage'   : float(artifact_percentage)
        }
        return response

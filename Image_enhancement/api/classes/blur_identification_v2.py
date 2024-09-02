from flask import jsonify, make_response
import numpy as np
import cv2
import json
# import requests
# from io import BytesIO

class BlurIdentify():

    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    # Function to check if an image is blurry
    def is_blurry(image_path, threshold=100):
        with open(image_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # std_lap = np.std(laplacian)
        variance = laplacian.var()
        ground_truth = variance < threshold
        # print(ground_truth)
        return ground_truth, variance


    def all_blur_std(image_path):
        with open(image_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

        np.random.seed(0)
        lap = []
        fourier = []
        gradient = []

        grid_size = 4
        image_height, image_width, _ = img.shape
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


    def blurValues(image_path):
        with open(image_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

        np.random.seed(0)
        
        # print(img)
        is_blurry_result, overall_variance = BlurIdentify.is_blurry(image_path) 
        # print('overall_variance:', overall_variance)

        lvb,ftb,gmb = BlurIdentify.all_blur_std(image_path)
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


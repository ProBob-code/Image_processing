from flask import jsonify, make_response
import numpy as np
import cv2
import json

class BlurIdentify():

    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    def blurValues(image_path):
        with open(image_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

        np.random.seed(0)

        lap = []
        fourier = []
        gradient = []
        grid_values = []  # Store blur values for each grid

        laplacian_thresholds = [(0, 140), (140, 610), (610, float('inf'))]

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

        selected_boxes = np.random.choice(grid_size * grid_size, size=10, replace=False)

        for box_index in selected_boxes:
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

            for start, end in laplacian_thresholds:
                if start <= laplacian_var <= end:
                    laplacian_category = f"{start}-{end}"

            # Append grid value to the appropriate category
            if laplacian_category == "0-140":
                blur_grid_values.append({
                    "lvb": laplacian_var,
                    "ftb": blur,
                    "gmb": blur_metric
                })
            elif laplacian_category == "140-610":
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

            # Count the number of grids in each category
            num_blur_grids = len(blur_grid_values)
            num_partial_blur_grids = len(partial_blur_grid_values)
            num_clear_grids = len(clear_grid_values)

        avg_blur1 = np.std(lap)
        Laplacian_variance_blur = avg_blur1

        avg_blur2 = np.std(fourier)
        Fourier_Transform_blur = avg_blur2

        avg_blur3 = np.std(gradient)
        Gradient_magnitude_blur = avg_blur3


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


        return earlier_blur_values,grid_count_result,grid_values

    @staticmethod
    # Function to check if an image is blurry
    def is_blurry(image_path, threshold=100):
        with open(image_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # std_lap = np.std(laplacian)
        variance = laplacian.var()
        ground_truth = variance < threshold
        print(ground_truth)
        return ground_truth, variance

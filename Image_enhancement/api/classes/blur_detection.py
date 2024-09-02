from flask import jsonify, make_response
import numpy as np
import cv2 as cv2
import json


class BlurDetect():

    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    def blurValues(image_path):
        with open(image_path, 'rb') as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

        # img = image_path     
        np.random.seed(0)

        lap = []
        fourier = []
        gradient = []
        # Define the grid parameters
        grid_size = 4 # number of rows and columns in the grid
        image_height, image_width, _ = img.shape # get image height and width
        grid_height = int(image_height / grid_size) # height of each grid box
        grid_width = int(image_width / grid_size) # width of each grid box

        # Draw the grid on the image
        for i in range(0, image_height, grid_height):
            cv2.line(img, (0, i), (image_width, i), (0, 255, 0), 1)
        for j in range(0, image_width, grid_width):
            cv2.line(img, (j, 0), (j, image_height), (0, 255, 0), 1)


        # Calculate the blur level for randomly selected grid boxes
        selected_boxes = np.random.choice(grid_size * grid_size, size=10, replace=False)

        for box_index in selected_boxes:
            i, j = divmod(box_index, grid_size)
            # get the ROI for the current grid box
            x1, y1 = j * grid_width, i * grid_height
            x2, y2 = x1 + grid_width, y1 + grid_height
            roi = img[y1:y2, x1:x2]
            # calculate the Laplacian variance
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            lap.append(laplacian_var)

            # calculate the Fourier transform
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dft_roi = cv2.dft(np.float32(gray_roi), flags=cv2.DFT_COMPLEX_OUTPUT)
            mag_dft_roi = 20 * np.log(cv2.magnitude(dft_roi[:, :, 0], dft_roi[:, :, 1]))
            blur = np.mean(mag_dft_roi)
            fourier.append(blur)

            # calculate the gradient magnitude
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)

            # calculate a blur metric from the gradient magnitude values
            blur_metric = np.mean(mag)
            gradient.append(blur_metric)

        avg_blur1 = np.std(lap)
        Laplacian_variance_blur = avg_blur1

        avg_blur2 = np.std(fourier)
        Fourier_Transform_blur = avg_blur2

        avg_blur3 = np.std(gradient)
        Gradient_magnitude_blur = avg_blur3

        # print(Laplacian_variance_blur)

        # # return jsonify(
        #     Laplacian_variance_blur = Laplacian_variance_blur,
        #     Fourier_Transform_blur  = Fourier_Transform_blur,
        #     Gradient_magnitude_blur = Gradient_magnitude_blur
        # # )
        return Laplacian_variance_blur, Fourier_Transform_blur, Gradient_magnitude_blur
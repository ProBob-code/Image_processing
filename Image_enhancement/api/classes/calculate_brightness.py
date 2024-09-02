import os
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify
from PIL import Image

class CalculateBrighntess():

    def __init__(self):
        pass

    def safeSerialize(obj):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(obj, default=default)

    def extractBrightnessMatrix(img):
        # Open the image
        image = Image.open(img)

        # Convert the image to grayscale
        grayscale_image = image.convert("L")

        # Extract the pixel values as a 2D matrix
        brightness_matrix = list(grayscale_image.getdata())
        width, height = grayscale_image.size
        brightness_matrix = [brightness_matrix[i:i+width] for i in range(0, len(brightness_matrix), width)]
        
        return brightness_matrix

    def calculateOverallBrightness(image_path):

        # with open(image_path, 'rb') as f:
        #     img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        
        brightness_matrix = CalculateBrighntess.extractBrightnessMatrix(image_path)
        total_pixels = 0
        brightness_sum = 0

        # Iterate over each pixel in the brightness matrix
        for row in brightness_matrix:
            for pixel in row:
                brightness_sum += pixel
                total_pixels += 1

        # Calculate the average brightness
        overall_brightness = brightness_sum / total_pixels

        return overall_brightness
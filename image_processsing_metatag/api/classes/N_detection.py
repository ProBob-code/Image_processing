import os
import time
from PIL import Image
import tensorflow as tf
from nudenet import NudeDetector
import cv2
import mediapipe as mp
import numpy as np
import requests
import io
from urllib.parse import urlparse

# Check for GPU availability
if tf.config.list_physical_devices('GPU'):
    print("GPU detected. TensorFlow will use it for inference.")
else:
    print("GPU not detected! Ensure TensorFlow-GPU is installed for faster processing.")


class NDetection:
    def __init__(self):
        # Initialize the NudeDetector model
        self.detector = NudeDetector()

    def resize_image(self, image_url, max_width=1000, image_path=""):
        """
        Resize an image based on its width. If the width is smaller than max_width, 
        resize it to max_width while maintaining aspect ratio, and overwrite the 
        image at the original image_path.

        Parameters:
            image_url (str): URL of the image.
            max_width (int): The maximum width for the resized image.
            image_path (str): The path to overwrite with the resized image.

        Returns:
            tuple: (bool, str) - Whether resizing was done and the path of the saved image.
        """
        try:
            # Fetch the image from the URL
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                # Open the image using PIL (Pillow)
                with Image.open(io.BytesIO(response.content)) as img:
                    width, height = img.size
                    print("width:", width)
                    print("height:", height)

                    # Check if resizing is needed (resize if width is smaller than max_width)
                    if width < max_width:
                        new_width = max_width
                        new_height = int((height / width) * new_width)  # Maintain aspect ratio
                        print('new_width:',new_width)
                        print('new_height:',new_height)

                        # Resize the image
                        img = img.resize((new_width, new_height), Image.ANTIALIAS)  # Resize to new dimensions

                        # Overwrite the image at the original image path
                        img.convert('RGB').save(image_path)  # Convert to RGB and save
                        print(f"Resized image saved at: {image_path}")

                        # Return the path of the resized image
                        return True, image_path
                    else:
                        # No resizing needed, overwrite the original image at the given path
                        img.save(image_path)  # Save the original image at the same path
                        print(f"Original image saved at: {image_path}")

                        # Return the path of the original image
                        return False, image_path
            else:
                print(f"Failed to download image from {image_url}, status code: {response.status_code}")
                return None, None

        except Exception as e:
            print(f"Error resizing image from {image_url}: {e}")
            return None, None

    def classify_image(self, predictions):
        """
        Classify the image based on predictions.
        Returns 'Nude', 'Semi-Nude', or 'Safe'.
        """
        nude_threshold = 0.5  # Adjust threshold for explicit content
        semi_nude_threshold = 0.3
        explicit_classes = [
            "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED",
            "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"
        ]
        secondary_classes = ["BELLY_EXPOSED", "FEET_EXPOSED", "ARMPITS_EXPOSED"]
        covered_classes = ["FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED"]

        # Scores for different categories
        explicit_scores = [p['score'] for p in predictions if p['class'] in explicit_classes]
        secondary_scores = [p['score'] for p in predictions if p['class'] in secondary_classes]
        covered_scores = [p['score'] for p in predictions if p['class'] in covered_classes]

        # Find maximum scores
        max_explicit = max(explicit_scores, default=0)
        max_secondary = max(secondary_scores, default=0)
        max_covered = max(covered_scores, default=0)

        # Classification logic
        if max_explicit >= nude_threshold:
            return "Nude"
        if max_explicit >= semi_nude_threshold or max_secondary >= semi_nude_threshold:
            return "Semi-Nude"
        return "Safe"

    def handle_nude_classification(self, predictions, image):
        """
        Handle additional logic for images classified as 'Nude'.
        Uses detection thresholds for explicit and secondary features.
        """
        try:
            # Thresholds
            explicit_threshold = 0.30

            # Explicit classes
            explicit_classes = [
                "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
                "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"
            ]

            # Secondary classes
            secondary_classes = ["BELLY_EXPOSED", "FEET_EXPOSED", "ARMPITS_EXPOSED"]

            # Check explicit features
            explicit_detected = [
                p for p in predictions 
                if p.get("class") in explicit_classes and p.get("score", 0) >= explicit_threshold
            ]

            # Check secondary features
            secondary_detected = [
                p for p in predictions 
                if p.get("class") in secondary_classes and p.get("score", 0) >= explicit_threshold
            ]

            # Check for male genitalia specifically
            male_genitalia_detected = any(
                p.get("class") == "MALE_GENITALIA_EXPOSED" and p.get("score", 0) >= explicit_threshold
                for p in predictions
            )

            # Check for hands/fingers using skin detection and Mediapipe
            hand_detected = any(p.get("hand_finger_identified") == 1 for p in predictions)
            if not hand_detected:
                skin_mask = self.detect_skin(image)
                contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hands_contour, _ = self.process_contours(contours)
                mediapipe_hands = self.detect_hands_mediapipe(image)
                hand_detected = hands_contour > 0 or mediapipe_hands > 0

            # Classification Logic
            if explicit_detected:
                print("Explicit features detected with score >= 0.30. Classified as Nude.")
                return "Nude", 0

            if secondary_detected:
                print("Secondary features detected with score >= 0.30. Classified as Semi-Nude.")
                return "Semi-Nude", 0

            if male_genitalia_detected:
                print("Male genitalia detected with score >= 0.30.")
                if hand_detected:
                    print("Hands or fingers detected. Classified as Safe.")
                    return "Safe", 1
                else:
                    print("No hands or fingers detected. Classified as Nude.")
                    return "Nude", 0

            # Default Safe classification
            print("No explicit or secondary features detected. Classified as Safe.")
            return "Safe", 0

        except Exception as e:
            print(f"Error in handle_nude_classification: {e}")
            return "Error", None



    def process_image(self, image_path, url):
        """Process the image and classify it."""
        start_time = time.time()

        # Check if the image exists on the local path
        if not os.path.exists(image_path):
            return {"error": f"Image file {image_path} not found."}

        # Resize the image (if necessary)
        resized, resized_image = self.resize_image(url, image_path=image_path)
        print('resized_image:',resized_image)
        print('resized:',resized)
        
        if resized_image is None:
            return {"error": f"Error resizing image {image_path}"}
        elif resized:
            print(f"Image resized to the maximum width of 500px.")
        else:
            print("Image does not need resizing.")

        try:
            # resized_image_np = np.array(resized_image)
            # Detect using the resized (or original) image object
            predictions = self.detector.detect(resized_image)
            print(f"Detector predictions: {predictions}")

            # Classify based on the predictions
            classification = self.classify_image(predictions)
            print(f"Classification: {classification}")

            # Initialize hand_finger_identified as 0
            hand_finger_identified = 0

            if classification == "Nude":
                # Invoke the new function for Nude classification
                classification, hand_finger_identified = self.handle_nude_classification(
                    predictions, cv2.imread(resized_image)  # Convert PIL image to OpenCV format if needed
                )
                # Add hand_finger_identified to predictions if classification is "Nude"
                predictions.append({"hand_finger_identified": hand_finger_identified})

            # Calculate processing time
            end_time = time.time()

            result = {
                "classification": classification,
                "processing_time": end_time - start_time,
                "predictions": predictions
            }

            return result
        except Exception as e:
            return {"error": f"Error processing resized image: {e}"}


    def detect_skin(self, image):
        """Detects skin regions in the image."""
        kernel = np.ones((5, 5), np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin_hsv = np.array([20, 255, 255], dtype=np.uint8)
        lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

        skin_mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
        skin_mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
        skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycrcb)

        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    def process_contours(self, contours):
        """Detect hands and fingers based on contours."""
        hands_contour = 0
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Threshold area to filter noise
                hands_contour += 1
        return hands_contour, contours

    def detect_hands_mediapipe(self, image):
        """Detect hands in an image using MediaPipe."""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        hands_count = 0
        if results.multi_hand_landmarks:
            hands_count = len(results.multi_hand_landmarks)
        return hands_count

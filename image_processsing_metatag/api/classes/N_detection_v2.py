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


class NDetection_V2:
    def __init__(self):
        # Initialize the NudeDetector model
        self.detector = NudeDetector()

        # Load the pre-trained MobileNet SSD model (Caffe version)
        self.net = cv2.dnn.readNetFromCaffe('./deploy.prototxt', './mobilenet_iter_73000.caffemodel')

        # Pascal VOC Labels
        self.PASCAL_VOC_LABELS = {
            0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
            7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
            15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
        }

        # COCO Labels
        self.COCO_LABELS = {
            0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
            8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter',
            14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant',
            22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie',
            29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
            36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle', 41: 'wine glass',
            42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich',
            50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake',
            57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'TV',
            64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven',
            71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors',
            78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'
        }

        # Combine labels with unique IDs
        self.COMBINED_LABELS = {**self.PASCAL_VOC_LABELS}
        offset = max(self.PASCAL_VOC_LABELS.keys()) + 1  # Avoid ID conflicts
        for k, v in self.COCO_LABELS.items():
            if k not in self.PASCAL_VOC_LABELS.values():
                self.COMBINED_LABELS[k + offset] = v

    def resize_image(self, image_url, max_width, image_path=""):
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

    # def classify_image(self, predictions):
    #     """
    #     Classify the image based on predictions.
    #     Returns 'Nude', 'Semi-Nude', or 'Safe'.
    #     """
    #     nude_threshold = 0.5  # Adjust threshold for explicit content
    #     semi_nude_threshold = 0.25
    #     male_breast_exposed_threshold = 0.40  # Threshold for male breast exposure
    #     female_breast_exposed_threshold = 0.5
    #     explicit_classes = [
    #         "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
    #         "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED",
    #         "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"
    #     ]
    #     secondary_classes = ["BELLY_EXPOSED", "FEET_EXPOSED", "ARMPITS_EXPOSED"]
    #     covered_classes = ["FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED"]

    #     # Scores for different categories
    #     explicit_scores = [p['score'] for p in predictions if p['class'] in explicit_classes]
    #     secondary_scores = [p['score'] for p in predictions if p['class'] in secondary_classes]
    #     covered_scores = [p['score'] for p in predictions if p['class'] in covered_classes]

    #     # Special check for Male Breast Exposed
    #     male_breast_exposed_scores = [p['score'] for p in predictions if p['class'] == "MALE_BREAST_EXPOSED"]
    #     if male_breast_exposed_scores and male_breast_exposed_scores[0] > male_breast_exposed_threshold:
    #         return "Semi-Nude"  # If male breast exposed score is above 40%, classify as Semi-Nude

    #     # Special check for "FEMALE_BREAST_EXPOSED"
    #     female_breast_score = next((p['score'] for p in predictions if p['class'] == "FEMALE_BREAST_EXPOSED"), 0)
    #     if female_breast_score >= female_breast_exposed_threshold:
    #         return "Semi-Nude"  # If female breast exposure score is above threshold, classify as Semi-Nude


    #     # Find maximum scores
    #     max_explicit = max(explicit_scores, default=0)
    #     max_secondary = max(secondary_scores, default=0)
    #     max_covered = max(covered_scores, default=0)

    #     # Classification logic
    #     if max_explicit >= nude_threshold:
    #         return "Nude"
    #     if max_explicit >= semi_nude_threshold or max_secondary >= semi_nude_threshold:
    #         return "Semi-Nude"
    #     return "Safe"

    def classify_image(self, predictions):
        """
        Classify the image based on predictions.
        Returns 'Nude', 'Semi-Nude', or 'Safe'.
        """
        # Thresholds
        nude_threshold = 0.5  # Threshold for explicit content
        semi_nude_threshold = 0.25
        female_breast_exposed_threshold = 0.5
        male_breast_exposed_threshold = 0.4

        # Class categories
        explicit_classes = [
            "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED",
            "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"
        ]
        secondary_classes = ["BELLY_EXPOSED", "FEET_EXPOSED", "ARMPITS_EXPOSED"]
        covered_classes = ["FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED"]

        # Extract scores for each category
        explicit_scores = [p['score'] for p in predictions if p['class'] in explicit_classes]
        secondary_scores = [p['score'] for p in predictions if p['class'] in secondary_classes]
        covered_scores = [p['score'] for p in predictions if p['class'] in covered_classes]

        # Individual scores for specific explicit classes
        female_breast_score = next((p['score'] for p in predictions if p['class'] == "FEMALE_BREAST_EXPOSED"), 0)
        male_breast_score = next((p['score'] for p in predictions if p['class'] == "MALE_BREAST_EXPOSED"), 0)
        other_explicit_scores = [
            p['score'] for p in predictions
            if p['class'] in explicit_classes and p['class'] not in ["FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED"]
        ]

        # 1. Detect "Nude"
        if (
            (female_breast_score >= female_breast_exposed_threshold or male_breast_score >= male_breast_exposed_threshold)
            and any(p['class'] in explicit_classes for p in predictions if p['class'] not in ["FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED"])
        ):
            return "Nude"

        # 2. Detect "Semi-Nude"
        if (
            (female_breast_score >= female_breast_exposed_threshold or male_breast_score >= male_breast_exposed_threshold)
            or max(explicit_scores, default=0) >= semi_nude_threshold
        ):
            return "Semi-Nude"
        

        # 3. Default to "Safe"
        return "Safe"

    def handle_nude_classification(self, predictions, image):
        try:
            explicit_threshold = 0.20
            explicit_classes = [
                "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
                "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"
            ]
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

            # Detect hands
            hand_detected = any(p.get("hand_finger_identified") == 1 for p in predictions)
            if not hand_detected:
                skin_mask = self.detect_skin(image)
                contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hands_contour, _ = self.process_contours(contours)
                mediapipe_hands = self.detect_hands_mediapipe(image)
                hand_detected = hands_contour > 0 or mediapipe_hands > 0

            # Debugging logs
            print(f"Explicit detected: {explicit_detected}")
            print(f"Secondary detected: {secondary_detected}")
            print(f"Hand detected: {hand_detected}")

            if explicit_detected:
                print("Explicit Detected:", explicit_detected)
                if all(p.get("class") in ["MALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_EXPOSED"] for p in explicit_detected):
                    if hand_detected:
                        print("Hand detected with genitalia exposed. Classified as Safe.")
                        return "Safe", 1
                    else:
                        print("No hand detected with genitalia exposed. Classified as Nude.")
                        return "Nude", 0
                # Other explicit features detected
                print("Explicit features detected. Classified as Nude.")
                return "Nude", 0

            # Secondary classification logic
            if secondary_detected:
                print("Secondary features detected. Classified as Semi-Nude.")
                return "Semi-Nude", 0

            # Default classification
            print("No explicit or secondary features detected. Classified as Safe.")
            return "Safe", 0

        except Exception as e:
            print(f"Error in handle_nude_classification: {e}")
            return "Error", None

    

    def detect_objects(self, image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)

        self.net.setInput(blob)
        detections = self.net.forward()

        detected_objects = []
        highest_confidence = 0
        highest_label = "unknown"
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence score for the detection
            if confidence > 0.5:  # Confidence threshold to filter weak detections
                class_id = int(detections[0, 0, i, 1])  # Class ID
                label = self.COMBINED_LABELS.get(class_id, 'unknown')
                detected_objects.append({'label': label, 'confidence': confidence})

                # Track the highest confidence and corresponding label
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    highest_label = label

                # Draw a bounding box around the detected object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(image, f"{label} ({confidence:.2f})", (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Determine the verdict based on confidence thresholds

        print("highest_confidence",highest_confidence)
        if highest_label == "person":
            if highest_confidence >= 0.7:
                image_tag = "Nude"
            else:
                image_tag = "Semi-Nude"
        elif highest_confidence >= 0.86:
            image_tag = "object"
        else:
            image_tag = "Semi-Nude"

        # Return the image with bounding boxes and the verdict
        return detected_objects, image_tag

    def process_image(self, image_path, url):
        """Process the image and classify it."""
        start_time = time.time()

        # Initialize classifications array
        classifications = []

        try:
            # Step 1: Check if the image exists
            if not os.path.exists(image_path):
                return {"error": f"Image file {image_path} not found."}

            # Step 2: Resize the image
            resized, resized_image = self.resize_image(url, max_width=1000, image_path=image_path)
            if resized_image is None:
                return {"error": f"Error resizing image {image_path}"}
            print(f"Image resized: {resized}. Resized image path: {resized_image}")

            # Step 3: Detect objects
            predictions = self.detector.detect(resized_image)
            print(f"Detector Predictions: {predictions}")

            # Step 4: Classify based on predictions
            if not predictions:
                print("No predictions were made by the detector. Marking the image as 'Safe'.")
                initial_classification = "Safe"
            else:
                initial_classification = self.classify_image(predictions)
            classifications.append(initial_classification)  # Record initial classification
            print(f"Initial Classification: {initial_classification}")

            hand_finger_identified = 0

            # Step 5: Handle "Nude" or "Semi-Nude" classifications
            if initial_classification in ["Nude", "Semi-Nude"]:
                updated_classification, hand_finger_identified = self.handle_nude_classification(
                    predictions, cv2.imread(resized_image)
                )
                classifications.append(updated_classification)  # Record classification from handle_nude_classification
                predictions.append({"hand_finger_identified": hand_finger_identified})
                print(f"Updated Classification (handle_nude_classification): {updated_classification}")

            # Step 6: Detect objects for additional verification
            detected_objects, image_tag = [], None
            detect_objects_classification = initial_classification  # Default fallback
            resized, resized_image_small = self.resize_image(url, max_width=500, image_path=image_path)
            if resized_image_small:
                detected_objects, image_tag = self.detect_objects(cv2.imread(resized_image_small))
                print(f"Detected Objects: {detected_objects}, Image Tag: {image_tag}")

                # Update classification based on object detection
                if not detected_objects:
                    detect_objects_classification = "Send to moderation"
                elif image_tag == "person":
                    detect_objects_classification = initial_classification
                elif image_tag in ["Nude", "Semi-Nude"]:
                    detect_objects_classification = image_tag
                else:
                    detect_objects_classification = "Safe"
                classifications.append(detect_objects_classification)  # Record detect_objects_classification
                print(f"Detect Objects Classification: {detect_objects_classification}")

            # Step 7: Determine the final verdict
            classification_counts = {c: classifications.count(c) for c in classifications}
            print(f"Classification Votes: {classification_counts}")

            # If only "Safe" and "Nude" are present with equal votes, choose "Safe"
            if set(classification_counts.keys()) == {"Safe", "Nude"} and classification_counts["Safe"] == 1 and classification_counts["Nude"] == 1:
                final_classification = "Safe"
            else:
                # Determine majority verdict
                majority_classification = max(classification_counts, key=classification_counts.get)
                majority_count = classification_counts[majority_classification]

                if majority_count >= 2:
                    final_classification = majority_classification
                else:
                    # Default to detect_objects_classification when no majority
                    final_classification = "Safe"  # Default fallback

            print(f"Final Classification: {final_classification}")

            # Step 8: Additional checks for "Safe"
            if final_classification == "Safe":
                exposed_items = {
                    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", 
                    "BELLY_EXPOSED", "BUTTOCKS_EXPOSED", 
                    "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED", "ARMPITS_EXPOSED"
                }
                
                # Count unique exposed items
                unique_exposed_classes = {
                    pred["class"] for pred in predictions if "class" in pred and pred["class"] in exposed_items
                }
                exposed_count = len(unique_exposed_classes)

                # Reclassify based on exposed_count
                if exposed_count > 2:
                    final_classification = "Nude"
                elif exposed_count == 2:
                    final_classification = "Semi-Nude"
                elif exposed_count == 1:
                    final_classification = "Send to moderation"
                else:
                    final_classification == "Safe"

            # Step 9: Construct the result
            processing_time = time.time() - start_time
            result = {
                "classification": final_classification,
                "processing_time": processing_time,
                "predictions": predictions,
                "classification_steps": classifications,
                "detected_objects": detected_objects,
                "image_tag": image_tag,
            }
            return result


        except Exception as e:
            print(f"Error occurred while processing the image: {e}")
            return {"error": f"Error processing image: {str(e)}"}



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


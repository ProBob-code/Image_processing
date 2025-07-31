import os
import time
from PIL import Image
import tensorflow as tf
from nudenet import NudeDetector

# Check for GPU availability
if tf.config.list_physical_devices('GPU'):
    print("GPU detected. TensorFlow will use it for inference.")
else:
    print("GPU not detected! Ensure TensorFlow-GPU is installed for faster processing.")


def resize_image(path, max_width=250):
    """
    Resize image to the specified width while maintaining aspect ratio.
    Overwrite the original image.
    """
    try:
        img = Image.open(path)
        width, height = img.size

        if width > max_width:
            new_width = max_width
            new_height = int((height / width) * new_width)
            img = img.resize((new_width, new_height))
            img.convert('RGB').save(path)
            return True, path  # Image was resized
        return False, path  # No resizing needed
    except Exception as e:
        print(f"Error resizing image {path}: {e}")
        return None, None


def classify_image(predictions):
    """
    Classify an image based on predictions.
    """
    nude_threshold = 0.75
    semi_nude_threshold = 0.50
    explicit_classes = [
        "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
        "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"
    ]
    secondary_classes = ["BELLY_EXPOSED", "FEET_EXPOSED", "ARMPITS_EXPOSED"]
    covered_classes = ["FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED"]

    explicit_scores = [p['score'] for p in predictions if p['class'] in explicit_classes]
    secondary_scores = [p['score'] for p in predictions if p['class'] in secondary_classes]
    covered_scores = [p['score'] for p in predictions if p['class'] in covered_classes]

    max_explicit = max(explicit_scores, default=0)
    max_secondary = max(secondary_scores, default=0)
    max_covered = max(covered_scores, default=0)

    if max_explicit >= nude_threshold:
        return "Nude"
    if max_explicit >= 0.5:
        return "Nude"
    if max_covered >= semi_nude_threshold or max_secondary >= semi_nude_threshold:
        return "Semi-Nude"
    return "Safe"


# Initialize NudeDetector
detector = NudeDetector()

# Prompt for a local image path
image_path = input("Enter the full path to the image: ").strip()

# Validate and process the image
if not os.path.isfile(image_path):
    print(f"Error: File not found at {image_path}")
else:
    start_time = time.time()

    # Check if resizing is needed
    resized, resized_path = resize_image(image_path)
    if resized is None:
        print(f"Error resizing image {image_path}")
    elif resized:
        print(f"Image resized to the maximum width of 250px.")
    else:
        print("Image does not need resizing.")
    
    try:
        # Detect and classify image
        predictions = detector.detect(resized_path)
        classification = classify_image(predictions)
        print(f"Image Classification: {classification}")
        
        # You can uncomment this section to censor the image if needed
        # if classification != "Safe":
        #     censored_img_path = detector.censor(resized_path)
        #     print(f"Image censored and saved to: {censored_img_path}")

    except Exception as e:
        print(f"Error processing {resized_path}: {e}")

    # Calculate and print the time taken for the image processing
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Time taken to process the image: {processing_time:.2f} seconds.")

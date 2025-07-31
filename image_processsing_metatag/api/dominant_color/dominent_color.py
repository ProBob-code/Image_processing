import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import requests
from urllib.parse import urlparse

# --- START: Debugging File Paths ---
print("--- Checking file paths ---")
current_dir = os.getcwd()
model_dir_path = os.path.join(current_dir, 'model')
u2net_file_path = os.path.join(model_dir_path, 'u2net.py')

print(f"Current working directory: {current_dir}")

if os.path.exists(model_dir_path):
    print(f"✅ 'model/' directory found at: {model_dir_path}")
    if os.path.isdir(model_dir_path):
        print(f"Contents of 'model/' directory: {os.listdir(model_dir_path)}")
        if os.path.exists(u2net_file_path):
            print(f"✅ 'u2net.py' file found at: {u2net_file_path}")
        else:
            print(f"❌ 'u2net.py' NOT found inside 'model/' directory. Please ensure it's named 'u2net.py' and placed correctly.")
            print("Expected path: " + u2net_file_path)
    else:
        print(f"❌ 'model' exists but is not a directory. Please check your file system.")
else:
    print(f"❌ 'model/' directory NOT found at: {model_dir_path}. Please create this directory.")
print("--- Path check complete ---")
# --- END: Debugging File Paths ---

# Import your U2NETP model definition from the correct path: model/u2net.py
from model.u2net import U2NETP

# Configuration
MODEL_PATH = "/home/justdial/content_processsing/api/dominent color/saved_models/u2net/u2netp.pth"
NEW_RESULTS_DIR = "new_results_latest4" # Directory for results
MASK_SUFFIX = "_mask.png"
TEMP_DOWNLOAD_DIR = "temp_downloads_latest4" # Directory to store downloaded images

# Create results and temporary download directories if they don't exist
os.makedirs(NEW_RESULTS_DIR, exist_ok=True)
os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)

# Load U2NETP model
print("⏳ Loading U2NETP model...")
net = U2NETP(3, 1)

try:
    net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    print("✅ Model weights loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model weights file not found at '{MODEL_PATH}'.")
    print("Please ensure the 'saved_models/u2net/u2netp.pth' file exists in the correct path.")
    exit()
except Exception as e:
    print(f"❌ An error occurred while loading model weights: {e}")
    exit()

net.eval()
print("✅ Model is in evaluation mode.")

# Define the preprocessing pipeline for input images
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def download_image(image_url, save_path):
    """
    Downloads an image from a given URL and saves it to a specified path.
    Returns the path to the downloaded image if successful, None otherwise.
    """
    try:
        print(f"Attempting to download: {image_url}")
        response = requests.get(image_url, stream=True, timeout=10) # Added timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {image_url} to {save_path}")
        return save_path
    except requests.exceptions.Timeout:
        print(f"❌ Timeout occurred while downloading {image_url}. Skipping.")
        return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error while downloading {image_url}. Skipping.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {image_url}: {e}. Skipping.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred while downloading {image_url}: {e}. Skipping.")
        return None

def process_image_from_path(image_path, output_dir):
    """
    Processes a single image from a local path to generate a background-removed image.
    Saves the mask and the transparent subject image to the specified output directory.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}. Skipping.")
        return

    print(f"\n🔍 Processing image: {image_path}")
    basename = os.path.splitext(os.path.basename(image_path))[0] # Get filename without extension

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Could not open image {image_path}: {e}. Skipping.")
        return

    original_size = image.size

    input_tensor = transform(image).unsqueeze(0)
    input_tensor = Variable(input_tensor)

    with torch.no_grad():
        d1, *_ = net(input_tensor)
        pred = d1[:, 0, :, :].squeeze().cpu().numpy()

        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred = (pred * 255).astype(np.uint8)

    mask = cv2.resize(pred, original_size)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    mask_filename = os.path.join(output_dir, f"{basename}{MASK_SUFFIX}")
    cv2.imwrite(mask_filename, binary_mask)
    print(f"✅ Saved mask: {mask_filename}")

    image_cv = cv2.imread(image_path)
    mask_cv = cv2.imread(mask_filename, 0)

    if image_cv.shape[0] != mask_cv.shape[0] or image_cv.shape[1] != mask_cv.shape[1]:
        mask_cv = cv2.resize(mask_cv, (image_cv.shape[1], image_cv.shape[0]))

    b, g, r = cv2.split(image_cv)
    rgba = cv2.merge((b, g, r, mask_cv))

    output_path = os.path.join(output_dir, f"{basename}_subject.png")
    cv2.imwrite(output_path, rgba)
    print(f"✅ Transparent subject image saved: {output_path}")

# --- Main execution for URL processing ---
print("\nStarting image processing from provided URLs...")

image_urls = [
    'https://testingimgsmum77710.s3.ap-south-1.amazonaws.com/testImages/image+680.png',
    'https://testingimgsmum77710.s3.ap-south-1.amazonaws.com/testImages/image.png',
    'https://testingimgsmum77710.s3.ap-south-1.amazonaws.com/testImages/image-2.png',
    'https://testingimgsmum77710.s3.ap-south-1.amazonaws.com/testImages/image-1.png',
    'https://testingimgsmum77710.s3.ap-south-1.amazonaws.com/testImages/image+681.png'
]

for url in image_urls:
    # Extract filename from URL, ensuring a valid default if parsing fails
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename or not any(filename.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
        # Attempt to derive a filename from the full path if the last segment is not a file
        # Or assign a generic name with a timestamp/UUID if no good filename is found
        filename = f"downloaded_image_{len(os.listdir(TEMP_DOWNLOAD_DIR)) + 1}.jpg" # Simple unique name

    local_image_path = os.path.join(TEMP_DOWNLOAD_DIR, filename)

    downloaded_path = download_image(url, local_image_path)
    if downloaded_path:
        process_image_from_path(downloaded_path, NEW_RESULTS_DIR)
        # Optional: Remove the downloaded image after processing to save space
        # os.remove(downloaded_path)
    else:
        print(f"Skipping processing for {url} due to download failure or invalid image.")

print("\n✨ All images processed. Check the 'new_results' directory for outputs.")
print(f"Downloaded images are temporarily stored in '{TEMP_DOWNLOAD_DIR}'. You can delete this folder manually.")



# ---------------------

import os
import pandas as pd
import numpy as np
import json
from colormap import rgb2hex
import extcolors
from PIL import Image
from skimage.color import rgb2lab # Import rgb2lab from scikit-image

class ExtractColor:
    @staticmethod
    def colorToDf(input):
        """
        Converts the raw color extraction output into a pandas DataFrame.

        Args:
            input: The raw output from extcolors.extract_from_path.

        Returns:
            A pandas DataFrame with 'c_code' (hex color) and 'occurence' (percentage).
        """
        colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
        df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
        df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]

        # Convert RGB tuples to hex color codes
        df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                                int(i.split(", ")[1]),
                                int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

        df = pd.DataFrame(zip(df_color_up, df_percent),
                            columns=['c_code', 'occurence'])
        return df

    @staticmethod
    def isNearBlackOrWhite(hex_color: str) -> bool:
        """
        Determines if a given hex color is near black or near white based on its luminance.

        Args:
            hex_color: A string representing the hex color (e.g., "#RRGGBB").

        Returns:
            True if the color's luminance is less than 0.05 (near black) or
            greater than 0.95 (near white), False otherwise.
        """
        def linearize(channel: float) -> float:
            """
            Linearizes an sRGB channel value.
            """
            return channel / 12.92 if channel < 0.03928 else pow((channel + 0.055) / 1.055, 2.4)

        hex_clean = hex_color.replace("#", "")
        if len(hex_clean) != 6:
            # Invalid hex code length
            return False

        try:
            r = int(hex_clean[0:2], 16)
            g = int(hex_clean[2:4], 16)
            b = int(hex_clean[4:6], 16)
        except ValueError:
            # Invalid hex characters
            return False

        r_norm = float(r) / 255.0
        g_norm = float(g) / 255.0
        b_norm = float(b) / 255.0

        r_lin = linearize(r_norm)
        g_lin = linearize(g_norm)
        b_lin = linearize(b_norm)

        # Calculate luminance using the sRGB luminance formula
        luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

        # Check if the luminance is near black or near white
        return luminance < 0.05 or luminance > 0.95

    @staticmethod
    def hex_to_rgb_norm(hex_color: str) -> np.ndarray:
        """
        Converts a hex color string to a normalized RGB numpy array.

        Args:
            hex_color: A string representing the hex color (e.g., "#RRGGBB").

        Returns:
            A numpy array of shape (1, 3) with normalized RGB values.
        """
        hex_color = hex_color.strip().lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return np.array([[r, g, b]]) # shaped for rgb2lab

    @staticmethod
    def chroma_from_lab(lab: np.ndarray) -> float:
        """
        Calculates the chroma (vibrancy) from a LAB color value.

        Args:
            lab: A numpy array representing the LAB color (e.g., from rgb2lab).

        Returns:
            The chroma value.
        """
        a, b = lab[0][1], lab[0][2]
        return np.sqrt(a**2 + b**2) # Corrected formula for chroma

    @staticmethod
    def most_vibrant_color(hex1: str, hex2: str) -> str:
        """
        Compares two hex colors and returns the one with higher vibrancy (chroma).

        Args:
            hex1: The first hex color string.
            hex2: The second hex color string.

        Returns:
            The hex color string that is more vibrant.
        """
        rgb1 = ExtractColor.hex_to_rgb_norm(hex1)
        rgb2 = ExtractColor.hex_to_rgb_norm(hex2)
        lab1 = rgb2lab(rgb1)
        lab2 = rgb2lab(rgb2)
        return hex1 if ExtractColor.chroma_from_lab(lab1) > ExtractColor.chroma_from_lab(lab2) else hex2

    @staticmethod
    def extractColor(path):
        """
        Extracts the single most vibrant dominant color from an image,
        excluding near black/white colors.

        Args:
            path: The file path to the image.

        Returns:
            A list containing the single most vibrant hex color, or an empty list if none found.
        """
        tolerance = 11

        img = Image.open(path)
        width, height = img.size
        if width > 250:
            new_width = 250
            new_height = int((height / width) * new_width)
            img = img.resize((new_width, new_height))
            # Save the resized image back to the path to be processed by extcolors
            img.convert('RGB').save(path)

        colors_x = extcolors.extract_from_path(path, tolerance=tolerance, limit=13)
        df_color = ExtractColor.colorToDf(colors_x)

        # Filter out near black or white colors
        filtered_colors = [
            color for color in df_color['c_code'].tolist()
            if not ExtractColor.isNearBlackOrWhite(color)
        ]

        # If there are no filtered colors, return an empty list
        if not filtered_colors:
            return []
        # If there's only one filtered color, return it
        elif len(filtered_colors) == 1:
            return [filtered_colors[0]]
        # If there are two or more filtered colors, find the most vibrant among the top 2
        else:
            vibrant_color = ExtractColor.most_vibrant_color(filtered_colors[0], filtered_colors[1])
            return [vibrant_color]

# ========================== Pipeline Execution ==========================

if __name__ == "__main__":
    # Process images from the 'new_results' folder
    image_extensions = ('.png',)
    new_results_dir = "new_results_latest4"
    csv_path = "/home/justdial/Videos/archive/Master Data_Go_JD_latest3 copy 2.csv"

    if not os.path.exists(new_results_dir):
        print(f"❌ Folder '{new_results_dir}' not found.")
        exit()

    files = [f for f in os.listdir(new_results_dir) if f.lower().endswith('_subject.png')]

    if not files:
        print(f"❌ No '_subject' images found in '{new_results_dir}'.")
        exit()

    print(f"🔍 Found {len(files)} subject images to process for color extraction...")

    # Load the CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded successfully: {csv_path}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        exit()

    # Add dominant_colors column if not present
    if 'dominant_colors' not in df.columns:
        df['dominant_colors'] = ''

    for image_file in files:
        try:
            image_path = os.path.join(new_results_dir, image_file)
            # Now extracting the single most vibrant dominant color
            top_colors = ExtractColor.extractColor(image_path)
            print(f"{image_file} ➤ Most Vibrant Dominant Color (filtered): {top_colors}")

            # Get the base name to match (removing _subject.png)
            base_filename = image_file.replace('_subject.png', '')

            # Find matching row
            matched_idx = df[df['product_thumb'].str.contains(base_filename, na=False)].index

            if not matched_idx.empty:
                for idx in matched_idx:
                    df.at[idx, 'dominant_colors'] = str(top_colors)
                print(f"✅ Updated dominant_colors for: {base_filename}")
            else:
                print(f"⚠️ No match found in CSV for: {base_filename}")

        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")

    # Save the updated CSV
    try:
        df.to_csv(csv_path, index=False)
        print(f"\n✅ CSV successfully updated with dominant colors: {csv_path}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

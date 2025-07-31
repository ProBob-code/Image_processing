import os
import sys # <--- Add this import
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import requests
from urllib.parse import urlparse
import pandas as pd
from colormap import rgb2hex
import extcolors
from skimage.color import rgb2lab
import colorsys
from helper import getConfigInfo
import traceback
from collections import OrderedDict

class DominantColorProcess:
    def __init__(self, model_path="/opt/airflow/models/saved_models/u2net/u2netp.pth"):
        """
        Initializes the DominantColorProcess with the U2NETP model.
        """
        self.MODEL_PATH = model_path
        self.NEW_RESULTS_DIR = getConfigInfo('NFS_path.video_input')
        self.MASK_SUFFIX = "_mask.png"
        self.TEMP_DOWNLOAD_DIR = getConfigInfo('NFS_path.video_input')

        # Create directories if they don't exist
        os.makedirs(self.NEW_RESULTS_DIR, exist_ok=True)
        os.makedirs(self.TEMP_DOWNLOAD_DIR, exist_ok=True)

        self._load_model()
        self._define_transforms()

    def _load_model(self):
        """
        Loads the U2NETP model and sets it to evaluation mode.
        """
        print("⏳ Loading U2NETP model...")

        # Add the directory containing u2net.py to sys.path
        # This makes 'model' (or whatever u2net.py is part of) discoverable
        # Ensure that '/opt/airflow/models' is the parent directory of 'u2net.py'
        # If u2net.py is directly in /opt/airflow/models
        u2net_parent_dir = '/opt/airflow/models'
        if u2net_parent_dir not in sys.path:
            sys.path.append(u2net_parent_dir)

        # Now, you can directly import u2net, no need for model.u2net
        # Assuming u2net.py defines U2NETP at its top level
        try:
            from u2net import U2NETP # <--- CHANGE THIS IMPORT
        except ImportError as e:
            print(f"❌ Error: Could not import U2NETP. Is 'u2net.py' in '{u2net_parent_dir}'?")
            print(f"Current sys.path: {sys.path}")
            raise e


        self.net = U2NETP(3, 1)

        try:
            self.net.load_state_dict(torch.load(self.MODEL_PATH, map_location='cpu'))
            print("✅ Model weights loaded successfully.")
        except FileNotFoundError:
            print(f"❌ Error: Model weights file not found at '{self.MODEL_PATH}'.")
            print("Please ensure the 'saved_models/u2net/u2netp.pth' file exists in the correct path.")
            raise FileNotFoundError(f"Model weights file not found at '{self.MODEL_PATH}'")
        except Exception as e:
            print(f"❌ An error occurred while loading model weights: {e}")
            raise RuntimeError(f"Error loading model weights: {e}")

        self.net.eval()
        print("✅ Model is in evaluation mode.")

    def _define_transforms(self):
        """
        Defines the preprocessing pipeline for input images.
        """
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
        )        ])

    def download_image(self, image_url: str) -> str or None:
        """
        Downloads an image from a given URL and saves it to a specified path.
        Returns the path to the downloaded image if successful, None otherwise.
        """
        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        if not filename or not any(filename.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png')):
            filename = f"downloaded_image_{len(os.listdir(self.TEMP_DOWNLOAD_DIR)) + 1}.jpg"

        save_path = os.path.join(self.TEMP_DOWNLOAD_DIR, filename)

        try:
            print(f"Attempting to download: {image_url}")
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()

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

    def process_image_for_background_removal(self, image_path: str) -> str or None:
        """
        Processes a single image from a local path to generate a background-removed image.
        Saves the mask and the transparent subject image to the specified output directory.
        Returns the path to the subject image if successful, None otherwise.
        """
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}. Skipping.")
            return None

        print(f"\n🔍 Processing image for background removal: {image_path}")
        basename = os.path.splitext(os.path.basename(image_path))[0]

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ Could not open image {image_path}: {e}. Skipping.")
            return None

        original_size = image.size

        input_tensor = self.transform(image).unsqueeze(0)
        input_tensor = Variable(input_tensor)

        with torch.no_grad():
            d1, *_ = self.net(input_tensor)
            pred = d1[:, 0, :, :].squeeze().cpu().numpy()

            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            pred = (pred * 255).astype(np.uint8)

        mask = cv2.resize(pred, original_size)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        mask_filename = os.path.join(self.NEW_RESULTS_DIR, f"{basename}{self.MASK_SUFFIX}")
        cv2.imwrite(mask_filename, binary_mask)
        print(f"✅ Saved mask: {mask_filename}")

        image_cv = cv2.imread(image_path)
        mask_cv = cv2.imread(mask_filename, 0)

        if image_cv is None:
            print(f"❌ Could not read image_cv from {image_path}. Skipping background removal.")
            return None
        if mask_cv is None:
            print(f"❌ Could not read mask_cv from {mask_filename}. Skipping background removal.")
            return None

        if image_cv.shape[0] != mask_cv.shape[0] or image_cv.shape[1] != mask_cv.shape[1]:
            mask_cv = cv2.resize(mask_cv, (image_cv.shape[1], image_cv.shape[0]))

        b, g, r = cv2.split(image_cv)
        rgba = cv2.merge((b, g, r, mask_cv))

        output_path = os.path.join(self.NEW_RESULTS_DIR, f"{basename}_subject.png")
        cv2.imwrite(output_path, rgba)
        print(f"✅ Transparent subject image saved: {output_path}")
        return output_path

    @staticmethod
    def _color_to_df(input_colors) -> pd.DataFrame:
        """
        Converts the raw color extraction output into a pandas DataFrame.
        """
        colors_pre_list = str(input_colors).replace('([(','').split(', (')[0:-1]
        df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
        # Parse occurence as integer, not string
        df_percent = [int(i.split('), ')[1].replace(')','')) for i in colors_pre_list]

        df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                                int(i.split(", ")[1]),
                                int(i.split(", ")[2].replace(")",""))) for i in df_rgb]

        df = pd.DataFrame(zip(df_color_up, df_percent), columns=['c_code', 'occurence'])
        return df
    
    @staticmethod
    def _rgb_to_luminance(r: int, g: int, b: int) -> float:
        """
        Calculates luminance for an RGB color.
        Formula based on WCAG 2.0 (relative luminance).
        """
        def linearize(channel: float) -> float:
            srgb = channel / 255.0
            return srgb / 12.92 if srgb < 0.03928 else pow((srgb + 0.055) / 1.055, 2.4)

        r_lin = linearize(r)
        g_lin = linearize(g)
        b_lin = linearize(b)

        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    @staticmethod
    def _is_near_black_or_white(hex_color: str) -> bool:
        """
        Determines if a given hex color is near black or near white based on its luminance.
        """
        hex_clean = hex_color.replace("#", "")
        if len(hex_clean) != 6:
            return False

        try:
            r = int(hex_clean[0:2], 16)
            g = int(hex_clean[2:4], 16)
            b = int(hex_clean[4:6], 16)
        except ValueError:
            return False

        luminance = DominantColorProcess._rgb_to_luminance(r, g, b)
        return luminance < 0.05 or luminance > 0.95

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """Converts a hex color string to an RGB tuple (0-255)."""
        hex_color = hex_color.strip().lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        """Converts an RGB tuple (0-255) to a hex color string."""
        # Ensure RGB values are within 0-255 range
        r, g, b = max(0, min(255, int(rgb[0]))), max(0, min(255, int(rgb[1]))), max(0, min(255, int(rgb[2])))
        return '#%02x%02x%02x' % (r, g, b)

    @staticmethod
    def _hex_to_rgb_norm(hex_color: str) -> np.ndarray:
        """
        Converts a hex color string to a normalized RGB numpy array.
        """
        hex_color = hex_color.strip().lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return np.array([[r, g, b]])

    @staticmethod
    def _chroma_from_lab(lab: np.ndarray) -> float:
        """
        Calculates the chroma (vibrancy) from a LAB roma  value.        """
        a, b = lab[0][1], lab[0][2]
        return np.sqrt(a**2 + b**2)

    def extract_dominant_color(self, image_path: str) -> dict or None:
        """
        Extracts the single most vibrant dominant color from an image,
        excluding near black/white colors, and applies luminance adjustment.
        Returns a dictionary with 'dominant_color_hex', 'initial_luminance',
        'final_dominant_color_hex', and 'final_luminance'.
        Returns None if no suitable color is found.
        """
        tolerance = 11
        luminance_threshold = 0.179
        value_reduction_step = 0.05 # Reduce V by 5% in each iteration (0.05 for 5%)
        max_iterations = 20 # Safety break to prevent infinite loops

        try:
            img = Image.open(image_path)
            width, height = img.size
            if width > 250:
                new_width = 250
                new_height = int((height / width) * new_width)
                img = img.resize((new_width, new_height))
                # Save the resized image back to the path to be processed by extcolors
                img.convert('RGB').save(image_path)

            colors_x = extcolors.extract_from_path(image_path, tolerance=tolerance, limit=13)
            df_color = self._color_to_df(colors_x)

            # Filter out near black or white colors
            filtered_colors = [
                color for color in df_color['c_code'].tolist()
                if not self._is_near_black_or_white(color)
            ]

            print("filtered_colors:", filtered_colors)

            # Map hex codes to percentages (assuming df_color has 'c_code' & 'occurence' columns)
            total_pixels = df_color['occurence'].sum()
            filtered_df = df_color[df_color['c_code'].isin(filtered_colors)]
            filtered_df['percentage'] = filtered_df['occurence'] / total_pixels * 100


            color_dict = OrderedDict(
                (row['c_code'], float(row['percentage']))
                for _, row in filtered_df.sort_values('percentage', ascending=False).iterrows()
            )

            if not filtered_colors:
                return None
            elif len(filtered_colors) == 1:
                dominant_color_hex = filtered_colors[0]
            else:
                # Find the most vibrant among the top 2
                rgb1 = self._hex_to_rgb_norm(filtered_colors[0])
                rgb2 = self._hex_to_rgb_norm(filtered_colors[1])
                rgb3 = self._hex_to_rgb_norm(filtered_colors[2])
                lab1 = rgb2lab(rgb1)
                lab2 = rgb2lab(rgb2)
                lab3 = rgb2lab(rgb3)

                dominant_color_hex = filtered_colors[0] if self._chroma_from_lab(lab1) > self._chroma_from_lab(lab2) and self._chroma_from_lab(lab1) > self._chroma_from_lab(lab3) else filtered_colors[1]
                print(f"🔍 Selected dominant color: {dominant_color_hex} based on vibrancy.")


            # --- Calculate Initial Luminance ---
            r_initial, g_initial, b_initial = self._hex_to_rgb(dominant_color_hex)
            initial_luminance = self._rgb_to_luminance(r_initial, g_initial, b_initial)
            print(f"📊 Initial Dominant Color: {dominant_color_hex}, Initial Luminance: {initial_luminance:.3f}")

            # --- Apply Luminance-based adjustment (Iterative) ---
            current_hex = dominant_color_hex
            current_luminance = initial_luminance
            
            # Convert to HSV (normalized 0-1 range) for adjustment
            h, s, v = colorsys.rgb_to_hsv(r_initial / 255.0, g_initial / 255.0, b_initial / 255.0)

            iterations = 0
            while current_luminance > luminance_threshold and iterations < max_iterations:
                # Reduce V by a step
                v -= value_reduction_step
                v = max(0.0, v) # Ensure V doesn't go below 0

                # Convert back to RGB and then to hex
                new_r_norm, new_g_norm, new_b_norm = colorsys.hsv_to_rgb(h, s, v)
                new_r = int(new_r_norm * 255)
                new_g = int(new_g_norm * 255)
                new_b = int(new_b_norm * 255)
                
                current_hex = self._rgb_to_hex((new_r, new_g, new_b))
                current_luminance = self._rgb_to_luminance(new_r, new_g, new_b)
                
                print(f"💡 Iteration {iterations+1}: Adjusted color to {current_hex} (luminance: {current_luminance:.3f}, V: {v:.3f})")
                
                iterations += 1
                if v == 0.0: # If V hits zero, no further reduction is possible
                    break
            
            final_dominant_color_hex = current_hex
            final_luminance = current_luminance # This is the final luminance after adjustments
            
            print(f"🎨 Final Dominant Color: {final_dominant_color_hex} (Final Luminance: {final_luminance:.3f})")

            return {
                "dominant_color_hex": dominant_color_hex,
                "initial_luminance": initial_luminance,
                "final_dominant_color_hex": final_dominant_color_hex,
                "final_luminance": final_luminance,
                "color": color_dict,
            }
        except Exception as e:
            print(f"❌ Error extracting dominant color from {image_path}: {e}")
            return None

    def process_urls_and_extract_colors(self, image_urls: list[str]) -> list[dict]:
        """
        Main function to download images from URLs, remove backgrounds,
        and extract dominant colors.

        Args:
            image_urls: A list of image URLs.

        Returns:
            A list of dictionaries, each containing 'file_name', 'subject_image_path',
            'dominant_color_hex', 'initial_luminance', 'final_dominant_color_hex',
            and 'final_luminance'.
        """
        all_results = []
        print("\n--- Starting URL processing and color extraction pipeline ---")

        for url in image_urls:
            downloaded_path = self.download_image(url)
            if downloaded_path:
                subject_image_path = self.process_image_for_background_removal(downloaded_path)
                if subject_image_path:
                    dominant_color_info = self.extract_dominant_color(subject_image_path)
                    
                    if dominant_color_info:
                        all_results.append({
                            "original_url": url,
                            "file_name": os.path.basename(subject_image_path),
                            "subject_image_path": subject_image_path,
                            "dominant_color_hex": dominant_color_info.get("dominant_color_hex"),
                            "initial_luminance": dominant_color_info.get("initial_luminance"),
                            "final_dominant_color_hex": dominant_color_info.get("final_dominant_color_hex"),
                            "final_luminance": dominant_color_info.get("final_luminance"),
                            "color": dominant_color_info.get("color", {})
                        })
                        print(f"✅ Processed {os.path.basename(subject_image_path)}: Original Dominant Color: {dominant_color_info.get('dominant_color_hex')} (L: {dominant_color_info.get('initial_luminance'):.3f}), Final Dominant Color: {dominant_color_info.get('final_dominant_color_hex')} (L: {dominant_color_info.get('final_luminance'):.3f})")
                    else:
                        all_results.append({
                            "original_url": url,
                            "file_name": os.path.basename(subject_image_path),
                            "subject_image_path": subject_image_path,
                            "dominant_color_hex": "Color extraction failed",
                            "initial_luminance": None,
                            "final_dominant_color_hex": "Color extraction failed",
                            "final_luminance": None,
                            "color": {}
                        })
                else:
                    all_results.append({
                        "original_url": url,
                        "file_name": os.path.basename(downloaded_path) if downloaded_path else "N/A",
                        "subject_image_path": None,
                        "dominant_color_hex": "Background removal failed",
                        "initial_luminance": None,
                        "final_dominant_color_hex": "Background removal failed",
                        "final_luminance": None,
                        "color": {}
                    })
            else:
                all_results.append({
                    "original_url": url,
                    "file_name": "N/A",
                    "subject_image_path": None,
                    "dominant_color_hex": "Download failed",
                    "initial_luminance": None,
                    "final_dominant_color_hex": "Download failed",
                    "final_luminance": None,
                    "color": {}
                })
            
            # Optional: Clean up the downloaded raw image after processing
            if downloaded_path and os.path.exists(downloaded_path):
                os.remove(downloaded_path)
                print(f"🗑️ Removed temporary download: {downloaded_path}")

        print("\n--- Pipeline execution complete ---")
        return all_results
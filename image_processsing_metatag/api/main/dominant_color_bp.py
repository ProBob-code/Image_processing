import os
import re
import requests
from flask import Blueprint, request
from datetime import datetime
from class_api_response import ApiResponse
from classes.manage_logs import Manage_Logs
from common import is_corrupted

dominant_color_bp = Blueprint('dominant_color_bp', __name__, url_prefix='/cp/api/v1')
logging_enabled = True

def log(message):
    global logging_enabled, logs
    if logging_enabled:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(f"[{current_time}] {message}")
        logs.append(f"[{current_time}] {message}")

@dominant_color_bp.route('/hello',methods=['GET'])
def check_test():
    apiResponse = ApiResponse()
    return apiResponse.responseSuccess("Hello World!")

@dominant_color_bp.route('/extract-dominant-color', methods=['GET', 'POST'])
def extract_dominant_color():
    global logs
    logs = []  # Clear logs for each new request
    start_time_request = datetime.now()
    log(f"API Request Start Time: {start_time_request}")

    # Import or instantiate DominantColorProcess if not already done
    try:
        from classes.dominant_color import DominantColorProcess
        from PIL import Image
        dominant_color_processor = DominantColorProcess()
    except ImportError:
        dominant_color_processor = None
        Image = None  # Ensure Image is defined

    # Initialize paths to None for cleanup in finally block
    downloaded_path = None
    subject_image_path = None

    try:
        if request.method == 'GET':
            apiResponse = ApiResponse()
            return apiResponse.responseSuccess(
                "This endpoint expects a POST request with image data.",
                {"example_usage": {"method": "POST", "body": {"local_path": "/path/to/your/image.png", "product_url": "http://example.com/image.jpg"}}}
            )

        elif request.method == 'POST':
            # Check if the dominant_color_processor instance was successfully initialized
            if dominant_color_processor is None:
                apiResponse = ApiResponse()
                return apiResponse.responseBadRequest("Image processing service is not initialized. Please check server logs.")

            post_data = request.form.to_dict()
            image_to_process_path = None # This will hold the path to the image (local or downloaded)
            api_name = "extract_dominant_color" # Define a specific API name for logging

            # Determine image source: local_path or product_url
            if 'local_path' in post_data and post_data['local_path']:
                image_to_process_path = re.sub(r"\\\/", "/", post_data['local_path'])
                log(f"Received local_path: {image_to_process_path}")
                
                if not os.path.isfile(image_to_process_path):
                    Manage_Logs.input_request(image_to_process_path, None, api_name, "Local file not found")
                    apiResponse = ApiResponse()
                    return apiResponse.responseNotFound(f"Image file not found at: {image_to_process_path}")
                
                if not is_corrupted(image_to_process_path):
                    Manage_Logs.input_request(image_to_process_path, None, api_name, "Image corrupted")
                    apiResponse = ApiResponse()
                    return apiResponse.responseNotFound(f"Image corrupted or deleted at: {image_to_process_path}")
            
            elif 'product_url' in post_data and post_data['product_url']:
                product_url = post_data['product_url']
                log(f"Received product_url: {product_url}")
                
                # Use the DominantColorProcess's own download_image method
                downloaded_path = dominant_color_processor.download_image(product_url)
                if not downloaded_path:
                    Manage_Logs.input_request(None, product_url, api_name, "Download failed")
                    apiResponse = ApiResponse()
                    return apiResponse.responseNotFound(f"Failed to download image from URL: {product_url}. Check URL or network connectivity.")
                
                image_to_process_path = downloaded_path # Set the path to the downloaded file
                log(f"Image downloaded to: {image_to_process_path}")
                
                # Add is_corrupted check for downloaded images
                if not is_corrupted(image_to_process_path, product_url):
                    Manage_Logs.input_request(None, product_url, api_name, "Downloaded image corrupted")
                    apiResponse = ApiResponse()
                    # Clean up the downloaded file immediately if corrupted
                    if downloaded_path and os.path.exists(downloaded_path):
                        os.remove(downloaded_path)
                        log(f"🗑️ Removed corrupted downloaded image: {downloaded_path}")
                    return apiResponse.responseNotFound(f"Downloaded image corrupted: {product_url}")

            else:
                apiResponse = ApiResponse()
                return apiResponse.responseBadRequest('Please provide either "local_path" or "product_url" in the POST request.')

            # --- Image Processing Pipeline ---
            start_time_processing = datetime.now()
            
            # Step 1: Process image for background removal
            subject_image_path = dominant_color_processor.process_image_for_background_removal(image_to_process_path)
            
            dominant_color_info = None
            if subject_image_path:
                # Step 2: Extract dominant color from the subject image
                dominant_color_info = dominant_color_processor.extract_dominant_color(subject_image_path)
            else:
                log(f"Background removal failed for {image_to_process_path}. Cannot proceed with color extraction.")
                apiResponse = ApiResponse()
                return apiResponse.responseNotFound(f"Background removal failed for image: {image_to_process_path}. Cannot extract dominant color.")

            end_time_processing = datetime.now()
            time_taken_processing = end_time_processing - start_time_processing
            log(f"Time taken for image processing (BR + Color): {time_taken_processing}")

            # Prepare the response based on dominant_color_info
            if dominant_color_info:
                response_data = {
                    'local_path': subject_image_path, # Path to the processed subject image on the server
                    'dominant_color_hex': dominant_color_info.get("dominant_color_hex"),
                    'initial_luminance': dominant_color_info.get("initial_luminance"),
                    'final_dominant_color_hex': dominant_color_info.get("final_dominant_color_hex"),
                    'final_luminance': dominant_color_info.get("final_luminance"),
                    'color': dominant_color_info.get("color", {})
                }
                message = "Dominant color extracted successfully."
                status_code = 200
            else:
                # If extract_dominant_color returns None, always try to get color from fallback API
                fallback_color = None
                fallback_luminance = None
                fallback_final_hex = None
                fallback_final_luminance = None

                # Try to detect if the image is a solid (single tone) color: white, black, or any other
                detected_color = None
                detected_luminance = None
                try:
                    with Image.open(subject_image_path) as img:
                        img = img.convert("RGB")
                        # Increase maxcolors for large images to ensure detection
                        colors = img.getcolors(maxcolors=1024*1024)
                        if colors and len(colors) == 1:
                            # Single tone color image (white, black, or any other)
                            count, rgb = colors[0]
                            detected_color = '#%02x%02x%02x' % rgb
                            # Calculate luminance
                            r, g, b = rgb
                            detected_luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            log(f"Detected single tone color image: {detected_color}, luminance: {detected_luminance}")
                except Exception as e:
                    log(f"Single tone color detection failed: {e}")

                if detected_color:
                    response_data = {
                        'local_path': subject_image_path,
                        'dominant_color_hex': detected_color,
                        'initial_luminance': detected_luminance,
                        'final_dominant_color_hex': detected_color,
                        'final_luminance': detected_luminance,
                        'color': {
                            'hex': detected_color,
                            #'rgb': detected_color[1:],  # Remove '#' for RGB
                            'luminance': detected_luminance
                        }
                    }
                    message = "Single tone color image detected (white, black, or other). Returning detected color."
                    status_code = 200
                else:
                    # Always call the fallback API first
                    product_url = post_data.get("product_url")
                    fallback_color = None

                    if product_url:
                        try:
                            resp = requests.post(
                                "http://192.168.131.170/cp/api/v1/extract-color",
                                data={"product_url": product_url},
                                timeout=10
                            )
                            if resp.ok:
                                resp_json = resp.json()
                                print(resp_json)  # Debugging line to see the response structure
                                color_list = []
                                # Try to get color list from new structure
                                if (
                                    isinstance(resp_json, dict)
                                    and "data" in resp_json
                                    and isinstance(resp_json["data"], dict)
                                    and "response" in resp_json["data"]
                                    and isinstance(resp_json["data"]["response"], dict)
                                ):
                                    color_list = resp_json["data"]["response"].get("color", [])
                                # Fallback to old structure if needed
                                if not color_list and "data" in resp_json:
                                    color_list = resp_json.get("data", {}).get("dominant_colors", [])
                                if color_list and isinstance(color_list, list):
                                    # Pick the color with max percentage, handle missing keys
                                    try:
                                        max_color = max(
                                            color_list,
                                            key=lambda x: x.get("percentage", 0) if isinstance(x, dict) else 0
                                        )
                                        fallback_color = (
                                            max_color.get("hex")
                                            or max_color.get("color_hex")
                                            or None
                                        )
                                    except Exception as e:
                                        log(f"Error extracting color from color_list: {e}")
                        except Exception as e:
                            log(f"Fallback color API call failed: {e}")

                    # If fallback API gave a color, use it
                    if fallback_color:
                        response_data = {
                            'local_path': subject_image_path,
                            'dominant_color_hex': fallback_color,
                            'initial_luminance': None,
                            'final_dominant_color_hex': fallback_color,
                            'final_luminance': None,
                            'color': {
                                'hex': fallback_color,
                                'luminance': None  # Luminance not available from fallback
                            }
                        }
                        message = "Dominant color extracted successfully using fallback API."
                        status_code = 200
                    else:
                        response_data = {
                            'local_path': subject_image_path,
                            'dominant_color_hex': "#FFFFFF",  # Default to white or any fallback color
                            'initial_luminance': None,
                            'final_dominant_color_hex': "#FFFFFF",
                            'final_luminance': None,
                            'color': {
                                'hex': "#FFFFFF",
                                'luminance': None
                            }
                        }
                        message = "Could not extract dominant color. Returning default color."
                        status_code = 200
                   
            # Log the request and response
            if logging_enabled:
                Manage_Logs.input_request(post_data.get("local_path", "N/A"), post_data.get("product_url", "N/A"), api_name)
                logs.append({"request": post_data, "response": response_data, "overall_logs": logs})
            
            apiResponse = ApiResponse()
            if status_code == 200:
                return apiResponse.responseSuccess(message, response_data)
            else:
                return apiResponse.responseNotFound(message) # Using NotFound for consistency with your original example

    except Exception as e:
        # Catch any unexpected errors during processing
        error_message = f"An unexpected error occurred during image processing: {e}"
        log(f"ERROR: {error_message}")
        # Ensure post_data is available for logging even if error occurs early
        log_local_path = post_data.get("local_path", "N/A") if 'post_data' in locals() else "N/A"
        log_product_url = post_data.get("product_url", "N/A") if 'post_data' in locals() else "N/A"
        Manage_Logs.input_request(log_local_path, log_product_url, api_name if 'api_name' in locals() else "extract_dominant_color", error=str(e))
        apiResponse = ApiResponse()
        return apiResponse.responseBadRequest(error_message)
    
    finally:
        # Clean up temporary downloaded original image file
        if downloaded_path and os.path.exists(downloaded_path):
            try:
                os.remove(downloaded_path)
                log(f"🗑️ Removed temporary downloaded original image: {downloaded_path}")
            except Exception as e:
                log(f"⚠️ Error cleaning up temporary file {downloaded_path}: {e}")
        
        # Clean up processed subject image file if it was created and is no longer needed
        if subject_image_path and os.path.exists(subject_image_path):
            try:
                os.remove(subject_image_path)
                log(f"🗑️ Removed processed subject image: {subject_image_path}")
                
                # Attempt to remove the mask file too, assuming its name pattern
                mask_filename = os.path.splitext(subject_image_path)[0].replace('_subject', '') + dominant_color_processor.MASK_SUFFIX
                if os.path.exists(mask_filename):
                    os.remove(mask_filename)
                    log(f"🗑️ Removed associated mask file: {mask_filename}")
            except Exception as e:
                log(f"⚠️ Error cleaning up processed subject image or mask {subject_image_path}: {e}")
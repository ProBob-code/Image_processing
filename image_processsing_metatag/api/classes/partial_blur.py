import numpy as np
import cv2
from PIL import Image
import requests
import io
import logging
from classes.imagemeta_tag import ImageMeta


class Partial_Blur():
    def __init__(self):
        pass


    def f_sharpness_score(img):
        try:
            # Check if image has 3 channels and is it not None before conversion
            if img is not None and len(img.shape)==3 and (img.shape[2] == 3 or img.shape[2] == 4):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img is not None:
                # Handle unexpected shape
                # logging.info(f"Partial_BLur: img.shape={img.shape}")
                if(len(img.shape)==2):
                    gray = img
                    # logging.info(f"Partial_BLur:Image already grayscale, img.shape = {img.shape}")
                elif(len(img.shape)==3 and img.shape[2]==1):
                    # Remove the last dimension if it has only 1 element
                    gray = np.squeeze(img)
                else:
                    logging.info(f"Partial_BLur:Image has unexpected number of channels, img.shape={img.shape}")
                    raise AttributeError(f"Partial_BLur: Image has unexpected number of channels, img.shape={img.shape}")
            else:
                logging.info(f"Partial_BLur: img={img}")
                raise AttributeError(f"Partial_BLur: Image is of NoneType, img={img}")
        except Exception as e:
            # Handle cases where img might not have a shape attribute
            logging.info(f"Partial_BLur:Image loading failed or has unexpected format, type(img)={type(img)}")
            raise AttributeError(e)
        # Calculate gradients using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate Tenengrad (gradient magnitude)
        tenengrad = np.sqrt(grad_x**2 + grad_y**2)

        # Calculate the sharpness score as the average of Tenengrad values
        new_sharpness_score = float(np.mean(tenengrad))
        return new_sharpness_score


    def fetchImageHeightWidthV2(image_url):
        response = requests.get(image_url, stream=True)
        chunk_size = 1024  # You can adjust this based on the image file's format
        if response.status_code == 200:
            # Open the image using PIL (Pillow)
            with Image.open(io.BytesIO(response.content)) as img:
                width, height = img.size
                size = int(response.headers.get('Content-Length', 0))
        else:
            print("Failed to fetch the image")
            output = {
                'error_code': 1,
                'url' : image_url,
                'msg' : 'Invalid url or not exist'
            }
            return output
        # Close the response
        out = {
            'error_code': 0,
            'size': size,
            'height': height,
            'width': width
            # 'image_path': tata
        }
        return out


    def find_clusters(matrix, val, mask):
        """
        Finds clusters of values less than or equal to 'val' in a 2D NumPy array,
        considering all adjacent neighbors (row-wise, column-wise, and diagonal).

        Args:
            matrix (np.ndarray): The 2D NumPy array of values.
            val (float): The threshold value for cluster identification.

        Returns:
            list: A list of tuples, where each tuple contains:
                - (int, int): The coordinates (row, col) of the cluster's starting point.
                - int: The number of elements within the cluster.
        """

        rows, cols = matrix.shape
        visited = np.zeros_like(matrix, dtype=bool)  # Track visited elements

        def dfs(row, col):
            """
            Depth-First Search to explore a cluster.

            Args:
                row (int): The current row index.
                col (int): The current column index.

            Returns:
                int: The total number of elements within the cluster.
            """
            # global mask
            if 0 <= row < rows and 0 <= col < cols and not visited[row, col] and matrix[row, col] >= val:
                visited[row, col] = True  # Mark as visited
                count = 1  # Count the current element
                mask[row, col] = 1
                # Explore adjacent neighbors (all directions)
                for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    count += dfs(row + drow, col + dcol)
                return count
            return 0

        clusters = []
        # Iterate through each element and start DFS if unvisited and below threshold
        for row in range(rows):
            for col in range(cols):
                if not visited[row, col] and matrix[row, col] >= val:
                    count = dfs(row, col)
                    mask[row, col] = 1
                    clusters.append((count, (row, col)))
        sorted_clusters = sorted(clusters, reverse=True)
        if(len(sorted_clusters)>0):
            mask[sorted_clusters[0][1]] = 2
        return sorted_clusters


    def create_grid(size, resized_img, grid_pixel, p_score_min, p_score_max, size_cutoff, sharpness_score_cutoff, top_percent=0.3):
        """
        Creates a grids of the given resized image.

        Args:
            resized_img (np.ndarray): The resized image as a NumPy array.
            grid_size (int, optional): The number of rows and columns in the grid.

        Returns:
            np.ndarray: The grid image as a NumPy array.
        """
        # print(type(resized_img))
        if len(resized_img.shape) == 3:
            height, width, channels = resized_img.shape
        else:
            height, width = resized_img.shape
            channels = 1  # Default value for grayscale image
        # Calculate grid cell dimensions
        grid_width = grid_pixel
        if(height!=0 and grid_width!=0 and height//grid_width!=0):
            grid_cell_height = grid_width + (height%grid_width)//(height//grid_width) 
        else:
            return _, _, "False"
        grid_cell_width = grid_width

        # Create an empty grid image
        grid_images = []
        list_ss = []
        num = 0
        m = height//grid_cell_height
        n = width//grid_cell_width
        array = np.zeros((m, n))
        mask = np.zeros((m, n))
        # Loop through each grid cell
        for i in range(m):
            for j in range(n):
                # width//grid_cell_width
                # Calculate starting coordinates for the current cell
                y_start = i * grid_cell_height
                x_start = j * grid_cell_width

                # Extract the sub-image for the current cell
                cell_img = resized_img[y_start:min(height, y_start + grid_cell_height), x_start:min(width, x_start + grid_cell_width)]
                ss_grid = Partial_Blur.f_sharpness_score(cell_img)
                
                list_ss.append(ss_grid)
                array[i][j] = ss_grid
                grid_images.append(cell_img)
                num+=1

        clusters = Partial_Blur.find_clusters(array, sharpness_score_cutoff, mask)
        # print(clusters)  # Output: [((1, 0), 2), ((2, 0), 1)]
        p_score = (100*clusters[0][0]/num) if len(clusters)>0 else 0
        # print("proportions: ", "{:.2f}".format(p_score), "%") 
        # print("size: ", "{:.2f}".format(size/1000), "KB") 
        if(p_score_max>=p_score>=p_score_min and size>=size_cutoff):
            blur_type = "True"
        else:    
            blur_type = "False"

        return array, mask, blur_type

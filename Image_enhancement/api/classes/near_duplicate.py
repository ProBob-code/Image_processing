from PIL import Image
import imagehash
import pandas as pd
from pymongo import MongoClient

class ImageLSH:
    def __init__(self, num_hash_tables, threshold):
        self.num_hash_tables = num_hash_tables
        self.threshold = threshold
        self.hash_tables = [{} for _ in range(num_hash_tables)]
        
    def hash_function(self, image_hash):
        return int(str(image_hash), 16) % self.num_hash_tables
    
    def index(self, hash_values):
        for hash_value in hash_values:
            for table_idx in range(self.num_hash_tables):
                bucket = self.hash_function(hash_value)
                if bucket not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][bucket] = []
                self.hash_tables[table_idx][bucket].append(hash_value)


    def hamming_distance(hash1, hash2):
        return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

# Read your DataFrame with 'product_id' and 'hash_value'
# Assuming the DataFrame is named 'df' with columns 'product_id' and 'hash_values'
# Example:

    def Dataconversion(docid):
        DocId = docid
        # Connect to MongoDB server
        client = MongoClient("mongodb://192.168.13.201:27017/")

        # Access database080PXX80.XX80.110218174640.L5D9
        db = client['db_product']

        # Authenticate with username and password
        username = "content_process"
        password = "C0nTent_pR0$3S"
        auth_db = "admin"
        db.authenticate(username, password, source=auth_db)

        # Access collection
        collection = db['tbl_catalogue_details']

        # Query data
        query = {"docid": DocId}
        print(query)
        query = {"docid": {"$in": docids}}

        # Fetch data and create a DataFrame
        cursor = collection.find(query)
        df = pd.DataFrame(list(cursor))

        df.dropna(inplace=True)

        df['keyword_ct'] = df['keywords'].apply(lambda x: x.get('ct', ''))
        df['keyword_cc'] = df['keywords'].apply(lambda x: x.get('cc', ''))
        df['keyword_cm'] = df['keywords'].apply(lambda x: x.get('cm', ''))
        df['keyword_dt'] = df['keywords'].apply(lambda x: x.get('dt', ''))
        df['keyword_cf'] = df['keywords'].apply(lambda x: x.get('cf', ''))
        df['quality'] = df['score'].apply(lambda x: x.get('quality', ''))
        df['source'] = df['score'].apply(lambda x: x.get('source', ''))
        df['classification'] = df['score'].apply(lambda x: x.get('classification', ''))
        df['like'] = df['score'].apply(lambda x: x.get('like', ''))
        df['report'] = df['score'].apply(lambda x: x.get('report', ''))

        extract_keywords = lambda x: x['parent']['keywords'] 
        extract_image_name = lambda x: x['parent']['image_name'] 
        extract_height = lambda x: x['parent']['height'] 
        extract_width = lambda x: x['parent']['width'] 
        extract_size = lambda x: x['parent']['size'] 
        extract_image_format = lambda x: x['parent']['image_format'] 
        extract_image_mode = lambda x: x['parent']['image_mode'] 
        extract_description = lambda x: x['parent']['description'] 
        extract_author = lambda x: x['parent']['author'] 
        extract_copyright = lambda x: x['parent']['copyright'] 
        extract_location = lambda x: x['parent']['location'] 
        extract_image_shape = lambda x: x['parent']['image_shape'] 

        # #derived
        extract_pixel_1 = lambda x: x['derived']['pixels'] 
        extract_megapixels = lambda x: x['derived']['megapixels'] 
        extract_ppi = lambda x: x['derived']['ppi'] 
        extract_exif_dict = lambda x: x['derived']['exif_dict']
        extract_red = lambda x: x['derived']['red'] 
        extract_green = lambda x: x['derived']['green'] 
        extract_blue = lambda x: x['derived']['blue'] 
        extract_laplacian_variance_blur = lambda x: x['derived']['laplacian_variance_blur'] 
        extract_fourier_transform_blur = lambda x: x['derived']['fourier_transform_blur'] 
        extract_gradient_magnitude_blur = lambda x: x['derived']['gradient_magnitude_blur'] 
        extract_brightness_score = lambda x: x['derived']['brightness_score']
        extract_colourfulness = lambda x: x['derived']['colourfulness']
        extract_sharpness_score = lambda x : x['derived']['sharpness_score']
        extract_image_metric = lambda x: x['derived']['image_metric'] 
        extract_hash_value = lambda x : x['derived']['hash_value']
        extract_duplicate = lambda x: x['derived']['duplicate'] 
        extract_color = lambda x: x['color']

        # Apply the lambda functions to the meta column and create new columns
        df['meta_keywords'] = df['meta'].apply(extract_keywords)
        df['image_name'] = df['meta'].apply(extract_image_name)
        df['height'] = df['meta'].apply(extract_height)
        df['width'] = df['meta'].apply(extract_width)
        df['size'] = df['meta'].apply(extract_size)
        df['image_format'] = df['meta'].apply(extract_image_format)
        df['image_mode'] = df['meta'].apply(extract_image_mode)
        df['description'] = df['meta'].apply(extract_description)
        df['author'] = df['meta'].apply(extract_author)
        df['copyright'] = df['meta'].apply(extract_copyright)
        df['location'] = df['meta'].apply(extract_location)
        df['image_shape'] = df['meta'].apply(extract_image_shape)

        df['pixels'] = df['meta'].apply(extract_pixel_1)
        df['megapixels'] = df['meta'].apply(extract_megapixels)
        df['ppi'] = df['meta'].apply(extract_ppi)
        df['exif_dict'] = df['meta'].apply(extract_exif_dict)
        df['red'] = df['meta'].apply(extract_red)
        df['green'] = df['meta'].apply(extract_green)
        df['blue'] = df['meta'].apply(extract_blue)
        df['laplacian_variance_blur'] = df['meta'].apply(extract_laplacian_variance_blur)
        df['fourier_transform_blur'] = df['meta'].apply(extract_fourier_transform_blur)
        df['gradient_magnitude_blur'] = df['meta'].apply(extract_gradient_magnitude_blur)
        df['brightness_score'] = df['meta'].apply(extract_brightness_score)
        df['colourfulness'] = df['meta'].apply(extract_colourfulness)
        df['sharpness_score'] = df['meta'].apply(extract_sharpness_score)
        df['color'] = df['meta'].apply(extract_color)
        df['image_metric'] = df['meta'].apply(extract_image_metric)
        df['hash_value'] = df['meta'].apply(extract_hash_value)
        df['duplicate'] = df['meta'].apply(extract_duplicate)

        # Drop the original meta column
        df = df.drop(columns=['meta','keywords','score'], axis=1)
        df = df.drop(df.loc[df['image_name']=='0'].index)
        df.reset_index(inplace=True,drop=True)

        new = df.convert_dtypes(infer_objects=True,convert_string=True,convert_integer=True,convert_boolean=True)
        df = new

        return df


doc = input('Enter docid:')
docids = list(str(doc).split(","))
print(docids)

final_data = ImageLSH.Dataconversion(docids)
# df = dataframe 
df = final_data[['product_id','hash_value']]

# Extract hash values from the DataFrame
hash_values = df['hash_value'].tolist()

# Create an instance of ImageLSH with desired number of hash tables and threshold
lsh = ImageLSH(num_hash_tables=1, threshold=10)

# Index the hash values
lsh.index(hash_values)

# Create a list to store rows for the DataFrame
data = []

# Iterate through each row in the DataFrame to calculate similarities
for index, row in df.iterrows():
    query_hash = row['hash_value']
    
    # Calculate similarity with each hash value in the DataFrame
    similarity_scores = []
    for _, candidate_row in df.iterrows():
        candidate_hash = candidate_row['hash_value']
        distance = ImageLSH.hamming_distance(query_hash, candidate_hash)
        similarity = 1 - (distance / (len(query_hash) * 4))  # Similarity calculation
        similarity_scores.append((candidate_row['product_id'], similarity))
    
    # Filter and store results within the specified similarity score range
    filtered_similar_images = [(similar_image, similarity) for similar_image, similarity in similarity_scores if 0.75 <= similarity <= 0.99]
    
    if filtered_similar_images:
        similar_images = filtered_similar_images
    else:
        similar_images = [(None, None)]  # Store as blank if no similar images
        
    data.append({'Product_ID': row['product_id'], 'Hash': query_hash, 'Similar_Images': similar_images})

# Create a DataFrame from the list of rows
result_df = pd.DataFrame(data)

from PIL import Image
# import imagehash
import pandas as pd
from pymongo import MongoClient
from helper import getConfigData
import concurrent.futures
import time

class ImageLSH:
    def __init__(self, num_hash_tables):
        self.num_hash_tables = num_hash_tables
        self.hash_tables = [{} for _ in range(num_hash_tables)]
        
    def hash_function(self, image_hash):
        if not image_hash:
            return 0
        return int(str(image_hash), 16) % self.num_hash_tables
    
    def index(self, hash_values):
        for hash_value in hash_values:
            for table_idx in range(self.num_hash_tables):
                bucket = self.hash_function(hash_value)
                if bucket not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][bucket] = []
                self.hash_tables[table_idx][bucket].append(hash_value)

    def dhash_similarity(hash1, hash2):
        if not hash1 or not hash2:
            return 0.0 
        hamming_distance = bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
        max_distance = 64  # dHash is 64 bits
        similarity = 1 - (hamming_distance / max_distance)
        return similarity

    def ahash_similarity(hash1, hash2):
        if not hash1 or not hash2:
            return 0.0 
        hamming_distance = bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
        max_distance = 64  # AHash is 64 bits
        similarity = 1 - (hamming_distance / max_distance)
        return similarity
    
    def phash_similarity(hash1, hash2):
        if not hash1 or not hash2:
            return 0.0 
        hamming_distance = bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
        max_distance = 64  # AHash is 64 bits
        similarity = 1 - (hamming_distance / max_distance)
        return similarity

    def chash_similarity(hash1, hash2):
        if not hash1 or not hash2:
            return 0.0 
        hamming_distance = bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
        max_distance = 64  # AHash is 64 bits
        similarity = 1 - (hamming_distance / max_distance)
        return similarity

    def calculate_similarity(self, hash1, hash2, method):
        if method == 'dhash':
            return ImageLSH.dhash_similarity(hash1, hash2)
        elif method == 'ahash':
            return ImageLSH.ahash_similarity(hash1, hash2)
        elif method == 'phash':
            return ImageLSH.phash_similarity(hash1, hash2)
        elif method == 'chash':
            return ImageLSH.phash_similarity(hash1, hash2)
        else:
            raise ValueError("Invalid similarity calculation method")

    def data_conversion(docid, switch, content_status):
        if switch:
            docid = int(docid)
        else:
            docids = [item.lower() for item in docid]
        
        # Connect to MongoDB server
        username = getConfigData("mongo.username")
        password = getConfigData("mongo.password")
        auth_db = getConfigData("mongo.auth_db")

        with MongoClient(getConfigData('mongo.host'), username=username, password=password, authSource=auth_db) as client:
            # Access database
            db = client[getConfigData('mongo.db')]

            # Access collection based on contentstatus
            if content_status == 'category':
                collection = db[getConfigData('mongo.collection_category')]
                query = {"national_catid": docid}

            elif content_status == 'approved':
                collection = db[getConfigData('mongo.collection')]
                query = {
                    "$or": [
                        {"did": {"$in": docids}, "df": 1, "dlf": 0, "ap": {"$in": [0, 1, 2]}}
                    ]
                }
            else:
                # Handle invalid contentstatus
                return {"msg": "Invalid contentstatus", "error_code": 1}

            # Fetch data and create a DataFrame
            cursor = collection.find(query, {'_id': 0, 'pid': 1, 'meta': 1, 'id': 1}).batch_size(500)
            df = pd.DataFrame(list(cursor))

            if df.empty:
                output = {"msg": "Docid not found", "error_code": 1}
                return output

            # Extract "average_hash" values from the "derived" node
            df['average_hash'] = df['meta'].apply(lambda x: x.get('derived', {}).get('ahash', '0') if isinstance(x, dict) else '0')
            df['difference_hash'] = df['meta'].apply(lambda x: x.get('derived', {}).get('dhash', '0') if isinstance(x, dict) else '0')
            df['perceptual_hash'] = df['meta'].apply(lambda x: x.get('derived', {}).get('phash', '0') if isinstance(x, dict) else '0')
            df['color_hash'] = df['meta'].apply(lambda x: x.get('derived', {}).get('chash', '0') if isinstance(x, dict) else '0')

        new = df.convert_dtypes(infer_objects=True, convert_string=True, convert_integer=True, convert_boolean=True)

        # Use the correct column names based on the value of 'switch'
        df = new[['id', 'average_hash', 'difference_hash', 'perceptual_hash','color_hash']] if switch else new[['pid', 'average_hash', 'difference_hash', 'perceptual_hash','color_hash']]

        # Extract hash values from the DataFrame
        ahash_values = df['average_hash'].tolist()
        dhash_values = df['difference_hash'].tolist()
        phash_values = df['perceptual_hash'].tolist()
        chash_values = df['color_hash'].tolist()

        # Create an instance of ImageLSH with desired number of hash tables and threshold
        lsh = ImageLSH(num_hash_tables=1)

        # Index the hash values
        lsh.index(ahash_values)
        lsh.index(dhash_values)
        lsh.index(phash_values)
        lsh.index(chash_values)
        # Create a list to store rows for the DataFrame

        data = []

        # Iterate through each row in the DataFrame to calculate similarities
        for index, row in df.iterrows():
            if switch == True:
                row['pid'] = row['id']
            query_ahash = row['average_hash']
            query_dhash = row['difference_hash']
            query_phash = row['perceptual_hash']
            query_chash = row['color_hash']

            # Calculate similarity with each hash value in the DataFrame
            ahash_similarity_scores = []
            dhash_similarity_scores = []
            phash_similarity_scores = []
            chash_similarity_scores = []

            for _, candidate_row in df.iterrows():
                if switch == True:
                    if row['id'] == candidate_row['id']:
                        continue
                else:
                    if row['pid'] == candidate_row['pid']:
                        continue
                
                candidate_ahash = candidate_row['average_hash']
                candidate_dhash = candidate_row['difference_hash']
                candidate_phash = candidate_row['perceptual_hash']
                candidate_chash = candidate_row['color_hash']

                ahash_similarity = lsh.calculate_similarity(query_ahash, candidate_ahash, 'ahash')
                dhash_similarity = lsh.calculate_similarity(query_dhash, candidate_dhash, 'dhash')
                phash_similarity = lsh.calculate_similarity(query_phash, candidate_phash, 'phash')
                chash_similarity = lsh.calculate_similarity(query_chash, candidate_chash, 'chash')

                if switch == True:
                    ahash_similarity_scores.append((candidate_row['id'], ahash_similarity))
                    dhash_similarity_scores.append((candidate_row['id'], dhash_similarity))
                    phash_similarity_scores.append((candidate_row['id'], phash_similarity))
                    chash_similarity_scores.append((candidate_row['id'], chash_similarity))
                else:
                    ahash_similarity_scores.append((candidate_row['pid'], ahash_similarity))
                    dhash_similarity_scores.append((candidate_row['pid'], dhash_similarity))
                    phash_similarity_scores.append((candidate_row['pid'], phash_similarity))
                    chash_similarity_scores.append((candidate_row['pid'], chash_similarity))
                    

            # Filter and store results within the specified similarity score range (including 1.0)
            filtered_ahash_similar_images = [{'pid': str(pid), 'h_score': round(float(similarity),3), 'h_hash': query_ahash, 'hash_name': 'ahash'} for pid, similarity in ahash_similarity_scores if 0.65 <= float(similarity) <= 1.00]
            filtered_dhash_similar_images = [{'pid': str(pid), 'h_score': round(float(similarity),3), 'h_hash': query_dhash, 'hash_name': 'dhash'} for pid, similarity in dhash_similarity_scores if 0.65 <= float(similarity) <= 1.00]
            filtered_phash_similar_images = [{'pid': str(pid), 'h_score': round(float(similarity),3), 'h_hash': query_phash, 'hash_name': 'phash'} for pid, similarity in phash_similarity_scores if 0.65 <= float(similarity) <= 1.00]
            filtered_chash_similar_images = [{'pid': str(pid), 'h_score': round(float(similarity),3), 'h_hash': query_chash, 'hash_name': 'chash'} for pid, similarity in chash_similarity_scores if 0.95 <= float(similarity) <= 1.00]


            if not filtered_ahash_similar_images:
                filtered_ahash_similar_images = []  # Store as blank if no similar images
                
            if not filtered_dhash_similar_images:
                filtered_dhash_similar_images = []  # Store as blank if no similar images

            if not filtered_phash_similar_images:
                filtered_phash_similar_images = []

            if not filtered_chash_similar_images:
                filtered_chash_similar_images = []
                
            # Update the corresponding entry in doc_data with similarity information
            data.append({'pid': str(row['pid']), 'AHash': query_ahash, 'DHash': query_dhash, 'PHash': query_phash, 'CHash': query_chash, 'AHash Similarities': filtered_ahash_similar_images, 'DHash Similarities': filtered_dhash_similar_images, 'PHash Similarities': filtered_phash_similar_images, 'CHash Similarities': filtered_chash_similar_images})
            
        # Create a DataFrame from the list of rows
        result_df = pd.DataFrame(data)

        # Filter images based on the combined similarity (both AHash and DHash)
        result_df['AHash Similarity Count'] = result_df['AHash Similarities'].apply(len)
        result_df['DHash Similarity Count'] = result_df['DHash Similarities'].apply(len)
        result_df['PHash Similarity Count'] = result_df['PHash Similarities'].apply(len)
        result_df['CHash Similarity Count'] = result_df['CHash Similarities'].apply(len)

        # Initialize the 'h_data' column
        result_df['h_data'] = ''

        # Loop to determine which similarity values to store in 'h_data' based on your conditions
        for i in range(len(result_df)):
            ahash_sim_count = result_df['AHash Similarity Count'].values[i]
            dhash_sim_count = result_df['DHash Similarity Count'].values[i]
            phash_sim_count = result_df['PHash Similarity Count'].values[i]
            chash_sim_count = result_df['CHash Similarity Count'].values[i]
            
            if phash_sim_count < ahash_sim_count:
                result_df.at[i, 'h_data'] = result_df['PHash Similarities'].values[i]
            elif dhash_sim_count == ahash_sim_count:
                result_df.at[i, 'h_data'] = result_df['AHash Similarities'].values[i]
            elif dhash_sim_count < ahash_sim_count:
                result_df.at[i, 'h_data'] = result_df['DHash Similarities'].values[i]
            elif chash_sim_count < ahash_sim_count:
                result_df.at[i, 'h_data'] = result_df['CHash Similarities'].values[i]
            else:
                result_df.at[i, 'h_data'] = result_df['AHash Similarities'].values[i]

        similar_images = result_df[['pid', 'h_data']]
        
        # Remove blank values from 'h_data' column
        
        similar_images['pid'] = similar_images['pid'].astype(str)
        similar_images.loc[:, 'h_data'] = similar_images['h_data'].apply(lambda x: x if isinstance(x, list) else [])

        # Filter rows with non-empty 'h_data' column
        similar_images = similar_images[similar_images['h_data'].apply(len) > 0]

        return similar_images.to_json(orient='records')


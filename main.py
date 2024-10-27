# ==========================================================
# Project: Dataset-label
# Author: Harrison E. Muchnic
# License: Apache License 2.0
#
# Copyright (c) [2024] Harrison E. Muchnic
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Attribution: If you use or modify this code, you must provide proper
# attribution to Harrison E. Muchnic as the original author.
#
# This project uses the CLIP server provided by [Jina AI](https://github.com/jina-ai/clip-as-service) 
# through a Docker container, which is licensed under the **Apache License 2.0**.
#
# Portions of the CLIP server model, specifically `model.py` and `simple_tokenizer.py`, 
# are licensed under the **MIT License** via [OpenCLIP](https://github.com/mlfoundations/open_clip).
#
# For more details on the license terms, please refer to:
# - [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
# - [MIT License](https://opensource.org/licenses/MIT)
#
# ==========================================================

#NOTE: it processes faster to use optimized_main.py instead of main.py even if your just using CPU. But if your using CPU, you probably dont care too much about processing time.
import os
import argparse
from clip_client import Client
from docarray import DocumentArray, Document
import zipfile
import csv
import tempfile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Default image size for CLIP model
CLIP_IMAGE_SIZE = (224, 224)

client = Client('grpc://0.0.0.0:51009')  # Create a client to the Docker container running the CLIP model

def main():
    parser = argparse.ArgumentParser(description='Process files with CLIP.')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file (zip or csv).')
    parser.add_argument('--categories', type=str, required=True, help='Comma-separated list of categories.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing data.')
    parser.add_argument('--stats', action='store_true', help='Generate statistics file.')
    args = parser.parse_args()

    file_path = args.file
    categories = args.categories.split(',')
    batch_size = args.batch_size
    stats = args.stats

    # Process data
    process_data(file_path, categories, client, batch_size, stats)

def process_data(file_path, categories, client, batch_size, stats=False):
    try:
        # Load categories
        category_docs = DocumentArray([Document(text=cat.strip()) for cat in categories])
        category_embeddings = client.encode(category_docs)
        category_embedding = category_embedding / np.linalg.norm(category_embedding) #forgot to normalize :/

        # Prepare to write results incrementally
        output_file = 'results.csv'
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['file', 'label'])
            writer.writeheader()

        # Initialize stats data if needed
        stats_data = {} if stats else None
        total_samples = 0  # To keep track of total number of samples processed

        # Process file
        if file_path.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract all files
                with zipfile.ZipFile(file_path) as zip_ref:
                    zip_ref.extractall(tmpdir)
                # Get list of image files with supported extensions
                image_paths = []
                for root, _, files in os.walk(tmpdir):
                    for name in files:
                        if name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for supported file types
                            image_paths.append(os.path.join(root, name))

                total = len(image_paths)
                for i in range(0, total, batch_size):
                    batch = image_paths[i:i+batch_size]
                    num_processed = process_batch(
                        batch, categories, category_embeddings, client, output_file, i, total, stats_data, is_text=False
                    )
                    total_samples += num_processed

        elif file_path.endswith('.csv'):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                texts = [row[0] for row in reader]

            total = len(texts)
            for i in range(0, total, batch_size):
                batch = texts[i:i+batch_size]
                num_processed = process_batch(
                    batch, categories, category_embeddings, client, output_file, i, total, stats_data, is_text=True
                )
                total_samples += num_processed

        else:
            print("Error: Unsupported file type.")
            return

        # Compute and write stats if required
        if stats and stats_data is not None:
            compute_and_write_stats(stats_data, total_samples, output_file='stats.csv')

    except Exception as e:
        print(f"Error: {e}")

def process_batch(
    batch, categories, category_embeddings, client, output_file, start_idx, total, stats_data=None, is_text=False
):
    try:
        embeddings = []
        if is_text:
            # Process text data
            for idx, text in enumerate(batch):
                doc = Document(text=text)
                embedding = client.encode([doc])[0].embedding
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append((text, embedding))
                # Print progress
                percentage = int(((start_idx + idx + 1) / total) * 100)
                print(f"Processing text {start_idx + idx + 1}/{total}: {text} ({percentage}%)")
        else:
            # Use ThreadPoolExecutor for parallel image loading and resizing
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, img_path) for img_path in batch]
                for idx, future in enumerate(futures):
                    result = future.result()
                    if result:
                        img_path, embedding = result
                        embeddings.append((img_path, embedding))
                        # Print progress
                        percentage = int(((start_idx + idx + 1) / total) * 100)
                        print(f"Processing image {start_idx + idx + 1}/{total}: {os.path.basename(img_path)} ({percentage}%)")
                    else:
                        # Failed to process image
                        pass

        # Assign labels and write to CSV
        results = []
        for uri, embedding in embeddings:
            scores = (embedding @ category_embeddings.embeddings.T).flatten()
            best_idx = scores.argmax()
            label = categories[best_idx]
            results.append({'file': uri, 'label': label} if not is_text else {'text': uri, 'label': label})

            # Collect stats if stats_data is not None
            if stats_data is not None:
                if label not in stats_data:
                    # Initialize stats_data for this label
                    stats_data[label] = {
                        'count': 0,
                        'dot_product_sum': 0.0,
                        'dot_product_sq_sum': 0.0,
                        'competitor_scores': {}
                    }

                # Update assigned category stats
                stats_data[label]['count'] += 1
                assigned_score = scores[best_idx]
                stats_data[label]['dot_product_sum'] += assigned_score
                stats_data[label]['dot_product_sq_sum'] += assigned_score ** 2

                # Update competitor scores
                for i, score in enumerate(scores):
                    if i != best_idx:
                        competitor_label = categories[i]
                        if competitor_label not in stats_data[label]['competitor_scores']:
                            stats_data[label]['competitor_scores'][competitor_label] = {
                                'sum': 0.0,
                                'count': 0
                            }
                        stats_data[label]['competitor_scores'][competitor_label]['sum'] += score
                        stats_data[label]['competitor_scores'][competitor_label]['count'] += 1

        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['file', 'label'] if not is_text else ['text', 'label'])
            writer.writerows(results)

        return len(embeddings)

    except Exception as e:
        print(f"Error processing batch: {e}")
        return 0

def process_image(img_path):
    try:
        # Open and resize the image to 224x224 for CLIP using Lanczos resampling
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # Ensure image is RGB
            img = img.resize(CLIP_IMAGE_SIZE, Image.LANCZOS)  # Use Lanczos for high-quality downscaling

        doc = Document(uri=img_path)
        doc.tensor = img  # Directly set the tensor with resized image
        embedding = client.encode([doc])[0].embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return img_path, embedding
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def compute_and_write_stats(stats_data, total_samples, output_file='stats.csv'):
    import csv
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['category', 'fraction', 'average_dot_product', 'variance', 'second_best_category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for label, data in stats_data.items():
            count = data['count']
            fraction = count / total_samples if total_samples > 0 else 0
            dot_product_sum = data['dot_product_sum']
            dot_product_sq_sum = data['dot_product_sq_sum']
            mean = dot_product_sum / count if count > 0 else 0
            variance = (dot_product_sq_sum - (dot_product_sum ** 2) / count) / count if count > 0 else 0

            # Compute average competitor scores
            competitor_scores = data['competitor_scores']
            competitor_averages = {}
            for comp_label, comp_data in competitor_scores.items():
                comp_sum = comp_data['sum']
                comp_count = comp_data['count']
                comp_mean = comp_sum / comp_count if comp_count > 0 else 0
                competitor_averages[comp_label] = comp_mean

            # Find the competitor category with highest average dot product
            if competitor_averages:
                second_best_category = max(competitor_averages.items(), key=lambda x: x[1])[0]
            else:
                second_best_category = None

            writer.writerow({
                'category': label,
                'fraction': fraction,
                'average_dot_product': mean,
                'variance': variance,
                'second_best_category': second_best_category
            })

if __name__ == "__main__":
    main()

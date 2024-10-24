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

import os
import argparse
from clip_client import Client
from docarray import DocumentArray, Document
import zipfile
import csv
import tempfile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Default image size for CLIP model
CLIP_IMAGE_SIZE = (224, 224)

client = Client('grpc://0.0.0.0:51009') #Createa client to docker container that is running clip model

def main():
    parser = argparse.ArgumentParser(description='Process files with CLIP.')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file (zip or csv).')
    parser.add_argument('--categories', type=str, required=True, help='Comma-separated list of categories.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing data.')
    args = parser.parse_args()
    
    file_path = args.file
    categories = args.categories.split(',')
    batch_size = args.batch_size

    # Process data
    process_data(file_path, categories, client, batch_size)

def process_data(file_path, categories, client, batch_size):
    try:
        # Load categories
        category_docs = DocumentArray([Document(text=cat.strip()) for cat in categories])
        category_embeddings = client.encode(category_docs)

        # Prepare to write results incrementally
        output_file = 'results.csv'
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['file', 'label'])
            writer.writeheader()

        # Process file
        if file_path.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract images
                with zipfile.ZipFile(file_path) as zip_ref:
                    zip_ref.extractall(tmpdir)
                # Get list of image files
                image_paths = []
                for root, _, files in os.walk(tmpdir):
                    for name in files:
                        if name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for supported file types
                            image_paths.append(os.path.join(root, name))

                total = len(image_paths)
                for i in range(0, total, batch_size):
                    batch = image_paths[i:i+batch_size]
                    process_batch(batch, categories, category_embeddings, client, output_file, i, total)

        elif file_path.endswith('.csv'):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                texts = [row[0] for row in reader]

            total = len(texts)
            for i in range(0, total, batch_size):
                batch = texts[i:i+batch_size]
                process_batch(batch, categories, category_embeddings, client, output_file, i, total, is_text=True)

        else:
            print("Error: Unsupported file type.")
    except Exception as e:
        print(f"Error: {e}")

def process_batch(batch, categories, category_embeddings, client, output_file, start_idx, total, is_text=False):
    try:
        embeddings = []
        if is_text:
            # Process text data
            for idx, text in enumerate(batch):
                doc = Document(text=text)
                embedding = client.encode([doc])[0].embedding
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

        # Assign labels and write to CSV
        results = []
        for uri, embedding in embeddings:
            scores = (embedding @ category_embeddings.embeddings.T).flatten()
            best_idx = scores.argmax()
            label = categories[best_idx]
            results.append({'file': uri, 'label': label} if not is_text else {'text': uri, 'label': label})

        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['file', 'label'] if not is_text else ['text', 'label'])
            writer.writerows(results)

    except Exception as e:
        print(f"Error processing batch: {e}")

def process_image(img_path):
    try:
        # Open and resize the image to 224x224 for CLIP using Lanczos resampling
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # Ensure image is RGB
            img = img.resize(CLIP_IMAGE_SIZE, Image.LANCZOS)  # Use Lanczos for high-quality downscaling

        doc = Document(uri=img_path)
        doc.tensor = img  # Directly set the tensor with resized image
        embedding = client.encode([doc])[0].embedding
        return img_path, embedding
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

if __name__ == "__main__":
    main()

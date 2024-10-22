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
#This project uses the CLIP server provided by [Jina AI](https://github.com/jina-ai/clip-as-service) through a Docker container, which is licensed under the **Apache License 2.0**.
#
#Portions of the CLIP server model, specifically `model.py` and `simple_tokenizer.py`, are licensed under the **MIT License** via [OpenCLIP](https://github.com/mlfoundations/open_clip).
#
#For more details on the license terms, please refer to:
#- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
#- [MIT License](https://opensource.org/licenses/MIT)
#
# ==========================================================


import os
import argparse
from clip_client import Client
from docarray import DocumentArray, Document
import zipfile
import csv
import tempfile

def main():
    parser = argparse.ArgumentParser(description='Process files with CLIP.')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file (zip or csv).')
    parser.add_argument('--categories', type=str, required=True, help='Comma-separated list of categories.')
    args = parser.parse_args()
    
    file_path = args.file
    categories = args.categories.split(',')

    # Create a client to the docker container
    client = Client('grpc://0.0.0.0:51009')  # Adjust the address if needed

    # Process data
    process_data(file_path, categories, client)

def process_data(file_path, categories, client):
    try:
        # Load categories
        category_docs = DocumentArray([Document(text=cat.strip()) for cat in categories])
        category_embeddings = client.encode(category_docs)

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
                        image_paths.append(os.path.join(root, name))
                total = len(image_paths)
                embeddings = []
                for idx, img_path in enumerate(image_paths):
                    doc = Document(uri=img_path)
                    doc.load_uri_to_image_tensor()
                    embedding = client.encode([doc])[0].embedding
                    embeddings.append((doc.uri, embedding))
                    # Print progress
                    percentage = int(((idx + 1) / total) * 100)
                    print(f"Processing image {idx + 1}/{total}: {os.path.basename(img_path)} ({percentage}%)")
                # Assign labels
                results = []
                for uri, embedding in embeddings:
                    scores = (embedding @ category_embeddings.embeddings.T).flatten()
                    best_idx = scores.argmax()
                    label = categories[best_idx]
                    results.append({'file': uri, 'label': label})
                # Save results
                output_file = 'results.csv'
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['file', 'label'])
                    writer.writeheader()
                    writer.writerows(results)
                print(f"Results saved to {output_file}")
        elif file_path.endswith('.csv'):
            # Process text data
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                texts = [row[0] for row in reader]
            total = len(texts)
            embeddings = []
            for idx, text in enumerate(texts):
                doc = Document(text=text)
                embedding = client.encode([doc])[0].embedding
                embeddings.append((text, embedding))
                # Print progress
                percentage = int(((idx + 1) / total) * 100)
                print(f"Processing text {idx + 1}/{total}: {text} ({percentage}%)")
            # Assign labels
            results = []
            for text, embedding in embeddings:
                scores = (embedding @ category_embeddings.embeddings.T).flatten()
                best_idx = scores.argmax()
                label = categories[best_idx]
                results.append({'text': text, 'label': label})
            # Save results
            output_file = 'results.csv'
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['text', 'label'])
                writer.writeheader()
                writer.writerows(results)
            print(f"Results saved to {output_file}")
        else:
            print("Error: Unsupported file type.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

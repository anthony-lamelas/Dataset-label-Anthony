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
import zipfile
import csv
import tempfile
import numpy as np
from clip_client import Client
from docarray import DocumentArray, Document
import asyncio

# Initialize the CLIP client. If you are insane and CPU or network becomes bottleneck, and other finetuning is not enough, make multiple client objects 
client = Client('grpc://0.0.0.0:51009')

async def main():
    parser = argparse.ArgumentParser(description='Process files with CLIP.')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file (zip or csv).')
    parser.add_argument('--categories', type=str, required=True, help='Comma-separated list of categories.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for processing data.')
    parser.add_argument('--prefetch', type=int, default=10, help='Number of prefetch batches.')
    parser.add_argument('--stats', action='store_true', help='Generate statistics file.')
    args = parser.parse_args()

    file_path = args.file
    categories = [cat.strip() for cat in args.categories.split(',')]
    batch_size = args.batch_size
    prefetch = args.prefetch
    stats = args.stats

    # Process data
    await process_data(file_path, categories, client, batch_size, prefetch, stats)

async def process_data(file_path, categories, client, batch_size, prefetch, stats=False):
    try:
        # Load and encode category embeddings
        category_docs = [cat.strip() for cat in categories]
        category_embeddings = await client.aencode(category_docs, batch_size=batch_size, prefetch=prefetch)
        category_embeddings = np.array([emb / np.linalg.norm(emb) for emb in category_embeddings])

        # Prepare to write results incrementally
        output_file = 'results.csv'
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['file' if file_path.endswith('.zip') else 'text', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Initialize stats data if needed
        stats_data = {} if stats else None
        total_samples = 0  # To keep track of total number of samples processed

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
                        if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_paths.append(os.path.join(root, name))

                total = len(image_paths)
                if total == 0:
                    print("No supported image files found in the zip archive.")
                    return

                # Process images asynchronously
                await process_items(
                    items=image_paths,
                    categories=categories,
                    category_embeddings=category_embeddings,
                    client=client,
                    output_file=output_file,
                    batch_size=batch_size,
                    prefetch=prefetch,
                    stats_data=stats_data,
                    total_samples=total_samples,
                    total=total,
                    is_text=False
                )

        elif file_path.endswith('.csv'):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                texts = [row[0] for row in reader]

            total = len(texts)
            if total == 0:
                print("No texts found in the CSV file.")
                return

            # Process texts asynchronously
            await process_items(
                items=texts,
                categories=categories,
                category_embeddings=category_embeddings,
                client=client,
                output_file=output_file,
                batch_size=batch_size,
                prefetch=prefetch,
                stats_data=stats_data,
                total_samples=total_samples,
                total=total,
                is_text=True
            )
        else:
            print("Error: Unsupported file type.")
            return

        # Compute and write stats if required
        if stats and stats_data is not None:
            compute_and_write_stats(stats_data, total, output_file='stats.csv')

    except Exception as e:
        print(f"Error: {e}")

async def process_items(
    items, categories, category_embeddings, client, output_file, batch_size, prefetch,
    stats_data=None, total_samples=0, total=0, is_text=False
):
    try:
        # Use async generator to process items
        num_batches = (len(items) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch_items = items[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            # Encode batch items
            embeddings = await client.aencode(
                batch_items,
                batch_size=batch_size,
                prefetch=prefetch
            )
            # Normalize embeddings
            embeddings = np.array([emb / np.linalg.norm(emb) for emb in embeddings])

            # Write embeddings to tsv file for PCA
            with open("embeddings.tsv", "a", newline="") as embfile:
                emb_writer = csv.writer(embfile, delimiter="\t")
                for idx, emb_vec in enumerate(embeddings):
                    row = [batch_items[idx]] + emb_vec.tolist()
                    emb_writer.writerow(row)

            # Assign labels and collect stats
            results = []
            for idx, embedding in enumerate(embeddings):
                uri = batch_items[idx]
                scores = np.dot(embedding, category_embeddings.T)
                best_idx = scores.argmax()
                label = categories[best_idx]
                result = {'file' if not is_text else 'text': uri, 'label': label}
                results.append(result)

                # Collect stats if stats_data is not None
                if stats_data is not None:
                    assigned_score = scores[best_idx]
                    if label not in stats_data:
                        stats_data[label] = {
                            'count': 0,
                            'dot_product_sum': 0.0,
                            'dot_product_sq_sum': 0.0,
                            'competitor_scores': {}
                        }

                    stats_data[label]['count'] += 1
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

            # Write results to CSV
            with open(output_file, 'a', newline='') as csvfile:
                fieldnames = ['file' if not is_text else 'text', 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(results)

            # Update total_samples
            total_samples += len(embeddings)

            # Print progress
            processed = min((batch_idx + 1) * batch_size, total)
            percentage = int((processed / total) * 100)
            print(f"Processed {processed}/{total} items ({percentage}%)")

    except Exception as e:
        print(f"Error processing items: {e}")

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
    asyncio.run(main())

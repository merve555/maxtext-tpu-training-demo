#!/usr/bin/env python3
"""
Data preparation script for Gemma-2-27B fine-tuning demo.
This script downloads and prepares the Alpaca dataset for instruction tuning.
"""

import os
import json
import argparse
from datasets import load_dataset
from google.cloud import storage

def prepare_alpaca_dataset(output_dir: str, num_samples: int = 1000):
    """
    Prepare Alpaca instruction-following dataset.
    This dataset is well-tested and requires minimal processing.
    """
    print(f"Loading Alpaca dataset with {num_samples} examples...")
    
    # Load the Stanford Alpaca dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Format data for MaxText training
    formatted_data = []
    for example in dataset:
        # Use Alpaca's standard format
        text = f"### Instruction:\n{example['instruction']}\n"
        if example['input']:
            text += f"### Input:\n{example['input']}\n"
        text += f"### Response:\n{example['output']}"
        
        formatted_data.append({"text": text})
    
    # Save to JSONL format
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "train_data.jsonl")
    
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(formatted_data)} examples to {output_file}")
    print(f"Dataset info: Stanford Alpaca - Good for instruction tuning and chat interactions")
    return output_file

def upload_to_gcs(local_file: str, bucket_name: str, gcs_path: str):
    """Upload file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    print(f"Uploading {local_file} to gs://{bucket_name}/{gcs_path}")
    blob.upload_from_filename(local_file)
    print("Upload completed!")

def main():
    parser = argparse.ArgumentParser(description="Prepare Alpaca dataset for Gemma-2-27B fine-tuning")
    parser.add_argument("--output_dir", default="/tmp/dataset", help="Local output directory")
    parser.add_argument("--gcs_bucket", required=True, help="GCS bucket name")
    parser.add_argument("--gcs_path", default="datasets/train_data.jsonl", help="GCS path")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to prepare")
    
    args = parser.parse_args()
    
    print(f"Preparing Alpaca dataset for training:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Output: {args.output_dir}")
    print(f"  GCS Path: gs://{args.gcs_bucket}/{args.gcs_path}")
    print()
    
    # Prepare Alpaca dataset
    output_file = prepare_alpaca_dataset(args.output_dir, args.num_samples)
    
    # Upload to GCS
    upload_to_gcs(output_file, args.gcs_bucket, args.gcs_path)

if __name__ == "__main__":
    main()

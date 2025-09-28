#!/usr/bin/env python3
"""
Data preparation script for Gemma-2-27B fine-tuning demo with ArrayRecord format.
This script downloads and prepares the Alpaca dataset in ArrayRecord format for Grain.
"""

import os
import json
import argparse
import tensorflow as tf
from datasets import load_dataset
from google.cloud import storage
from transformers import AutoTokenizer

def _create_tf_example(features):
    """Convert features to TensorFlow Example protobuf."""
    tf_features = {}
    for key, value in features.items():
        # Convert TensorFlow tensors to numpy arrays first, then to lists
        if hasattr(value, 'numpy'):
            value = value.numpy().flatten().tolist()
        elif isinstance(value, (list, tuple)):
            value = list(value)
        elif isinstance(value, int):
            value = [value]
        else:
            # Skip non-integer values or convert them appropriately
            continue
            
        # Ensure all values are integers
        try:
            value = [int(v) for v in value]
            tf_features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        except (ValueError, TypeError):
            # Skip values that can't be converted to integers
            continue
            
    return tf.train.Example(features=tf.train.Features(feature=tf_features))

def prepare_alpaca_arrayrecord(output_dir: str, num_samples: int = 500, tokenizer_name: str = "google/gemma-2-27b"):
    """
    Prepare Alpaca instruction-following dataset in ArrayRecord format for Grain.
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    # Get HuggingFace token from environment
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is required")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Alpaca dataset with {num_samples} examples...")
    
    # Load the Stanford Alpaca dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Tokenization and formatting logic for SFT
    def tokenize_and_format(element):
        # Format the text using the Alpaca template
        text = f"### Instruction:\n{element['instruction']}\n"
        if element['input']:
            text += f"### Input:\n{element['input']}\n"
        text += f"### Response:\n{element['output']}"
        
        # Tokenize the full text
        tokenized_full = tokenizer(text, return_tensors="tf", padding='max_length', truncation=True, max_length=2048)
        
        # Tokenize only the instruction part to find its length
        instruction_part = f"### Instruction:\n{element['instruction']}\n"
        if element['input']:
            instruction_part += f"### Input:\n{element['input']}\n"
        instruction_part += f"### Response:\n"
        tokenized_instruction = tokenizer(instruction_part, return_tensors="tf")
        instruction_length = tf.shape(tokenized_instruction['input_ids'])[1]

        # Create target masks: -1 for instruction, token_id for response
        labels = tf.identity(tokenized_full['input_ids'])
        # We need to cast labels to a mutable type to modify it
        labels_numpy = labels.numpy()
        labels_numpy[0, :instruction_length] = -1
        labels = tf.convert_to_tensor(labels_numpy)

        return {
            'inputs': tokenized_full['input_ids'][0], 
            'targets': labels[0],
            'inputs_segmentation': tokenized_full['attention_mask'][0],
            'targets_segmentation': tokenized_full['attention_mask'][0]
        }

    print("Tokenizing and formatting dataset...")
    tokenized_dataset = dataset.map(tokenize_and_format, batched=False)

    # Save to ArrayRecord format
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "train_data.array_record")
    
    print(f"Writing {len(tokenized_dataset)} records to {output_file}...")
    # Use TFRecord writer (Grain can read TFRecord format)
    with tf.io.TFRecordWriter(output_file) as writer:
        for i, features in enumerate(tokenized_dataset):
            try:
                example = _create_tf_example(features)
                if example:  # Only write if example was created successfully
                    writer.write(example.SerializeToString())
                if i % 50 == 0:
                    print(f"Processed {i+1}/{len(tokenized_dataset)} records...")
            except Exception as e:
                print(f"Warning: Skipping record {i} due to error: {e}")
                continue
            
    print(f"Saved {len(tokenized_dataset)} examples to {output_file}")
    print(f"Dataset info: Stanford Alpaca - ArrayRecord format for Grain")
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
    parser = argparse.ArgumentParser(description="Prepare Alpaca dataset in ArrayRecord format for Gemma-2-27B fine-tuning")
    parser.add_argument("--output_dir", default="/tmp/dataset", help="Local output directory")
    parser.add_argument("--gcs_bucket", required=True, help="GCS bucket name")
    parser.add_argument("--gcs_path", default="datasets/train_data.array_record", help="GCS path")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to prepare")
    parser.add_argument("--tokenizer_name", default="google/gemma-2-27b", help="Tokenizer to use")
    
    args = parser.parse_args()
    
    print(f"Preparing Alpaca dataset in ArrayRecord format:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Output: {args.output_dir}")
    print(f"  GCS Path: gs://{args.gcs_bucket}/{args.gcs_path}")
    print(f"  Tokenizer: {args.tokenizer_name}")
    print()
    
    # Prepare Alpaca dataset in ArrayRecord format
    output_file = prepare_alpaca_arrayrecord(args.output_dir, args.num_samples, args.tokenizer_name)
    
    # Upload to GCS
    upload_to_gcs(output_file, args.gcs_bucket, args.gcs_path)

if __name__ == "__main__":
    main()

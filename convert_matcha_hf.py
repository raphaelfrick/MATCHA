#!/usr/bin/env python3
"""
Conversion script for MATCH-A dataset.

This script downloads the MATCH-A dataset from HuggingFace and converts it to the target format used in the framework.

Usage:
    python convert_matcha_hf.py --output_dir ./output
"""

import argparse
import hashlib
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MATCH-A Dataset from HuggingFace to target format."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./local_matcha_dataset",
        help="Output directory for the converted dataset"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)"
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Download the full HuggingFace dataset without processing (optimization to avoid streaming overhead)"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="rfsit/MATCH-A",
        help="HuggingFace dataset name (default: rfsit/MATCH-A)"
    )
    return parser.parse_args()


# Global variables
args = None
output_dir = None
_existing_files = None
_existing_files_lock = Lock()


def init_existing_files_cache():
    """Initialize the cache of existing files."""
    global _existing_files
    _existing_files = set()
    if output_dir.exists():
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                rel_path = Path(root) / f
                _existing_files.add(str(rel_path.relative_to(output_dir)))


def should_save_image(output_path: Path) -> bool:
    """Check if image should be saved."""
    global _existing_files
    with _existing_files_lock:
        if _existing_files is not None:
            rel_path = str(output_path.relative_to(output_dir))
            if rel_path in _existing_files:
                return False
    return True


def mark_file_saved(output_path: Path):
    """Mark a file as saved in the cache."""
    global _existing_files
    with _existing_files_lock:
        if _existing_files is not None:
            _existing_files.add(str(output_path.relative_to(output_dir)))


def save_image_pil(image_pil: Image.Image, output_path: Path) -> bool:
    """Save PIL image to file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save without optimize for speed
        image_pil.save(output_path, quality=95)
        mark_file_saved(output_path)
        return True
    except Exception:
        return False


def copy_image(src_path: Path, output_path: Path) -> bool:
    """Copy an image file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(src_path, 'rb') as src, open(output_path, 'wb') as dst:
            dst.write(src.read())
        mark_file_saved(output_path)
        return True
    except Exception:
        return False


def process_files_parallel(files: List, process_func, num_workers: int) -> List:
    """Process files in parallel using ThreadPoolExecutor."""
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_func, f): f for f in files}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception:
                pass
    
    return results


def process_hf_gallery_image(args_tuple: Tuple[int, Dict, Path]) -> List[Dict]:
    """Process a single gallery image from HuggingFace dataset.
    
    Args:
        args_tuple: Tuple of (idx, item, output_dir)
    
    Returns:
        List of dicts, one for each split the image belongs to (train, val, test)
    """
    idx, item, output_dir = args_tuple
    try:
        image = item['image']
        image_pil = None
        
        # Handle different HuggingFace image formats
        if isinstance(image, Image.Image):
            # Already a PIL Image
            image_pil = image
        elif isinstance(image, dict):
            # Dict format with 'bytes' or 'path' key
            if 'bytes' in image:
                image_pil = Image.open(io.BytesIO(image['bytes']))
            elif 'path' in image and image['path']:
                image_pil = Image.open(image['path'])
        elif hasattr(image, 'path') and image.path:
            # Lazy loading object with path attribute
            image_pil = Image.open(image.path)
        elif isinstance(image, bytes):
            # Raw bytes
            image_pil = Image.open(io.BytesIO(image))
        elif isinstance(image, str) and image:
            # String path
            image_pil = Image.open(image)
        
        if image_pil is not None:
            # Extract original filename from item, or generate hash-based filename
            filename = None
            for field in ['filename', 'image_id', 'id', 'name']:
                if field in item and item[field]:
                    filename = str(item[field])
                    break
            
            if filename:
                # Use original filename, ensure it starts with 'I_'
                if not filename.startswith('I_'):
                    filename = f"I_{filename}" if not filename.startswith('I_') else filename
            else:
                # Generate hash-based filename from image content
                img_bytes = io.BytesIO()
                image_pil.save(img_bytes, format='PNG')
                img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()
                filename = f"I_{img_hash}.png"
            
            # Ensure correct extension for gallery (png)
            if not filename.endswith('.png'):
                filename = filename.rsplit('.', 1)[0] + '.png'
            
            output_path = output_dir / "reference_db" / filename
            
            # Check if file already exists
            if not should_save_image(output_path):
                # Determine which splits this image belongs to
                splits = []
                if item.get('train', False):
                    splits.append('train')
                if item.get('val', False):
                    splits.append('val')
                if item.get('test', False):
                    splits.append('test')
                
                # If no split specified, default to train
                if not splits:
                    splits = ['train']
                
                # Return one entry per split
                return [
                    {
                        'image_path': f"reference_db/{filename}",
                        'split': split
                    }
                    for split in splits
                ]
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image_pil.save(output_path, quality=95)
            mark_file_saved(output_path)
            
            # Determine which splits this image belongs to
            splits = []
            if item.get('train', False):
                splits.append('train')
            if item.get('val', False):
                splits.append('val')
            if item.get('test', False):
                splits.append('test')
            
            # If no split specified, default to train
            if not splits:
                splits = ['train']
            
            # Return one entry per split
            return [
                {
                    'image_path': f"reference_db/{filename}",
                    'split': split
                }
                for split in splits
            ]
    except Exception as e:
        pass  # Silently skip failed images
    return []


def process_hf_query_image(args_tuple: Tuple[int, Dict, str, Path]) -> Optional[Dict]:
    """Process a single query image from HuggingFace dataset.
    
    Args:
        args_tuple: Tuple of (idx, item, split, output_dir)
    """
    idx, item, split, output_dir = args_tuple
    try:
        image = item['query']
        image_pil = None
        
        # Handle different HuggingFace image formats
        if isinstance(image, Image.Image):
            # Already a PIL Image
            image_pil = image
        elif isinstance(image, dict):
            # Dict format with 'bytes' or 'path' key
            if 'bytes' in image:
                image_pil = Image.open(io.BytesIO(image['bytes']))
            elif 'path' in image and image['path']:
                image_pil = Image.open(image['path'])
        elif hasattr(image, 'path') and image.path:
            # Lazy loading object with path attribute
            image_pil = Image.open(image.path)
        elif isinstance(image, bytes):
            # Raw bytes
            image_pil = Image.open(io.BytesIO(image))
        elif isinstance(image, str) and image:
            # String path
            image_pil = Image.open(image)
        
        if image_pil is not None:
            # Extract original filename from item, or generate hash-based filename
            # Try to get filename from various possible fields
            filename = None
            for field in ['filename', 'image_id', 'id', 'name']:
                if field in item and item[field]:
                    filename = str(item[field])
                    break
            
            if filename:
                # Use original filename, ensure it starts with 'I_' and has correct extension
                if not filename.startswith('I_'):
                    filename = f"I_{filename}" if not filename.startswith('I_') else filename
            else:
                # Generate hash-based filename from image content
                img_bytes = io.BytesIO()
                image_pil.save(img_bytes, format='PNG')
                img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()
                filename = f"I_{img_hash}.jpg"
            
            # Ensure correct extension for query (jpg)
            if not filename.endswith('.jpg'):
                filename = filename.rsplit('.', 1)[0] + '.jpg'
            
            output_path = output_dir / "queries" / split / filename
            
            # Check if file already exists
            if not should_save_image(output_path):
                positive_id = item.get('positive_id', '')
                positive_rel_path = f"reference_db/{positive_id}.png" if positive_id else ""
                return {
                    'query_path': f"queries/{split}/{filename}",
                    'positive_path': positive_rel_path,
                    'has_positive': 1 if positive_id else 0,
                    'query_transforms': item.get('query_transforms', '') or '',
                    'quality': item.get('quality', 0) or 0,
                    'split': split
                }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image_pil.save(output_path, quality=95)
            mark_file_saved(output_path)
            
            positive_id = item.get('positive_id', '')
            positive_rel_path = f"reference_db/{positive_id}.png" if positive_id else ""
            
            return {
                'query_path': f"queries/{split}/{filename}",
                'positive_path': positive_rel_path,
                'has_positive': 1 if positive_id else 0,
                'query_transforms': item.get('query_transforms', '') or '',
                'quality': item.get('quality', 0) or 0,
                'split': split
            }
    except Exception as e:
        pass  # Silently skip failed images
    return None


def write_csv(data: List[Dict], output_path: Path, columns: List[str]) -> int:
    """Write data to CSV file."""
    if not data:
        return 0
    
    df = pd.DataFrame(data)
    df = df[columns]
    df.to_csv(output_path, index=False)
    return len(df)


def main():
    """Main conversion function."""
    global args, output_dir
    args = parse_args()
    output_dir = Path(args.output_dir)
    num_workers = args.num_workers
    
    # Handle download_only mode - download HuggingFace dataset without processing
    if args.download_only:
        print("=" * 60)
        print("Download Only Mode")
        print("=" * 60)
        print(f"Downloading HuggingFace dataset: {args.hf_dataset}")
        print(f"Output directory: {output_dir}")
        
        try:
            from datasets import load_dataset
            
            # Download the full dataset (not streaming)
            print("Downloading dataset (this may take a while)...")
            ds = load_dataset(
                args.hf_dataset,
                num_proc=num_workers
            )
            
            print("\nDataset downloaded successfully!")
            print(f"Dataset splits: {list(ds.keys())}")
            for split in ds.keys():
                print(f"  - {split}: {len(ds[split])} samples")
            
            # Save dataset info
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "download_info.txt", "w") as f:
                f.write(f"HuggingFace Dataset: {args.hf_dataset}\n")
                f.write(f"Downloaded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Splits: {list(ds.keys())}\n")
                for split in ds.keys():
                    f.write(f"  - {split}: {len(ds[split])} samples\n")
            
            print(f"\nDownload info saved to: {output_dir / 'download_info.txt'}")
            print("\nDownload complete! Run without --download_only to process the dataset.")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)
        
        return
    
    # Download from HuggingFace and process
    print("=" * 60)
    print("HuggingFace Download & Process Mode")
    print("=" * 60)
    print(f"Downloading HuggingFace dataset: {args.hf_dataset}")
    print(f"Output directory: {output_dir}")
    
    try:
        from datasets import load_dataset
        
        # Download both configs: 'queries' and 'trusted_db'
        print("Downloading 'queries' config...")
        ds_queries = load_dataset(
            args.hf_dataset,
            name="queries",
            num_proc=num_workers
        )
        print(f"  Queries config splits: {list(ds_queries.keys())}")
        
        print("\nDownloading 'trusted_db' config...")
        ds_trusted_db = load_dataset(
            args.hf_dataset,
            name="trusted_db",
            num_proc=num_workers
        )
        print(f"  Trusted DB config splits: {list(ds_trusted_db.keys())}")
        
        # Process both configs
        process_hf_dataset(ds_queries, ds_trusted_db, output_dir, num_workers)
        
    except Exception as e:
        print(f"Error downloading/processing dataset: {e}")
        sys.exit(1)


def process_hf_dataset(ds_queries, ds_trusted_db, output_dir: Path, num_workers: int):
    """Process a HuggingFace dataset directly.
    
    Args:
        ds_queries: Dataset with 'queries' config (contains query images with train/val/test splits)
        ds_trusted_db: Dataset with 'trusted_db' config (contains gallery/reference database images)
        output_dir: Output directory for converted dataset
        num_workers: Number of workers for parallel processing
    """
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "queries" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "queries" / "test").mkdir(parents=True, exist_ok=True)
    (output_dir / "queries" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "reference_db").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 40)
    print("Processing Gallery Images from HuggingFace (trusted_db config)")
    print("=" * 40)
    
    # Process gallery/reference images from trusted_db config
    
    gallery_results = []
    if 'train' in ds_trusted_db:
        gallery_data = ds_trusted_db['train']
        total_images = len(gallery_data)
        print(f"  Found {total_images} gallery images in trusted_db/train")
        
        # Process in batches to avoid loading all items into memory at once
        batch_size = 1000  # Process 1000 items at a time
        print(f"  Processing in batches of {batch_size} with {num_workers} workers...")
        
        start_time = time.time()
        processed_count = 0
        
        with tqdm(total=total_images, desc="  Gallery images", unit="img") as pbar:
            for batch_start in range(0, total_images, batch_size):
                batch_end = min(batch_start + batch_size, total_images)
                
                # Prepare batch arguments
                gallery_args = []
                for idx in range(batch_start, batch_end):
                    item = gallery_data[idx]
                    gallery_args.append((idx, item, output_dir))
                
                # Process batch in parallel
                batch_results = process_files_parallel(gallery_args, process_hf_gallery_image, num_workers)
                # Flatten the list of lists
                for result_list in batch_results:
                    gallery_results.extend(result_list)
                
                processed_count = batch_end
                pbar.update(len(gallery_args))
        
        gallery_time = time.time() - start_time
        print(f"  Processed in {gallery_time:.2f}s ({len(gallery_results)/gallery_time:.2f} images/sec)")
        print(f"  Results count: {len(gallery_results)}")
    
    # Write gallery CSVs for each split
    print("  Writing gallery CSVs...")
    gallery_columns = ["image_path", "split"]
    
    for split in ["train", "test", "val"]:
        split_data = [r for r in gallery_results if r and r.get('split') == split]
        if split_data:
            output_path = output_dir / f"gallery_{split}.csv"
            count = write_csv(split_data, output_path, gallery_columns)
            print(f"    {output_path.name}: {count} rows")
        else:
            # Create empty CSV for this split
            pd.DataFrame(columns=gallery_columns).to_csv(output_dir / f"gallery_{split}.csv", index=False)
            print(f"    gallery_{split}.csv: 0 rows")
    
    # Process query images from queries config
    print("\n" + "=" * 40)
    print("Processing Query Images from HuggingFace (queries config)")
    print("=" * 40)
    
    query_results = []
    total_query_time = 0.0  # Track total time across all splits
    split_results = {}  # Track results per split
    
    for split in ['train', 'test', 'val']:
        if split in ds_queries:
            query_data = ds_queries[split]
            total_samples = len(query_data)
            print(f"  Processing {split} queries: {total_samples} samples")
            
            # Process in batches to avoid loading all items into memory at once
            batch_size = 1000  # Process 1000 items at a time
            print(f"  Processing in batches of {batch_size} with {num_workers} workers...")
            
            start_time = time.time()
            split_results[split] = []  # Initialize list for this split
            
            with tqdm(total=total_samples, desc=f"  {split} queries", unit="img") as pbar:
                for batch_start in range(0, total_samples, batch_size):
                    batch_end = min(batch_start + batch_size, total_samples)
                    
                    # Prepare batch arguments
                    query_args = []
                    for idx in range(batch_start, batch_end):
                        item = query_data[idx]
                        query_args.append((idx, item, split, output_dir))
                    
                    # Process batch in parallel
                    batch_results = process_files_parallel(query_args, process_hf_query_image, num_workers)
                    query_results.extend(batch_results)
                    split_results[split].extend(batch_results)
                    
                    pbar.update(len(query_args))
            
            query_time = time.time() - start_time
            total_query_time += query_time  # Accumulate total time
            print(f"  Processed in {query_time:.2f}s ({len(split_results[split])/query_time:.2f} images/sec)")
    
    # Write query CSVs
    print("  Writing query CSVs...")
    query_columns = ["query_path", "positive_path", "has_positive", "query_transforms", "quality", "split"]
    for split in ["train", "test", "val"]:
        split_data = [r for r in query_results if r and r.get('split') == split]
        if split_data:
            output_path = output_dir / f"queries_{split}.csv"
            count = write_csv(split_data, output_path, query_columns)
            print(f"    {output_path.name}: {count} rows")
    
    # Create data_splits.csv - combines all splits into a single file
    print("\n" + "=" * 40)
    print("Creating data_splits.csv")
    print("=" * 40)
    
    # Collect all query data
    all_query_data = [r for r in query_results if r and 'query_path' in r]
    
    # Collect all gallery data
    all_gallery_data = [r for r in gallery_results if r and 'image_path' in r] if gallery_results else []
    
    # Create combined DataFrame for data_splits.csv
    datasplits_data = []
    
    # Add query data
    for item in all_query_data:
        datasplits_data.append({
            'type': 'query',
            'path': item.get('query_path', ''),
            'positive_path': item.get('positive_path', ''),
            'has_positive': item.get('has_positive', 0),
            'query_transforms': item.get('query_transforms', ''),
            'quality': item.get('quality', 0),
            'split': item.get('split', '')
        })
    
    # Add gallery data
    for item in all_gallery_data:
        datasplits_data.append({
            'type': 'gallery',
            'path': item.get('image_path', ''),
            'positive_path': '',
            'has_positive': 0,
            'query_transforms': '',
            'quality': 0,
            'split': item.get('split', '')
        })
    
    if datasplits_data:
        datasplits_df = pd.DataFrame(datasplits_data)
        datasplits_df.to_csv(output_dir / "data_splits.csv", index=False)
        print(f"  data_splits.csv: {len(datasplits_df)} rows")
    else:
        pd.DataFrame(columns=['type', 'path', 'positive_path', 'has_positive', 'query_transforms', 'quality', 'split']).to_csv(output_dir / "data_splits.csv", index=False)
        print(f"  data_splits.csv: 0 rows")
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nTotal images processed:")
    print(f"  - Query images: {len([r for r in query_results if r and 'query_path' in r])}")
    print(f"  - Gallery images: {len(gallery_results)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import hashlib
import pandas as pd
from collections import defaultdict
from PIL import Image
import h5py
import torch
from utils.attn_utils import AttentionStore

def compute_image_hash(image_path):
    """Compute a hash for an image file using MD5 and extract image dimensions."""
    hasher = hashlib.md5()
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            hasher.update(img.tobytes())
            width, height = img.size  # Get dimensions
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None
    return hasher.hexdigest(), width, height

def scan_folders(base_folder, composition_categories, frame_size_categories):
    """Scan folders and organize images by hash, tracking composition, frame size labels, and dimensions."""
    image_data = defaultdict(lambda: {
        "composition": set(),
        "frame_size": set(),
        "file_paths": [],
        "width": None,
        "height": None
    })
    
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(root, file)
                img_hash, width, height = compute_image_hash(file_path)
                if img_hash:
                    image_data[img_hash]["file_paths"].append(file_path)

                    # Store dimensions only the first time
                    if image_data[img_hash]["width"] is None:
                        image_data[img_hash]["width"] = width
                        image_data[img_hash]["height"] = height

                    for category in composition_categories:
                        if category.lower() in root.lower():
                            image_data[img_hash]["composition"].add(category)
                    
                    for category in frame_size_categories:
                        if os.path.basename(root).lower() == category.lower():
                            image_data[img_hash]["frame_size"].add(category)
    
    return image_data

def create_dataframe(image_data):
    """Create a DataFrame from the collected image data."""
    data = []
    for img_hash, img_metadata in image_data.items():
        composition = ", ".join(img_metadata["composition"]) if img_metadata["composition"] else "Unknown"
        frame_size = ", ".join(img_metadata["frame_size"]) if img_metadata["frame_size"] else "Unknown"
        width = img_metadata["width"]
        height = img_metadata["height"]
        data.append([img_hash, composition, frame_size, img_metadata["file_paths"], width, height])
    
    df = pd.DataFrame(data, columns=['image_hash', 'composition', 'frame_size', 'file_paths', 'width', 'height'])
    
    # Compute aspect ratio on the whole DataFrame and round to 2 decimal places
    df["aspect_ratio"] = (df["width"] / df["height"]).round(2)
    
    return df

def update_dataframe(df, image_data):
    # Update existing images by checking for valid file paths and updating metadata
    for idx, row in df.iterrows():
        img_hash = row['image_hash']
        if img_hash in image_data:
            # Only keep file paths that exist
            valid_paths = [path for path in image_data[img_hash]["file_paths"] if os.path.exists(path)]
            df.at[idx, "file_paths"] = ",".join(valid_paths)  # Update the file paths column

            # Update the composition and frame size
            df.at[idx, "composition"] = ",".join(image_data[img_hash]["composition"]) if image_data[img_hash]["composition"] else "Unknown"
            df.at[idx, "frame_size"] = ",".join(image_data[img_hash]["frame_size"]) if image_data[img_hash]["frame_size"] else "Unknown"
    
    # Handle new images (those not in the original DataFrame)
    existing_hashes = set(df['image_hash'])
    for img_hash, metadata in image_data.items():
        if img_hash not in existing_hashes:
            # Add new row for this image
            new_row = {
                'image_hash': img_hash,
                'composition': ",".join(metadata["composition"]) if metadata["composition"] else "Unknown",
                'frame_size': ",".join(metadata["frame_size"]) if metadata["frame_size"] else "Unknown",
                'file_paths': ",".join(metadata["file_paths"]),
                'width': metadata["width"],
                'height': metadata["height"],
                'aspect_ratio': round(metadata["width"] / metadata["height"], 2) if metadata["width"] and metadata["height"] else None
            }
            df = df.append(new_row, ignore_index=True)

    return df

def load_attention_maps(hdf5_path, attn_type="cross", index=-1, aggregate=False, aggregation_mode="max", downsample=False, sampling_mode="max", res=16):
    """
    Loads and optionally aggregates attention maps from an HDF5 file.
    """
    if aggregate:
        attn = AttentionStore()

    with h5py.File(hdf5_path, "r") as f:
        data = []
        
        # Loop through each image group (image_hash)
        for image_hash in f.keys():
            row_data = {"image_hash": image_hash}
            grouped_maps = {}  # Stores maps for aggregation if `aggregate_flag=True`

            # Loop through all layers for this image
            for layer_key in f[image_hash].keys():
                if layer_key.split("_")[0] == attn_type:
                    group_key = "_".join(layer_key.split("_")[:-1])  # Extract group key for aggregation

                    # Convert to Torch tensor
                    layer = f[image_hash][layer_key][:, :, index] if index is not None else f[image_hash][layer_key][:]
                    attn_map = torch.tensor(layer.copy())
                    H, W = attn_map.shape[:2]
                    if len(attn_map.shape) == 2:
                        attn_map = attn_map.unsqueeze(-1)

                    # Downsample if necessary
                    if downsample:
                        if H > res:
                            scale_factor = res / H
                            attn_map = attn_map.unsqueeze(0)
                            resized_map = attn.rescale_attention(attn_map, scale_factor=scale_factor, sampling_mode=sampling_mode)
                            attn_map = resized_map.squeeze(0)

                    # 🔹 Store Map Based on `aggregate_flag`
                    if aggregate:
                        grouped_maps.setdefault(group_key, []).append(attn_map)
                    else:
                        row_data[layer_key] = attn_map  # Directly store the individual layer

            # 🔹 Aggregate Maps if `aggregate_flag=True`
            if aggregate:
                for group, attn_maps in grouped_maps.items():
                    shapes = [attn.shape for attn in attn_maps]
                    if len(set(shapes)) > 1:  # If there are multiple unique shapes
                        print(f"⚠️ Runtime Error: Mismatched attention map sizes in image '{image_hash}', group '{group}'")
                        print(f"Mismatched shapes found: {shapes}")  # Output the different shapes
                        continue  # Skip this group to prevent crashes

                    stacked_maps = torch.stack(attn_maps, dim=0)
                    
                    # Apply chosen aggregation mode
                    if aggregation_mode == "mean":
                        aggregated_map = stacked_maps.mean(dim=0)
                    # elif aggregation_mode == "sum":
                    #     aggregated_map = stacked_maps.sum(dim=0)
                    elif aggregation_mode == "max":
                        aggregated_map = stacked_maps.max(dim=0)[0]
                    else:
                        raise ValueError(f"Invalid aggregation mode: {aggregation_mode}. Choose 'mean' or 'max'")

                    row_data[group] = aggregated_map  # Store aggregated map

            data.append(row_data)

    attn_df = pd.DataFrame(data)
    attn_df = attn_df.dropna()
    attn_df = attn_df.set_index("image_hash")
    
    return attn_df

if __name__ == "__main__":
    # Define composition and frame size categories
    composition_categories = ["balanced", "center", "left", "right", "symmetrical"]
    frame_size_categories = ["ECU", "CU", "MCU", "MS", "MWS", "WS", "EWS"]
    
    base_folder = "shotdeck_data"  # Root folder containing subfolders for composition and frame size
    image_data = scan_folders(base_folder, composition_categories, frame_size_categories)
    output_dir = "shotdeck_csv"
    output_csv = os.path.join(output_dir, "shotdeck_updated.csv")
    data_file = os.path.join(output_dir, "shotdeck_data.csv")
    # df = create_dataframe(image_data)
    df = pd.read_csv(data_file)
    df = update_dataframe(df, image_data)
    # # Save or display the DataFrame
    df.to_csv(output_csv, index=False)
    print("DataFrame saved to shotdeck_updated.csv")
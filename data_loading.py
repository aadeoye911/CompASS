#!/usr/bin/env python3

import os
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict
from PIL import Image
import h5py
import torch
from fractions import Fraction
from sklearn.preprocessing import MultiLabelBinarizer

def load_encoded_labels(labels_path, category): 
    labels_df = pd.read_csv(labels_path, usecols=["image_hash", category], index_col=0)

    # Filter out unknown labels
    labels_df[category] = labels_df[category].apply(lambda x: x.split(",") if isinstance(x, str) else x)

    mlb = MultiLabelBinarizer()
    labels_encoded = pd.DataFrame(
        mlb.fit_transform(labels_df[category]),
        columns=mlb.classes_,
        index=labels_df.index  # Keep the original index (image_hash)
    )
    num_dropped = (labels_encoded["Unknown"] == 1).sum()
    print(f"Dropped {num_dropped} samples with unknown {category} labels.")

    labels_encoded = labels_encoded[labels_encoded["Unknown"] != 1]
    labels_encoded = labels_encoded.drop(columns="Unknown")

    return labels_encoded

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

def load_attention_maps(hdf5_path, attn_type="cross", index=-1, keep_dims=False, verbose=True):
    """
    Loads and optionally aggregates attention maps from an HDF5 file.
    """

    with h5py.File(hdf5_path, "r") as f:
        valid_data = []
        dropped_hashes = []
        
        # Loop through each image group (image_hash)
        for image_hash in f.keys():
            row_data = {"image_hash": image_hash}
            heights = []
            widths = []

            # Loop through all layers for this image
            for layer_key in f[image_hash].keys():
                if layer_key.split("_")[0] == attn_type:

                    # Convert to Torch tensor
                    layer = f[image_hash][layer_key][:, :, index] if index is not None else f[image_hash][layer_key][:]
                    if keep_dims and layer.ndim == 2:
                        layer = layer[:, :, np.newaxis]
                    
                    H, W = layer.shape[:2]
                    heights.append(H)
                    widths.append(W)

                    attn_map = torch.tensor(layer.copy())
                    row_data[layer_key] = attn_map

            first_ratio = Fraction(heights[0], widths[0])
            ratios = [Fraction(h, w) for h, w in zip(heights, widths)]

            # Check proportionality of all maps (same aspect ratio)
            if all(r == first_ratio for r in ratios):
                row_data["output_dims"] = (heights[0], widths[0])
                valid_data.append(row_data)
            else:
                dropped_hashes.append(image_hash)
                
    if verbose:
        print(f"✅ Loaded {len(valid_data)} valid samples.")
        print(f"❌ Dropped {len(dropped_hashes)} samples due to mismatched aspect ratios.")
        if dropped_hashes:
            print("Dropped hashes (first 10):", dropped_hashes[:10])
            
    attn_df = pd.DataFrame(valid_data)
    attn_df = attn_df.dropna()
    attn_df = attn_df.set_index("image_hash")
    
    return attn_df

def extract_attention_dataset(pipe, images, diffused_dir, hdf5_path):
    results = []

    for _, row in images.iterrows():
        image_hash = row["Image Hash"]
        image_path = row["Resized Path"]

        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue

        try:
            # pipe.extract_reference_attn_maps(image_path)  # GPU-accelerated
            diffused_image = pipe.diffused_images[0]  # Extract the diffused image
            diffused_path = os.path.join(diffused_dir, f"{image_hash}.png")
            diffused_image.save(diffused_path)
            results.append((image_hash, pipe.attention_store))  # Store attention data
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

    # Write the processed batch to HDF5
    with h5py.File(hdf5_path, "a") as f:  # Append mode to avoid overwriting
        for image_hash, attn_data in results:
            if attn_data is None:
                continue
            img_group = f.create_group(image_hash)
            for layer_key, attn_map in attn_data.items():
                img_group.create_dataset(layer_key, data=attn_map, compression="lzf")

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
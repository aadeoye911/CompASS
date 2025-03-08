#!/usr/bin/env python3

import os
import hashlib
import pandas as pd
from collections import defaultdict
from PIL import Image

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
    
    df = pd.DataFrame(data, columns=['Image Hash', 'Composition', 'Frame Size', 'File Paths', 'Width', 'Height'])
    
    # Compute aspect ratio on the whole DataFrame and round to 2 decimal places
    df["Aspect Ratio"] = (df["Width"] / df["Height"]).round(2)
    
    return df

def update_dataframe(df, image_data):
    # Update existing images by checking for valid file paths and updating metadata
    for idx, row in df.iterrows():
        img_hash = row['Image Hash']
        if img_hash in image_data:
            # Only keep file paths that exist
            valid_paths = [path for path in image_data[img_hash]["file_paths"] if os.path.exists(path)]
            df.at[idx, "File Paths"] = ", ".join(valid_paths)  # Update the file paths column

            # Update the composition and frame size
            df.at[idx, "Composition"] = ", ".join(image_data[img_hash]["composition"]) if image_data[img_hash]["composition"] else "Unknown"
            df.at[idx, "Frame Size"] = ", ".join(image_data[img_hash]["frame_size"]) if image_data[img_hash]["frame_size"] else "Unknown"
    
    # Handle new images (those not in the original DataFrame)
    existing_hashes = set(df['Image Hash'])
    for img_hash, metadata in image_data.items():
        if img_hash not in existing_hashes:
            # Add new row for this image
            new_row = {
                'Image Hash': img_hash,
                'Composition': ", ".join(metadata["composition"]) if metadata["composition"] else "Unknown",
                'Frame Size': ", ".join(metadata["frame_size"]) if metadata["frame_size"] else "Unknown",
                'File Paths': ", ".join(metadata["file_paths"]),
                'Width': metadata["width"],
                'Height': metadata["height"],
                'Aspect Ratio': round(metadata["width"] / metadata["height"], 2) if metadata["width"] and metadata["height"] else None
            }
            df = df.append(new_row, ignore_index=True)

    return df

if __name__ == "__main__":
    # Define composition and frame size categories
    composition_categories = ["balanced", "center", "left", "right", "symmetrical"]
    frame_size_categories = ["ECU", "CU", "MCU", "MS", "MWS", "WS", "EWS"]
    
    base_folder = "shotdeck_data"  # Root folder containing subfolders for composition and frame size
    image_data = scan_folders(base_folder, composition_categories, frame_size_categories)
    output_csv = "test.csv"
    # df = create_dataframe(image_data)
    # df = pd.read_csv("shotdeck_data.csv")
    # df = update_dataframe(df, image_data)
    # # Save or display the DataFrame
    # df.to_csv(output_csv, index=False)
    print("DataFrame saved to shotdeck_data.csv")
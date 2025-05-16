#!/usr/bin/env python3

import os
import hashlib
import pandas as pd
import numpy as np
import h5py
from collections import defaultdict
from PIL import Image
import h5py
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pandas as pd

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
        composition = ",".join(img_metadata["composition"]) if img_metadata["composition"] else None
        frame_size = ",".join(img_metadata["frame_size"]) if img_metadata["frame_size"] else None
        width = img_metadata["width"]
        height = img_metadata["height"]
        ref_path = img_metadata["file_paths"][0]
        file_paths = ",".join(img_metadata["file_paths"])
        data.append([img_hash, composition, frame_size, width, height, ref_path, file_paths])
    
    df = pd.DataFrame(data, columns=['image_hash', 'composition', 'frame_size', 'width', 'height', 'ref_path', 'file_paths'])
        
    return df

def update_dataframe(df, image_data):
    # Update existing images by checking for valid file paths and updating metadata
    for idx, row in df.iterrows():
        img_hash = row['image_hash']
        if img_hash in image_data:
            # Only keep file paths that exist
            valid_paths = [path for path in image_data[img_hash]["file_paths"] if os.path.exists(path)]
            df.at[idx, "composition"] = ",".join(image_data[img_hash]["composition"]) if image_data[img_hash]["composition"] else None
            df.at[idx, "frame_size"] = ",".join(image_data[img_hash]["frame_size"]) if image_data[img_hash]["frame_size"] else None
            df.at[idx, "ref_path"] = valid_paths[0]
            df.at[idx, "file_paths"] = ",".join(valid_paths)  # Update the file paths column

    # Handle new images (those not in the original DataFrame)
    existing_hashes = set(df['image_hash'])
    for img_hash, img_metadata in image_data.items():
        if img_hash not in existing_hashes:
            # Add new row for this image
            new_row = {
                'image_hash': img_hash,
                'composition': ",".join(img_metadata["composition"]) if img_metadata["composition"] else None,
                'frame_size': ",".join(img_metadata["frame_size"]) if img_metadata["frame_size"] else None,
                'width': img_metadata["width"],
                'height': img_metadata["height"],
                'ref_path': img_metadata["file_paths"][0],
                'file_paths': ",".join(img_metadata["file_paths"]),
            }
            df = df.append(new_row, ignore_index=True)

    return df

def set_layer_keys(col, layer_dict=None):
    if not layer_dict:
        layer_dict = {
            "cross_down_0": "down_64", 
            "cross_down_1": "down_32",
            "cross_down_2": "down_16",
            "cross_mid_0": "mid_8",
            "cross_up_1": "up_16",
            "cross_up_2": "up_32",
            "cross_up_3": "up_64"}
    
    for old, new in layer_dict.items():
        if col.startswith(old):
            return col.replace(old, new)
    return col  # leave unchanged if no match

def load_attention_maps(hdf5_path, use_resolution_keys=True):
    valid_data = []
    with h5py.File(hdf5_path, "r") as f:
        for image_hash in f.keys():
            row_data = {"image_hash": image_hash}
            for layer_key in f[image_hash].keys():
                attn_map = f[image_hash][layer_key][:] 
                row_data[layer_key] = torch.tensor(attn_map.copy())

            valid_data.append(row_data)
    
    attn_df = pd.DataFrame(valid_data)
    attn_df = attn_df.dropna()
    attn_df = attn_df.set_index("image_hash")

    if use_resolution_keys:
        attn_df.columns = [set_layer_keys(col) for col in attn_df.columns]

    print(f"Loaded {len(attn_df)} valid samples.")          
    return attn_df

def load_encoded_labels(labels_path, category, drop_unknown=True): 
    labels_df = pd.read_csv(labels_path, usecols=["image_hash", category], index_col=0)

    # Turn comma separate entries into binarized multilabels
    labels_df[category] = labels_df[category].apply(lambda x: x.split(",") if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    labels_encoded = pd.DataFrame(
        mlb.fit_transform(labels_df[category]),
        columns=mlb.classes_,
        index=labels_df.index  # Keep the original index (image_hash)
    )

    if drop_unknown:
        num_dropped = (labels_encoded.sum(axis=1) == 0).sum()
        print(f"Dropped {num_dropped} samples with unknown {category} labels.")
        labels_encoded = labels_encoded[labels_encoded.sum(axis=1) > 0]

    return labels_encoded

def filter_maps_by_modal_dims(attn_df, labels_df):
    modal_dim = labels_df['resized_width'].mode()[0]
    matching_indices = labels_df[labels_df['resized_width'] == modal_dim].index
    attn_filtered = attn_df.loc[attn_df.index.isin(matching_indices)]

    return attn_filtered

def filter_selected_labels(labels_encoded, selected_labels, allow_multilabel=True):
    subset_encoded = labels_encoded[selected_labels]
    valid_rows = subset_encoded.sum(axis=1) > 0 if allow_multilabel else subset_encoded.sum(axis=1) == 1
    subset_encoded = subset_encoded[valid_rows]

    return subset_encoded

def align_samples_and_labels(attn_df, labels_encoded):
    # Load the attention maps into a DataFrame
    valid_samples = attn_df.index.intersection(labels_encoded.index)
    attn_aligned = attn_df.loc[valid_samples]
    labels_aligned = labels_encoded.loc[valid_samples]

    return attn_aligned, labels_aligned

def random_sampling(labels_encoded, label, num_samples=1, seed=42):
    """
    Sample training data by label
    """
    filtered_labels = labels_encoded[labels_encoded[label] == 1]
    num_samples = min(num_samples, len(filtered_labels))
    sampled_df = filtered_labels.sample(num_samples, random_state=seed)
    
    return sampled_df

def mlp_classifier(X, y, hidden_layer_sizes=(128, 128), class_names=None, activation="relu"):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=5000, activation=activation, learning_rate="adaptive")
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=class_names, zero_division=0)

    return report

def per_feature_classifier(X_df, y_df, hidden_layer_size=(128, 128), activation="relu", multilabel=False):
    feature_results = {}
    attn_features = [col for col in X_df.columns]
    class_names = [col for col in y_df.columns]    
    for layer_idx, col in enumerate(attn_features):
        X = X_df[col].apply(lambda x: x.reshape(x.shape[0], -1)).item().numpy()
        y = y_df.values
        if not multilabel:
            y = np.argmax(y, axis=1)
        report = mlp_classifier(X, y, hidden_layer_sizes=hidden_layer_size, activation=activation, class_names=class_names)
        feature_results[col] = report

    results_df = read_model_results(feature_results, class_names)
    return results_df

def read_model_results(feature_results, class_names):
    combined_data = {}
    for feature, report in feature_results.items():
        for key, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    combined_data.setdefault((key, metric_name), {})[feature] = value
            else:
                # Handle scalar entries like 'accuracy' if present
                combined_data.setdefault(('accuracy', key), {})[feature] = metrics

    # Convert the nested dictionary to a DataFrame
    df = pd.DataFrame(combined_data).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["Label", "Metric"])
    return df

def balanced_multilabel_sampling(labels_df, seed=42, class_count=None, verbose=True):
    """
    Sample multi-label dataset so each class has roughly same number of positive examples.
    """

    if class_count is None:
        class_count = labels_df.sum(axis=0).min()
        if verbose:
            print(f"[INFO] Using {class_count} samples per label (minimum across classes)")


    for i, label in enumerate(labels_df.columns):
        subsampled_df = random_sampling(labels_df, label, num_samples=class_count, seed=seed).reset_index()
        if verbose:
            print(f"[INFO] Sampled {len(subsampled_df)} rows for label '{label}'")
        if i == 0:
            sampled_df = subsampled_df
        else:
            sampled_df = pd.concat([sampled_df, subsampled_df], ignore_index=True)
    if verbose:
        label_coverage = sampled_df[labels_df.columns].sum().sort_values(ascending=False)
        print("\n[INFO] Label counts in the final merged sampled set:")
        print(label_coverage)
    
    sampled_df = sampled_df.drop_duplicates().sample(frac=1, random_state=42)

    return sampled_df.set_index("image_hash")

def sample_by_label(attn_df, labels_encoded, selected_labels, multilabel=False):
    subset_encoded = filter_selected_labels(labels_encoded, selected_labels, allow_multilabel=multilabel)
    filtered_df, subset_encoded = align_samples_and_labels(attn_df, subset_encoded)
    balanced_multilabel = balanced_multilabel_sampling(subset_encoded, verbose=False)
    sampled_df, y_df = align_samples_and_labels(filtered_df, balanced_multilabel)
    return sampled_df, y_df

# if __name__ == "__main__":
#     # Define composition and frame size categories
#     composition_labels = ["balanced", "center", "left", "right", "symmetrical"]
#     frame_size_labels = ["ECU", "CU", "MCU", "MS", "MWS", "WS", "EWS"]
    
#     base_folder = "shotdeck_data"  # Root folder containing subfolders for composition and frame size
#     image_data = scan_folders(base_folder, composition_labels, frame_size_labels)
    
#     df = create_dataframe(image_data)
#     csv_path = "shotdeck_master.csv"

#     df.to_csv(csv_path, index=False)
#     print(f"DataFrame saved to {csv_path}")
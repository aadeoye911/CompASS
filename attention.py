import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

class AttentionStore():

    def __init__(self, save_global_store=True):
        """
        Initializes an AttentionStore that tracks attention maps with structured keys.
        """
        self.attn_store = defaultdict(list)  # Stores attention in a nested dictionary format
        self.attn_metadata = defaultdict(list)


    def reset(self):
        """
        Reset attention storage.
        """
        self.attn_store = self.get_empty_store()


    def get_empty_store(self):
        """
        Returns an empty attention store.
        """
        return {layer_key: [] for layer_key in self.attn_metadata.keys()}
    

    def print_attention_metadata(self):
        """
        Print layer metadata.
        """
        print(f"Total Cross-Attention Layers: {len(self.attn_metadata)}")
        print("\nCross-Attention Layers:")
        for layer_key, value in self.attn_metadata.items():
            print(f"Key: {layer_key}:, Downsample Factor: {value[0]}, Module Name: {value[1]}")


    def store(self, attn_probs, layer_key, latent_height, latent_width):
        """
        Store attention scores using a dictionary-based key format.
        """
        res_factor = self.attn_metadata[layer_key][0]
        attn_map = self.reshape_attention_map(attn_probs, latent_height, latent_width, res_factor)
        print(f"Storing attn for layer: {layer_key} with shape {attn_map.shape}")
        self.attn_store[layer_key].append(attn_map)  # Store as tuple (attn_map, res_factor)


    def reshape_attention_map(self, attn_probs, latent_height, latent_width, res_factor):
        if not latent_height % res_factor != 0 or not latent_width % res_factor != 0:
            raise ValueError(f"Downsampling produced non-integer dimensions.")
        attn_height, attn_width = int(latent_height / res_factor), int(latent_width / res_factor)
        attn_map = attn_probs.reshape(attn_probs.shape[0], attn_height, attn_width, -1)

        return attn_map
    
import torch
import torch.nn.functional as F

class AttentionAggregator:
    def __init__(self, attn_store, attn_metadata):
        """
        Initialize the Attention Aggregator.
        
        Args:
            attn_store (dict): Dictionary storing attention maps with keys.
            attn_metadata (dict): Metadata dictionary containing downsample factors.
        """
        self.attn_store = attn_store
        self.attn_metadata = attn_metadata

    def filter_attention_maps(self, place_in_unet, level=None):
        """
        Filters attention maps based on place in unet (down, mid, up) and level.
        """
        filtered_maps = defaultdict(list)
        res_factors = []
        # Select attention maps based on filter keys
        for layer_key, attn_map in self.attn_store.items():
            if layer_key[0] == place_in_unet and (level is None or layer_key[1] == level):
                filtered_maps[(place_in_unet, level)].append(attn_map)
                res_factors.append(self.attn_metadata[layer_key][0])
        
        if not filtered_maps:
            raise ValueError(f"No attention maps found for place_in_unet='{place_in_unet}', level={level}")

        return filtered_maps, list(set(res_factors))

    def resize_attention_maps(self, attn_maps, resolution=None, downsample_mode="bilinear"):
        """
        Resizes attention maps to a common resolution.

        Args:
            attn_maps (list of torch.Tensor): List of attention maps.
            resolution (tuple, optional): Target resolution (H, W). Defaults to the minimum resolution.
            downsample_mode (str): Downsampling mode - "bilinear" (default) or "max".

        Returns:
            list of torch.Tensor: Resized attention maps.
        """
        # Determine target resolution if not provided
        if resolution is None:
            min_H = min([attn.shape[1] for attn in attn_maps])
            min_W = min([attn.shape[2] for attn in attn_maps])
            resolution = (min_H, min_W)

        resized_maps = []
        for attn_map in attn_maps:
            H, W = attn_map.shape[1:3]  # Original size
            if (H, W) != resolution:
                if H > resolution[0] or W > resolution[1]:  # Downsampling
                    if downsample_mode == "max":
                        pool_size = (H // resolution[0], W // resolution[1])
                        attn_map = F.adaptive_max_pool2d(attn_map.permute(0, 3, 1, 2), resolution).permute(0, 2, 3, 1)
                    else:
                        attn_map = F.interpolate(attn_map.permute(0, 3, 1, 2), size=resolution, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                else:  # Upsampling
                    attn_map = F.interpolate(attn_map.permute(0, 3, 1, 2), size=resolution, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            resized_maps.append(attn_map.unsqueeze(0))  # Unsqueeze at dim=0 for aggregation

        return resized_maps

    def aggregate_attention(self, filter_keys=None, resolution=None, mode="mean", downsample_mode="bilinear"):
        """
        Aggregates filtered attention maps at a specified resolution.

        Args:
            filter_keys (tuple or list of tuples, optional): Keys to filter attention maps.
            resolution (tuple, optional): Target resolution (H, W). Defaults to minimum resolution.
            mode (str): Aggregation mode - "mean" or "max".
            downsample_mode (str): Downsampling mode - "bilinear" (default) or "max".

        Returns:
            torch.Tensor: Aggregated attention map of shape (B, H, W, C).
        """
        # Step 1: Filter attention maps
        selected_maps, _ = self.filter_attention_maps(filter_keys)

        # Step 2: Resize to common resolution
        resized_maps = self.resize_attention_maps(selected_maps, resolution, downsample_mode)

        # Step 3: Aggregate
        stacked_maps = torch.cat(resized_maps, dim=0)  # Stack along new dimension for aggregation

        if mode == "mean":
            aggregated_map = stacked_maps.mean(dim=0)
        elif mode == "max":
            aggregated_map = stacked_maps.max(dim=0)[0]
        else:
            raise ValueError(f"Invalid aggregation mode: {mode}. Choose 'mean' or 'max'.")

        return aggregated_map

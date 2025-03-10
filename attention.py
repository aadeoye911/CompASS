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
        if (latent_height % res_factor != 0) or (latent_width % res_factor != 0):
            raise ValueError(f"Downsampling produced non-integer dimensions.")
        attn_height, attn_width = latent_height // res_factor, latent_width // res_factor
        attn_map = attn_probs.reshape(attn_probs.shape[0], attn_height, attn_width, -1)

        return attn_map
    

    def filter_layer_keys(self, place_in_unet, level=None):
        """
        Filters keys for attention maps based on place in unet (down, mid, up) and level.
        """
        filtered_keys = {}
        for layer_key in self.attn_store.keys():
            if layer_key[0] == place_in_unet and (level is None or layer_key[1] == level):
                filter = (place_in_unet,) if level is None else (place_in_unet, level)
                filtered_keys.setdefault(filter, []).append(layer_key)

        if not filtered_keys:
            raise ValueError(f"No attention map keys found for place_in_unet='{place_in_unet}', level={level}")

        return filtered_keys

    def resize_attention_map(self, attn_map, scale_factor, downsample_mode="bilinear", upsample_mode="bilinear"):
        """
        Up/downsample to change resolution of attention maps
        """
        if scale_factor <= 0:
            raise ValueError(f"Invalid scale_factor={scale_factor}. Must be a positive value.")
        height, width = attn_map.shape[1:3]
        target_res = int(height * scale_factor), int(width * scale_factor)
        
        # Permute to (B, C, H, W) format for interpolation
        attn_map = attn_map.permute(0, 3, 1, 2)

        # Handle upsampling
        if scale_factor > 1:
            if upsample_mode not in ["bilinear", "nearest", "bicubic"]:
                raise ValueError(f"Invalid upsample_mode='{upsample_mode}'. Choose 'bilinear', 'nearest', or 'bicubic'.")
            resized_map = F.interpolate(attn_map, size=target_res, mode=upsample_mode)

        # Handle downsampling
        else:
            if downsample_mode == "max":
                resized_map = F.adaptive_max_pool2d(attn_map, output_size=target_res)
            elif downsample_mode in ["bilinear", "nearest", "bicubic"]:
                resized_map = F.interpolate(attn_map, size=target_res, mode=downsample_mode)
            else:
                raise ValueError(f"Invalid downsample_mode='{downsample_mode}'. Choose 'bilinear', 'nearest', 'bicubic', or 'max'.")

        # Permute back to (B, H, W, C)
        return resized_map.permute(0, 2, 3, 1)
    
    def aggregate_attention(self, place_in_unet, level=True,res=None, aggregation_mode="mean", downsample_mode="bilinear"):
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

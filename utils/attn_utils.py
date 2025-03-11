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

        return
    
    def store(self, attn_probs, layer_key, latent_height, latent_width):
        """
        Store attention scores using a dictionary-based key format.
        """
        res_factor = self.attn_metadata[layer_key][0]
        attn_map = self.reshape_attention_map(attn_probs, latent_height, latent_width, res_factor)
        print(f"Storing attn for layer: {layer_key} with shape {attn_map.shape}")
        self.attn_store[layer_key].append(attn_map)  # Store as tuple (attn_map, res_factor)

        return
    
    def reshape_attention_map(self, attn_probs, latent_height, latent_width, res_factor):
        if (latent_height % res_factor != 0) or (latent_width % res_factor != 0):
            raise ValueError(f"Downsampling produced non-integer dimensions.")
        attn_height, attn_width = latent_height // res_factor, latent_width // res_factor
        attn_map = attn_probs.reshape(attn_probs.shape[0], attn_height, attn_width, -1)

        return attn_map
    

    def filter_layer_keys(self, separate_levels=False):
        """
        Filters keys for attention maps based on place in unet (down, mid, up) and level.
        """
        filtered_layers = defaultdict(list)
        for layer_key in self.attn_store.keys():
            group_key = layer_key[:2] if separate_levels else layer_key[:1]
            filtered_layers[group_key].append(layer_key)

        return filtered_layers


    def modify_attention_resolution(self, attn_map, scale_factor, mode="bilinear"):
        """
        Up/downsample to change resolution of attention maps
        """
        if scale_factor <= 0:
            raise ValueError(f"Invalid scale_factor={scale_factor}. Must be a positive value.")
        if scale_factor == 1:
            return attn_map
        
        height, width = attn_map.shape[1:3]
        target_res = int(height * scale_factor), int(width * scale_factor)
        
        attn_map = attn_map.permute(0, 3, 1, 2) # Permute to (B, C, H, W) 
        if mode not in ["bilinear", "nearest", "bicubic"]:
            if scale_factor > 1:
                raise ValueError(f"Invalid upsamping mode='{mode}'. Choose 'bilinear', 'nearest', or 'bicubic' ")
            elif mode != "max":
                raise ValueError(f"Invalid downsamping mode='{mode}'. Choose 'bilinear', 'nearest', or 'bicubic' ")   
            else:
                resized_map = F.adaptive_max_pool2d(attn_map, output_size=target_res)
        else:
            resized_map = F.interpolate(attn_map, size=target_res, mode=mode)

        return resized_map.permute(0, 2, 3, 1)
    

    def aggregate_attention(self, place_in_unet, level=None, res_factor=None, aggregation_mode="mean", sampling_mode="bilinear"):
        """
        Aggregates the attention across filtered layers at the specified resolution.
        """
        filtered_layers = self.filter_attention_maps(place_in_unet, level)  # {layer_name: metadata_tuple}
        if not filtered_layers:
            raise ValueError(f"No attention maps found for place_in_unet='{place_in_unet}', level={level}")

        if res_factor == None:
            filtered_factors = [self.attn_metadata[layer_key][0] for layer_key in filtered_layers[(place_in_unet, level)]]
            res_factor = min(filtered_factors)

        # Step 5: Resize attention maps using computed scale factors
        resized_maps = []
        for layer_key in filtered_layers:
            scale_factor = self.attn_metadata[layer_key][0] / res_factor   # Get corresponding scale factor
            attn_map = self.attn_store[layer_key]  # Retrieve actual tensor [B, H, W, C]
            resized_map = self.modify_attention_resolution(attn_map, scale_factor, sampling_mode)
            resized_maps.append(resized_map)

        stacked_maps = torch.cat(resized_maps, dim=0)  # Stack along new dimension for aggregation
        if aggregation_mode == "mean":
            aggregated_map = stacked_maps.mean(dim=0)
        elif aggregation_mode == "max":
            aggregated_map = stacked_maps.max(dim=0)[0]
        else:
            raise ValueError(f"Invalid aggregation mode: {aggregation_mode}. Choose 'mean' or 'max'.")

        return aggregated_map
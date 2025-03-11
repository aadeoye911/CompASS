import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

class AttentionStore:
    def __init__(self, save_global_store=True):
        """
        Initializes an AttentionStore that tracks attention maps with structured keys.
        """
        self.cross_attention_maps = defaultdict(list)  # Stores cross-attention maps
        self.self_attention_maps = defaultdict(list)   # Stores self-attention maps

        # Ensures layer metadata is properly structured
        self.attn_metadata = { "cross": {}, "self": {} }


    def reset(self):
        """
        Reset attention storage.
        """
        self.cross_attention_maps = self.get_empty_store("cross")
        self.self_attention_maps = self.get_empty_store("self")


    def get_empty_store(self, attn_type="cross"):
        """
        Returns an empty attention store.
        """
        return {layer_key: [] for layer_key in self.attn_metadata[attn_type].keys()}


    def print_attn_metadata(self):
        """
        Print formatted metadata for stored attention layers.
        """
        # Print Cross-Attention Metadata
        print(f"Total Cross-Attention Layers: {len(self.attn_metadata['cross'])}")
        print("\nCross-Attention Layers:")
        for layer_key, (res_factor, name) in self.attn_metadata["cross"].items():
            print(f"Layer Key: {layer_key}, Resolution Downsampling Factor: {res_factor}, Module Name: {name}")

        # Print Self-Attention Metadata
        print(f"Total Self-Attention Layers: {len(self.attn_metadata['self'])}")
        print("\nSelf-Attention Layers:")
        for layer_key, (res_factor, name) in self.attn_metadata["self"].items():
            print(f"Layer Key: {layer_key}, Resolution Downsampling Factor: {res_factor}, Module Name: {name}")
    

    def store(self, attn_probs, layer_key, latent_height, latent_width):
        """
        Store attention scores using a dictionary-based key format.
        """
        attn_type = layer_key[0]
        if attn_type == "cross":
            res_factor = self.attn_metadata[attn_type][layer_key][0]
            attn_map = self.reshape_attention(attn_probs, latent_height, latent_width, res_factor)
        else:
            attn_map = attn_probs
        print(f"Storing attention map for layer: {layer_key} with shape {attn_map.shape}")
        getattr(self, f"{attn_type}_attention_maps")[layer_key].append(attn_map)
    

    def reshape_attention(self, attn_probs, latent_height, latent_width, res_factor):
        if (latent_height % res_factor != 0) or (latent_width % res_factor != 0):
            raise ValueError(f"Downsampling produced non-integer dimensions.")
        attn_height, attn_width = latent_height // res_factor, latent_width // res_factor
        attn_map = attn_probs.reshape(attn_probs.shape[0], attn_height, attn_width, -1)

        return attn_map
    

    def group_attention_layers(self, attn_type, group_by_level=True):
        """
        Filters keys for attention maps based on place in unet (down, mid, up) and level.
        """
        grouped_layers = defaultdict(list)
        for layer_key in self.attn_metadata[attn_type].keys():
            group_key = layer_key[:3] if group_by_level else layer_key[:2]
            grouped_layers[group_key].append(layer_key)

        return grouped_layers


    def rescale_attention(self, attn_map, scale_factor, sampling_mode="bilinear"):
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
        if sampling_mode not in ["bilinear", "nearest", "bicubic"]:
            if scale_factor > 1:
                raise ValueError(f"Invalid upsamping mode='{sampling_mode}'. Choose 'bilinear', 'nearest', or 'bicubic' ")
            elif sampling_mode != "max":
                raise ValueError(f"Invalid downsamping mode='{sampling_mode}'. Choose 'bilinear', 'nearest', or 'bicubic' ")   
            else:
                resized_map = F.adaptive_max_pool2d(attn_map, output_size=target_res)
        else:
            resized_map = F.interpolate(attn_map, size=target_res, mode=sampling_mode)

        return resized_map.permute(0, 2, 3, 1)
    

    def aggregate_attention(self, layer_keys, res_factor=None, aggregation_mode="max", downsampling_mode="max", upsampling_mode="bilinear"):
        """
        Aggregates the attention across subset of layers at the specified resolution factor.
        """
        if res_factor == None:
            res_factor = min([self.attn_metadata[layer_key[0]][layer_key][0] for layer_key in layer_keys])

        resized_maps = []
        for layer_key in layer_keys:
            attn_type = layer_key[0]
            scale_factor = res_factor / self.attn_metadata[attn_type][layer_key][0]
            attn_map = getattr(self, f"{attn_type}_attention_maps")[layer_key]
            sampling_mode = downsampling_mode if scale_factor <  1 else upsampling_mode
            resized_map = self.rescale_attention(attn_map, scale_factor, mode=sampling_mode)
            resized_maps.append(resized_map)

        stacked_maps = torch.cat(resized_maps, dim=0)
        if aggregation_mode == "mean":
            aggregated_map = stacked_maps.mean(dim=0)
        elif aggregation_mode == "max":
            aggregated_map = stacked_maps.max(dim=0)[0]
        elif aggregation_mode == "sum":
            aggregated_map = stacked_maps.sum(dim=0)
        else:
            raise ValueError(f"Invalid aggregation mode: {aggregation_mode}. Choose 'mean', 'max', or 'sum.")

        return aggregated_map
      

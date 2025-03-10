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
    
    def aggregate_attention(self, resolution, pooling_method):
        """
        Aggregates attention maps from AttentionStore, using (branch_type, level) keys,
        and resizing based on stored res_factors.
        """
        grouped_maps = defaultdict(list)
        min_resolution = float('inf')  # Track smallest resolution if not specified

        # Group attention maps by (branch_type, level)
        for layer_key, maps in self.attn_store.items():
            base_layer = "-".join(layer_key.split("-")[:-1])  # Extract base key
            for attn_map, res_factor in maps:
                grouped_maps[base_layer].append((attn_map, res_factor))
                min_resolution = min(min_resolution, attn_map.shape[1])  # Keep track of min res

        # Determine final resolution
        if resolution is None:
            resolution = min_resolution

        # Aggregate maps
        aggregated_maps = {}
        for group, maps in grouped_maps.items():
            resized_maps = []
            
            for attn_map, res_factor in maps:
                # Compute effective size
                target_size = (resolution, resolution)

                # Resize attention maps
                resized_map = F.interpolate(
                    attn_map.permute(0, 3, 1, 2),  # Convert [B, H, W, T] -> [B, T, H, W]
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                ).permute(0, 2, 3, 1)  # Convert back to [B, H, W, T]

                resized_maps.append(resized_map)

            # Stack and pool
            stacked_maps = torch.cat(resized_maps, dim=0)  # Stack across batch
            if pooling_method == "max":
                aggregated_maps[group] = stacked_maps.max(dim=0).values
            else:  # Default to sum pooling
                aggregated_maps[group] = stacked_maps.sum(dim=0)

        return aggregated_maps

    def query_attention(self, timestep=None, block_type=None, level=None, instance=None, is_cross=True):
        """
        Retrieve stored attention maps using structured filtering.
        """
        # Build a dictionary of filters, removing None values for flexibility
        filters = {k: v for k, v in {
            "timestep": timestep,
            "is_cross": is_cross,
            "block_type": block_type,
            "level": level,
            "instance": instance
        }.items() if v is not None}

        # Search and filter based on the provided keys
        return [
            attn_map for key, attn_list in self.attention_maps.items()
            if all(filters[k] == key[i] for i, k in enumerate(filters))
            for attn_map in attn_list  # Flatten list of lists into a single list
        ]

    # Query the list of attentino maps
    def get_average_attention(self, block_type=None, attention_type=None):
        """
        Compute the average attention over stored steps.
        """
        attn_maps = self.query_attention(block_type=block_type, attention_type=attention_type)
        if not attn_maps:
            raise ValueError("No attention maps found for given filters.")

        stacked_maps = torch.stack(sum(attn_maps, []))  # Flatten list of lists and stack
        return stacked_maps.mean(dim=0)  # Compute mean attention map
    

    # Visualise cross attention maps
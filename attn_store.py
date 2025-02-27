import numpy as np
import torch
from PIL import Image

class AttentionStore:
    def __init__(self, save_global_store=True):
        """
        Initializes an AttentionStore that tracks attention maps with structured keys.
        """
        self.attention_maps = {}

    def reset(self):
        """
        Reset attention storage.
        """
        self.attention_maps = {}

    def store_attention(self, attention_probs, timestep, is_cross, block_type, level, instance):
        """
        Store attention scores using a dictionary-based key format.
        """
        key = (timestep, is_cross, block_type, level, instance)
        self.attention_maps[key].append(attention_probs.detach().cpu())

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

    # Visualise self attention maps

    # aggregate the attention maps

    #
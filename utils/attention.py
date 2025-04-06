from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.utils import deprecate
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

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
        self.layer_metadata = { "cross": {}, "self": {} }


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
        return {layer_key: [] for layer_key in self.layer_metadata[attn_type].keys()}


    def print_attention_metadata(self):
        """
        Print formatted metadata for stored attention layers.
        """
        # Print Cross-Attention Metadata
        print(f"Total Cross-Attention Layers: {len(self.layer_metadata['cross'])}")
        print("\nCross-Attention Layers:")
        for layer_key, (res_factor, name) in self.layer_metadata["cross"].items():
            print(f"Layer Key: {layer_key}, Resolution Downsampling Factor: {res_factor}, Module Name: {name}")

        # Print Self-Attention Metadata
        print(f"Total Self-Attention Layers: {len(self.layer_metadata['self'])}")
        print("\nSelf-Attention Layers:")
        for layer_key, (res_factor, name) in self.layer_metadata["self"].items():
            print(f"Layer Key: {layer_key}, Resolution Downsampling Factor: {res_factor}, Module Name: {name}")
    

    def store(self, attn_probs, layer_key, img_height, img_width):
        """
        Store attention scores using a dictionary-based key format.
        """
        attn_type = layer_key[0]
        attn_probs = attn_probs.clone().detach()
        if attn_type == "cross":
            attn_features = self.reduce_token_dimension(attn_probs)
        else:
            attn_pca = self.reduce_dimensionality_pca(attn_probs)
            attn_given = attn_probs.mean(dim=-1).unsqueeze(-1)    # [B, seq_len, 1] — how each token gives attention
            attn_received = attn_probs.mean(dim=-2).unsqueeze(-1) # [B, seq_len, 1] — how each token receives attention
            attn_features = torch.cat([attn_pca, attn_given, attn_received], dim=-1)
        
        attn_map = self.reshape_attention(attn_features, img_height, img_width)
        getattr(self, f"{attn_type}_attention_maps")[layer_key].append(attn_map.cpu())


    def reshape_attention(self, attn_probs, img_height, img_width):
        """
        Reshape attention to spatial dimensions
        """
        batch_size, seq_len, _ = attn_probs.shape
        downsample_factor = img_height * img_width / seq_len
        if (img_height % downsample_factor != 0) or (img_width % downsample_factor != 0):
            raise ValueError(f"Downsampling produced non-integer dimensions.")
        map_height, map_width = img_height // downsample_factor, img_width // downsample_factor
        attn_map = attn_probs.reshape(batch_size, map_height, map_width, -1)

        return attn_map
    
    def reduce_token_dimension(self, attn_probs, eot_idx=1, sum_padding=True):
        """
        Reduces the num_tokens dimension by:
        """
        prompt_tokens = attn_probs[:, :, :eot_idx]  
        summed_padding = attn_probs[:, :, eot_idx:].sum(dim=-1, keepdim=True)  
        special_token_probs = torch.cat([prompt_tokens, summed_padding], dim=-1)

        return special_token_probs

    def reduce_dimensionality_pca(self, attn_probs, n_components=3):
        """
        Applies PCA to self-attention maps 
        """
        batch_size, seq_len, _ = attn_probs.shape  # Get dimensions
        pca_reduced = []
        for i in range(batch_size):
            try:
                # ✅ Apply PCA safely (with error handling)
                U, S, V = torch.pca_lowrank(attn_probs[i], q=n_components)
                reduced_map = attn_probs[i] @ V  # Project to new PCA basis
            except RuntimeError as e:
                print(f"⚠️ PCA failed for sample {i}: {e}")
                reduced_map = attn_probs[i][..., :n_components]  # Fallback to slicing

            pca_reduced.append(reduced_map)

        return torch.stack(pca_reduced, dim=0)
    

    def group_attention_layers(self, attn_type, group_by_level=True):
        """
        Filters keys for attention maps based on place in unet (down, mid, up) and level.
        """
        grouped_layers = defaultdict(list)
        for layer_key in self.layer_metadata[attn_type].keys():
            group_key = layer_key[:3] if group_by_level else layer_key[:2]
            grouped_layers[group_key].append(layer_key)

        return grouped_layers


    def rescale_attention(self, attn_map, scale_factor):
        """
        Up/downsample to change resolution of attention maps
        """
        if scale_factor == 1:
            return attn_map
        else:
            height, width = attn_map.shape[1:3]
            target_res = int(height * scale_factor), int(width * scale_factor)
            attn_map = attn_map.permute(0, 3, 1, 2) # Permute to (B, C, H, W) 
            rescaled_map = F.interpolate(attn_map, size=target_res, mode="bilinear")

        return rescaled_map.permute(0, 2, 3, 1)
    

    def aggregate_attention(self, layer_keys, res_factor=None, aggregation_mode="max"):
        """
        Aggregates the attention across subset of layers at the specified resolution factor.
        """
        if res_factor == None:
            res_factor = min([self.layer_metadata[layer_key[0]][layer_key][0] for layer_key in layer_keys])

        resized_maps = []
        for layer_key in layer_keys:
            attn_type = layer_key[0]
            scale_factor = res_factor / self.layer_metadata[attn_type][layer_key][0]
            attn_map = getattr(self, f"{attn_type}_attention_maps")[layer_key]
            resized_map = self.rescale_attention(attn_map, scale_factor)
            resized_maps.append(resized_map)

        stacked_maps = torch.cat(resized_maps, dim=0)
        if aggregation_mode == "mean":
            aggregated_map = stacked_maps.mean(dim=0)
        elif aggregation_mode == "max":
            aggregated_map = stacked_maps.max(dim=0)[0]
        else:
            raise ValueError(f"Invalid aggregation mode: {aggregation_mode}. Choose 'mean' or 'max'")

        return aggregated_map
    

class MyCustomAttnProcessor(AttnProcessor2_0):
    """
    Copied heavily from https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/attention_processor.py
    """
    def __init__(self, attnstore, layer_key, img_height, img_width):
        super().__init__()
        self.attnstore = attnstore
        self.layer_key = layer_key
        self.img_height = img_height
        self.img_width = img_width
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        #### CUSTOM LOGIC ######
        print(key.shape)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        self.attnstore.store(attention_probs, 
                             self.layer_key, 
                             self.img_height, 
                             self.img_width)

        ## INJECT HERE IF NECESSARY
        # ########################

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    # def __init__(self,
    #              scheduler: object = None,
    #              injection_steps: int = None,
    #              token_injection_tensors: IndexTensorPair = None,
    #              attn_store: dict = None,
    #              nu: float = 0.0) -> None:
        
    #     self.injection_steps = injection_steps
    #     self.nu = nu
    #     self.scheduler = scheduler
    #     self.token_injection_tensors = token_injection_tensors
    #     self.attn_store = attn_store

    # def get_injection_scale(self):
    #     return self.nu * np.log(1 + self.scheduler.sigmas[self.scheduler.step_index].cpu().numpy())
    
    # def resize_injection_tensors(self, attention_dim):
    #     resize_factor = int(64 // np.sqrt(attention_dim))
    #     token_injection_tensors = copy.deepcopy(self.token_injection_tensors) 

    #     for token_injection_tensor in token_injection_tensors:
            
    #         token_injection_tensor.tensor = token_injection_tensor.tensor[0::resize_factor, 0::resize_factor]           
            
    #     return token_injection_tensors
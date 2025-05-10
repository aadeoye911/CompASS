from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from collections import defaultdict
from composition import generate_grid

class AttentionStore:
    def __init__(self, latent_height, latent_width, eot_tensor, device, save_maps=True):
        """
        Initializes an AttentionStore that tracks attention maps with structured keys.
        """
        self.save_maps = save_maps
        self.layer_metadata = {}
        self.resolutions = {}      # resolution tracking per layer
        self.device = device
         
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.eot_tensor = eot_tensor

        self.attention_maps = defaultdict(list)
        self.centroids = defaultdict(list)
        self.grid_cache = {}
        
        self.initialized = False
    
    def get_empty_store(self):
        return {layer_key: [] for layer_key in self.layer_metadata.keys()}
    
    def register_keys(self):
        """
        Called once after layer keys are known
        """
        self.attention_maps = self.get_empty_store()
        self.resolutions = {k: None for k in self.layer_metadata.keys()}
        self.initialized = True
   
    def reset(self):
        self.resolutions = {k: None for k in self.layer_metadata}
        self.attention_maps = self.get_empty_store()
        self.centroids = self.get_empty_store
        self.grid_cache = {}
        
    def get_meshgrid(self, batch_size, H, W, flatten=True):
        if (H, W) not in self.grid_cache:
            grid = generate_grid(H, W, centered=True, grid_aspect="equal") # Shape [H, W, 2]
            grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, W, 2]
            if flatten:
                grid = grid.reshape(batch_size, -1, 2) #[B, seq_len, 2]
            self.grid_cache[(H, W)] = grid.to(self.device)
    
    def __call__(self, attn_probs, layer_key):
        if not self.initialized:
            raise RuntimeError("AttentionStore not initialized.")

        # Store resolution and initialise meshgrid for centroid computations if unknown
        if self.resolutions[layer_key] is None:
            batch_size, seq_len, _ = attn_probs.shape
            H, W = seq_len_to_spatial_dims(seq_len, self.latent_height, self.latent_width)
            self.resolutions[layer_key] = (H, W)
            self.get_meshgrid(batch_size, H, W)
        
        padding = aggregate_padding_tokens(attn_probs, self.eot_tensor, self.device)
        padding = normalize(padding)
        grid = self.grid_cache[self.resolutions[layer_key]]
        step_centroids = (padding * grid).sum(dim=1) / padding.sum(dim=1)
        self.centroids[layer_key].append(step_centroids)

        if self.save_maps:
            with torch.no_grad():
                self.attention_maps[layer_key].append(padding.detach())
        
"""
Adapted from https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/attention_processor.py
"""
class MyCustomAttnProcessor(AttnProcessor2_0):
    def __init__(self, attention_store, layer_key):
        super().__init__()
        self.store = attention_store
        self.layer_key = layer_key
        
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

        ################ CUSTOM LOGIC ########################################
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.store(attention_probs, self.layer_key)
        ######################################################################

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
    
def aggregate_padding_tokens(attn_probs, eot_tensor, device):
    batch_size, seq_len, num_tokens = attn_probs.shape
    # Apply mask to isolate EoT paddings along token dimension
    token_range = torch.arange(num_tokens, device=device).unsqueeze(0) # shape: [1, num_tokens]
    token_mask = token_range >= eot_tensor.unsqueeze(1)  # shape: [batch_size, num_tokens]
    token_mask = token_mask.unsqueeze(1)
    masked_attn = attn_probs * token_mask.float()

    return masked_attn.sum(dim=-1, keepdim=True)  # shape: [batch_size, seq_len, 1]

def apply_pca_reduction(attn_probs, n_components=3):
    batch_size, seq_len, _ = attn_probs.shape
    pca_reduced = []
    for i in range(batch_size):
        _, _, V = torch.pca_lowrank(attn_probs[i], q=n_components)
        # Project to new PCA basis
        reduced_map = attn_probs[i] @ V  
        pca_reduced.append(reduced_map)

    return torch.stack(pca_reduced, dim=0)

def seq_len_to_spatial_dims(seq_len, ref_height, ref_width):
    scale_factor = np.sqrt(ref_height * ref_width // seq_len)
    return int(ref_height // scale_factor), int(ref_width // scale_factor)

def normalize(attn_probs):
    """
    Min-max normalize each [seq_len, 1] map independently per batch item.
    """
    min_vals = attn_probs.min(dim=1, keepdim=True)[0]  # [B, 1, 1]
    max_vals = attn_probs.max(dim=1, keepdim=True)[0]  # [B, 1, 1]

    return (attn_probs - min_vals) / (max_vals - min_vals) #[B, seq_len, 1]

def reshape_attention(attn_probs, ref_height, ref_width):
    batch_size, seq_len, _ = attn_probs.shape
    height, width = seq_len_to_spatial_dims(seq_len, ref_height, ref_width)
    return attn_probs.reshape(batch_size, height, width, -1)

def rescale_attention(attn_map, resolution=16):
    height, width = attn_map.shape[1:3] # shape: [B, H, W, C]
    min_dim = min(height, width)
    if min_dim == resolution:
        return attn_map

    scale_factor = resolution / min_dim
    target_dims = int(height * scale_factor), int(width * scale_factor)
    attn_map = attn_map.permute(0, 3, 1, 2) # Permute to (B, C, H, W) 
    rescaled_map = F.interpolate(attn_map, size=target_dims, mode="bilinear")

    return rescaled_map.permute(0, 2, 3, 1)
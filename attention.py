from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from collections import defaultdict
from utils.attn_utils import aggregate_padding_tokens

class AttentionStore:
    def __init__(self, save_global_store=True):
        """
        Initializes an AttentionStore that tracks attention maps with structured keys.
        """
        self.layer_metadata = {}
        self.resolutions = {}      # resolution tracking per layer
        self.save_global_store = save_global_store
        self.attention_store = {}
        self.global_store = {}
        
        self.centroids = []
        self.initialized = False
        
    def register_keys(self):
        """
        Called once after layer keys are known
        """
        self.step_store = self.get_empty_store()
        self.attention_store = self.get_empty_store()
        self.resolutions = {k: None for k in self.layer_metadata.keys()}
        self.initialized = True

    def reset(self):
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def get_empty_store(self):
        return {layer_key: [] for layer_key in self.layer_metadata.keys()}
    
    def __call__(self, attn_probs, layer_key):
        if not self.initialized:
            raise RuntimeError("AttentionStore not initialized.")

        # Set resolution if it's not recorded yet
        batch_size, seq_len, num_tokens = attn_probs.shape
        if self.resolutions[layer_key] is None:
            self.resolutions[layer_key] = attention_map.shape[1:3]  # e.g., (height, width)

        # Store the attention map
        self.step_store[layer_key].append(attention_map)
        self.attention_store[layer_key].append(attention_map.cpu())

    def store(self, attn_probs, layer_key):
        """
        Store attention scores using a dictionary-based key format.
        """
        # attn_probs = attn_probs.clone().detach()
        
        eot_indices = torch.ones(batch_size)  # [B, seq_len, num_tokens]
        eot_maps = aggregate_padding_tokens(attn_probs, eot_indices)
        
        self.attention_store[layer_key].append(eot_maps.cpu())
    
"""
Adapted from https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/attention_processor.py
"""
class MyCustomAttnProcessor(AttnProcessor2_0):
    def __init__(self, attnstore, layer_key):
        super().__init__()
        self.attnstore = attnstore
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
        self.attnstore.store(attention_probs, self.layer_key)
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
import torch
import torch.nn.functional as F

def aggregate_padding_tokens(attn_probs, eot_indices):
    _, _, num_tokens = attn_probs.shape
    device = attn_probs.device

    token_range = torch.arange(num_tokens, device=device).unsqueeze(0) # [1, num_tokens]
    token_mask = token_range >= eot_indices.unsqueeze(1)  # [B, num_tokens]

    # Apply mask (convert bool to float for multiplication)
    token_mask = token_mask.unsqueeze(1)
    masked_attn = attn_probs * token_mask.float()

    return masked_attn.sum(dim=-1, keepdim=True)  # → [B, seq_len, 1]

def apply_pca_reduction(attn_probs, n_components=3):
    batch_size, seq_len, _ = attn_probs.shape  # Get dimensions
    pca_reduced = []
    for i in range(batch_size):
        _, _, V = torch.pca_lowrank(attn_probs[i], q=n_components)
        reduced_map = attn_probs[i] @ V  # Project to new PCA basis
        pca_reduced.append(reduced_map)

    return torch.stack(pca_reduced, dim=0)

def reshape_attention(attn_probs, img_height, img_width):
    batch_size, seq_len, _ = attn_probs.shape
    downsample_factor = img_height * img_width / seq_len
    if (img_height % downsample_factor != 0) or (img_width % downsample_factor != 0):
        raise ValueError(f"Downsampling produced non-integer dimensions.")
    map_height, map_width = img_height // downsample_factor, img_width // downsample_factor
    attn_map = attn_probs.reshape(batch_size, map_height, map_width, -1)

    return attn_map

def rescale_attention(attn_map, resolution=16):
    height, width = attn_map.shape[1:3]
    min_dim = min(height, width)
    if min_dim == resolution:
        return attn_map

    scale_factor = resolution / min_dim
    target_dims = int(height * scale_factor), int(width * scale_factor)
    attn_map = attn_map.permute(0, 3, 1, 2) # Permute to (B, C, H, W) 
    rescaled_map = F.interpolate(attn_map, size=target_dims, mode="bilinear")

    return rescaled_map.permute(0, 2, 3, 1)
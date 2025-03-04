import matplotlib as plt

def visualize_latents(sampler):
    T = len(sampler.decoded_images)
    cols = min(10, T)  # Max 10 columns
    rows = (T + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Plot images in grid
    for i, ax in enumerate(axes.flatten()):
        if i < len(sampler.decoded_images):
            ax.imshow(sampler.decoded_images[i])
            ax.set_title(f"$z_{{{T -i}}}$", fontsize=12)
        ax.axis("off")

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()

def visualize_cross_attention(sampler, token_index, aggregation=False, group_by_level=False, cumulative_timestep=False, timestep_aggregation="max"):
    """
    Visualize cross-attention maps for a specific token index with optional aggregation.

    Rows: Timesteps (or cumulative if cumulative_timestep=True)
    Columns: Layers grouped by block type and level (if group_by_level=True)
    """

    # Reorder layers based on 'down', 'mid', 'up' order
    def order_layers(layers):
        def layer_sort_key(name):
            block_type, level, sub_level = sampler.parse_layer_name(name)
            return ({"down": 0, "mid": 1, "up": 2}[block_type], level, sub_level)
        return sorted(layers, key=layer_sort_key)

    ordered_layers = order_layers(sampler.cross_attention_maps.keys())

    # Group layers by block type & level if enabled
    if group_by_level:
        grouped_layers = {}
        for layer in ordered_layers:
            block_type, level, _ = sampler.parse_layer_name(layer)
            key = f"{block_type}[{level}]" if block_type != "mid" else "mid"

            if key not in grouped_layers:
                grouped_layers[key] = []
            grouped_layers[key].append(layer)

        ordered_layers = list(grouped_layers.keys())  # Replace with grouped keys

    # Determine number of rows (timesteps) and columns (grouped layers or individual layers)
    max_timesteps = max(len(sampler.cross_attention_maps[layer]) for layer in sampler.cross_attention_maps)
    num_columns = len(ordered_layers)

    # Create a figure
    fig, axes = plt.subplots(
        nrows=max_timesteps,
        ncols=num_columns,
        figsize=(num_columns * 2, max_timesteps * 2),
        squeeze=False
    )

    # Store cumulative maps if needed
    cumulative_maps = {}

    for col, layer_group in enumerate(ordered_layers):
        for row in range(max_timesteps):
            ax = axes[row, col]
            ax.axis("off")

            # Retrieve and aggregate attention maps
            if group_by_level:
                attn_maps = []
                for sub_layer in grouped_layers[layer_group]:
                    if row < len(sampler.cross_attention_maps[sub_layer]):
                        attn_maps.append(sampler.cross_attention_maps[sub_layer][row])

                if attn_maps:
                    attn_stack = torch.stack(attn_maps)
                    attn_map = attn_stack.max(dim=0).values  # Aggregate sublevels using max
                else:
                    continue  # Skip if no valid attention maps
            else:
                attn_map = sampler.cross_attention_maps[layer_group][row]

            attn_map = attn_map.squeeze(0)  # Remove batch dimension
            if aggregation:
                processed_attn = attn_map[:, token_index:].max(dim=1).values

            # Cumulative aggregation over timesteps
            if cumulative_timestep:
                if layer_group not in cumulative_maps:
                    cumulative_maps[layer_group] = []

                cumulative_maps[layer_group].append(processed_attn)

                if timestep_aggregation == "max":
                    processed_attn = torch.stack(cumulative_maps[layer_group]).mean(dim=0)  # Mean over timesteps
                # elif timestep_aggregation == "max":
                #     processed_attn = torch.stack(cumulative_maps[layer_group]).max(dim=0).values  # Max over timesteps
                else:
                    raise ValueError("Invalid timestep_aggregation method. Use 'mean' or 'max'.")

            square_attn_map = reshape_attention_map(processed_attn, sampler)

            # Choose color map
            if aggregation is None:
                cmap = "Greens"
            elif token_index == 0:
                cmap = "Blues"
            else:
                cmap = "Reds"

            ax.imshow(square_attn_map, cmap)
            if row == 0:
                ax.set_title(f"{layer_group}", fontsize=8, rotation=90, pad=10)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def reshape_attention_map(processed_attn, sampler):

    latent_shape = sampler.latent_history[-1].shape
    latent_height, latent_width = latent_shape[-2], latent_shape[-1]

    attn_size = processed_attn.shape[0]
    latent_size = latent_height * latent_width

    res_factor = np.sqrt(attn_size / latent_size)
    attn_height = int(latent_height * res_factor)
    attn_width = int(latent_width * res_factor)

    reshaped_attn = processed_attn[:attn_height * attn_width].reshape(1, 1, attn_height, attn_width)

    # resized_attn = F.interpolate(reshaped_attn, size=(latent_height, latent_width), mode="bilinear", align_corners=False)
    return reshaped_attn.squeeze().cpu().numpy()

def visualize_self_attention(attention_maps, top_k=3, downsample_size=64, max_timesteps=10):
    """
    Visualize self-attention maps using SVD to extract dominant components.

    Args:
        attention_maps: Dict[layer_name, List[attention_map]] - Self-attention maps.
        top_k: int - Number of singular values/components to use for reconstruction.
        downsample_size: int - Size to downsample for visualization.
        max_timesteps: int - Number of timesteps to visualize (rows).
    """
    layers = list(attention_maps.keys())  # Layer names
    num_layers = len(layers)
    num_timesteps = min(len(next(iter(attention_maps.values()))), max_timesteps)

    fig, axes = plt.subplots(num_timesteps, num_layers, figsize=(num_layers * 3, num_timesteps * 3))
    fig.suptitle(f"Self-Attention Maps (SVD Top-{top_k} Approximation)", fontsize=16)

    for t in range(num_timesteps):
        for l, layer_name in enumerate(layers):
            attn_map = attention_maps[layer_name][t]  # Shape: [1, query_len, query_len]
            attn_map = attn_map.squeeze(0).cpu()  # Remove batch dim -> Shape: [query_len, query_len]

            # Apply SVD
            U, S, Vh = torch.linalg.svd(attn_map)
            S_k = torch.diag(S[:top_k])  # Top-k singular values
            U_k = U[:, :top_k]  # Top-k left singular vectors
            Vh_k = Vh[:top_k, :]  # Top-k right singular vectors

            # Low-rank reconstruction
            attn_reconstructed = U_k @ S_k @ Vh_k

            # Downsample for visualization
            attn_resized = torch.nn.functional.interpolate(
                attn_reconstructed.unsqueeze(0).unsqueeze(0),  # Add batch & channel dims
                size=(downsample_size, downsample_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0).numpy()

            # Plot heatmap
            ax = axes[t, l] if num_timesteps > 1 else axes[l]
            ax.imshow(attn_resized, cmap="viridis", aspect="auto")
            ax.set_title(f"Layer: {layer_name}\nTimestep: {t}")
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
import matplotlib as plt

def view_images(input_image):
    plt.figure(figsize=(6, 3))  # Adjust overall figure size
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis('off')

    plt.show()

# def view_images(input_image, diffused_image)
#     plt.figure(figsize=(6, 3))  # Adjust overall figure size

#     # # Original Image
#     plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
#     plt.imshow(input_image)
#     plt.title("Input Image")
#     plt.axis('off')

#     # # Resized Image
#     # plt.subplot(1, 2, 2)  # 1 row, 2 columns, position 2
#     # plt.imshow(test_resized)
#     # plt.title("Resized Image")
#     # plt.axis('off')

#     # plt.show()


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
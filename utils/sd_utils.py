import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union
import matplotlib.pyplot as plt


def resize_image(image, min_dim=512, factor=64):
    """
    Resize PIL image dimensions to multiples of 'factor'.
    """
    width, height = image.size
    min_dim = round_to_multiple(min_dim, factor, mode="up")
    scale = min_dim / min(width, height)
    if scale == 1 and width % factor == 0 and height % factor == 0:
        return image # Return image as is if dimensions are already correct
    
    new_width = round_to_multiple(width * scale, factor)
    new_height = round_to_multiple(height * scale, factor)

    return image.resize((new_width, new_height), Image.LANCZOS)

def round_to_multiple(value, factor, mode="nearest"):
    """
    Round a given value to the nearest multiple of `factor`.
    """
    if mode == "up":
        return int((value + factor - 1) // factor * factor)
    elif mode == "down":
        return int(value // factor * factor)
    elif mode == "nearest":
        return int(round(value / factor) * factor)
    else:
        raise ValueError("mode must be 'up', 'down', or 'nearest'")

def generate_seeds(num_seeds=1):
    """
    Generate a list of random seeds.
    """
    return [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_seeds)]

def seed2generator(device, seed=42, batch_size=1):
    """
    Generate list of generators from random seeds.
    """
    if isinstance(seed, int):
        seed = [seed] * batch_size  # Duplicate same seed for all batch elements
    elif seed is None:
        seed = generate_seeds(batch_size)  # Generate random seeds
    elif not isinstance(seed, list):
        raise TypeError(f"`seed` must be an int, list of ints, or None, but got {type(seed)}")

    if len(seed) != batch_size:
        raise ValueError(f"Seed list length ({len(seed)}) does not match batch size ({batch_size}).")

    return [torch.Generator(device=device).manual_seed(s) for s in seed]

def token2idx(tokenizer, prompts, eot_only=True):
    """
    Retrives EoT token indices for prompts or dictionary of prompt mappings (SoT, EoT, and words) --> indices.
    """
    if isinstance(prompts, str):
        prompts = [prompts]  # Convert to list for uniform processing

    encodings = tokenizer(prompts, add_special_tokens=True)
    if eot_only:
        return [{"eot": len(tokens) - 1} for tokens in encodings["input_ids"]]

    token_indices = []
    tokenized_prompts = [tokenizer.convert_ids_to_tokens(token_ids) for token_ids in encodings["input_ids"]]
    for tokens in tokenized_prompts:
        prompt_map = {"sot": [0]}
        prompt_map["eot"] = [len(tokens) - 1]
        for i in range(1, len(tokens) - 1):
            word = tokens[i].replace("##", "").replace("</w>", "")  # Remove BPE subword markers
            prompt_map.setdefault(word, []).append(i)

        token_indices.append(prompt_map)

    return token_indices

def extract_attention_metadata(module_name):
    """
    Parse attention module name to determine its place-in-Unet ("down", "up", "mid"), level, and instance.
    """
    try:
        place_in_unet = module_name.split("_")[0]
        if place_in_unet not in {"down", "up", "mid"}:
            raise ValueError(f"Invalid place-in-Unet: {place_in_unet} in layer '{module_name}'")
        numbers = [int(s) for s in module_name if s.isdigit()]
        level = numbers[0] if len(numbers) > 0 else -1
        instance = numbers[1] if len(numbers) > 1 else -1

        return place_in_unet, level, instance
    except Exception as e:
        raise ValueError(f"Failed to parse module name '{module_name}': {e}")
    
# def visualize_latents(sampler):
#     T = len(sampler.decoded_images)
#     cols = min(10, T)  # Max 10 columns
#     rows = (T + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

#     # Plot images in grid
#     for i, ax in enumerate(axes.flatten()):
#         if i < len(sampler.decoded_images):
#             ax.imshow(sampler.decoded_images[i])
#             ax.set_title(f"$z_{{{T -i}}}$", fontsize=12)
#         ax.axis("off")

#     # Tight layout for better spacing
#     plt.tight_layout()
#     plt.show()
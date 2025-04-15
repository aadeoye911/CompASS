import torch
from PIL import Image
import matplotlib.pyplot as plt

def scale_resolution_to_multiple(height, width, factor, target_dim=None):
    scale = 1 if target_dim is None else target_dim / min(height, width)
    height, width = (int(round(x * scale / factor) * factor) for x in (height, width))
    return height, width

def resize_image(image, factor=64, target_dim=512):
    """
    Resize PIL image dimensions to model compatible dimensions
    """
    width, height = image.size
    new_width, new_height = scale_resolution_to_multiple(width, height, factor, target_dim=target_dim)
    if new_width == width and new_height == height:
        return image # Return image as is if dimensions are already correct
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def generate_seeds(num_seeds=1):
    return [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_seeds)]

def seed2generator(device, seed=42, batch_size=1):
    if isinstance(seed, int):
        seed = [seed] * batch_size  # Duplicate same seed for all batch elements
    elif seed is None:
        seed = generate_seeds(batch_size)  # Generate random seeds
    elif not isinstance(seed, list):
        raise TypeError(f"`seed` must be an int, list of ints, or None, but got {type(seed)}")

    if len(seed) != batch_size:
        raise ValueError(f"Seed list length ({len(seed)}) does not match batch size ({batch_size}).")

    return [torch.Generator(device=device).manual_seed(s) for s in seed]

def prompt2idx(tokenizer, prompts, eot_only=True):
    if isinstance(prompts, str):
        prompts = [prompts]  # Convert to list for uniform processing

    encodings = tokenizer(prompts, add_special_tokens=True)
    if eot_only:
        return [len(tokens) - 1 for tokens in encodings.input_ids]

    token_indices = []
    tokenized_prompts = [tokenizer.convert_ids_to_tokens(token_id) for token_id in encodings.input_ids]
    for tokens in tokenized_prompts:
        prompt_map = {"sot": [0], "eot": [len(tokens) - 1]}
        for i in range(1, len(tokens) - 1):
            word = tokens[i].replace("##", "").replace("</w>", "")  # Remove BPE subword markers
            prompt_map.setdefault(word, []).append(i)

        token_indices.append(prompt_map)

    return token_indices

def parse_module_name(name):
    try:
        place_in_unet = name.split("_")[0]
        if place_in_unet not in {"down", "up", "mid"}:
            raise ValueError(f"Invalid place-in-unet: {place_in_unet} in layer '{name}'")
        numbers = [int(s) for s in name if s.isdigit()]
        level = numbers[0] if len(numbers) > 0 else -1
        instance = numbers[1] if len(numbers) > 1 else -1

        return place_in_unet, level, instance
    except Exception as e:
        raise ValueError(f"Failed to parse module name '{name}': {e}")
import torch
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, Compose
from typing import Any, Callable, Dict, List, Optional, Union

def resize_image(image, target_dim=512, factor=64):
    """
    Resize PIL image dimensions to multiples of 'factor'.
    """
    width, height = image.size
    min_dim = round_to_multiple(target_dim, factor, mode="up")
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


def preprocess_image(image, dtype=torch.float32, min_dim=512, factor=64):
    """
    Convert PIL image into a torch.Tensor with model-compatible dimensions.
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = resize_image(image, min_dim, factor)
    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

    return transform(image).unsqueeze(0).to(dtype)


def generate_seeds(num_seeds: int = 1):
    """
    Generate a list of random seeds.
    """
    if not isinstance(num_seeds, int) or num_seeds < 1:
        raise ValueError(f"`num_seeds` must be a positive integer, but got {num_seeds}.")

    return [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_seeds)]


def seed2generator(device, seed=None, batch_size=1):
    """
    Generate list of generators from random seeds.
    """
    if isinstance(seed, int):
        seed = [seed] * batch_size  # Duplicate same seed for all batch elements
    elif seed is None:
        seed = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(batch_size)]  # Generate random seeds
    elif not isinstance(seed, list):
        raise TypeError(f"`seed` must be an int, list of ints, or None, but got {type(seed)}")

    if len(seed) != batch_size:
        raise ValueError(f"Seed list length ({len(seed)}) does not match batch size ({batch_size}).")

    return [torch.Generator(device=device).manual_seed(s) for s in seed]


def init_latent(batch_size, num_channels, height, width, generator=None, dtype=torch.float32):
    """
    Generate random noise latent tensor for Stable Diffusion.
    """
    if isinstance(generator, list):
        if len(generator) > batch_size:
            print(f'generator longer than batch size. truncationg list to match batch')
            generator = generator[:batch_size]

    return torch.randn((batch_size, num_channels, height, width), generator=generator, dtype=dtype)

# Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "mode":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "mean":
        return encoder_output.latent_dist.mean()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
        
def get_token_indices(tokenizer, prompts, eot_only=True):
    """
    Retrives EoT token index for prompts or dictionary of word-to-token index maps.
    """
    if isinstance(prompts, str):
        prompts = [prompts]  # Convert to list for uniform processing

    encodings = tokenizer(prompts, add_special_tokens=True)
    if eot_only:
        return [{"eot": len(tokens) - 1} for tokens in encodings["input_ids"]]

    token_indices = []
    tokenized_prompts = [tokenizer.convert_ids_to_tokens(token_ids) for token_ids in encodings["input_ids"]]
    for tokens in tokenized_prompts:
        token_map = {"sot": [0]}
        token_map["eot"] = [len(tokens) - 1]
        for i in range(1, len(tokens) - 1):
            word = tokens[i].replace("##", "").replace("</w>", "")  # Remove BPE subword markers
            token_map.setdefault(word, []).append(i)
        
        token_indices.append(token_map)

    return token_indices


def parse_layer_name(layer_name):
    """
    Parse attention layer name to determine block type ("down", "up", "mid"), level, and instance.
    """
    try:
        block_type = layer_name.split("_")[0]
        if block_type not in {"down", "up", "mid"}:
            raise ValueError(f"Invalid block type: {block_type} in layer '{layer_name}'")
        numbers = [int(s) for s in layer_name if s.isdigit()]
        level = numbers[0] if len(numbers) > 0 else -1
        instance = numbers[1] if len(numbers) > 1 else -1

        return block_type, level, instance
    except Exception as e:
        raise ValueError(f"Failed to parse layer name '{layer_name}': {e}")
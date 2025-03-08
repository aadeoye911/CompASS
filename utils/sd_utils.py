import torch
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, Compose

def resize_image(image, min_dim=512, factor=64):
    """
    Resize PIL image to maintain aspect ratio, ensuring dimensions are multiples of 64.
    """
    width, height = image.size
    min_dim = round_to_multiple(min_dim, factor, mode="up")
    scale = min_dim / min(width, height)
    # Return image as-is if dimensions are already correct
    if scale == 1 and width % factor == 0 and height % factor == 0:
        return image 
    
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
                         

def preprocess_image(image, dtype=torch.float32):
    """
    Convert PIL image into a torch.Tensor.
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = resize_image(image)
    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

    return transform(image).unsqueeze(0).to(dtype)


def prepare_latents(self, image, batch_size, generator):
       
    image = image.repeat(batch_size, 1, 1, 1)
    latents = self.vae.encode(image).latent_dist.sample(generator=generator)
    latents = latents * self.vae.config.scaling_factor
    
    return latents


def init_latent(unet, generator, image=None, batch_size=1):
    """
    Initialize latent for Stable Diffusion.
    """
    if image is not None:
        width, height = image.size
    else:
        width, height = unet.
    latents = torch.randn(
        (batch_size, unet.in_channels, height, width),
        generator=generator,
        device=pipe.device,
        dtype=pipe.dtype
    ) * pipe.scheduler.init_noise_sigma

    return latents


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
    Parse the layer name to determine block type ("down", "up", "mid"), level, and instance.
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
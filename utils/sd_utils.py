import torch
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, Resize, Compose, ToPILImage

def resize_image(image, max_dim=512, factor=64):
    """
    Resize PIL image to maintain aspect ratio, ensuring dimensions are multiples of 64.
    """
    width, height = image.size
    scale = max_dim / max(width, height)
    width = int(round(width * scale / factor) * factor)
    height = int(round(height * scale / factor) * factor)

    return image.resize((width, height), Image.LANCZOS)

def image2latent(pipe, image):
    """
    Encode PIL image into SD latent space.
    """
    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
    image_tensor = transform(image).unsqueeze(0).to(pipe.device, dtype=pipe.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.mean

    return latents * pipe.vae.config.scaling_factor

def latent2image(pipe, latent):
    """
    Decode SD latents to PIL image.
    """
    with torch.no_grad():
        image_tensor = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
        image = (image_tensor / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

    return pipe.numpy_to_pil(image)[0]

def init_latent(pipe, generator, height, width, batch_size=1):
    """
    Initialize latent for Stable Diffusion.
    """
    latents = torch.randn(
        (batch_size, pipe.unet.in_channels, height, width),
        generator=generator,
        device=pipe.device,
        dtype=pipe.dtype
    ) * pipe.scheduler.init_noise_sigma

    return latents

def store_token_indices(pipe, prompt):
    """
    Store token indices for SoT (Start-of-Text), EoT (End-of-Text), and words.
    """
    token_ids = pipe.tokenizer.encode(prompt, add_special_tokens=True)
    tokenized_words = pipe.tokenizer.convert_ids_to_tokens(token_ids)

    word_indices = {"sot": [0]}  # SoT is always at index 0
    eot_positions = (torch.tensor(token_ids) == pipe.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    word_indices["eot"] = [eot_positions[0].item() if len(eot_positions) > 0 else len(token_ids) - 1]

    word_tracker = {}
    for token_idx, token in enumerate(tokenized_words):
        word = token.replace("##", "")
        if word not in word_indices:
            word_indices[word] = []
        word_indices[word].append(token_idx)

    return word_indices

def encode_prompt(pipe, prompt, batch_size=1):
    """
    Computes text embeddings for conditional and unconditional text (for CFG).
    """
    # Unconditional (empty prompt for CFG guidance)
    uncond_input = pipe.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    ).to(pipe.device)
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]

    # Encode actual prompt
    cond_input = pipe.tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(pipe.device)
    cond_embeddings = pipe.text_encoder(cond_input.input_ids)[0]

    return torch.cat([uncond_embeddings, cond_embeddings], dim=0)

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
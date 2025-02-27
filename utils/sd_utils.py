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

def get_token_indices(pipe, prompt):
    """
    Get token indices for SoT (Start-of-Text), EoT (End-of-Text), and words.
    """
    # Tokenize the prompt
    token_ids = pipe.tokenizer.encode(prompt, add_special_tokens=True)
    tokenized_words = pipe.tokenizer.convert_ids_to_tokens(token_ids)

    token_indices = {}
    for token_idx, token in enumerate(tokenized_words):
        # Clean tokens to regular text
        if token == pipe.tokenizer.bos_token:
            word = "sot"
        elif token == pipe.tokenizer.eos_token:
            word = "eot"
        else:
            word = token.replace("##", "").replace("</w>", "") # Remove BPE subword markers
        # Store token indices for prompt words and special tokens
        token_indices.setdefault(word, []).append(token_idx)

    return token_indices

def tokenize_prompt(pipe, prompt, max_length=None):
    """ 
    Tokenize a prompt or list of prompts, handling padding and truncation.
    """
    if max_length is None:
        max_length = pipe.tokenizer.model_max_length

    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,  # Detects if truncation occurs
    )
    if "overflowing_tokens" in text_inputs:
        truncated_tokens = pipe.tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0])
        last_word = truncated_tokens[-2]  # Get last token before truncation
        print(f"⚠️ Warning: Prompt was truncated at token {max_length}. Last word kept: '{last_word}'.")

    return text_inputs.input_ids.to(pipe.device)
    
def get_text_embeddings(pipe, prompt, batch_size=1):
    """
    Computes text embeddings for conditional and unconditional text (for CFG).
    """
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    # Unconditional (empty prompt for CFG guidance)
    uncond_input = tokenize_prompt(pipe, "")
    uncond_embeddings = pipe.text_encoder(uncond_input)[0]
    print(uncond_embeddings.shape)

    return uncond_embeddings

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
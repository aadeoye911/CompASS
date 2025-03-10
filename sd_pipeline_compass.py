import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision.transforms import Normalize, ToTensor, Compose
from attention import AttentionStore
from utils.sd_utils import resize_image, extract_attention_info, init_latent
from collections import defaultdict

class CompASSPipeline(StableDiffusionPipeline):
    """
    Initialize the CompASSPipeline, inheriting from StableDiffusionPipeline.
    """
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        # Load resolution defaults
        self.default_output_resolution = None # Default output resolution for model
        self.unet_depth = None
        self.total_downsample_factor = None # Total VAE+UNet downscaling
        self.get_resolution_defaults()

        # Additional components for attention tracking
        self.attn_store = AttentionStore()
        self.setup_hooks()
        
        self.diffused_images = []
        self.height = None
        self.width = None
        self.empty_embeds = None
        self.get_empty_embeddings()


    def get_resolution_defaults(self):
        """
        Compute and store default model parameters.
        """
        self.default_output_resolution = self.unet.config.sample_size * self.vae_scale_factor
        self.unet_depth = len(self.unet.config.block_out_channels)-1
        self.total_downsample_factor = 2**self.unet_depth * self.vae_scale_factor


    def setup_hooks(self):
        """
        Set up hooks on all cross-attention layers.
        """
        self.hooks = []
        res_factor = 0  # Initialize resolution factor
        max_factor = res_factor
        for name, module in self.unet.named_modules():

            # Place hook on cross attention modules
            if getattr(module, "is_cross_attention", False):
                place_in_unet, level, instance = extract_attention_info(name)
                layer_key = (place_in_unet, level, instance)
                self.attn_store.attn_metadata[layer_key] = (2**res_factor, name)
                self.hooks.append(module.register_forward_hook(self._hook_fn(layer_key)))

            # Track resolution through downsampling/upsampling modules
            if "sample" in name.split(".")[-1]:
                res_factor = res_factor + 1 if "down" in name else res_factor - 1
                max_factor = max(max_factor, res_factor)

        # Ensure consistency between up/down resolutions
        max_downblock_factor = max(value[0] for layer_key, value in self.attn_store.attn_metadata.items() if layer_key[0] == "down")
        max_upblock_factor = max(value[0] for layer_key, value in self.attn_store.attn_metadata.items() if layer_key[0] == "up")
        assert (max_upblock_factor == max_downblock_factor, "Mismatch between upblock and downblock resolution factors")
        assert (max_factor == self.unet_depth, "Invalid UNet depth calculation")
        
        # Reassign resolution factor for midblocks
        for layer_key in self.attn_store.attn_metadata:
            if layer_key[0] == "mid":
                _, name = self.attn_store.attn_metadata[layer_key]
                self.attn_store.attn_metadata[layer_key] = (2**max_factor, name)  # Reassign the tuple
  
        print(f"Number of hooks initialised: {len(self.hooks)}")

    
    def _hook_fn(self, layer_key):
        """
        Hook function to capture attention scores.
        """
        def hook(module, input, output):
            try:
                query = module.to_q(input[0])
                key = module.to_k(self.empty_embeddings)
                # key = module.to_k(self.text_embeddings.chunk(2, dim=0)[1] if is_cross else input[0])
                attn_probs = (module.get_attention_scores(query, key)).detach().cpu()
                self.attn_store.store(attn_probs, layer_key, self.height, self.width)
            except Exception as e:
                print(f"Error processing attention scores for layer {layer_key}: {e}")
        return hook
    

    def clear_hooks(self):
        """
        Remove all hooks from the model.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


    def reset(self):
        """
        Reset stored latents, images, and attention maps.
        """
        self.diffused_images.clear()
        self.attn_store.reset()


    def get_empty_embeddings(self, prompt="", batch_size=1):
        """
        Tokenize the prompt and get text embeddings.
        """
        self.empty_embeds = self.encode_prompt(prompt, self.device, batch_size, False)
        # Output is tuple
        print("Initiatilized empty embeddings")


    def preprocess_image(self, image, min_dim=None, factor=None):
        """
        Convert PIL image into a torch.Tensor with model-compatible dimensions.
        """
        min_dim = min_dim if min_dim is not None else self.default_output_resolution
        factor = factor if factor is not None else self.total_downsample_factor
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = resize_image(image, min_dim, factor)
        transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

        return transform(image).unsqueeze(0).to(self.dtype)
    

    def image2latent(self, image, timesteps, seed=42):
        """
        Prepare latents from an image or random noise.
        """
        if image.shape[0] != len(timesteps):
            if image.shape[1] == 1:
                image = image.repeat(len(timesteps), 1, 1, 1)
            else:
                raise ValueError(f"Image shape {image.shape} does not match timesteps {timesteps}")

        latents = self.vae.encode(image).latent_dist.mean * self.vae.config.scaling_factor
        
        # Set seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        batch_size, num_channels, height, width = latents.shape
        noise = init_latent(batch_size, num_channels, height, width, generator=generator, dtype=self.dtype)  # Generate noise deterministically
        
        # Apply timestep-dependent noise across batch channel
        latents = self.scheduler.add_noise(latents, noise, timesteps)


    def extract_reference_attn_maps(self, image, timesteps, seed=42):
        batch_size = len(timesteps)
        if image.shape[0] != batch_size:
            if image.shape[0] == 1:
                image = image.repeat(batch_size, 1, 1, 1)
            else:
                raise ValueError(f"Image shape {image.shape} does not match timesteps {batch_size}")
        image = image.to(self.device)

        if self.empty_embeds[0].shape[0] != batch_size:
            self.empty_embeds = (self.empty_embeds[0].repeat(batch_size, 1, 1), self.empty_embeds[1])

        latents = self.image2latent(image, timesteps, seed)
        with torch.no_grad():
            unet_output = self.pipe.unet(latents, timesteps, self.text_embeddings, return_dict=True)
            noise_pred = unet_output["sample"]
            latents = self.pipe.scheduler.step(noise_pred, timesteps, latents)["prev_sample"]
            torch.cuda.empty_cache()